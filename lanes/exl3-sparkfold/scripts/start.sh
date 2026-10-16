#!/usr/bin/env bash
# Qwen3.8-Flash-Next, EXL3 4.05bpw (turboderp/Qwen3.8-Flash-Next-exl3, branch 4.05bpw_h6_ng6) on TensorFold 0.6.5 + SparkFold + the
# patches in ../patches, in a container: the image supplies PyTorch and the dependencies, the patched source tree is mounted over it.
#
#   TF_SRC=~/tensorfold-src ./start.sh                  # PROFILE=spark (default): SparkFold's settings + a 6 GiB memory floor + NVMe spill
#   SPILL=0 TF_SRC=... ./start.sh                       # no NVMe spill
#   DRAFT_VOCAB= TF_SRC=... ./start.sh                  # MTP drafts from the stock draft list instead of draft_vocab_ko.txt (patch 0009)
#   PROFILE=safe TF_SRC=... ./start.sh                  # SparkFold's code with the old decoding policy (no draft-cost stop, eager shared rounds)
#   KNOBS="TF_KEEP=16 TF_FRESH_SLACK=4096" TF_SRC=... ./start.sh   # extra TF_* env on top of the profile (later entries win)
#   DRY_RUN=1 TF_SRC=... ./start.sh                     # print the docker command only
#
# Paths and names (all overridable):
#   TF_SRC      the patched TensorFold tree, mounted read-only at /w/TensorFold (build it with ../patches/apply.sh)  REQUIRED
#   MODELS_DIR  the directory holding the pack, mounted read-only at /models   (default ~/models; pack at $MODELS_DIR/qwen3.8-flash-next-exl3-4.05bpw)
#   CACHE       extension builds, Triton cache, the converted vision file and the NVMe spill (default ~/.cache/tensorfold-sparkfold)
#   HF_CACHE    Hugging Face cache (default ~/.cache/huggingface)
#   IMAGE       the container image (default tensorfold-qwen38:v0.3.6.3, built by MiaAI-Lab's scripts/prepare.sh)
#   NET         docker network (default bridge)
#   BIND        host address the port is published on (default 127.0.0.1; BIND= with an empty value publishes on every interface).
#               The server has no API key unless you add one: do not expose it to a network you do not trust.
#
# TF_SOLO_ROWS is 64K rows (1.1 GiB) by default, SparkFold's setting: a 262144-row graph slot would hold 4.5 GiB of the memory this box
# keeps free (see TENSORFOLD_MEMORY_RESERVE_GIB below); SOLO_ROWS=131072 covers the 128K bucket for 2.2 GiB.
#
# PROFILE=safe pins everything that changes decoding or scheduling policy back to the old policy, so a first A/B isolates the code
# changes: TF_DRAFT_COST=off (no expected-time draft stop), TF_MULTI_GRAPHS=0 (shared rounds eager). It adds only TF_SOLO_ROWS
# (the lone-stream graph slot is pre-sized and every context bucket captured at start: TensorFold issue #336) and the miss-reason logs.
#
# Both profiles carry TENSORFOLD_MEMORY_RESERVE_GIB=6 and PIN_EXTRA_GIB=-6.2 (RESERVE_GIB / PIN_EXTRA_GIB override; RESERVE_GIB=12.2 and
# PIN_EXTRA_GIB=0 give the stock floor on a 128 GB box). Why: the stream-cache gate refuses growth unless host MemAvailable is above a floor
# of max(4 GiB, a tenth of RAM = 12.2 GiB) plus 2 GiB, a Spark idles near 15 GiB, so a second long conversation evicts the first
# (../docs/memory-gate.md). The startup lock room is budget minus estimate and the budget moves with the floor, hence the matching
# PIN_EXTRA; check that "GiB of pages locked" in the startup line stays near 20-25. The price: the margin to earlyoom's SIGTERM line
# (3% = 3.6 GiB of MemAvailable on a 121 GiB box) falls from ~11 to ~5 GiB.
set -euo pipefail
IMAGE="${IMAGE:-tensorfold-qwen38:v0.3.6.3}"
TF_SRC="${TF_SRC:?set TF_SRC to the patched TensorFold tree (../patches/apply.sh builds it)}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
MODEL="${MODEL:-/models/qwen3.8-flash-next-exl3-4.05bpw}"
NAME="${NAME:-llm}"
NET="${NET:-bridge}"
PORT="${PORT:-8888}"
BIND="${BIND-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"
CACHE="${CACHE:-${HOME}/.cache/tensorfold-sparkfold}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
VISION="${VISION:-1}"
PARALLEL="${PARALLEL:-5}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
PROFILE="${PROFILE:-spark}"
KNOBS="${KNOBS:-}"
DRAFT_VOCAB="${DRAFT_VOCAB-draft_vocab_ko.txt}"   # stock draft list + its 6,641 missing Hangul tokens (patch 0009): speed only, drafts are verified; empty = stock list

case "${PROFILE}" in
  safe)  PROFILE_ENV="TF_DRAFT_COST=off TF_MULTI_GRAPHS=0 TF_SOLO_ROWS=${SOLO_ROWS:-65536} TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE_GIB:-6}"; PIN_EXTRA_GIB="${PIN_EXTRA_GIB:--6.2}" ;;
  spark) PROFILE_ENV="TF_SOLO_ROWS=${SOLO_ROWS:-65536} TF_KEEP=16 TF_FIRST_PASS=1024 TF_PASS_MIN=1024 TENSORFOLD_PREFILL_ROWS=4096 TF_EXL3_MOE_WINDOW=4096 TENSORFOLD_MEMORY_RESERVE_GIB=${RESERVE_GIB:-6}"; PIN_EXTRA_GIB="${PIN_EXTRA_GIB:--6.2}" ;;
  bare)  PROFILE_ENV="" ;;
  *) echo "FATAL: PROFILE must be safe, spark or bare" >&2; exit 1 ;;
esac
SPILL="${SPILL:-1}"            # idle conversations' kept states go to NVMe instead of being dropped (../docs/nvme-spill.md); SPILL=0 off
SPILL_ENV=""
if [[ "${SPILL}" == 1 ]]; then SPILL_ENV="TF_SPILL_DIR=/cache/spill TF_SPILL_GIB=${SPILL_GIB:-48} TF_SPILL_MIN=${SPILL_MIN:-16384}"; fi
VOCAB_ENV=""
if [[ -n "${DRAFT_VOCAB}" ]]; then VOCAB_ENV="TF_DRAFT_VOCAB=${DRAFT_VOCAB}"; fi   # part of the spill signature: changing it drops stored spill states once
KNOB_ARGS=()
for kv in ${PROFILE_ENV} ${SPILL_ENV} ${VOCAB_ENV} ${KNOBS}; do
  [[ "${kv}" == *=* ]] || { echo "FATAL: knob '${kv}' is not NAME=VALUE" >&2; exit 1; }
  KNOB_ARGS+=(-e "${kv}")
done

VISION_ARGS=()
VISION_ENV=()
if [[ "${VISION}" == 1 ]]; then
  [[ -f "${CACHE}/vision-f16-v3.safetensors" ]] || { echo "FATAL: ${CACHE}/vision-f16-v3.safetensors missing: convert the pack's vision_k6.safetensors once with 'python -m tensorfold.vision.exl3_convert' (see ../README.md, step 4), or run with VISION=0" >&2; exit 1; }
  VISION_ARGS=(--vision --vision-max-images "${VISION_MAX_IMAGES:-50}" --vision-image-tokens "${VISION_IMAGE_TOKENS:-16384}")
  VISION_ENV=(-e TENSORFOLD_VISION_WEIGHTS=/cache/vision-f16-v3.safetensors)
fi
MEM_ENV=(-e TENSORFOLD_PIN_EXTRA_GIB="${PIN_EXTRA_GIB:-0}")       # 0 = stock lock (budget - startup estimate, ~25 GiB of the n-gram table); locking all of it let earlyoom SIGKILL the server
[[ -n "${MEMORY_LIMIT_GB:-}" ]] && MEM_ENV+=(-e TENSORFOLD_MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB}")
[[ -d "${TF_SRC}/src/tensorfold" ]] || { echo "FATAL: TensorFold sources missing at ${TF_SRC}" >&2; exit 1; }
[[ -f "${MODELS_DIR}${MODEL#/models}/model.safetensors.index.json" ]] || { echo "FATAL: pack missing under ${MODELS_DIR}" >&2; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || { echo "FATAL: image ${IMAGE} not present (MiaAI-Lab's scripts/prepare.sh builds it)" >&2; exit 1; }
mkdir -p "${CACHE}/torch_extensions" "${CACHE}/triton"

PUBLISH="${BIND:+${BIND}:}${PORT}:${PORT}"
CMD=(docker run -d
  --name "${NAME}"
  --gpus all --ipc=host
  --network "${NET}"
  -p "${PUBLISH}"
  --restart no
  --ulimit memlock=-1 --ulimit stack=67108864
  -e HF_HUB_OFFLINE=1
  -e TENSORFOLD_NO_UPDATE_CHECK=1
  -e TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions
  -e TRITON_CACHE_DIR=/cache/triton
  -e PYTHONPATH=/w/TensorFold/src
  "${VISION_ENV[@]}" "${MEM_ENV[@]}" "${KNOB_ARGS[@]}"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${CACHE}:/cache"
  -v "${MODELS_DIR}:/models:ro"
  -v "${TF_SRC}:/w/TensorFold:ro"
  "${IMAGE}"
  python3 -m tensorfold serve "${MODEL}"
    --host 0.0.0.0 --port "${PORT}"
    --name "${SERVED_NAME}"
    --parallel "${PARALLEL}" --context "${CONTEXT}" --kv-dtype "${KV_DTYPE}"
    --max-tokens "${MAX_TOKENS:-${CONTEXT}}"
    --mtp-drafts "${MTP_DRAFTS:-6}" --mtp-confidence "${MTP_CONFIDENCE:-0.60}"
    --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-0.95}" --top-k "${TOP_K:-10}"
    --thinking
    "${VISION_ARGS[@]}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
# a restart deletes the old container's log: keep it first if you want it (docker logs ${NAME} > somewhere)
docker rm -f "${NAME}" >/dev/null 2>&1 || true
"${CMD[@]}" >/dev/null
echo "started ${NAME} (PROFILE=${PROFILE}, model=${MODEL}, vision=${VISION}, parallel=${PARALLEL}, context=${CONTEXT}, kv=${KV_DTYPE}, publish=${PUBLISH}, knobs: ${PROFILE_ENV} ${SPILL_ENV} ${VOCAB_ENV} ${KNOBS})"
echo "follow with: docker logs -f ${NAME}"
