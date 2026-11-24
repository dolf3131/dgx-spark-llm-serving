#!/usr/bin/env bash
# Hy4-flash (tencent/Hy4-flash) from our EXL3 pack on the busitdol engine (https://github.com/dolf3131/busitdol), in a container:
# the image supplies PyTorch and the dependencies, the busitdol tree is mounted over it.
#
#   TF_SRC=~/busitdol ./start.sh                     # port 8888 on 127.0.0.1, the settings this lane is served with
#   CONTEXT=262144 TF_SRC=~/busitdol ./start.sh      # a smaller context window (less memory for the KV pool)
#   VISION=0 TF_SRC=~/busitdol ./start.sh            # text only (the vision tower takes ~0.9 GiB of GPU memory)
#   KNOBS="-e TF_HY4_RELAX_MARGIN=0" TF_SRC=...   # extra docker -e options, added last (later entries win)
#   DRY_RUN=1 TF_SRC=~/busitdol ./start.sh           # print the docker command only
#
# Paths and names (all overridable):
#   TF_SRC      the busitdol tree, mounted read-only at /w/TensorFold                                   (default ~/busitdol)
#   MODELS_DIR  the directory holding the pack, mounted read-only at /models                         (default ~/models)
#   MODEL       the pack as the container sees it; it must contain engram/engram.json                (default /models/hy4-flash-exl3)
#   CACHE       extension builds and the Triton / inductor caches                                     (default ~/.cache/busitdol-hy4)
#   IMAGE       the container image (default tensorfold-qwen38:v0.3.6.3-xgrammar, built as in the EXL3 Qwen lane's README)
#   NET         docker network (default bridge)
#   BIND        host address the port is published on (default 127.0.0.1; BIND= with an empty value publishes on every
#               interface). The server has no API key unless you add one: do not expose it to a network you do not trust.
#
# The settings below are the ones the lane is served with (2026-11-24). Each has a one-line note; the value that turns it
# off or restores the earlier behaviour is in brackets.
set -euo pipefail
TF_SRC="${TF_SRC:-${HOME}/busitdol}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
MODEL="${MODEL:-/models/hy4-flash-exl3}"
IMAGE="${IMAGE:-tensorfold-qwen38:v0.3.6.3-xgrammar}"
CACHE="${CACHE:-${HOME}/.cache/busitdol-hy4}"
NAME="${NAME:-hy4}"
NET="${NET:-bridge}"
PORT="${PORT:-8888}"
BIND="${BIND-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-hy4-flash}"
CONTEXT="${CONTEXT:-1048576}"          # the model's 1M window; the latent KV cache is int8
ENGRAM_CACHE_GIB="${ENGRAM_CACHE_GIB:-8}"   # DRAM cache for Engram rows read from NVMe
VISION="${VISION:-1}"
KNOBS="${KNOBS:-}"

DRAFT_INT8="${DRAFT_INT8:-1}"          # drafter matrices as int8 rows (H32-rotated, fp16 scale a group of 32); drafts only [0]
ENGRAM_LATE="${ENGRAM_LATE:-2}"        # Engram rows arrive by copy engine while the decode graph runs (same bits) [0]
PREFILL_ROWS="${PREFILL_ROWS:-4096}"   # prompt chunk rows [2048]
PREFILL_CONC="${PREFILL_CONC:-2048}"   # prompt chunk rows while another stream decodes [0 = PREFILL_ROWS]
COPY_EXT="${COPY_EXT:-4}"              # a 4..9-token history match extends DFly2's drafts [0]
P4="${P4:-8}"                          # EXL3 prompt expert kernel p4, variant 8 [0 = the stock prompt kernels]
ROUTE_FUSED="${ROUTE_FUSED:-1}"        # router selection in one Triton kernel (same scores) [0]
MLA_TUNE="${MLA_TUNE:-1}"              # MLA decode tile table for the int8 KV cache (numerically equivalent) [0]
IDX_PROMPT="${IDX_PROMPT:-1}"          # prompt indexer kernel [0]
SWA_FA2="${SWA_FA2:-1}"                # sliding-window prompt chunks through FlashAttention-2 [0]
RELAX_GUARD="${RELAX_GUARD:-1}"        # relaxed acceptance never keeps a Chinese/Japanese draft the target refused [0]
SCREEN_ROT="${SCREEN_ROT:-1}"          # head screen's int8 rescoring copy from Hadamard-rotated rows [0]
SCREEN_G="${SCREEN_G:-32}"             # ... with an fp16 scale a group of 32 weights [0 = one scale a row]
SCREEN_BLOCKS="${SCREEN_BLOCKS:-16}"   # rescored 128-row blocks, drafter and target screen [24]
TARGET_ROWS="${TARGET_ROWS:-12}"       # the target reads its head through the screen up to this many rows [8]

[[ -d "${TF_SRC}/src/tensorfold/families/hy_v4_flash" ]] || { echo "FATAL: no busitdol tree at ${TF_SRC}" >&2; exit 1; }
[[ -f "${MODELS_DIR}${MODEL#/models}/engram/engram.json" ]] || { echo "FATAL: pack (with engram/) missing at ${MODELS_DIR}${MODEL#/models}" >&2; exit 1; }
VISION_ARGS=()
[[ "${VISION}" == 1 ]] && VISION_ARGS=(--vision --vision-max-images "${VISION_MAX_IMAGES:-50}" --vision-image-tokens "${VISION_IMAGE_TOKENS:-16384}")
mkdir -p "${CACHE}/torch_extensions" "${CACHE}/triton" "${CACHE}/inductor"

CMD=(docker run -d --name "${NAME}" --gpus all --ipc=host --network "${NET}" -p "${BIND:+${BIND}:}${PORT}:${PORT}" --restart no
  --oom-score-adj 1000 --ulimit memlock=-1 --ulimit stack=67108864
  -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e PYTHONPATH=/w/TensorFold/src
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TORCHINDUCTOR_COMPILE_THREADS=4
  -e TF_HY4_CONTEXT="${CONTEXT}" -e TF_HY4_ENGRAM_CACHE_GIB="${ENGRAM_CACHE_GIB}"
  -e TF_HY4_DRAFT_INT8="${DRAFT_INT8}" -e TF_HY4_ENGRAM_LATE="${ENGRAM_LATE}"
  -e TF_HY4_PREFILL_ROWS="${PREFILL_ROWS}" -e TF_HY4_PREFILL_ROWS_CONC="${PREFILL_CONC}" -e TF_HY4_COPY_EXT="${COPY_EXT}"
  -e TF_EXL3_P4="${P4}" -e TF_HY4_ROUTE_FUSED="${ROUTE_FUSED}" -e TF_HY4_MLA_TUNE="${MLA_TUNE}"
  -e TF_HY4_IDX_PROMPT="${IDX_PROMPT}" -e TF_HY4_SWA_FA2="${SWA_FA2}" -e TF_HY4_RELAX_GUARD="${RELAX_GUARD}"
  -e TF_HY4_SCREEN_I8ROT="${SCREEN_ROT}" -e TF_HY4_SCREEN_I8G="${SCREEN_G}"
  -e TF_HY4_DRAFT_SCREEN_TWO_BLOCKS="${SCREEN_BLOCKS}" -e TF_HY4_TARGET_SCREEN_BLOCKS="${SCREEN_BLOCKS}"
  -e TF_HY4_TARGET_SCREEN_ROWS="${TARGET_ROWS}" ${KNOBS}
  -v "${CACHE}:/cache" -v "${MODELS_DIR}:/models:ro" -v "${TF_SRC}:/w/TensorFold:ro"
  "${IMAGE}"
  python3 -m tensorfold serve "${MODEL}" --host 0.0.0.0 --port "${PORT}" --name "${SERVED_NAME}"
    --parallel 1 --context "${CONTEXT}" --max-tokens "${MAX_TOKENS:-${CONTEXT}}"
    --temperature "${TEMPERATURE:-0.9}" --top-p "${TOP_P:-1.0}" --top-k "${TOP_K:-10}" --thinking "${VISION_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
docker rm -f "${NAME}" >/dev/null 2>&1 || true
"${CMD[@]}" >/dev/null
echo "started ${NAME} (Hy4-flash on busitdol at ${TF_SRC}, model=${MODEL}, context=${CONTEXT}, vision=${VISION}, port=${BIND:-*}:${PORT})"
echo "follow with: docker logs -f ${NAME}"
