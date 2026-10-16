#!/usr/bin/env bash
# NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 with the LiLiCorr drafter on one DGX Spark, in vLLM.
# The engine and model flags are the model card's DGX Spark Quick Start, unchanged:
#   https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4#quick-start
#
#   ./start.sh                     # container nemotron-vl, port 8000 on 127.0.0.1
#   DRY_RUN=1 ./start.sh           # print the docker command only
#   docker logs -f nemotron-vl     # boot log; ready when `curl -s localhost:8000/v1/models` answers
#
# Paths and names (all overridable):
#   MODELS_DIR  where ./download.sh put the two repos (default ~/models)
#   CACHE       vLLM, FlashInfer and Triton caches and the remote-code modules, mounted at /root/.cache (default ~/.cache/nemotron-vl)
#   IMAGE       default: the card's nightly; the card's alternative, v0.32.0, was not on Docker Hub on 2026-10-16
#   NAME, NET (default bridge), PORT (default 8000)
#   BIND        host address the port is published on (default 127.0.0.1; BIND= with an empty value publishes on every interface).
#               The server has no API key: do not expose it to a network you do not trust.
#   GPU_UTIL    the card's 0.75 (about 91 GiB of a Spark's 121 GiB)
#
# Differences from the card's command, environment only: local paths instead of Hub ids, HF_HUB_OFFLINE=1, --restart no,
# --oom-score-adj 1000 (if memory runs out, the OOM killer and earlyoom pick this container before the desktop), its own cache directory,
# and the pack's page cache is dropped first (on GB10, CUDA counts reclaimable page cache as used, and vLLM then fails its free-memory check).
#
# Nothing else that holds GPU memory may run: the script refuses to start unless MemAvailable covers GPU_UTIL plus 6 GiB (FORCE=1 overrides).
set -euo pipefail
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
PACK="${PACK:-${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4}"
DRAFT="${DRAFT:-${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4-lilicorr}"
CACHE="${CACHE:-${HOME}/.cache/nemotron-vl}"
NAME="${NAME:-nemotron-vl}"
NET="${NET:-bridge}"
PORT="${PORT:-8000}"
BIND="${BIND-127.0.0.1}"
GPU_UTIL="${GPU_UTIL:-0.75}"

[[ -f "${PACK}/model.safetensors.index.json" ]] || { echo "FATAL: ${PACK} incomplete (run ./download.sh)" >&2; exit 1; }
[[ -f "${DRAFT}/model.safetensors" ]] || { echo "FATAL: ${DRAFT}/model.safetensors missing (run ./download.sh)" >&2; exit 1; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 || { echo "FATAL: image ${IMAGE} not present (docker pull ${IMAGE})" >&2; exit 1; }

ARGS=( /models/target
  --served-model-name nemotron-3.5-super-vl
  --host 0.0.0.0 --port 8000
  --max-num-seqs 1
  --trust-remote-code
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-num-batched-tokens 8192
  --speculative_config.model /models/lilicorr
  --speculative_config.num_speculative_tokens 7
  --speculative_config.method dflash
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
  --enable-mamba-shared-prefix-checkpoint
  --video-pruning-rate 0.5
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
  --block-size 64
  --prefix-match-unit 64
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3 )

PUBLISH="${BIND:+${BIND}:}${PORT}:8000"
CMD=(docker run -d
  --name "${NAME}"
  --gpus all --ipc host
  --network "${NET}"
  -p "${PUBLISH}"
  --restart no
  --oom-score-adj 1000
  -e HF_HUB_OFFLINE=1
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -v "${PACK}:/models/target:ro"
  -v "${DRAFT}:/models/lilicorr:ro"
  -v "${CACHE}:/root/.cache"
  "${IMAGE}"
  "${ARGS[@]}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

# GB10 unified memory: the pack's page cache must not count against CUDA-free
python3 - "${PACK}" "${DRAFT}" <<'PY'
import glob, os, sys
n = 0
for d in sys.argv[1:]:
    for fn in glob.glob(os.path.join(d, "*.safetensors")):
        fd = os.open(fn, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
        finally:
            os.close(fd)
print(f"page cache dropped for {n} pack files")
PY
need=$(awk -v u="${GPU_UTIL}" '/^MemTotal:/{printf "%d", u*$2/1048576 + 6}' /proc/meminfo)
avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
echo "MemAvailable ${avail} GiB (need ${need})"
if [[ "${avail}" -lt "${need}" && "${FORCE:-0}" != 1 ]]; then
  echo "REFUSED: not enough free memory; another model server or job still holds it (docker ps; nvidia-smi). FORCE=1 overrides." >&2
  exit 1
fi

# a restart deletes the old container's log: keep it first if you want it (docker logs ${NAME} > somewhere)
docker rm -f "${NAME}" >/dev/null 2>&1 || true
mkdir -p "${CACHE}"
"${CMD[@]}" >/dev/null
echo "started ${NAME} (${IMAGE##*:}, util ${GPU_UTIL}, LiLiCorr k=7, publish ${PUBLISH}, served name nemotron-3.5-super-vl)"
echo "follow: docker logs -f ${NAME}     ready: curl -s localhost:${PORT}/v1/models"
