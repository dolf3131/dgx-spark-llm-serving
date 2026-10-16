#!/usr/bin/env bash
# Download the target model and the LiLiCorr drafter at the revisions this lane was prepared with (2026-10-16).
#   ./download.sh                      # into ~/models
#   MODELS_DIR=/data/models ./download.sh
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
BASE_REV=82d93699acd3be9e4617c53360fe5b3f5dbeb1b3     # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4, 88 shards, ~75 GB
DRAFT_REV=7c7221aeb4cb3139307aa457f7dce11a8fb6ceb2    # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr, one 1.3 GB file
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr --revision "${DRAFT_REV}" \
  --local-dir "${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4-lilicorr"
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --revision "${BASE_REV}" \
  --local-dir "${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4"
# checksums against the Hub (reads every byte once)
hf cache verify nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr --revision "${DRAFT_REV}" \
  --local-dir "${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4-lilicorr" --fail-on-missing-files
hf cache verify nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --revision "${BASE_REV}" \
  --local-dir "${MODELS_DIR}/nemotron-3.5-super-vl-nvfp4" --fail-on-missing-files
