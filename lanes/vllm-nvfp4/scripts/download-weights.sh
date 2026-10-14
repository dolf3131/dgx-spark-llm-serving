#!/bin/bash
# Download nvidia/Qwen3.8-Flash-Next-NVFP4 (123.6 GiB, 24 files).
#
# Qwen3.8-Flash-Next: Qwen4 architecture preview, arch Qwen4ExpForConditionalGeneration /
# qwen4_exp. 125B total with 6B activated, PLUS a 51B n-gram embedding table and a 4B MTP
# layer -- 180B params, 335 GiB in BF16. 48 layers as 12 x (3 x Gated DeltaNet -> 1 x Qwen
# Sparse Attention), 512 experts top-10 + shared, hidden 2560, GQA 24/2 at head_dim 256,
# interleaved M-RoPE, VLM, native 262144 context (YaRN factor 4 -> 1M per the card).
#
# WHY THIS BUILD (changed 2026-09-06; this script used to fetch Inferact):
#   - nvidia NVFP4 (this one): quant_algo=MIXED_PRECISION, described per layer --
#     NVFP4 routed experts (gs 16), FP8 PLE, FP8_PB_WO MTP experts (gs 128). 123.6 GiB,
#     46.7 GiB less than Inferact, and the PLE is 47.7 GiB rather than 95.37. Officially
#     published, and it ships its own eval numbers. Needs patch-nv-mixed.py: the pinned
#     image cannot load a mixed-precision PLE and cannot draft with an FP8_PB_WO MTP.
#   - Inferact NVFP4 (170.3 GiB, PLE BF16 95.4 GiB) is what this repo served until now and
#     still loads with no source changes. Slower here: 32.65 tok/s against 33.02, and
#     ~98 GiB of swap against 55.
#   - RadixArk NVFP4 (126.0 GiB, PLE fp8) loads only with the same ple_layer.py gate
#     relaxed; superseded by the official build, which needs the fix anyway.
#   - Qwen/Qwen3.8-Flash-Next-FP8 (172.8 GiB): body alone is ~125 GiB, does not fit.
#
# MEMORY PLAN on the Spark (121 GiB unified, ~7 GiB OS):
#   body 75.9 GiB resident + KV 24 KiB/token (only 12 of 48 layers hold KV: 2 kv heads x
#   (256+256) x 2B x 12) => 6 GiB at 262144, 24 GiB at 1M. PLE 47.7 GiB lives in the
#   offload process and is paged to /swap-ple.img (128 GiB). Measured steady state: 55 GiB
#   of swap in use. Three separate attempts to make the PLE resident instead all failed --
#   see the README; the kernel treats a table touched 18 rows at a time as cold no matter
#   how much room you give it.
#   Needs VLLM_PLE_CPU_OFFLOAD=1 and the image vllm/vllm-openai:qwen38-flash-next-arm64-cu130
#   (vllm 0.1.dev20073+g8e685d198, ships vllm/v1/ple_offload/ and the env var).
#
# Sequential curl -C -. Parallel pulls did not beat ~10 MB/s on this host and the hf
# CLI stalled at 0 B/s on Xet, so this keeps it simple and resumable.
set -uo pipefail

REPO="${REPO:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
DEST="${DEST:-${MODELS_DIR:-$HOME/models}/qwen3.8-flash-next-nvidia}"
BASE="https://huggingface.co/${REPO}/resolve/main"
mkdir -p "$DEST" || exit 1

# size AND lfs.sha256: a size-correct but truncated/corrupt shard loads cleanly, reports
# sane shapes, produces correct-magnitude activations, and yields fluent garbage that
# survives every configuration change. Someone lost a day to exactly that on this model.
manifest=$(curl -sL "https://huggingface.co/api/models/${REPO}?blobs=true" | python3 -c "
import sys, json
for f in json.load(sys.stdin).get('siblings', []):
    n = f['rfilename']
    if n.startswith('.'):
        continue
    lfs = f.get('lfs') or {}
    print(n, f.get('size') or 0, lfs.get('sha256') or '-')
")

[[ -z "$manifest" ]] && { echo 'FATAL: could not read file manifest' >&2; exit 1; }

verify() {   # $1 path  $2 expected size  $3 expected sha256 ("-" to skip)
    [[ -f "$1" ]] || return 1
    [[ "$(stat -c %s "$1")" == "$2" ]] || return 1
    [[ "$3" == "-" ]] && return 0
    [[ "$(sha256sum "$1" | cut -d" " -f1)" == "$3" ]]
}

fail=0
while read -r name size sha; do
    [[ -z "$name" ]] && continue
    out="${DEST}/${name}"
    mkdir -p "$(dirname "$out")"
    if verify "$out" "$size" "$sha"; then
        printf '  ok    %s\n' "$name"
        continue
    fi
    if [[ -f "$out" ]] && [[ "$(stat -c %s "$out")" == "$size" ]]; then
        printf '  BAD   %s (size matches, sha256 does not) -- refetching\n' "$name" >&2
        rm -f "$out"
    fi
    printf '  get   %s (%.2f GiB)\n' "$name" "$(echo "$size" | awk '{print $1/1073741824}')"
    curl -fL -C - --retry 5 --retry-delay 5 --retry-all-errors --no-progress-meter \
         -o "$out" "${BASE}/${name}" || { echo "  FAIL  $name" >&2; fail=1; continue; }
    verify "$out" "$size" "$sha" || { echo "  FAIL  $name (verification failed after download)" >&2; fail=1; }
done <<< "$manifest"

echo
if [[ "$fail" == 0 ]]; then
    echo "done -> ${DEST}  ($(du -sh "$DEST" | cut -f1))"
else
    echo "FINISHED WITH ERRORS -- rerun to resume (curl -C - continues partial files)" >&2
    exit 1
fi
