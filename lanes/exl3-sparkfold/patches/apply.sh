#!/usr/bin/env bash
# Build the patched TensorFold tree this lane runs:
#   SparkFold spark-exl3 @ 806b345 (itself TensorFold 0.6.5, commit 609ca41, plus 89 commits)  ->  0001 applied as one commit  ->  0002..0007 with git am
#
#   ./apply.sh ~/tensorfold-src
#   SPARKFOLD_URL=/path/to/a/local/SparkFold/clone ./apply.sh ~/tensorfold-src      # skip the download
#
# The result is the tree start.sh mounts (TF_SRC). SparkFold has moved on since 806b345 (see ../README.md, "Newer upstream work");
# the pin is what the measurements in ../results were made on.
set -euo pipefail
OUT="${1:?usage: apply.sh <output dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SPARKFOLD_URL="${SPARKFOLD_URL:-https://github.com/grearjake-star/TensorFold.git}"
BASE=806b345

[[ -e "${OUT}" ]] && { echo "FATAL: ${OUT} exists" >&2; exit 1; }
git init -q "${OUT}"
cd "${OUT}"
git fetch -q "${SPARKFOLD_URL}" spark-exl3
git checkout -q "${BASE}"
git -c user.name=apply -c user.email=apply@localhost apply --index "${HERE}/0001-carry-over-tf-opt-local-fixes.patch"
git -c user.name=apply -c user.email=apply@localhost commit -q -m "carry-over of local fixes (see patches/0001)"
for p in "${HERE}"/000[2-7]*.patch; do
  git -c user.name=apply -c user.email=apply@localhost am -q "${p}"
done
echo "built ${OUT} at $(git rev-parse --short HEAD) (SparkFold ${BASE} + 7 patches)"
