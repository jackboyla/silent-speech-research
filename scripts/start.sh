#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-inference/bin/python ]; then
  echo "Run ./scripts/setup.sh first." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec .venv-inference/bin/python scripts/serve.py --upstream vendor/auto_avsr --checkpoint "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth" "$@"
