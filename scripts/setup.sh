#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for executable in uv git ffmpeg ffprobe; do
  command -v "$executable" >/dev/null || { echo "Install $executable before running setup." >&2; exit 1; }
done

uv venv --python 3.11 --allow-existing .venv-inference
if [ "$(uname -s)" = Linux ]; then
  uv pip install --python .venv-inference/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
else
  uv pip install --python .venv-inference/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi
uv pip install --python .venv-inference/bin/python -r scripts/requirements-auto-avsr.txt -r pyproject.toml gdown==6.5.1

revision=182b62837773ab01052d4ac21ef1d2203ea7d267
mkdir -p vendor
if [ ! -d vendor/auto_avsr ]; then
  git clone https://github.com/mpc001/auto_avsr.git vendor/auto_avsr
fi
if [ -n "$(git -C vendor/auto_avsr status --porcelain)" ]; then
  echo "vendor/auto_avsr has local changes; preserve them before running setup." >&2
  exit 1
fi
git -C vendor/auto_avsr checkout "$revision"

checkpoint_dir="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
mkdir -p "$checkpoint_dir"
checkpoint="$checkpoint_dir/vsr_trlrs2lrs3vox2avsp_base.pth"
if [ ! -f "$checkpoint" ]; then
  .venv-inference/bin/python -m gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_ -O "$checkpoint.part"
  .venv-inference/bin/python - "$checkpoint.part" <<'PY'
import hashlib, sys
with open(sys.argv[1], "rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
if actual != "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb":
    raise SystemExit("Downloaded checkpoint hash mismatch")
PY
  mv "$checkpoint.part" "$checkpoint"
fi
.venv-inference/bin/python - "$checkpoint" <<'PY'
import hashlib, sys
with open(sys.argv[1], "rb") as handle:
    actual = hashlib.file_digest(handle, "sha256").hexdigest()
if actual != "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb":
    raise SystemExit("Cached checkpoint hash mismatch")
PY
echo "Setup complete. Start the app with ./scripts/start.sh"
