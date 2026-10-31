# Reproduce the Auto-AVSR integration check

This Linux CPU runner uses the original model and preprocessing at a fixed revision. It accepts audio-free, 25fps clips, verifies the checkpoint hash, and reports a transcript plus offline timing. It has no external language-model correction. It is a single-clip runner; batching, live capture, and personal adaptation are not implemented.

Prerequisites: Git, curl, ffmpeg, and uv. Use a separate Python 3.11 environment for the baseline's older inference dependencies. Run these commands from the project root; skip existing clones and reuse the cached checkpoint.

```bash
uv venv --python 3.11 progress/baseline-env
uv pip install --python progress/baseline-env/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python progress/baseline-env/bin/python -r scripts/requirements-auto-avsr.txt gdown==6.5.1
mkdir -p progress/vendor progress/samples progress/evaluations
git clone https://github.com/mpc001/auto_avsr.git progress/vendor/auto_avsr
git -C progress/vendor/auto_avsr checkout 182b62837773ab01052d4ac21ef1d2203ea7d267
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
```

Download the official README's checkpoint only if it is not already cached:

```bash
if [ ! -f "$HOME/.cache/torch/hub/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth" ]; then
  progress/baseline-env/bin/python -m gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_ -O "$HOME/.cache/torch/hub/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth"
fi
```

The runner requires SHA-256 `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`. A mismatch is an error, not a reason to bypass verification.

## Public sample

The source is AV-HuBERT's published demonstration GIF at a fixed Git revision. Crop out its displayed transcript before decoding; otherwise the model input would contain text correlated with the answer.

```bash
curl -fL https://raw.githubusercontent.com/facebookresearch/av_hubert/258fb50e155134eec2c4b49c2ae8de267075fd18/assets/lipreading.gif -o progress/samples/avhubert-demo.gif
ffmpeg -loglevel error -y -i progress/samples/avhubert-demo.gif -vf 'crop=176:204:0:8,fps=25' -an -c:v libx264 -pix_fmt yuv420p progress/samples/avhubert-face-only.mp4
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 progress/baseline-env/bin/python scripts/run_auto_avsr.py --upstream progress/vendor/auto_avsr --checkpoint "$HOME/.cache/torch/hub/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth" --video progress/samples/avhubert-face-only.mp4 --output progress/evaluations/auto-avsr-smoke.json
```

The source GIF SHA-256 is `8cf0498b502e01bd6eb72f0985854a64793a6b4f0513181a8bc474abc3e8e75f`. Preserve this hash and the output video hash in experiment records; ffmpeg versions may produce different encoded bytes. Inspect the crop to confirm it excludes the demonstration caption. The GIF has no audio, and the runner rejects any audio stream.

The public demo's displayed sentence is another model's output, not an independently verified reference. Its footage may overlap pretraining data. Use this sample solely to check execution; do not score ground-truth WER, tune hyperparameters on it, or claim personal/silent-speech accuracy.

## Personal clips

Replace `--video` and `--output` with the clip and result paths. Derive an audio-free constant-rate copy using ffmpeg while preserving the original recording and timestamps. Apply the checkpoint's frame rate without accelerating playback. A variable-frame-rate file whose average happens to be 25fps is insufficient.

The runner uses CPU tensors and four Torch threads by default. MediaPipe uses a CPU inference delegate but may initialize an OpenGL context; disabling CUDA does not disable every graphics initialization. Check resource ownership before larger work, and put long batches in tmux with logging.

Adapt each successful runner result to the [prediction contract](evaluation.md): retain the recording ID, map `prediction` to `text`, and use preprocessing plus inference seconds as `processing_s`. Preserve failed clips as empty predictions with an error field. Keep model load and whole-process startup separate. This conversion is currently manual; the runner's rich provenance JSON is not itself a benchmark prediction JSONL.

The code is Apache-2.0; the pretrained weights' and source datasets' conditions require separate review before product distribution. See the upstream [model documentation](https://github.com/mpc001/auto_avsr/tree/182b62837773ab01052d4ac21ef1d2203ea7d267).
