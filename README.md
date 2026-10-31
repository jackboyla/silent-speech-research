# Silent speech research

Record a silent sentence with your webcam, replay it, and turn the mouth movements into text. The browser requests video only; a local Auto-AVSR model performs recognition on the server's CPU. No glasses, microphone input, or external language-model API is required.

This is an experimental pretrained lip reader, not yet a personalized recognizer. It can miss or change words. The research target is free-form silent English dictation from a MacBook webcam, with accuracy measured on new sentences across separate recording days.

## Run the app

On the inference computer, install Git, ffmpeg (including ffprobe), and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/jackboyla/silent-speech-research.git
cd silent-speech-research
./scripts/setup.sh
./scripts/start.sh
```

Setup creates an isolated Python 3.11 environment, installs CPU inference dependencies, checks out the pinned upstream model code, and downloads verified weights into the shared Torch cache. The app listens at `http://127.0.0.1:18870`. The inference server is tested on Linux; a Mac can use its browser as the camera client without installing Python or model weights.

For a Mac on the same Tailscale network, run this on the inference computer:

```bash
tailscale serve --bg --https=11012 http://127.0.0.1:18870
```

Open the HTTPS URL printed by Tailscale on the Mac. Remote camera access requires HTTPS. Use a free port and preserve any existing Tailscale routes. The app is intended for localhost or a trusted tailnet; it does not provide public-internet authentication.

1. Select **Enable camera**, then **Record**, and mouth one short sentence.
2. Stop and replay the clip, then select **Read my lips**.
3. Check and copy the transcript. To collect calibration data, correct it, confirm the replay, and select **Save confirmed words**.

Recordings stop slightly before the server's 12-second limit. Audio-free MP4, MOV, and WebM uploads are also supported, up to 25 MiB. One request runs at a time; another client receives a busy response. Recognition uses a full clip, not streaming speech.

## Recordings and calibration

Transcribing stores the source clip, normalized video, raw prediction, and model provenance in `data/captures/` on the inference computer. **Discard clip** deletes the current uploaded clip, including its saved label. Starting another clip or closing the page leaves earlier successful uploads on disk. Nothing is sent to GitHub or a cloud teacher. Personal data, model weights, downloaded upstream code, environments, and logs are ignored by Git.

Saving a correction does not train the model. Since the model's answer is already visible, these labels are restricted to **training**, not validation or test. Export confirmed clips with:

```bash
uv sync --locked
uv run python scripts/export_captures.py --speaker my-speaker-id
```

This writes `data/practice-manifest.jsonl` and `outputs/practice-predictions.jsonl`. These can check training data and pipeline behavior, but cannot establish held-out accuracy. Follow the [collection protocol](docs/collection.md) for independent research recordings.

## Research

Two tracks share recordings and evaluation:

- **Dictation:** adapt a pretrained visual speech recognizer to transcribe sentences absent from the calibration set.
- **Phrase recognition (follow-up):** learn a fixed set of personal commands from examples, with an unknown/reject outcome.

The [research plan](docs/research-plan.md) defines hypotheses, baselines, training stages, and decision gates. The [collection protocol](docs/collection.md) defines the first recording sessions. The [evaluation contract](docs/evaluation.md) defines manifests and scoring.

The [Auto-AVSR baseline](docs/baseline.md) provides a pinned CLI inference path and public integration sample. Personal fine-tuning and a collector for sealed evaluation remain planned work.

## Evaluate predictions

Install [uv](https://docs.astral.sh/uv/), then run from this directory:

```bash
uv sync --locked
uv run python scripts/evaluate.py data/manifest.jsonl outputs/predictions.jsonl --split test
uv run python -m unittest discover -s tests
node --check web/app.js
```

The scorer checks session and phrase leakage before scoring, requires one prediction per selected clip, and reports errors for missing results. Raw recognizer and language-model-assisted predictions must be separate files. Tests include real ffmpeg media validation; model weights are not needed for the unit suite. See the [app operation notes](docs/app.md) for integration checks and storage details.

## Data and results

Keep recordings in `data/`, generated predictions in `outputs/`, and checkpoints in the shared model cache or `checkpoints/`. These paths and `progress/` are excluded from version control. `data/commands.json` contains only the proposed command vocabulary, not recordings.

`progress/plan.md` holds current work and blockers; `progress/experiment-log.md` records executed runs and their provenance. A public spoken-video smoke test validates an inference path. It does not establish performance on silently mouthed speech or on a new user.

## License and credits

This project's code is MIT licensed. [Auto-AVSR](https://github.com/mpc001/auto_avsr) supplies the pretrained visual speech model and preprocessing; its code is Apache-2.0. Model weights and training data have separate conditions that must be checked before redistribution or product use. This repository does not redistribute them. [Chaplin](https://github.com/amanvirparhar/chaplin) informed the interaction, and [Hunch](https://github.com/undrgrnd-labs/hunch) motivates the planned personal phrase-learning study.
