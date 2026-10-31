# App operation

The app serves a buildless browser client and one persistent CPU recognizer. The client requests a video track with `audio: false`, chooses a supported MediaRecorder format, and lets the user replay before uploading. Uploads and inference stay on the machine running the server. A Mac browser can reach a Linux inference host over Tailscale HTTPS.

## Server and media

`scripts/start.sh` runs the server in `.venv-inference`, sets CPU thread limits, and uses the pinned upstream checkout and persistent Torch checkpoint cache. `TORCH_HOME` can override the cache root. Pass `--port`, `--data-dir`, or `--threads` through the start script when needed. Bind to loopback and use Tailscale Serve for remote access.

For a persistent session:

```bash
mkdir -p progress/logs
tmux new-session -d -s silent-speech-app './scripts/start.sh > progress/logs/app.log 2>&1'
curl -f http://127.0.0.1:18870/api/health
tail -30 progress/logs/app.log
```

Only stop this app's process/session when maintaining it. An existing tmux session with that name should be inspected before changing it.

The upload endpoint accepts video containers, rejects audio tracks, and caps request size and decoded duration. It normalizes to 25fps without changing playback speed, bounds frame dimensions, and uses lossless encoding to avoid an additional lossy compression pass before recognition. Very short clips, overlong clips, unreadable media, and missing faces return an explicit error. The model processes one clip at a time; health requests remain responsive.

`processing_s` covers server-side normalization and recognition after upload. It does not include camera capture, upload, model startup, or human correction. The model uses full-utterance attention and is not a streaming recognizer.

## Stored records

Each successful request creates `data/captures/<UUID>/` with:

- `source.*`: the original uploaded bytes.
- `video.mp4`: the normalized, audio-free inference input.
- `result.json`: raw transcript, timing, source filename, and model/checkpoint/input provenance.
- `label.json`: optional confirmed reference, session, speech mode, and training-only label origin.

Files are not served by the web app. Successful uploads remain even if the page closes or a new clip starts; discard deletes the current clip. Interrupted or failed preprocessing cleans up partial captures. A successful inference can finish and persist its complete result after the browser disconnects.

The browser uses a session identifier kept in sessionStorage. Start a fresh browser tab/session for a distinct collection sitting. This identifier helps organize practice data, but does not prove independent recording days. The uploader defaults to `voiced_muted`; change it only when the recording really contains silent articulation.

Export only confirmed training clips with `scripts/export_captures.py`. It preserves original model predictions separately from corrected references. Automatic `phrase_family` values identify exact reference strings; group related sentence templates manually before any future multi-split benchmark. Exported paths point to local videos and must be rebased if moving the dataset. Deleting a capture does not update an earlier export; export again before using it.

## Verification

The unit suite checks media boundaries using actual ffmpeg, upload cleanup, concurrent request exclusion, label restrictions, deletion, and evaluation leakage. Its HTTP model is a small stub, so these tests do not establish recognition quality.

For a real integration check, follow [baseline.md](baseline.md) to derive the audio-free public demo, upload it through the browser, verify a nonempty transcript, confirm a practice label, export it, and discard it. Repeat in the intended Mac browser. A camera test must exercise getUserMedia and MediaRecorder; uploading a file alone does not test webcam capture. Synthetic-camera testing does not establish physical-camera permissions or silent-speech accuracy.

The app has no account system. Access should remain limited to localhost or a trusted tailnet. Cross-origin browser writes are rejected; that is not a substitute for authentication on a public deployment.
