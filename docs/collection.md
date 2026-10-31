# Personal recording protocol

## First check

The browser app supports practice capture and training corrections after viewing predictions. It does not yet implement sealed validation/test collection. Use a separate capture process for held-out sessions so their confirmed transcripts remain independent of model output.

Use the MacBook's built-in webcam at normal desk distance. Start with ten short, genuinely silent sentences to check visibility, focus, frame timing, and crop tracking. This is setup footage, not final test data. Capture the native video with timestamps; derive checkpoint-specific crops later. Prefer stable 30fps capture when available and retain observed rather than requested frame rate. Do not increase mouth motion just to make a demo work unless recording that as a separate condition.

Use an explicit start/stop control. The model receives only the camera frames. Do not burn the displayed prompt into the video, include the screen, supply transcripts as inference context, or expose labelled filenames to a multimodal model. Choose opaque clip IDs. Store raw footage locally; do not send it to an external teacher by default.

## Four-session dictation pilot

Use separate days where practical. Sessions 1 and 2 are training, session 3 validation, session 4 the sealed test. Move/reseat the camera naturally between sessions while keeping the intended deployment conditions. Record lighting, camera, distance, head pose, and any capture issues.

| Material per session | Count | Purpose |
| --- | ---: | --- |
| Novel sentences of about 4–12 words | 40 | Free-form dictation and personal adaptation |
| Idle, smiling, chewing, looking away, unrelated mouth motion | 10 | Unwanted text on non-speech input |

Total: 200 clips over four sessions, comprising 160 sentences and 40 non-speech clips. Budget roughly 10–20 minutes per session including review, and revise from actual collection time. This is a feasibility pilot; it does not guarantee enough data for strong dictation. Measure the unchanged D0 recognizer before personal adaptation.

Randomize sentence order within every session. Each take starts a new capture. Free-form sentence families must not cross split boundaries; retakes stay in their original split.

For novel sentences, use distinct wording in each split. Include ordinary messages, names, quantities, negation, and questions. Tag related variants with one `phrase_family` so “send this on Tuesday” and “send this on Thursday” stay in one split. Reserve genuinely new families for the final session. Personal vocabulary may occur across splits; whole sentence templates must not.

Confirm each take with playback and an explicit correct/re-record decision. The displayed prompt supplies intended words, but is not proof that the take matches them. Correct the reference if a different phrase was knowingly mouthed; mark uncertainty for exclusion/re-recording before running models. Never use model output to revise test labels without a blinded adjudication record.

## Paired speech modes

After the initial silent pilot, record a smaller paired subset in voiced and silent modes to measure the domain gap. Both takes of a phrase stay in the same split. Voiced recordings retain an optional separate audio track for training supervision or label checking; evaluation runners consume a video-only derivative. A silent take recorded with a microphone off is evidence only of no captured audio, not of how the user articulated it: record the instructed mode explicitly.

## Calibration expansion

The dictation pilot contains 80 training sentences, 40 validation sentences, and 40 sealed-test sentences. Expand to 5, 15, 30, and 60 actual minutes of distinct silent utterances only after validating crop quality and the pretrained baseline. Keep multiple new sessions for validation/test; additional repetitions of an old clip do not add independent data.

Retain failures and their reasons. A face tracker losing the mouth is a system failure, not permission to remove the clip from the reported denominator. Training cleanup and pre-model exclusion rules may remove corrupt or incorrectly recorded files, with counts recorded before model comparison.

## Optional fixed-command study

After the dictation pilot, test Hunch-style calibration separately. In each of four new sessions, collect five takes of each of the first 12 phrases in [commands.json](../data/commands.json), 20 unknown phrases, and 20 non-speech clips. Use two training sessions, one validation session, and one sealed test. Known commands intentionally recur across sessions; unknown sentence families stay disjoint across splits. Include similar-length unknowns and near-confusable commands. Randomize takes instead of recording one phrase in a block.

Use nested training subsets of 1, 3, 5, and 10 examples per known phrase, with a saved seed and sampled IDs. Do not reuse sealed dictation recordings for training this study. Command accuracy is a separate result from new-sentence transcription.

## Manifest

Write one JSON object per clip following [evaluation.md](evaluation.md). Keep the manifest separate from model inputs. Store source SHA-256, actual frame rate, camera details, mode, session, and split. Every crop/augmentation inherits its parent clip's ID relationship and split. No personal recordings exist in the initial project; do not fill the manifest with pretend predictions.
