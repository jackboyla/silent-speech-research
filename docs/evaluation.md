# Evaluation contract

## Manifest

One JSON object per clip, in one JSONL file containing all splits. This illustrative record is not a collected sample:

```json
{"id":"clip-0001","video":"data/clip-0001.mp4","speaker_id":"jack","session_id":"day4","split":"test","task":"dictation","mode":"silent","reference":"please send the revised document","phrase_family":"document_request","duration_s":3.6}
```

Required fields: `id`, `speaker_id`, `session_id`, `split`, `task`, `mode`, `reference`, `duration_s`. Splits are `train`, `validation`, `test`; tasks are `dictation`, `command`, `idle`; modes are `silent`, `voiced_muted`, `non_speech`. `idle` requires `non_speech` and an empty reference. Other tasks require a nonempty reference. Add recording provenance fields such as `source_sha256`, `fps`, `camera`, `parent_id`, and `video`; the scorer ignores them, but they are necessary in a real dataset audit.

Command/idle records also require `label`. Known command labels come from `data/commands.json`; unknown phrases and idle use `__unknown__`. A command unknown still has its actual mouthed words as `reference`. Dictation and unknown-phrase records require `phrase_family` to prevent template leakage. Known commands intentionally recur in different sessions.

The validator rejects sessions assigned to multiple splits, normalized novel sentences reused across splits (including overlap with trained commands), conflicting labels for the same command phrase, and novel phrase families crossing splits. Cross-split phrase repetition is allowed only when all occurrences are known commands with the same label. It cannot detect mislabeled sessions, semantic duplicates with different family IDs, burned-in prompts, or duplicated video bytes: inspect metadata and source hashes during collection. Store paired voiced/silent takes in the same split.

## Predictions

One result for every clip in the selected split, with no extra or duplicate IDs:

```json
{"id":"clip-0001","text":"please send the revised document","processing_s":0.9}
```

`text` is always required. Keep crop/decode failures as empty predictions and record an additional `error` string; do not drop them. Command/idle results also need the predicted `label`, using `__unknown__` for rejection or failure. A closed-command classifier should output the selected command's canonical text, or an empty string if rejected; its primary metrics are label accuracy, coverage, and false acceptance. Do not compare its transcript WER directly with unrestricted dictation.

Optional `processing_s` measures video decode/cropping plus recognition for that clip after model loading. Optional `endpoint_latency_s` measures elapsed time from actual utterance end to final visible text in a live system. Do not populate that field from an offline decoding timer. The scorer reports how many timings were supplied and computes aggregate processing real-time factor only when every clip in the group has a timing.

Keep base-model and external-LM outputs in separate files. Report model versions, preprocessing, beam settings, quantization, load time, and hardware alongside predictions. For online learning, save each prediction before revealing its reference; score that as a separately named chronological experiment.

## Metrics

The scorer groups by task and speech mode so silent and voiced-muted data cannot silently pool into one headline metric. It uses corpus word error rate from JiWER: total substitutions, deletions, and insertions divided by total reference words. It also reports exact utterance match. WER can exceed one. Null ratios mean no eligible denominator, never perfect performance.

Text normalization applies Unicode NFKC, lowercasing, curly-to-straight apostrophes, punctuation-to-space except apostrophes, and collapsed whitespace. It retains numbers as written and does not equate contractions or semantic paraphrases. Freeze reference spelling and normalization before comparison. Both wrong negation and wrong numbers count as ordinary word errors; tag those clips separately for a targeted audit.

For commands, report known-command accuracy including rejected clips, accepted coverage, accuracy among accepted known commands, per-label counts, and unknown false accepts with denominators. Idle clips report false accepted labels and clips with inserted text separately. Clip rejection rates are not false activations/hour. The pilot does not have enough negative exposure to establish a rare-event rate.

These summary metrics do not yet implement bootstrap confidence intervals, confusion charts, per-session dashboards, or user correction-time analysis. Add those when real multi-session recordings make the analyses meaningful; the research plan specifies the required decisions. Single public demo footage without independent transcription must not receive a claimed ground-truth WER.
