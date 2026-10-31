# Personal camera-only silent speech

## Objective and scope

Build useful English silent dictation from an ordinary camera at normal desk distance. The primary question is whether adaptation to one user's genuinely silent articulation improves transcription of new sentences across recording days. A separate fixed-command experiment tests fast learning from a few examples.

Confirmed target: MacBook built-in webcam and free-form silent dictation. Start with explicit capture start/stop, short utterances, and local storage. Run D0 and personal sequence adaptation first; C0–C3 are a follow-up experiment and do not gate dictation work. Smartphone transfer, automatic endpointing, and live streaming follow only after the recognition baseline works. No model sees audio at inference.

## What Hunch can contribute

[Hunch](https://github.com/undrgrnd-labs/hunch/tree/6541bd157e11a5e7d7434360e99ea81f40a8c181) learns structured decisions with a small classifier on fixed EmbeddingGemma 2 features. Its live interface accepts an image or text and supports human teaching. It does not update the embedding backbone. [The implementation](https://github.com/undrgrnd-labs/hunch/blob/6541bd157e11a5e7d7434360e99ea81f40a8c181/hunch/src/hunch/live.py#L272) fits a scaler and logistic regression.

The original [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) is text-only. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) supports video, but Hunch's current [eg2cpu runtime](https://github.com/undrgrnd-labs/hunch/blob/6541bd157e11a5e7d7434360e99ea81f40a8c181/eg2cpu/README.md#L61) excludes video and multiple-image inputs. A temporal adapter is additional work. A classifier choosing among taught phrases also cannot compose an arbitrary transcript.

Use human-confirmed recordings as labels. A general vision model's guessed transcript is not ground truth. Hunch's teacher agreement is useful for distillation but cannot establish recognition accuracy when the teacher misreads the mouth. Keep automatic teaching disabled during held-out evaluation.

## Hypotheses

1. A frozen speech-specific encoder with a small personal classifier distinguishes taught phrases across sessions better than generic visual embeddings under equal data and tuning budgets.
2. Small encoder adapters or limited fine-tuning improve unseen-sentence recognition over an unchanged pretrained VSR model.
3. Truly silent examples add value beyond the same amount of voiced video with its audio removed.
4. Language-model assistance reduces transcription errors without increasing errors on names, numbers, or negation. Fluency and semantic similarity alone cannot establish this.

All four can fail. If performance collapses on a new day, investigate capture, crop alignment, articulation differences, and label quality before growing the model.

## Baselines

| ID | System | What it tests |
| --- | --- | --- |
| C0 | Training-majority label; nearest class prototype | Minimum closed-command reference; prototypes use training data only |
| C1 | Frozen EG2 single central mouth frame + regularized linear classifier | Weak static-image control for scene/pose shortcuts |
| C2 | Frozen EG2 ordered clip features + the same classifier | Hunch-inspired personal phrase learning with motion order |
| C3 | Frozen Auto-AVSR temporal features + the same classifier | Value of a speech-specific representation |
| D0 | Pinned Auto-AVSR video-only pretrained model, fixed upstream decoder | First reproducible open-dictation baseline |
| D1 | D0 with an external text rescoring stage | Incremental value and harm of language assistance |
| D2 | D0 with personal adapters or limited fine-tuning | Whether calibration improves genuinely new utterances |

For C1–C3 fix the class list, classifier family, regularization search, training examples, and rejection policy. Compare ordered temporal bins with order-destroyed inputs on validation as a diagnostic. EG2 clip features initially concatenate four chronological bin means from a capped, recorded frame schedule. This adds order but is not a speech decoder. Do not blend those vectors with EG2 text vectors as though their dimensions and meanings still matched.

Native EG2 video encoding is a separate later ablation: its documented default samples sparsely, so a speech experiment must explicitly set and log the actual frame timestamps and context budget. Long clips must not silently truncate. Start with a capped short-clip study before attempting high-rate utterances.

[Auto-AVSR](https://github.com/mpc001/auto_avsr) supplies pretrained speech models and preprocessing guidance. [Chaplin](https://github.com/amanvirparhar/chaplin) is a useful interaction reference, but its corrected text mixes recognition with language-model rewriting. Keep the base recognizer visible in evaluation. "Raw" here means no additional text model; the VSR decoder itself may still contain a learned language prior.

[AV-HuBERT](https://github.com/facebookresearch/av_hubert) is a candidate second speech representation if C3/D0 warrants it; its archived framework and separate license increase integration cost. [VALLR](https://github.com/MarshallT-99/VALLR) supplies official code and a pretrained-weight link under CC BY-NC 4.0; its phoneme-to-language-model route is a later candidate after reproducing its preprocessing and inference. Published benchmark scores do not replace same-data comparisons.

## Data and evaluation design

Follow the [collection protocol](collection.md). Keep these claims separate:

- Public spoken-video smoke: validates loading, crop handling, and visual-only decoding.
- Personal known phrases on a new session: measures learning a command vocabulary.
- Personal novel sentences on a new session: measures dictation generalization.
- Additional participants: measures transfer beyond one person; not established by personal success.

Primary pilot: four sessions with two for training, one for validation, and one sealed test. Never split adjacent frames, augmentations, retakes, or one continuous recording across partitions. Keep open-dictation sentence families and unknown command phrases disjoint across partitions. The fixed known command vocabulary intentionally repeats across sessions.

Use nested command calibration budgets of 1, 3, 5, and 10 takes per phrase from training sessions only. For dictation, start with all available training utterances and report their actual minutes. Expand to nested 5/15/30/60-minute budgets only after collecting enough distinct recordings; those quantities are a proposed learning curve, not a promise of sufficient data.

## Training sequence

### Stage 0: establish the input and baseline

Pin model, processor, vocabulary, source revision, environment, hashes, and decoder settings. Preserve original videos and preprocessing metadata. Run one short public sample, then the personal pilot. Decode each clip independently without its reference text, prompt, or filename as a model feature. Inference should be reproducible with metadata stripped.

Use the upstream crop geometry, alignment, grayscale/normalization, and frame rate for the chosen checkpoint. For other models retain their own preprocessing requirements; fairness means equivalent source footage, not forcing incompatible crops. Record face/crop failures as failures in the complete task result, alongside a diagnosed subset.

### Optional command study: cheap personal learning

Extract and cache frozen training features. Fit nearest prototypes and a regularized linear classifier. Fit feature scaling on training data only. Tune regularization and rejection on the validation session, then evaluate the sealed test once. Retraining this head should remain a CPU task.

If C2 cannot beat the static C1 control across days, there is no evidence that EG2 preserves the motion needed here. If C3 wins, keep the human-teaching interaction but use the speech-specific encoder.

### Stage 2: adapt a speech sequence model

Retain the pretrained vocabulary and CTC/attention transcription objectives supported by the model. First train a small adapter or the last encoder block and recognition head; do not replace the decoder simply to introduce a general LLM. Compare against D0 and a decoder/head-only control to separate improved visual features from learning the user's common words.

Begin with one short configuration and one seed. A bounded starting search is learning rates 1e-5 and 5e-5 for unfrozen pretrained layers; new adapters may need a separate rate. Pick the exact modules and rates after inspecting the chosen implementation. Track validation word error rate and stop early. Confirm a promising setting with three seeds before a larger data collection. Seed repetition does not measure uncertainty across users or recording days.

Use modest spatial/brightness augmentation that matches observed camera variation. Keep all derivatives in the source split. Do not randomly reorder frames or horizontally flip the mouth without checking the task consequences. Avoid heavy temporal warping until a controlled validation ablation supports it.

When permitted public training data are available, compare personal-only adaptation with replay of that data to check forgetting. Without such data, report personal gains and loss of public generalization separately rather than assuming preservation.

### Stage 3: transfer from voiced to silent articulation

Record paired voiced and silent versions with confirmed transcripts. Put both versions in the same partition. The voiced audio may supply training supervision or a diagnostic ASR transcript after checking it; it is never an input to the deployed recognizer.

Use transcript-level supervision or an alignment method that tolerates different articulation timing. Do not assume frame-by-frame correspondence between separate voiced and silent takes. Compare equal visual-data budgets: voiced-only, silent-only, and combined. [Published work](https://arxiv.org/abs/2305.14203) documents this speech-mode gap and motivates testing it directly.

Contrastive video/text alignment is a later auxiliary objective, not a substitute for exact token supervision. Preserve distinctions in negation, numbers, word order, and names; a semantic sentence embedding can place materially different transcripts close together.

### Stage 4: active collection and deployment

Request human labels for uncertain or disagreeing training examples, while retaining a random collection fraction to limit selection bias. Keep the sealed test outside the correction loop. A separate online study must predict first, then reveal the label and update, in chronological order.

Only after useful offline recognition: measure camera-to-text latency, end-of-utterance delay, memory, power, and correction time on the intended laptop. Distill or quantize only with a held-out accuracy check. Add automatic endpointing and continuous idle tests as separate changes. A fast batch decoder is not evidence of low-latency streaming.

## Metrics and decision gates

The [scorer](evaluation.md) provides corpus word error rate, exact utterance match, command accuracy/coverage, and negative-clip acceptance, grouped by speech mode and task. Also inspect per-command confusion, names/numbers/negation, failure counts, and results by session. Log preprocessing plus inference time separately from model load, and do not present decoding time as full user latency.

Use paired bootstrap intervals by utterance for exploratory dictation comparisons, and by session when enough independent sessions exist. Report denominators and uncertainty for rejection. A few negative clips cannot establish rare false activations; later continuous idle evaluation needs hours of exposure and event counts.

Proposed decision gates, to fix before the sealed test:

1. **Continue command calibration:** at least 90% known-command coverage and 95% accuracy among accepted known commands; report all unknown and idle false accepts. These are pilot targets, not deployment certification.
2. **Continue sequence adaptation:** at least 15% relative reduction in novel-sentence WER over D0 on validation, followed by a separately reported sealed-test result. Also report the absolute reduction and counts; a large relative gain on a poor baseline may still be unusable.
3. **Build a daily-use app:** beat the user's measured typing time on matched message tasks after including capture, waiting, retries, and corrections. Ask whether silent articulation remains comfortable over a realistic session. No fixed words/minute claim before this study.

If an estimate is too uncertain or a gate fails, collect an independent session or revise the hypothesis and reserve a new test set. Do not tune repeatedly on the original final session.

## Compute and execution

Prefer frozen feature extraction plus CPU fitting for the first comparison. Check current GPU ownership before every launch. Use one free GPU deliberately for a small adaptation run; use both only for independent short comparisons when both are free. Keep pilot runs below 30 minutes where possible and record any exception.

Use persistent model caches. Run long work in tmux with logs and verify the process and output. Use Modal only when local GPUs remain busy, a single job needs more than 32GB, or temporary scale-out is justified; record the resource and cost bound before running. Never interrupt another experiment to free capacity.

Record actual runs in `progress/experiment-log.md`, including failed downloads and inference attempts, exact commands, machine, environment, code/model revisions, resource check, predictions, and result. Do not commit personal recordings or training logs. Review data/checkpoint terms before redistribution or use beyond the research scope; a code license does not settle the training data's terms.
