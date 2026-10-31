"""Score independent clip predictions after checking the research split contract."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import unicodedata

import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return " ".join("".join(
        c if not unicodedata.category(c).startswith("P") or c == "'" else " "
        for c in text
    ).split())


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError(f"{path}: need nonempty records with string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    return rows


def validate_manifest(rows):
    sessions = {}
    phrases = defaultdict(list)
    families = {}
    for r in rows:
        for key in ("id", "speaker_id", "session_id", "split", "task", "mode"):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError(f"Missing {key} in manifest")
        if r["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"{r['id']}: invalid split")
        if r["task"] not in {"dictation", "command", "idle"}:
            raise ValueError(f"{r['id']}: invalid task")
        if r["mode"] not in {"silent", "voiced_muted", "non_speech"}:
            raise ValueError(f"{r['id']}: invalid mode")
        if not isinstance(r.get("reference"), str):
            raise ValueError(f"{r['id']}: reference must be text")
        ref = normalize(r["reference"])
        if ref:
            phrases[ref].append(r)
        if (r["task"] == "idle") != (r["mode"] == "non_speech"):
            raise ValueError(f"{r['id']}: idle task requires non_speech mode")
        if (r["task"] == "idle") != (ref == ""):
            raise ValueError(f"{r['id']}: empty reference is allowed only for idle")
        duration = r.get("duration_s")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{r['id']}: duration_s must be positive and finite")
        if r["task"] in {"command", "idle"}:
            if not isinstance(r.get("label"), str) or not r["label"]:
                raise ValueError(f"{r['id']}: command/idle needs a label")
            if r["task"] == "idle" and r["label"] != "__unknown__":
                raise ValueError(f"{r['id']}: idle must use __unknown__")
        session = (r["speaker_id"], r["session_id"])
        if sessions.setdefault(session, r["split"]) != r["split"]:
            raise ValueError(f"Session crosses splits: {session}")
        # Known commands intentionally recur across sessions; new sentences do not.
        if r["task"] == "dictation" or (r["task"] == "command" and r["label"] == "__unknown__"):
            family = r.get("phrase_family")
            if not isinstance(family, str) or not family:
                raise ValueError(f"{r['id']}: novel phrases require phrase_family")
            if families.setdefault(family, r["split"]) != r["split"]:
                raise ValueError(f"Phrase family crosses splits: {family}")
    for ref, occurrences in phrases.items():
        known = [r for r in occurrences if r["task"] == "command" and r["label"] != "__unknown__"]
        labels = {r["label"] for r in occurrences if r["task"] == "command"}
        if len(labels) > 1:
            raise ValueError(f"Conflicting command labels for phrase: {ref}")
        if len({r["split"] for r in occurrences}) > 1 and len(known) != len(occurrences):
            raise ValueError(f"Novel phrase crosses splits: {ref}")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score(rows, predictions, split):
    validate_manifest(rows)
    selected = [r for r in rows if r["split"] == split]
    by_id = {p["id"]: p for p in predictions}
    expected = {r["id"] for r in selected}
    if not expected or len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Predictions must contain exactly one record per selected clip")
    groups = defaultdict(list)
    for r in selected:
        p = by_id[r["id"]]
        if not isinstance(p.get("text"), str):
            raise ValueError(f"{r['id']}: prediction text is required, including failures")
        if r["task"] in {"command", "idle"} and (not isinstance(p.get("label"), str) or not p["label"]):
            raise ValueError(f"{r['id']}: prediction label is required")
        for key in ("processing_s", "endpoint_latency_s"):
            value = p.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{r['id']}: {key} must be finite and nonnegative")
        groups[(r["task"], r["mode"])].append((r, p))
    result = {"split": split, "clips": len(selected), "groups": {}}
    for (task, mode), pairs in sorted(groups.items()):
        refs = [normalize(r["reference"]) for r, _ in pairs]
        hyps = [normalize(p["text"]) for _, p in pairs]
        metric = {"clips": len(pairs), "exact_matches": sum(a == b for a, b in zip(refs, hyps))}
        metric["exact_match_rate"] = metric["exact_matches"] / len(pairs)
        if task != "idle":
            words = jiwer.process_words(refs, hyps)
            metric.update(wer=words.wer, reference_words=sum(len(s.split()) for s in refs),
                          substitutions=words.substitutions, deletions=words.deletions,
                          insertions=words.insertions)
        else:
            metric["clips_with_inserted_text"] = sum(bool(h) for h in hyps)
        if task in {"command", "idle"}:
            counts = defaultdict(lambda: {"total": 0, "correct": 0})
            known = accepted_known = correct_accepted = negatives = false_accepts = 0
            for r, p in pairs:
                counts[r["label"]]["total"] += 1
                counts[r["label"]]["correct"] += p["label"] == r["label"]
                if r["label"] == "__unknown__":
                    negatives += 1
                    false_accepts += p["label"] != "__unknown__"
                else:
                    known += 1
                    accepted_known += p["label"] != "__unknown__"
                    correct_accepted += p["label"] == r["label"]
            metric.update(per_label=dict(counts), known_clips=known,
                          known_coverage=ratio(accepted_known, known),
                          accepted_known_accuracy=ratio(correct_accepted, accepted_known),
                          known_accuracy=ratio(correct_accepted, known), negative_clips=negatives,
                          false_accepts=false_accepts, false_accept_rate=ratio(false_accepts, negatives))
        for key in ("processing_s", "endpoint_latency_s"):
            times = sorted(p[key] for _, p in pairs if p.get(key) is not None)
            if times:
                metric[key] = {"n": len(times), "median": statistics.median(times),
                               "p95": times[math.ceil(0.95 * len(times)) - 1]}
        if all(p.get("processing_s") is not None for _, p in pairs):
            metric["processing_rtf"] = sum(p["processing_s"] for _, p in pairs) / sum(r["duration_s"] for r, _ in pairs)
        result["groups"][f"{task}/{mode}"] = metric
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args()
    try:
        result = score(load_jsonl(args.manifest), load_jsonl(args.predictions), args.split)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
