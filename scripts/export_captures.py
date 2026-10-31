"""Export confirmed practice recordings as training manifests and raw predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def export(captures, speaker):
    manifest, predictions = [], []
    for label_path in sorted(captures.glob("*/label.json")):
        directory = label_path.parent
        label = json.loads(label_path.read_text())
        result = json.loads((directory / "result.json").read_text())
        if label["split"] != "train" or label["id"] != result["id"] or label["id"] != directory.name:
            raise ValueError(f"Invalid training label: {label_path}")
        video = directory / "video.mp4"
        with video.open("rb") as handle:
            video_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        family = hashlib.sha256(label["reference"].strip().lower().encode()).hexdigest()[:16]
        manifest.append({"id": result["id"], "video": str(video.resolve()), "speaker_id": speaker,
                         "session_id": label["session_id"], "split": "train", "task": "dictation",
                         "mode": label["mode"], "reference": label["reference"],
                         "phrase_family": family, "duration_s": result["duration_s"], "fps": 25,
                         "source_sha256": video_hash, "label_source": label["label_source"]})
        predictions.append({"id": result["id"], "text": result["text"], "processing_s": result["processing_s"]})
    if not manifest:
        raise ValueError("No confirmed training recordings found. Save a correction in the app first.")
    return manifest, predictions


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("data/captures"))
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/practice-manifest.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/practice-predictions.jsonl"))
    args = parser.parse_args()
    try:
        rows, predictions = export(args.captures, args.speaker)
        write_jsonl(args.manifest, rows)
        write_jsonl(args.predictions, predictions)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(rows)} training clips; these are not a held-out evaluation.")
