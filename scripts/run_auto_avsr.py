"""Run pinned Auto-AVSR on a video-only clip using the upstream inference path."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REVISION = "182b62837773ab01052d4ac21ef1d2203ea7d267"
CHECKPOINT_SHA256 = "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb"
MODEL_NAME = "auto-avsr-vsr_trlrs2lrs3vox2avsp_base"


class Recognizer:
    """One loaded CPU model; callers must serialize predict calls."""

    def __init__(self, upstream, checkpoint, threads=4):
        if threads < 1:
            raise ValueError("threads must be positive")
        upstream, checkpoint = Path(upstream), Path(checkpoint)
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise ValueError(f"Expected pinned Auto-AVSR revision {REVISION}")
        if subprocess.check_output(["git", "-C", str(upstream), "diff", "HEAD", "--"], text=True):
            raise ValueError("Auto-AVSR checkout has modified tracked files")
        with checkpoint.open("rb") as handle:
            self.checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match the pinned visual-only weights")
        sys.path.insert(0, str(upstream.resolve()))
        import av
        import numpy as np
        import torch
        from datamodule.transforms import VideoTransform
        from lightning import ModelModule
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess

        self.av, self.np, self.torch = av, np, torch
        self.threads = threads
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        started = time.perf_counter()
        self.model = ModelModule(argparse.Namespace(modality="video"))
        self.model.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.detector = LandmarksDetector()
        self.crop = VideoProcess(convert_gray=False)
        self.transform = VideoTransform(subset="test")

    def close(self):
        self.detector.short_range_detector.close()
        self.detector.full_range_detector.close()

    def predict(self, video_path):
        video_path = Path(video_path)
        started = time.perf_counter()
        with self.av.open(str(video_path)) as container:
            if container.streams.audio:
                raise ValueError("Input must have no audio streams; strip audio before evaluation")
            if len(container.streams.video) != 1:
                raise ValueError("Input must have exactly one video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 0)
            if abs(fps - 25.0) > 0.01:
                raise ValueError(f"Expected 25 fps, got {fps}; resample before inference")
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            if not decoded:
                raise ValueError("Video contains no decodable frames")
            frames = self.np.stack(decoded)
        try:
            landmarks = self.detector(frames)
            crops = self.crop(frames, landmarks)
        except (AssertionError, OverflowError) as exc:
            raise ValueError("Could not track your mouth. Face the camera with your mouth visible.") from exc
        if crops is None:
            raise ValueError("Could not detect a face. Face the camera with your mouth visible.")
        video = self.transform(self.torch.from_numpy(crops).permute(0, 3, 1, 2))
        preprocess_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with self.torch.inference_mode():
            prediction = self.model(video)
        inference_seconds = time.perf_counter() - started
        with video_path.open("rb") as handle:
            video_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        return {
            "model": MODEL_NAME,
            "upstream_revision": REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "video_sha256": video_sha256,
            "prediction": prediction,
            "device": "cpu", "threads": self.threads,
            "torch_version": self.torch.__version__,
            "frames": len(frames), "fps": fps,
            "clip_seconds": len(frames) / fps,
            "model_load_seconds": self.load_seconds,
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "external_lm": False,
            "beam_size": 40,
            "note": "Offline full-clip inference. Not live latency or a silent-mouthing benchmark.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    recognizer = Recognizer(args.upstream, args.checkpoint, args.threads)
    try:
        result = recognizer.predict(args.video)
    finally:
        recognizer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
