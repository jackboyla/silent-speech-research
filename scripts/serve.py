"""Serve the local camera-to-text research app with one persistent CPU model."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

try:
    from .run_auto_avsr import MODEL_NAME, Recognizer
except ImportError:
    from run_auto_avsr import MODEL_NAME, Recognizer

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_S = 12


class Label(BaseModel):
    reference: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=100)
    split: Literal["train"] = "train"
    mode: Literal["silent", "voiced_muted"] = "silent"

    @field_validator("reference", "session_id")
    @classmethod
    def nonblank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Must not be blank")
        return value


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalize_video(source, destination):
    """Decode a bounded clip, checking decoded frame count rather than WebM duration metadata."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,matroska,webm",
         "-show_entries", "stream=codec_type,width,height", "-of", "json", str(source)],
        capture_output=True, text=True, timeout=10,
    )
    if probe.returncode:
        raise ValueError("Could not read this video. Record again or upload an MP4/WebM clip.")
    streams = json.loads(probe.stdout).get("streams", [])
    if any(stream.get("codec_type") == "audio" for stream in streams):
        raise ValueError("This app accepts video without audio. Record with the camera-only recorder.")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("The file must contain exactly one video stream.")
    if not (0 < videos[0].get("width", 0) <= 4096 and 0 < videos[0].get("height", 0) <= 4096):
        raise ValueError("Video dimensions must be at most 4096 pixels per side.")
    # One extra frame distinguishes overlong clips from exactly-at-limit clips.
    converted = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-protocol_whitelist", "file,pipe",
         "-format_whitelist", "mov,matroska,webm", "-i", str(source), "-map", "0:v:0", "-an",
         "-vf", "setpts=PTS-STARTPTS,fps=25,scale=w='min(1280,iw)':h='min(1280,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
         "-frames:v", str(MAX_DURATION_S * 25 + 1), "-c:v", "libx264", "-threads", "2",
         "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(destination)],
        capture_output=True, text=True, timeout=30,
    )
    if converted.returncode:
        raise ValueError("The video could not be decoded. Try a new recording.")
    counted = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "json", str(destination)],
        capture_output=True, text=True, timeout=10,
    )
    if counted.returncode:
        raise ValueError("The video contains no readable frames.")
    output_streams = json.loads(counted.stdout).get("streams", [])
    if not output_streams or not str(output_streams[0].get("nb_read_frames", "")).isdigit():
        raise ValueError("The video contains no readable frames.")
    count = int(output_streams[0]["nb_read_frames"])
    if count < 10 or count > MAX_DURATION_S * 25:
        raise ValueError(f"Use a clip between 0.4 and {MAX_DURATION_S} seconds long.")
    return count / 25


def create_app(upstream, checkpoint, data_dir=ROOT / "data" / "captures", threads=4):
    data_dir = Path(data_dir)

    @asynccontextmanager
    async def lifespan(app):
        data_dir.mkdir(parents=True, exist_ok=True)
        app.state.recognizer = await anyio.to_thread.run_sync(lambda: Recognizer(upstream, checkpoint, threads))
        app.state.busy = False
        try:
            yield
        finally:
            app.state.recognizer.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def write_boundary(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            origin_host = urlsplit(origin).netloc if origin else None
            if request.headers.get("x-silent-speech") != "1" or (origin and origin_host != request.headers.get("host")):
                return JSONResponse({"detail": "Use the app on this server to make changes."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Permissions-Policy"] = "microphone=(), camera=(self)"
        return response

    @app.get("/api/health")
    async def health():
        return {"ready": hasattr(app.state, "recognizer"), "model": MODEL_NAME,
                "max_duration_s": MAX_DURATION_S, "max_upload_bytes": MAX_UPLOAD_BYTES,
                "busy": getattr(app.state, "busy", False)}

    def clip_directory(clip_id):
        try:
            canonical = str(UUID(clip_id))
        except ValueError:
            raise HTTPException(404, "Clip not found")
        if canonical != clip_id:
            raise HTTPException(404, "Clip not found")
        directory = data_dir / canonical
        if not (directory / "result.json").is_file():
            raise HTTPException(404, "Clip not found")
        return directory

    def infer(source, directory, clip_id):
        started = time.perf_counter()
        normalized = directory / "video.mp4"
        duration = normalize_video(source, normalized)
        prediction = app.state.recognizer.predict(normalized)
        result = {"id": clip_id, "text": prediction["prediction"], "duration_s": duration,
                  "processing_s": time.perf_counter() - started,
                  "created_at": datetime.now(timezone.utc).isoformat(), "model_result": prediction,
                  "source_filename": source.name}
        write_json(directory / "result.json", result)
        return {key: result[key] for key in ("id", "text", "duration_s", "processing_s")}

    @app.post("/api/transcribe")
    async def transcribe(request: Request):
        media_type = request.headers.get("content-type", "").split(";")[0].strip()
        if media_type not in {"video/webm", "video/mp4", "application/octet-stream"}:
            raise HTTPException(415, "Send an MP4 or WebM video.")
        if app.state.busy:
            raise HTTPException(409, "The recognizer is busy. Please try again shortly.")
        app.state.busy = True
        clip_id = str(uuid4())
        directory = data_dir / clip_id
        success = False
        try:
            directory.mkdir()
            suffix = {"video/webm": ".webm", "video/mp4": ".mp4"}.get(media_type, ".bin")
            source = directory / ("source" + suffix)
            size = 0
            async with asyncio.timeout(30):
                with source.open("wb") as handle:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Video exceeds the 25 MiB upload limit.")
                        handle.write(chunk)
            if not size:
                raise HTTPException(422, "The recording is empty.")
            # Keep the busy flag until the worker finishes even if the browser disconnects.
            result = await anyio.to_thread.run_sync(lambda: infer(source, directory, clip_id))
            success = True
            return result
        except TimeoutError:
            raise HTTPException(408, "Upload timed out. Try a shorter recording.")
        except subprocess.TimeoutExpired:
            raise HTTPException(422, "Video processing timed out. Try a shorter recording.")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        finally:
            if not success:
                shutil.rmtree(directory, ignore_errors=True)
            app.state.busy = False

    @app.post("/api/clips/{clip_id}/label")
    async def label_clip(clip_id: str, label: Label):
        directory = clip_directory(clip_id)
        value = label.model_dump() | {"id": clip_id, "label_source": "user_correction_after_prediction",
                                      "labelled_at": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "label.json", value)
        return {"id": clip_id, "saved": True}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str):
        directory = clip_directory(clip_id)
        shutil.rmtree(directory)
        return {"id": clip_id, "deleted": True}

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/assets", StaticFiles(directory=ROOT / "web", check_dir=False), name="assets")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "vendor" / "auto_avsr")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "captures")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18870)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.upstream, args.checkpoint, args.data_dir, args.threads),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
