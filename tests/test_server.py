"""HTTP/storage contracts; real media decoding with the recognizer boundary stubbed."""
import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from scripts.serve import create_app, normalize_video


class StubRecognizer:
    def __init__(self, *args):
        pass

    def predict(self, path):
        return {"prediction": "HELLO THERE", "video_sha256": "test-only"}

    def close(self):
        pass


class ServerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        for name, seconds, audio in [("valid.mp4", "1", False), ("long.webm", "12.2", False), ("audio.mp4", "1", True)]:
            command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=30:d={seconds}"]
            if audio:
                command += ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-shortest"]
            command += ["-threads", "1", "-y", str(cls.root / name)]
            subprocess.run(command, check=True, timeout=20)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_video_duration_and_audio_validation(self):
        result = self.root / "normalized.mp4"
        self.assertEqual(normalize_video(self.root / "valid.mp4", result), 1)
        with self.assertRaisesRegex(ValueError, "between"):
            normalize_video(self.root / "long.webm", result)
        with self.assertRaisesRegex(ValueError, "without audio"):
            normalize_video(self.root / "audio.mp4", result)

    def test_capture_label_delete_and_privacy(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.serve.Recognizer", StubRecognizer):
            app = create_app("unused", "unused", directory)
            with TestClient(app) as client:
                self.assertTrue(client.get("/api/health").json()["ready"])
                headers = {"X-Silent-Speech": "1", "Content-Type": "video/mp4"}
                response = client.post("/api/transcribe", content=(self.root / "valid.mp4").read_bytes(), headers=headers)
                self.assertEqual(response.status_code, 200, response.text)
                clip_id = response.json()["id"]
                clip = Path(directory) / clip_id
                self.assertTrue((clip / "video.mp4").is_file())
                self.assertEqual(response.json()["text"], "HELLO THERE")
                self.assertEqual(client.get(f"/data/captures/{clip_id}/video.mp4").status_code, 404)
                payload = {"reference": "hello there", "session_id": "day1", "mode": "silent", "split": "test"}
                url = f"/api/clips/{clip_id}/label"
                self.assertEqual(client.post(url, json=payload, headers={"X-Silent-Speech": "1"}).status_code, 422)
                payload["split"] = "train"
                self.assertEqual(client.post(url, json=payload, headers={"X-Silent-Speech": "1"}).status_code, 200)
                label = json.loads((clip / "label.json").read_text())
                self.assertEqual(label["label_source"], "user_correction_after_prediction")
                self.assertEqual(client.delete(f"/api/clips/{clip_id}", headers={"X-Silent-Speech": "1"}).status_code, 200)
                self.assertFalse(clip.exists())

    def test_cancelled_request_keeps_worker_exclusive(self):
        started, release = threading.Event(), threading.Event()

        class WaitingRecognizer(StubRecognizer):
            def predict(self, path):
                started.set()
                release.wait(5)
                return super().predict(path)

        async def run():
            with tempfile.TemporaryDirectory() as directory, patch("scripts.serve.Recognizer", WaitingRecognizer):
                app = create_app("unused", "unused", directory)
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                        headers = {"X-Silent-Speech": "1", "Content-Type": "video/mp4"}
                        upload = asyncio.create_task(client.post("/api/transcribe", content=(self.root / "valid.mp4").read_bytes(), headers=headers))
                        self.assertTrue(await asyncio.to_thread(started.wait, 5))
                        upload.cancel()
                        try:
                            await asyncio.sleep(0.05)
                            self.assertTrue((await client.get("/api/health")).json()["busy"])
                            self.assertEqual((await client.post("/api/transcribe", content=b"x", headers=headers)).status_code, 409)
                        finally:
                            release.set()
                            with self.assertRaises(asyncio.CancelledError):
                                await upload
                        for clip in Path(directory).iterdir():
                            self.assertTrue((clip / "result.json").is_file())
                        self.assertFalse((await client.get("/api/health")).json()["busy"])
        asyncio.run(run())

    def test_failed_upload_cleanup_and_request_boundary(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.serve.Recognizer", StubRecognizer):
            app = create_app("unused", "unused", directory)
            with TestClient(app) as client:
                self.assertEqual(client.post("/api/transcribe", content=b"bad").status_code, 403)
                headers = {"X-Silent-Speech": "1", "Content-Type": "video/mp4", "Origin": "https://unrelated.example"}
                self.assertEqual(client.post("/api/transcribe", content=b"bad", headers=headers).status_code, 403)
                headers.pop("Origin")
                self.assertEqual(client.post("/api/transcribe", content=b"bad", headers=headers).status_code, 422)
                self.assertEqual(list(Path(directory).iterdir()), [])
                with patch("scripts.serve.MAX_UPLOAD_BYTES", 2):
                    self.assertEqual(client.post("/api/transcribe", content=b"bad", headers=headers).status_code, 413)
                self.assertEqual(list(Path(directory).iterdir()), [])
                self.assertFalse(client.get("/api/health").json()["busy"])


if __name__ == "__main__":
    unittest.main()
