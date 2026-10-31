"use strict";

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "service-status", "error", "video", "camera-placeholder", "capture-state",
  "capture-hint", "camera-button", "record-button", "stop-button", "file-input",
  "transcribe-button", "discard-button", "progress-status", "transcript",
  "result-detail", "copy-button", "confirm-reference", "save-button", "save-status",
  "speech-mode", "mode-field", "recording-indicator", "timer", "limits",
].map((id) => [id, $(id)]));

let stream = null;
let recorder = null;
let clip = null;
let clipUrl = null;
let clipPlayable = false;
let result = null;
let busy = false;
let cameraPending = false;
let ready = false;
let timer = null;
let stopTimer = null;
let recordingStarted = 0;
let maxSeconds = 12;
let maxBytes = 25 * 1024 * 1024;
let sourceMode = "silent";
let saved = false;
let generation = 0;
let leaving = false;
const sessionId = getSessionId();

function getSessionId() {
  const key = "silent-speech-session";
  try {
    let id = sessionStorage.getItem(key);
    if (!id) {
      id = `${new Date().toISOString().slice(0, 10)}-${crypto.randomUUID()}`;
      sessionStorage.setItem(key, id);
    }
    return id;
  } catch {
    return `${new Date().toISOString().slice(0, 10)}-${crypto.randomUUID()}`;
  }
}

function isRecording() { return recorder && recorder.state !== "inactive"; }
function isCurrent(version) { return version === generation && !leaving; }
function error(message) { ui.error.textContent = message; ui.error.hidden = !message; }
function status(message) { ui["progress-status"].textContent = message; }

function update() {
  const recording = Boolean(isRecording());
  ui["camera-button"].disabled = busy || cameraPending || recording;
  ui["camera-button"].textContent = cameraPending ? "Opening camera…" : stream ? "Turn camera off" : "Enable camera";
  ui["record-button"].disabled = busy || cameraPending || !stream || recording;
  ui["record-button"].hidden = recording;
  ui["stop-button"].hidden = !recording;
  ui["file-input"].disabled = busy || cameraPending || recording;
  ui["transcribe-button"].disabled = busy || recording || !clip || !clipPlayable || !ready || Boolean(result);
  ui["transcribe-button"].textContent = busy && !result ? "Reading your lips…" : "Read my lips →";
  ui["discard-button"].disabled = busy || recording || !clip;
  ui.transcript.disabled = busy || !result;
  ui["copy-button"].disabled = busy || !ui.transcript.value.trim();
  ui["confirm-reference"].disabled = busy || !result || saved;
  ui["speech-mode"].disabled = busy || saved;
  ui["save-button"].disabled = busy || !result || saved || !ui.transcript.value.trim() || !ui["confirm-reference"].checked;
  ui["save-button"].textContent = saved ? "Confirmed words saved" : "Save confirmed words";
}

function clearResult() {
  result = null;
  saved = false;
  ui.transcript.value = "";
  ui["confirm-reference"].checked = false;
  ui["result-detail"].textContent = "Experimental lip reading can miss or change words. Check the result.";
  ui["save-status"].textContent = "Saved examples are for training only. Saving does not train the model yet.";
}

function releaseClip() {
  generation += 1;
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = null;
  clip = null;
  clipPlayable = false;
  clearResult();
  ui["file-input"].value = "";
  ui["mode-field"].hidden = true;
}

async function showCamera() {
  ui.video.pause();
  ui.video.removeAttribute("src");
  ui.video.srcObject = stream;
  ui.video.controls = false;
  ui.video.classList.toggle("live", Boolean(stream));
  ui["camera-placeholder"].hidden = Boolean(stream);
  if (stream) await ui.video.play();
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (!clip) {
    ui.video.srcObject = null;
    ui.video.classList.remove("live");
    ui["camera-placeholder"].hidden = false;
  }
  ui["capture-state"].textContent = clip ? "Clip ready · camera off" : "Camera off";
}

ui["camera-button"].addEventListener("click", async () => {
  error("");
  if (stream) {
    stopCamera();
    update();
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    error("Camera access needs a secure connection. Open the HTTPS Tailscale address, or use localhost on this device.");
    return;
  }
  if (!window.MediaRecorder) {
    error("This browser cannot record video here. Try a current Safari or Chrome, or choose a video file.");
    return;
  }
  cameraPending = true;
  const version = generation;
  update();
  try {
    const opened = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    if (leaving || version !== generation) { opened.getTracks().forEach((track) => track.stop()); return; }
    stream = opened;
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (isRecording()) stopRecording();
      stopCamera();
      update();
    });
    if (!clip) await showCamera();
    if (!isCurrent(version)) return;
    ui["capture-state"].textContent = clip ? "Clip ready · camera on" : "Camera ready";
    status("Select Record, then silently mouth one short sentence.");
  } catch (e) {
    if (!isCurrent(version)) return;
    stopCamera();
    error(e.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access for this site in your browser, then try again." : `Could not open the camera: ${e.message}`);
  } finally {
    if (isCurrent(version)) { cameraPending = false; update(); }
  }
});

function finishTimers() {
  clearInterval(timer);
  clearTimeout(stopTimer);
  timer = stopTimer = null;
  ui["recording-indicator"].hidden = true;
}

function stopRecording() {
  if (!isRecording()) return;
  recorder.stop();
  finishTimers();
  busy = true;
  ui["capture-state"].textContent = "Preparing clip…";
  update();
}

function displayClip(blob, mode, uploaded) {
  clip = blob;
  clipPlayable = false;
  clipUrl = URL.createObjectURL(blob);
  sourceMode = mode;
  ui.video.pause();
  ui.video.srcObject = null;
  ui.video.classList.remove("live");
  ui.video.src = clipUrl;
  ui.video.controls = true;
  ui.video.load();
  ui["camera-placeholder"].hidden = true;
  ui["mode-field"].hidden = !uploaded;
  ui["speech-mode"].value = mode;
  ui["capture-state"].textContent = "Clip ready";
  ui["capture-hint"].textContent = "Replay your clip, then select Read my lips.";
  status("Ready to transcribe. Your clip has not been uploaded yet.");
  update();
}

ui["record-button"].addEventListener("click", async () => {
  if (!stream || busy || isRecording()) return;
  error("");
  releaseClip();
  const version = generation;
  busy = true;
  update();
  try {
    await showCamera();
    if (!isCurrent(version)) return;
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/mp4;codecs=avc1.42E01E", "video/mp4", "video/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    const chunks = [];
    let recordingFailed = false;
    const current = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recorder = current;
    current.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
    current.addEventListener("error", (event) => {
      if (!isCurrent(version)) return;
      recordingFailed = true;
      error(`Recording failed: ${event.error?.message || "the browser stopped the recorder"}`);
      stopRecording();
    });
    current.addEventListener("stop", () => {
      if (!isCurrent(version)) return;
      finishTimers();
      recorder = null;
      busy = false;
      if (recordingFailed) {
        ui["capture-state"].textContent = stream ? "Camera ready" : "Camera off";
        status("Recording failed. Please record another clip.");
        update();
        return;
      }
      const blob = new Blob(chunks, { type: current.mimeType || chunks[0]?.type || "video/mp4" });
      if (!blob.size || blob.size > maxBytes) {
        error(blob.size ? "That recording is too large. Try a shorter clip." : "The camera produced an empty recording. Please try again.");
        ui["capture-state"].textContent = "Camera ready";
        update();
        return;
      }
      displayClip(blob, "silent", false);
    });
    current.start(250);
    busy = false;
    recordingStarted = performance.now();
    ui.timer.textContent = "0.0s";
    ui["recording-indicator"].hidden = false;
    ui["capture-state"].textContent = "Recording";
    status("Mouth your sentence naturally. Select Stop when you finish.");
    timer = setInterval(() => { ui.timer.textContent = `${((performance.now() - recordingStarted) / 1000).toFixed(1)}s`; }, 100);
    // Leave a margin for the recorder's final encoded frame below the server limit.
    stopTimer = setTimeout(stopRecording, Math.max(500, (maxSeconds - 0.5) * 1000));
  } catch (e) {
    if (!isCurrent(version)) return;
    recorder = null;
    busy = false;
    error(`Could not start recording: ${e.message}`);
  }
  update();
});

ui["stop-button"].addEventListener("click", stopRecording);
document.addEventListener("visibilitychange", () => { if (document.hidden && isRecording()) stopRecording(); });

ui["file-input"].addEventListener("change", () => {
  const file = ui["file-input"].files[0];
  if (!file || busy || isRecording()) return;
  error("");
  if (file.size > maxBytes || !file.size) {
    error(file.size ? "That video is too large. Choose a clip under 25 MB." : "That video file is empty.");
    ui["file-input"].value = "";
    return;
  }
  releaseClip();
  displayClip(file, "voiced_muted", true);
});

ui.video.addEventListener("error", () => {
  if (clip) {
    clipPlayable = false;
    error("This browser could not replay the file. Try an MP4 or WebM clip that plays correctly before transcribing.");
    update();
  }
});
ui.video.addEventListener("loadeddata", () => {
  if (clip && ui.video.currentSrc === clipUrl) {
    clipPlayable = true;
    update();
  }
});

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : null;
    throw new Error(detail || `Request failed (${response.status}). Please try again.`);
  }
  return data;
}

ui["transcribe-button"].addEventListener("click", async () => {
  if (!clip || !clipPlayable || busy || result || !ready) return;
  error("");
  const version = generation;
  busy = true;
  status("Uploading and reading your clip. Longer sentences can take a little time.");
  update();
  try {
    const prediction = await api("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": /^(video\/webm|video\/mp4)(;|$)/.test(clip.type) ? clip.type : "application/octet-stream", "X-Silent-Speech": "1" },
      body: clip,
    });
    if (version !== generation || leaving) return;
    result = prediction;
    ui.transcript.value = prediction.text || "";
    ui["result-detail"].textContent = "Model prediction — replay the clip and correct any missed words.";
    status(prediction.text ? "Transcript ready. Your clip is stored on the workstation." : "No words were recognized. You can retry with a clearer, shorter sentence.");
    ui.transcript.focus();
  } catch (e) {
    if (version === generation && !leaving) {
      error(e.message);
      status("Could not transcribe this clip. You can retry or record another.");
    }
  } finally {
    if (isCurrent(version)) { busy = false; update(); }
  }
});

ui.transcript.addEventListener("input", () => {
  ui["confirm-reference"].checked = false;
  saved = false;
  ui["save-status"].textContent = "Check the replay and confirm the edited words before saving.";
  update();
});
ui["speech-mode"].addEventListener("change", () => {
  sourceMode = ui["speech-mode"].value;
  ui["confirm-reference"].checked = false;
  update();
});
ui["confirm-reference"].addEventListener("change", update);

ui["save-button"].addEventListener("click", async () => {
  const reference = ui.transcript.value.trim();
  if (!result || !reference || busy || saved || !ui["confirm-reference"].checked) return;
  const version = generation;
  busy = true;
  error("");
  update();
  try {
    await api(`/api/clips/${encodeURIComponent(result.id)}/label`, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Silent-Speech": "1" },
      body: JSON.stringify({ reference, session_id: sessionId, split: "train", mode: sourceMode }),
    });
    if (!isCurrent(version)) return;
    saved = true;
    ui["save-status"].textContent = "Saved on your workstation for future calibration. The model has not been retrained.";
  } catch (e) { if (isCurrent(version)) error(e.message); }
  finally { if (isCurrent(version)) { busy = false; update(); } }
});

ui["copy-button"].addEventListener("click", async () => {
  const version = generation;
  try {
    await navigator.clipboard.writeText(ui.transcript.value);
    if (isCurrent(version)) status("Transcript copied.");
  } catch {
    if (!isCurrent(version)) return;
    ui.transcript.focus();
    ui.transcript.select();
    status("Select Copy from your browser, or press Command-C to copy the selected text.");
  }
});

ui["discard-button"].addEventListener("click", async () => {
  if (busy || isRecording()) return;
  let version = generation;
  busy = true;
  error("");
  update();
  try {
    if (result) await api(`/api/clips/${encodeURIComponent(result.id)}`, { method: "DELETE", headers: { "X-Silent-Speech": "1" } });
    if (!isCurrent(version)) return;
    releaseClip();
    version = generation;
    await showCamera();
    if (!isCurrent(version)) return;
    ui["capture-state"].textContent = stream ? "Camera ready" : "Camera off";
    ui["capture-hint"].textContent = "Face the camera in good light, with your whole mouth visible.";
    status("Clip discarded. Ready for another sentence.");
  } catch (e) { if (isCurrent(version)) error(`Could not discard the clip: ${e.message}`); }
  finally { if (isCurrent(version)) { busy = false; update(); } }
});

window.addEventListener("pagehide", () => {
  leaving = true;
  if (isRecording()) recorder.stop();
  recorder = null;
  cameraPending = false;
  finishTimers();
  stopCamera();
  releaseClip();
  ui.video.srcObject = null;
  ui.video.removeAttribute("src");
  ui.video.classList.remove("live");
  ui["camera-placeholder"].hidden = false;
  busy = false;
});
window.addEventListener("pageshow", (event) => {
  leaving = false;
  if (event.persisted) {
    ui["capture-state"].textContent = "Camera off";
    ui["capture-hint"].textContent = "Face the camera in good light, with your whole mouth visible.";
    status("Enable your camera or choose a short video to begin.");
  }
  update();
});

async function checkHealth() {
  try {
    const health = await api("/api/health");
    ready = health.ready === true;
    maxSeconds = health.max_duration_s || 12;
    maxBytes = health.max_upload_bytes || 25 * 1024 * 1024;
    ui["service-status"].textContent = ready ? "Recognizer ready" : "Recognizer warming up";
    ui["service-status"].className = `service-status${ready ? " ready" : ""}`;
    ui.limits.textContent = `Up to ${maxSeconds} seconds · no audio track`;
  } catch {
    ready = false;
    ui["service-status"].textContent = "Recognizer unavailable";
    ui["service-status"].className = "service-status unavailable";
  }
  update();
}

checkHealth();
setInterval(checkHealth, 15000);
