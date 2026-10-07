"""Video probing and frame reading (OpenCV; ffmpeg/ffprobe for metadata and audio)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    has_audio: bool


def probe_video(path: Path | str) -> VideoInfo:
    path = Path(path)
    if shutil.which("ffprobe"):
        q = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if q.returncode == 0:
            streams = json.loads(q.stdout).get("streams", [])
            vid = next((s for s in streams if s.get("codec_type") == "video"), None)
            if vid is not None:
                num, den = vid["r_frame_rate"].split("/")
                fps = float(num) / float(den)
                dur = float(vid.get("duration") or 0.0)
                n = int(vid.get("nb_frames") or 0)
                if n == 0 and dur > 0:
                    n = round(dur * fps)
                if dur == 0 and n > 0:
                    dur = n / fps
                return VideoInfo(
                    path,
                    int(vid["width"]),
                    int(vid["height"]),
                    fps,
                    n,
                    dur,
                    any(s.get("codec_type") == "audio" for s in streams),
                )
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info = VideoInfo(path, w, h, fps, n, n / fps, False)
    cap.release()
    return info


def list_videos(video_dir: Path | str, pattern: str = "*.mp4", names: list[str] | None = None) -> dict[str, Path]:
    """Camera name -> video path. Names default to the file stems, in natural sort order."""
    paths = sorted(
        Path(video_dir).glob(pattern), key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.stem)]
    )
    if not paths:
        raise FileNotFoundError(f"no videos matching {pattern!r} in {video_dir}")
    if names is not None:
        if len(names) != len(paths):
            raise ValueError(f"{len(names)} camera names for {len(paths)} videos")
        return dict(zip(names, paths))
    return {p.stem: p for p in paths}


def iter_frames(
    path: Path | str, start: int = 0, stop: int | None = None, step: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode forward (frame-exact; no seeking) and yield (frame_index, BGR image) for frames start, start+step, ..."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    f = 0
    try:
        while True:
            if stop is not None and f >= stop:
                break
            if f < start or (f - start) % step:
                if not cap.grab():
                    break
                f += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            yield f, img
            f += 1
    finally:
        cap.release()


class FrameReader:
    """Random access by decoding forward from the nearest earlier position (seeks only for large jumps)."""

    def __init__(self, path: Path | str, seek_threshold: int = 90) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.pos = 0
        self.seek_threshold = seek_threshold

    def read(self, frame: int) -> np.ndarray | None:
        if frame < self.pos or frame - self.pos > self.seek_threshold:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.pos = frame
        while self.pos < frame:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, img = self.cap.read()
        self.pos += 1
        return img if ok else None

    def close(self) -> None:
        self.cap.release()


def read_audio(path: Path | str, sample_rate: int = 8000, max_s: float = 0.0) -> np.ndarray:
    """Mono float32 PCM decoded with ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_s > 0:
        cmd += ["-t", f"{max_s}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {out.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(out.stdout, np.float32).copy()


def trim_video(
    src: Path | str, dst: Path | str, start_s: float, duration_s: float | None, fps: float | None = None, gop: int = 60
) -> None:
    """Frame-accurate trim (re-encode) so that every camera starts on the same instant."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.6f}", "-i", str(src)]
    if duration_s is not None and duration_s > 0:
        cmd += ["-t", f"{duration_s:.6f}"]
    vf = [f"fps={fps}"] if fps else []
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-g", str(gop), "-pix_fmt", "yuv420p", "-an", str(dst)]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {out.stderr.decode(errors='ignore')[-300:]}")


def write_video(path: Path | str, size: tuple[int, int], fps: float, crf: int = 18) -> subprocess.Popen:
    """ffmpeg process accepting raw BGR frames on stdin."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    src = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd = ["ffmpeg", "-v", "error", "-y", *src, *enc, str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
