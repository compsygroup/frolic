"""Stage `sync`: time-align the camera videos and write work/videos.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frolic.config import FrolicConfig
from frolic.io.video import VideoInfo, list_videos, probe_video, read_audio, trim_video
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.sync.audio import PairLag, closure_errors, offsets_from_pairs, pairwise_lags

log = get_logger(__name__)


def audio_offsets(
    videos: dict[str, Path], cfg: FrolicConfig, reference: str
) -> tuple[dict[str, float], dict[str, Any]]:
    s = cfg.sync
    audio = {name: read_audio(path, s.sample_rate, s.max_audio_s) for name, path in videos.items()}
    short = [n for n, a in audio.items() if len(a) < s.min_overlap_s * s.sample_rate]
    if short:
        raise RuntimeError(f"audio shorter than sync.min_overlap_s in {short}")
    pairs = pairwise_lags(audio, s.sample_rate, s.min_overlap_s, s.fine_window_ms / 1000.0, s.band_hz, s.envelope_ms)
    if not pairs:
        raise RuntimeError("no camera pair could be correlated")
    offsets = offsets_from_pairs(pairs, list(videos), reference)
    closure = closure_errors(pairs)
    worst = max((abs(v) for v in closure.values()), default=0.0)
    report = {
        "method": "audio",
        "pairs": [p.__dict__ for p in pairs],
        "closure_s": closure,
        "max_closure_ms": round(1000 * worst, 2),
        "ok": bool(worst <= s.max_closure_ms / 1000.0),
    }
    if not report["ok"]:
        log.warning("triangle closure %.1f ms exceeds sync.max_closure_ms", 1000 * worst)
    return offsets, report


def _offsets(
    videos: dict[str, Path], infos: dict[str, VideoInfo], cfg: FrolicConfig, reference: str
) -> tuple[dict[str, float], dict[str, Any]]:
    if cfg.sync.manual_offsets_s is not None:
        missing = set(videos) - set(cfg.sync.manual_offsets_s)
        if missing:
            raise ValueError(f"sync.manual_offsets_s lacks {sorted(missing)}")
        return {n: float(cfg.sync.manual_offsets_s[n]) for n in videos}, {"method": "manual"}
    if cfg.input.synced:
        return dict.fromkeys(videos, 0.0), {"method": "assumed_synced"}
    if not all(infos[n].has_audio for n in videos):
        raise RuntimeError("input.synced is false but some videos have no audio; give sync.manual_offsets_s")
    return audio_offsets(videos, cfg, reference)


def common_window(
    infos: dict[str, VideoInfo], offsets: dict[str, float], start_s: float, duration_s: float
) -> tuple[float, float]:
    """Aligned time window [t0, t1) available in every camera (file time = aligned time + offset)."""
    t0 = max(0.0, max(-o for o in offsets.values())) + start_s
    t1 = min(infos[n].duration_s - offsets[n] for n in infos)
    if duration_s > 0:
        t1 = min(t1, t0 + duration_s)
    if t1 <= t0:
        raise RuntimeError(f"no common overlap (t0={t0:.2f}, t1={t1:.2f})")
    return t0, t1


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    videos = list_videos(cfg.input.video_dir, cfg.input.pattern, cfg.input.camera_names)
    names = list(videos)
    if len(names) < 2:
        raise RuntimeError("need at least two cameras")
    infos = {n: probe_video(p) for n, p in videos.items()}
    reference = cfg.sync.reference or names[0]
    if reference not in names:
        raise ValueError(f"sync.reference {reference!r} is not a camera ({names})")
    offsets, report = _offsets(videos, infos, cfg, reference)
    fps = cfg.input.fps or float(np.median([infos[n].fps for n in names]))
    if any(abs(infos[n].fps - fps) > 0.01 for n in names) and not cfg.sync.trim:
        log.warning(
            "frame rates differ between cameras (%s); enable sync.trim to resample", {n: infos[n].fps for n in names}
        )
    t0, t1 = common_window(infos, offsets, cfg.input.start_s, cfg.input.duration_s)
    n_frames = int(np.floor((t1 - t0) * fps + 1e-6))
    trim = cfg.sync.trim and (
        any(abs(v) > 1e-9 for v in offsets.values()) or t0 > 0 or cfg.input.duration_s > 0 or cfg.input.fps is not None
    )
    out_videos: dict[str, str] = {}
    frame_offset: dict[str, int] = {}
    sdir = work.stage_dir("sync")
    for n in names:
        file_t0 = t0 + offsets[n]
        if trim:
            dst = sdir / "videos" / f"{n}.mp4"
            log.info("trimming %s from %.3f s for %.3f s", n, file_t0, t1 - t0)
            trim_video(videos[n], dst, file_t0, t1 - t0, fps=fps)
            got = probe_video(dst).n_frames
            n_frames = min(n_frames, got)
            out_videos[n], frame_offset[n] = str(dst), 0
        else:
            out_videos[n], frame_offset[n] = str(videos[n]), round(file_t0 * fps)
    doc = {
        "cameras": names,
        "videos": out_videos,
        "frame_offset": frame_offset,
        "fps": fps,
        "width": {n: infos[n].width for n in names},
        "height": {n: infos[n].height for n in names},
        "n_frames": n_frames,
        "offsets_s": offsets,
        "reference": reference,
        "window_s": [t0, t1],
        "trimmed": trim,
        "sync": report,
    }
    work.write_videos(doc)
    with open(sdir / "offsets.json", "w") as fh:
        json.dump({"reference": reference, "offsets_s": offsets, "window_s": [t0, t1], "report": report}, fh, indent=1)
    return {"cameras": names, "n_frames": n_frames, "fps": fps, "method": report["method"], "trimmed": trim}


def _unused(_: PairLag) -> None:  # keeps the PairLag type exported for callers reading offsets.json
    return None
