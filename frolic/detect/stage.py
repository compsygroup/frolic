"""Detect stage: per-camera 2D boxes + Halpe-26 keypoints on the aligned frame timeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frolic.config import FrolicConfig
from frolic.detect.adapter import PoseDetector, make_detector
from frolic.detect.masks import masked_rows, static_rows
from frolic.io.detections import N_KP, Detections
from frolic.io.video import iter_frames, probe_video
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.skeleton import HALPE26_TORSO

log = get_logger(__name__)
TORSO_POLY = [5, 6, 12, 11]


def torso_colour(
    img_bgr: np.ndarray, kps: np.ndarray, ks: np.ndarray, hsv: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray]:
    """HSV median inside the shoulder/hip polygon; returns (hsv image cache, colour or NaN)."""
    if min(ks[j] for j in HALPE26_TORSO[:4]) < 0.3:
        return hsv, np.full(3, np.nan, np.float32)
    if hsv is None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(kps[TORSO_POLY]).astype(np.int32)], 1)
    if m.sum() < 100:
        return hsv, np.full(3, np.nan, np.float32)
    sel = m > 0
    return hsv, np.array([np.median(hsv[..., c][sel]) for c in range(3)], np.float32)


def detect_video(
    detector: PoseDetector, video: Path, camera: str, frame_offset: int, n_frames: int, step: int, meta: dict[str, Any]
) -> Detections:
    rows: list[tuple] = []
    start, stop = frame_offset, frame_offset + n_frames
    for f, img in iter_frames(video, start, stop, step):
        boxes, bs, kps, ks = detector(img)
        hsv = None
        for i in range(len(boxes)):
            hsv, col = torso_colour(img, kps[i], ks[i], hsv)
            rows.append((f - frame_offset, boxes[i], bs[i], kps[i], ks[i], col))
        if len(rows) and (f - start) % (500 * step) == 0:
            log.info("%s: frame %d/%d", camera, f - start, n_frames)
    if not rows:
        return Detections.empty(camera, meta)
    det = Detections(
        camera,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        [r[4] for r in rows],
        [r[5] for r in rows],
        meta,
    )
    return det


def postprocess(det: Detections, cfg: FrolicConfig) -> Detections:
    drop = masked_rows(det, cfg.detect.masks.get(det.camera, []), cfg.detect.mask_overlap)
    drop |= static_rows(
        det,
        cfg.detect.static_occupancy,
        cfg.detect.static_spread,
        noisy_occupancy=cfg.detect.noisy_occupancy,
        noisy_kp=cfg.detect.noisy_kp,
        noisy_box=cfg.detect.noisy_box,
    )
    if drop.any():
        log.info("%s: dropped %d masked/static detections", det.camera, int(drop.sum()))
    return det.subset(np.flatnonzero(~drop))


def _mock_copy(cfg: FrolicConfig, cam: str, videos: dict[str, Any], out: Path) -> Detections:
    src = Path(cfg.detect.mock_dir) / f"{cam}.npz"
    det = Detections.load(src)
    off = int(videos.get("frame_offset", {}).get(cam, 0))
    if off:
        keep = det.frame >= off
        det = det.subset(np.flatnonzero(keep))
        det.frame = det.frame - off
    det.meta.update(
        {
            "width": videos["width"][cam],
            "height": videos["height"][cam],
            "fps": videos["fps"],
            "step": 1,
            "n_frames": videos["n_frames"],
            "video": videos["videos"][cam],
        }
    )
    return det


def _complete(path: Path, n_frames: int) -> bool:
    side = path.with_suffix(".json")
    if not path.exists() or not side.exists():
        return False
    with open(side) as fh:
        meta = json.load(fh)
    return int(meta.get("n_frames", -1)) == n_frames and bool(meta.get("complete", False))


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    videos = work.read_videos()
    out_dir = work.stage_dir("detect")
    n_frames = int(videos["n_frames"])
    detector = None
    counts = {}
    for cam in videos["cameras"]:
        path = out_dir / f"{cam}.npz"
        if _complete(path, n_frames):
            log.info("%s: detections exist, skipped", cam)
            counts[cam] = len(Detections.load(path))
            continue
        if cfg.detect.detector == "mock":
            if cfg.detect.mock_dir is None:
                raise ValueError("detect.detector = mock needs detect.mock_dir")
            det = _mock_copy(cfg, cam, videos, out_dir)
        else:
            if detector is None:
                detector = make_detector(cfg.detect, cfg.compute.device)
            video = Path(videos["videos"][cam])
            info = probe_video(video)
            meta = {
                "width": info.width,
                "height": info.height,
                "fps": videos["fps"],
                "step": cfg.detect.step,
                "n_frames": n_frames,
                "video": str(video),
            }
            det = detect_video(detector, video, cam, int(videos["frame_offset"][cam]), n_frames, cfg.detect.step, meta)
        det = postprocess(det, cfg)
        det.meta["complete"] = True
        det.save(path)
        counts[cam] = len(det)
        log.info("%s: %d detections on %d frames", cam, len(det), len(det.frames))
    if cfg.detect.detector == "mock" and cfg.detect.mock_dir is not None:
        extra = Path(cfg.detect.mock_dir) / "joints.npz"
        if extra.exists():
            shutil.copy(extra, out_dir / "mock_joints.npz")
    return {"detections": counts, "n_keypoints": N_KP}
