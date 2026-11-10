"""Appearance re-identification embeddings (OSNet through torchreid) of track crops."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frolic.config import TrackConfig
from frolic.io.detections import Detections
from frolic.io.tracks import Tracks
from frolic.io.video import FrameReader
from frolic.log import get_logger

log = get_logger(__name__)
Embedding = dict[tuple[str, int], np.ndarray]


def weights_path(cfg: TrackConfig) -> Path | None:
    p = cfg.reid_weights or os.environ.get("FROLIC_OSNET_WEIGHTS")
    if p is None:
        return None
    p = Path(p)
    return p if p.exists() else None


class OsnetEmbedder:
    def __init__(self, weights: Path, device: str = "cpu", model: str = "osnet_ain_x1_0", threads: int = 8) -> None:
        try:
            import torch
            from torchreid.reid.utils import FeatureExtractor
        except ImportError as e:
            raise ImportError(f"torchreid is not available ({e}); pip install 'frolic[track]'") from e
        torch.set_num_threads(threads)
        dev = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.ex = FeatureExtractor(model_name=model, model_path=str(weights), device=dev)

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch

        with torch.no_grad():
            e = self.ex(crops).cpu().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def crop(img: np.ndarray, box: np.ndarray, min_w: int = 16, min_h: int = 32) -> np.ndarray | None:
    x1, y1, x2, y2 = [round(float(q)) for q in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    c = img[y1:y2, x1:x2]
    if c.size == 0 or x2 - x1 < min_w or y2 - y1 < min_h:
        return None
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def _flush(embedder: Any, pend: list[tuple[Any, np.ndarray]], out: dict[Any, np.ndarray]) -> None:
    if pend:
        e = embedder([c for _, c in pend])
        out.update({k: v for (k, _), v in zip(pend, e)})
        pend.clear()


def embed_crops(
    embedder: Any,
    jobs: dict[tuple[str, int], list[tuple[Any, np.ndarray]]],
    videos: dict[str, str],
    frame_offset: dict[str, int],
    batch: int = 64,
) -> dict[Any, np.ndarray]:
    """jobs: (camera, aligned frame) -> [(key, box)] -> {key: unit vector}. Frames are decoded in memory only."""
    out: dict[Any, np.ndarray] = {}
    for cam in sorted({c for c, _ in jobs}):
        reader = FrameReader(videos[cam])
        pend: list[tuple[Any, np.ndarray]] = []
        for f in sorted(f for c, f in jobs if c == cam):
            img = reader.read(f + int(frame_offset.get(cam, 0)))
            if img is None:
                continue
            for key, box in jobs[(cam, f)]:
                c = crop(img, box)
                if c is not None:
                    pend.append((key, c))
                if len(pend) >= batch:
                    _flush(embedder, pend, out)
        _flush(embedder, pend, out)
        reader.close()
    return out


def embed_tracks(
    tracks: Tracks,
    dets: dict[str, Detections],
    videos: dict[str, str],
    frame_offset: dict[str, int],
    embedder: Any,
    every_s: float = 1.0,
) -> Embedding:
    """{(track id, track frame): unit vector (mean over cameras)} sampled every `every_s` seconds."""
    step = max(1, round(every_s * tracks.fps))
    jobs: dict[tuple[str, int], list[tuple[tuple[str, int, str], np.ndarray]]] = {}
    for p, pid in enumerate(tracks.ids):
        for i in range(0, tracks.n_frames, step):
            if not tracks.has_3d[p, i]:
                continue
            f = tracks.source_frame(i)
            for v, cam in enumerate(tracks.cameras):
                r = int(tracks.support[p, i, v])
                if r < 0:
                    continue
                jobs.setdefault((cam, f), []).append(((pid, i, cam), dets[cam].box[r]))
    E = embed_crops(embedder, jobs, videos, frame_offset)
    acc: dict[tuple[str, int], list[np.ndarray]] = {}
    for (pid, i, _), e in E.items():
        acc.setdefault((pid, i), []).append(e)
    out = {}
    for k, es in acc.items():
        m = np.mean(es, 0)
        out[k] = m / (np.linalg.norm(m) + 1e-9)
    log.info("re-ID: %d (track, frame) embeddings from %d crops", len(out), len(E))
    return out


def centre_embeddings(E: Embedding) -> Embedding:
    """Remove the shared scene/lighting component and renormalise."""
    if not E:
        return E
    mu = np.mean(list(E.values()), 0)
    return {k: (v - mu) / (np.linalg.norm(v - mu) + 1e-9) for k, v in E.items()}
