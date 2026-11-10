"""Per-camera 2D detections (flat arrays, one row per detection) stored as `<camera>.npz` + `<camera>.json`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

N_KP = 26


@dataclass
class Detections:
    camera: str
    frame: np.ndarray  # (M,) int32 source frame index
    box: np.ndarray  # (M, 4) x1 y1 x2 y2 pixels (distorted image)
    box_score: np.ndarray  # (M,)
    kps: np.ndarray  # (M, 26, 2) pixels (distorted image)
    kps_score: np.ndarray  # (M, 26)
    colour: np.ndarray  # (M, 3) torso HSV median, NaN when unavailable
    meta: dict[str, Any] = field(default_factory=dict)  # width, height, fps, step, n_frames, video

    def __post_init__(self) -> None:
        self.frame = np.asarray(self.frame, np.int32)
        self.box = np.asarray(self.box, np.float32).reshape(-1, 4)
        self.box_score = np.asarray(self.box_score, np.float32)
        self.kps = np.asarray(self.kps, np.float32).reshape(-1, N_KP, 2)
        self.kps_score = np.asarray(self.kps_score, np.float32).reshape(-1, N_KP)
        self.colour = np.asarray(self.colour, np.float32).reshape(-1, 3)
        self._index: dict[int, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def frames(self) -> np.ndarray:
        return np.unique(self.frame)

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])

    def at(self, frame: int) -> np.ndarray:
        """Row indices of the detections in a source frame."""
        if self._index is None:
            order = np.argsort(self.frame, kind="stable")
            f = self.frame[order]
            starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
            ends = np.r_[starts[1:], len(f)]
            self._index = {int(f[s]): order[s:e] for s, e in zip(starts, ends)}
        return self._index.get(int(frame), np.zeros(0, np.int64))

    def subset(self, rows: np.ndarray) -> Detections:
        return Detections(
            self.camera,
            self.frame[rows],
            self.box[rows],
            self.box_score[rows],
            self.kps[rows],
            self.kps_score[rows],
            self.colour[rows],
            dict(self.meta),
        )

    @classmethod
    def empty(cls, camera: str, meta: dict[str, Any] | None = None) -> Detections:
        z = np.zeros
        return cls(camera, z(0), z((0, 4)), z(0), z((0, N_KP, 2)), z((0, N_KP)), z((0, 3)), meta or {})

    @classmethod
    def concat(cls, parts: list[Detections]) -> Detections:
        p0 = parts[0]
        return cls(
            p0.camera,
            np.concatenate([p.frame for p in parts]),
            np.concatenate([p.box for p in parts]),
            np.concatenate([p.box_score for p in parts]),
            np.concatenate([p.kps for p in parts]),
            np.concatenate([p.kps_score for p in parts]),
            np.concatenate([p.colour for p in parts]),
            dict(p0.meta),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame=self.frame,
            box=self.box,
            box_score=self.box_score,
            kps=self.kps,
            kps_score=self.kps_score,
            colour=self.colour,
        )
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({"camera": self.camera, **self.meta}, fh, indent=1)

    @classmethod
    def load(cls, path: Path | str) -> Detections:
        path = Path(path)
        z = np.load(path)
        with open(path.with_suffix(".json")) as fh:
            meta = json.load(fh)
        camera = meta.pop("camera", path.stem)
        return cls(camera, z["frame"], z["box"], z["box_score"], z["kps"], z["kps_score"], z["colour"], meta)


def load_all(det_dir: Path | str, cameras: list[str]) -> dict[str, Detections]:
    det_dir = Path(det_dir)
    return {c: Detections.load(det_dir / f"{c}.npz") for c in cameras}
