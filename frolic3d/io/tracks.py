"""Per-person 3D tracks on the track grid, stored as one `tracks.npz` (+ `people.json` for labels)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frolic3d.skeleton import N_JOINTS


@dataclass
class Tracks:
    ids: list[str]
    xyz: np.ndarray  # (P, N, 17, 3) float32, NaN where missing
    score: np.ndarray  # (P, N, 17) float32, 0 where missing
    cameras: list[str]
    support: np.ndarray  # (P, N, V) int32: detection row in that camera's Detections, -1 if none
    fps: float
    stride: int  # source frames per track frame
    frame0: int = 0  # source frame of track frame 0
    single_view: np.ndarray | None = None  # (P, N) bool: frame lifted from one camera
    observed: np.ndarray | None = (
        None  # (P, N) bool: some camera observed the person; OR-ed with has_3d on construction and save
    )
    labels: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ids = [str(i) for i in self.ids]
        self.xyz = np.asarray(self.xyz, np.float32).reshape(len(self.ids), -1, N_JOINTS, 3)
        P, N = self.xyz.shape[:2]
        self.score = np.asarray(self.score, np.float32).reshape(P, N, N_JOINTS)
        self.support = np.asarray(self.support, np.int32).reshape(P, N, len(self.cameras))
        if self.single_view is None:
            self.single_view = np.zeros((P, N), bool)
        self.single_view = np.asarray(self.single_view, bool).reshape(P, N)
        self.observed = (
            np.zeros((P, N), bool) if self.observed is None else np.asarray(self.observed, bool).reshape(P, N)
        )
        self.observed = self.observed | self.has_3d

    @property
    def n_people(self) -> int:
        return len(self.ids)

    @property
    def n_frames(self) -> int:
        return self.xyz.shape[1]

    @property
    def has_3d(self) -> np.ndarray:
        return np.isfinite(self.xyz[:, :, 0, 0])

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    def index(self, pid: str) -> int:
        return self.ids.index(pid)

    def person(self, pid: str) -> int:
        return self.index(pid)

    def source_frame(self, i: int) -> int:
        return self.frame0 + i * self.stride

    def n_views(self) -> np.ndarray:
        return (self.support >= 0).sum(-1)

    def subset(self, ids: list[str]) -> Tracks:
        k = [self.index(i) for i in ids]
        return Tracks(
            [self.ids[i] for i in k],
            self.xyz[k],
            self.score[k],
            list(self.cameras),
            self.support[k],
            self.fps,
            self.stride,
            self.frame0,
            self.single_view[k],
            self.observed[k],
            {i: self.labels[i] for i in ids if i in self.labels},
            dict(self.meta),
        )

    def renamed(self, mapping: dict[str, str]) -> Tracks:
        t = self.subset(self.ids)
        t.ids = [mapping.get(i, i) for i in self.ids]
        t.labels = {mapping.get(i, i): v for i, v in self.labels.items()}
        return t

    @classmethod
    def empty(
        cls, ids: list[str], n_frames: int, cameras: list[str], fps: float, stride: int, frame0: int = 0
    ) -> Tracks:
        P = len(ids)
        return cls(
            ids,
            np.full((P, n_frames, N_JOINTS, 3), np.nan, np.float32),
            np.zeros((P, n_frames, N_JOINTS), np.float32),
            cameras,
            np.full((P, n_frames, len(cameras)), -1, np.int32),
            fps,
            stride,
            frame0,
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=np.array(self.ids),
            xyz=self.xyz,
            score=self.score,
            cameras=np.array(self.cameras),
            support=self.support,
            fps=self.fps,
            stride=self.stride,
            frame0=self.frame0,
            single_view=self.single_view,
            observed=self.observed | self.has_3d,
        )
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({"labels": self.labels, "meta": self.meta}, fh, indent=1)

    @classmethod
    def load(cls, path: Path | str) -> Tracks:
        path = Path(path)
        z = np.load(path)
        side = path.with_suffix(".json")
        extra = {"labels": {}, "meta": {}}
        if side.exists():
            with open(side) as fh:
                extra = json.load(fh)
        return cls(
            list(z["ids"]),
            z["xyz"],
            z["score"],
            list(z["cameras"]),
            z["support"],
            float(z["fps"]),
            int(z["stride"]),
            int(z["frame0"]),
            z["single_view"],
            z["observed"],
            extra.get("labels", {}),
            extra.get("meta", {}),
        )
