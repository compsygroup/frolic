"""Detector adapter: a callable that turns one BGR image into person boxes and Halpe-26 keypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from frolic.config import DetectConfig

DetectorOutput = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class PoseDetector(Protocol):
    def __call__(self, img_bgr: np.ndarray) -> DetectorOutput:
        """-> boxes (M, 4) x1 y1 x2 y2, box_scores (M,), kps (M, 26, 2), kps_scores (M, 26)."""
        ...


DetectorFactory = Callable[[DetectConfig, str], PoseDetector]
_FACTORIES: dict[str, DetectorFactory] = {}


def register_detector(name: str, factory: DetectorFactory) -> None:
    _FACTORIES[name] = factory


def make_detector(cfg: DetectConfig, device: str) -> PoseDetector:
    if cfg.detector in _FACTORIES:
        return _FACTORIES[cfg.detector](cfg, device)
    if cfg.detector == "rtmlib":
        from frolic.detect.rtmlib_adapter import RtmlibDetector

        return RtmlibDetector(cfg, device)
    raise ValueError(f"no detector factory for {cfg.detector!r}")


def empty_output() -> DetectorOutput:
    return (
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.float32),
        np.zeros((0, 26, 2), np.float32),
        np.zeros((0, 26), np.float32),
    )
