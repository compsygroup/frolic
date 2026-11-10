from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from frolic.synthetic import person_track, project_people, ring_cameras

SCRATCH = Path(os.environ.get("FROLIC_TEST_SCRATCH", "")) if os.environ.get("FROLIC_TEST_SCRATCH") else None


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("FROLIC_TEST_CLIP"):
        skip = pytest.mark.skip(reason="set FROLIC_TEST_CLIP=<dir with camN.mp4> to run the real-model GPU test")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def cams():
    return ring_cameras(4)


@pytest.fixture(scope="session")
def two_people():
    """(adult, child) joint tracks (N, 17, 3) at 15 Hz, 45 frames, crossing paths."""
    n, fps = 45, 15.0
    adult = person_track(n, fps, 1.72, (-1.0, -0.6), (1.0, 0.6))
    child = person_track(n, fps, 0.95, (1.0, -0.6), (-1.0, 0.6), phase=0.3)
    return [adult, child], fps


@pytest.fixture(scope="session")
def gt_detections(cams, two_people):
    people, fps = two_people
    dets = project_people(cams, people, noise_px=0.5)
    for d in dets.values():
        d.meta["fps"] = fps
    return dets


@pytest.fixture
def work(tmp_path):
    from frolic.io.workdir import WorkDir

    return WorkDir(tmp_path / "work")


def have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def has_anny() -> bool:
    if not have_module("anny"):
        return False
    cache = os.environ.get("ANNY_CACHE_DIR")
    return bool(cache and Path(cache).exists())


def can_render() -> bool:
    if not have_module("pyrender"):
        return False
    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender

        r = pyrender.OffscreenRenderer(8, 8)
        r.delete()
        return True
    except Exception:
        return False
