"""Real models on a local multi-camera clip.

FROLIC_TEST_CLIP: folder with one video per camera (never copied into the repo).
FROLIC_TEST_CALIB: optional cameras.json; without it the clip is bootstrapped and FROLIC_TEST_HEIGHT (m) sets the scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from frolic3d.config import FrolicConfig
from frolic3d.io.workdir import WorkDir
from frolic3d.pipeline import run_pipeline

CLIP = os.environ.get("FROLIC_TEST_CLIP")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CLIP, reason="set FROLIC_TEST_CLIP to a folder of per-camera videos"),
]


def test_real_pipeline_on_clip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    calib = os.environ.get("FROLIC_TEST_CALIB")
    if calib:
        cal = {"mode": "given", "given": calib}
    else:
        cal = {"mode": "bootstrap", "reference_height_m": float(os.environ.get("FROLIC_TEST_HEIGHT", "1.7"))}
    cfg = FrolicConfig.model_validate(
        {
            "input": {"video_dir": CLIP, "work_dir": str(tmp_path / "work"), "duration_s": 10.0},
            "calib": cal,
            "track": {"reid": "none"},
            "select": {"mode": "auto"},
            "render": {"what": ["grid"]},
        }
    )
    work = WorkDir(cfg.work_dir)
    run_pipeline(cfg, work, stop="export")
    out = Path(work.stage_dir("export"))
    people = sorted(out.glob("p*.npz"))
    assert people
    for p in people:
        d = np.load(p)
        seen = np.isfinite(d["joints"]).all((1, 2))
        assert seen.any()
        assert 0.5 < float(d["stature_m"]) < 2.2
    assert json.loads((out / "metrics.json").read_text())
