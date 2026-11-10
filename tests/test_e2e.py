"""Every stage end to end on the synthetic scene with mocked detector, SAM 3D Body and body model."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from conftest import can_render

from frolic.config import FrolicConfig
from frolic.io.workdir import STAGES, WorkDir
from frolic.pipeline import run_pipeline
from frolic.skeleton import J
from frolic.synthetic import make_scene


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    people = [
        {"stature": 1.75, "start": (-1.0, -0.5), "end": (1.0, 0.5)},
        {"stature": 1.00, "start": (0.9, -0.7), "end": (-0.7, 0.6), "phase": 0.3},
    ]
    info = make_scene(d / "scene", n_cams=3, n_frames=30, fps=15.0, people=people, videos=True)
    render = can_render()
    cfg = FrolicConfig.model_validate(
        {
            "input": {"video_dir": str(d / "scene" / "videos"), "work_dir": str(d / "work")},
            "calib": {"mode": "given", "given": str(d / "scene" / "calib" / "cameras.json")},
            "detect": {"detector": "mock", "mock_dir": str(d / "scene" / "gt")},
            "track": {"reid": "none"},
            "select": {"mode": "auto", "min_track_s": 1.0},
            "sam3d": {"backend": "mock"},
            "fit": {"backend": "mock", "device": "cpu", "iters": 200},
            "render": {
                "what": ["overlay", "grid", "room", "trajectories"],
                "tile_width": 320,
                "room_width": 320,
                "room_height": 180,
            },
            "compute": {"device": "cpu"},
        }
    )
    work = WorkDir(cfg.work_dir)
    stop = "render" if render else "export"
    ran = run_pipeline(cfg, work, stop=stop)
    return cfg, work, info, ran, stop


def test_all_stages_ran(run):
    _, work, _, ran, stop = run
    assert ran == STAGES[: STAGES.index(stop) + 1]
    assert all(work.is_done(s) for s in ran)


def test_export_outputs(run):
    _, work, _info, _, _ = run
    out = work.stage_dir("export")
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["people"]) == 2
    gt = np.load(work.stage_dir("detect") / "mock_joints.npz")["xyz"]
    statures = []
    assert meta["schema"] == "frolic.export/1"
    for person in meta["people"]:
        z = np.load(out / f"{person['id']}.npz")
        n = len(z["frame_times"])
        assert z["vertices"].shape[0] == n and z["joints"].shape == (n, 17, 3)
        assert z["faces"].dtype == np.int32
        statures.append(float(z["stature_m"]))
        pel = z["joints"][:, J["pelvis"]]
        err = min(np.nanmedian(np.linalg.norm(pel - g[: len(pel), J["pelvis"]], axis=1)) for g in gt)
        assert err < 0.15
    assert np.allclose(sorted(statures), [1.00, 1.75], rtol=0.08)
    rows = list(csv.DictReader((out / "summary.csv").open()))
    assert len(rows) == 2
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["per_person"]) == {p["id"] for p in meta["people"]}


def test_rerun_is_noop_and_from_forces(run):
    cfg, work, _, _, stop = run
    assert run_pipeline(cfg, work, stop=stop) == []
    assert run_pipeline(cfg, work, start="export", stop="export", force=True) == ["export"]


@pytest.mark.skipif(not can_render(), reason="no offscreen OpenGL backend (pyrender EGL/OSMesa) available")
def test_render_videos(run):
    _, work, _, _, _ = run
    vids = sorted(p.name for p in work.stage_dir("render").glob("*.mp4"))
    assert "room.mp4" in vids and "grid.mp4" in vids and any(v.startswith("overlay_") for v in vids)
