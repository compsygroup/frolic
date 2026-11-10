"""detect(mock) -> track -> select(auto) -> singleview through the pipeline runner on the synthetic scene."""

from __future__ import annotations

import json

import numpy as np
import pytest

from frolic.config import FrolicConfig
from frolic.io.calib import load_cameras
from frolic.io.detections import Detections
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.pipeline import run_stage
from frolic.skeleton import J
from frolic.synthetic import make_scene


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    people = [
        {"stature": 1.70, "start": (-1.0, -0.5), "end": (1.0, 0.5)},
        {"stature": 0.95, "start": (0.8, -0.8), "end": (-0.6, 0.6), "phase": 0.3},
    ]
    info = make_scene(d, n_cams=4, n_frames=40, fps=15.0, people=people, videos=True)
    return d, info


@pytest.fixture(scope="module")
def pipeline(scene, tmp_path_factory):
    scene_dir, info = scene
    root = tmp_path_factory.mktemp("run")
    cfg = FrolicConfig.model_validate(
        {
            "input": {"video_dir": str(scene_dir / "videos"), "work_dir": str(root / "work")},
            "calib": {"mode": "given", "given": str(scene_dir / "calib" / "cameras.json")},
            "detect": {"detector": "mock", "mock_dir": str(scene_dir / "gt")},
            "track": {"reid": "none", "refine_smooth": 20.0},
            "select": {"mode": "auto", "min_track_s": 1.0},
        }
    )
    work = WorkDir(cfg.work_dir)
    work.root.mkdir(parents=True)
    cams = info["cameras"]
    work.write_videos(
        {
            "cameras": cams,
            "videos": info["videos"],
            "frame_offset": {c: 0 for c in cams},
            "fps": info["fps"],
            "width": {c: 640 for c in cams},
            "height": {c: 360 for c in cams},
            "n_frames": info["n_frames"],
            "offsets_s": {c: 0.0 for c in cams},
        }
    )
    work.mark_done("sync")
    # calibrate is another stage's job: provide its output directly
    cs = load_cameras(scene_dir / "calib" / "cameras.json")
    from frolic.io.calib import save_cameras

    save_cameras(cs, work.cameras)
    work.mark_done("calibrate")
    return cfg, work, info


def test_stages_run_in_order(pipeline):
    cfg, work, info = pipeline
    for st in ("detect", "track", "select", "singleview"):
        assert run_stage(cfg, work, st) is True
        assert work.is_done(st)
    assert (work.stage_dir("detect") / "cam1.npz").exists()
    tr = Tracks.load(work.tracks)
    assert tr.n_people == 2 and tr.n_frames == info["n_frames"] and tr.fps == 15.0
    assert (work.stage_dir("track") / "identities.json").exists() and work.cameras_level.exists()
    gt = np.load(work.stage_dir("detect") / "mock_joints.npz")["xyz"]
    for p in range(2):
        g = int(
            np.argmin([np.nanmedian(np.linalg.norm(tr.xyz[p, :, J["pelvis"]] - q[:, J["pelvis"]], axis=1)) for q in gt])
        )
        have = tr.has_3d[p]
        assert have.mean() > 0.85
        # floor alignment may shift the world vertically: compare up to a global translation
        off = np.nanmedian((tr.xyz[p][have] - gt[g][have]).reshape(-1, 3), axis=0)
        assert np.abs(off[:2]).max() < 0.02 and abs(off[2]) < 0.15
        assert np.nanmedian(np.linalg.norm(tr.xyz[p][have] - gt[g][have] - off, axis=-1)) < 0.04
    sel = Tracks.load(work.tracks_selected)
    assert sel.ids == ["p0", "p1"] and set(sel.labels) == {"p0", "p1"}
    with open(work.stage_dir("select") / "selection.json") as fh:
        s = json.load(fh)
    assert s["mode"] == "auto" and s["ids"] == ["p0", "p1"]
    sv = Tracks.load(work.tracks_singleview)
    assert sv.ids == sel.ids and sv.single_view.shape == sel.single_view.shape
    assert (work.stage_dir("singleview") / "svfill.json").exists()
    assert work.fit_input_tracks == work.tracks_singleview


def test_stages_idempotent_and_force(pipeline):
    cfg, work, _ = pipeline
    assert run_stage(cfg, work, "track") is False
    before = work.status()["track"]["time"]
    assert run_stage(cfg, work, "select", force=True) is True
    # forcing select clears every later marker
    assert work.is_done("select") and not work.is_done("singleview")
    assert work.status()["track"]["time"] == before
    assert run_stage(cfg, work, "singleview") is True


def test_detections_schema(pipeline):
    _, work, info = pipeline
    d = Detections.load(work.stage_dir("detect") / "cam2.npz")
    assert d.kps.shape[1:] == (26, 2) and d.meta["n_frames"] == info["n_frames"] and d.meta["complete"]


def test_sam3d_model_path_with_mock_estimator(pipeline, monkeypatch):
    """Prompts -> per-view estimates on the decoded videos -> multi-view MHR fit, with the model replaced by MockSam3d."""
    from frolic.sam3d import stage as sam3d_stage
    from frolic.sam3d.adapter import MockSam3d
    from frolic.sam3d.prompts import Prompts
    from frolic.sam3d.run import ViewResults

    cfg, work, info = pipeline
    if not work.is_done("singleview"):
        run_stage(cfg, work, "singleview")
    cfg = cfg.model_copy(deep=True)
    cfg.sam3d.backend = "sam3d"
    cfg.sam3d.iters = 60
    cfg.compute.device = "cpu"
    monkeypatch.setattr(sam3d_stage, "make_estimator", lambda *a, **k: MockSam3d())
    assert run_stage(cfg, work, "sam3d", force=True) is True
    out = work.stage_dir("sam3d")
    pr = Prompts.load(out / "prompts.npz")
    tr = Tracks.load(work.fit_input_tracks)
    assert pr.box.shape[:3] == (tr.n_people, len(info["cameras"]), tr.n_frames)
    vr = ViewResults.load(out / "views" / "cam1.npz")
    assert len(vr) > 0 and vr.arrays["pred_keypoints_3d"].shape[1:] == (70, 3)
    for pid in tr.ids:
        d = np.load(out / "mhr" / f"{pid}.npz")
        assert d["kp"].shape == (tr.n_frames, 70, 3) and d["have"].mean() > 0.8
        assert np.isfinite(d["kp"][d["have"]]).all()
        v = np.load(out / "mhr" / f"{pid}_verts.npy")
        assert v.shape[0] == tr.n_frames and np.isfinite(v[d["have"]].astype(float)).all()
