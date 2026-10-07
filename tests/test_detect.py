from __future__ import annotations

import json

import numpy as np
import pytest

from frolic3d.config import FrolicConfig
from frolic3d.detect import adapter, stage
from frolic3d.detect.masks import box_mask_fraction, masked_rows, polygon_mask, static_rows
from frolic3d.io.detections import Detections
from frolic3d.io.workdir import WorkDir
from frolic3d.synthetic import make_scene


def _det(frames, boxes, kps=None):
    M = len(frames)
    kps = np.zeros((M, 26, 2), np.float32) if kps is None else kps
    d = Detections("cam1", frames, boxes, np.ones(M), kps, np.ones((M, 26)), np.full((M, 3), np.nan))
    d.meta = {"width": 640, "height": 360}
    return d


def test_polygon_mask_fraction():
    m = polygon_mask([[[0, 0], [100, 0], [100, 100], [0, 100]]], 640, 360)
    assert box_mask_fraction(np.array([0, 0, 50, 50]), m) == 1.0
    assert box_mask_fraction(np.array([200, 200, 250, 250]), m) == 0.0
    assert 0.4 < box_mask_fraction(np.array([50, 0, 150, 100]), m) < 0.6
    assert polygon_mask([], 10, 10) is None


def test_masked_rows():
    d = _det([0, 0], [[0, 0, 50, 50], [300, 100, 350, 200]])
    drop = masked_rows(d, [[[0, 0], [100, 0], [100, 100], [0, 100]]], 0.6)
    assert drop.tolist() == [True, False]


def test_static_rows_frozen_poster_kept_person():
    n = 50
    frames = np.repeat(np.arange(n), 2)
    boxes, kps = [], []
    for i in range(n):
        boxes.append([100, 100, 160, 220])  # frozen poster
        kps.append(np.full((26, 2), [130, 160], np.float32))
        boxes.append([300 + i, 100, 360 + i, 220])  # moving person
        kps.append(np.full((26, 2), [330 + i, 160], np.float32) + np.random.default_rng(i).normal(0, 8, (26, 2)))
    d = _det(frames, boxes, np.array(kps, np.float32))
    drop = static_rows(d, 0.6, 0.01)
    assert drop[0::2].all() and not drop[1::2].any()


def _config(scene_dir, tmp_path, detector="mock"):
    cfg = FrolicConfig.model_validate(
        {
            "input": {"video_dir": str(scene_dir / "videos"), "work_dir": str(tmp_path / "work")},
            "calib": {"mode": "given", "given": str(scene_dir / "calib" / "cameras.json")},
            "detect": {"detector": detector, "mock_dir": str(scene_dir / "gt")},
        }
    )
    return cfg


def _videos_json(work: WorkDir, info, scene_dir):
    work.root.mkdir(parents=True, exist_ok=True)
    cams = info["cameras"]
    work.write_videos(
        {
            "cameras": cams,
            "videos": {c: str(scene_dir / "videos" / f"{c}.mp4") for c in cams},
            "frame_offset": {c: 0 for c in cams},
            "fps": info["fps"],
            "width": {c: 640 for c in cams},
            "height": {c: 360 for c in cams},
            "n_frames": info["n_frames"],
            "offsets_s": {c: 0.0 for c in cams},
        }
    )


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    info = make_scene(d, n_cams=3, n_frames=12, videos=True)
    return d, info


def test_mock_stage_copies_and_resumes(scene, tmp_path):
    scene_dir, info = scene
    cfg = _config(scene_dir, tmp_path)
    work = WorkDir(cfg.work_dir)
    _videos_json(work, info, scene_dir)
    out = stage.run(cfg, work)
    assert set(out["detections"]) == set(info["cameras"])
    d = Detections.load(work.stage_dir("detect") / "cam1.npz")
    assert d.meta["complete"] and d.meta["n_frames"] == info["n_frames"]
    assert d.frames.max() < info["n_frames"]
    # second run skips every camera (counts unchanged)
    out2 = stage.run(cfg, work)
    assert out2["detections"] == out["detections"]


def test_mock_frame_offset_reindex(scene, tmp_path):
    scene_dir, info = scene
    cfg = _config(scene_dir, tmp_path)
    work = WorkDir(cfg.work_dir)
    _videos_json(work, info, scene_dir)
    v = work.read_videos()
    v["frame_offset"]["cam1"] = 3
    work.write_videos(v)
    stage.run(cfg, work)
    d = Detections.load(work.stage_dir("detect") / "cam1.npz")
    gt = Detections.load(scene_dir / "gt" / "cam1.npz")
    assert d.frames.min() == 0 and len(d) == int((gt.frame >= 3).sum())


def test_real_loop_with_fake_detector(scene, tmp_path, monkeypatch):
    scene_dir, info = scene
    gt = {c: Detections.load(scene_dir / "gt" / f"{c}.npz") for c in info["cameras"]}
    state = {"cam": None, "f": 0}

    class Fake:
        def __call__(self, img):
            d = gt[state["cam"]]
            rows = d.at(state["f"])
            state["f"] += 1
            if len(rows) == 0:
                return adapter.empty_output()
            return d.box[rows], d.box_score[rows], d.kps[rows], d.kps_score[rows]

    def factory(cfg, device):
        return Fake()

    adapter.register_detector("fake", factory)
    monkeypatch.setattr(stage, "make_detector", lambda c, dev: factory(c, dev))
    cfg = _config(scene_dir, tmp_path, detector="rtmlib")
    work = WorkDir(cfg.work_dir)
    _videos_json(work, info, scene_dir)
    orig = stage.detect_video

    def wrapped(det, video, camera, off, n, step, meta):
        state["cam"], state["f"] = camera, 0
        return orig(det, video, camera, off, n, step, meta)

    monkeypatch.setattr(stage, "detect_video", wrapped)
    out = stage.run(cfg, work)
    d = Detections.load(work.stage_dir("detect") / "cam2.npz")
    assert out["detections"]["cam2"] == len(gt["cam2"]) == len(d)
    assert np.isfinite(d.colour).all(axis=1).mean() > 0.5
    with open(work.stage_dir("detect") / "cam2.json") as fh:
        assert json.load(fh)["complete"]
