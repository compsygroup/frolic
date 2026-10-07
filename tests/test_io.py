from __future__ import annotations

import json

import numpy as np
import pytest

from frolic3d.geometry import Camera
from frolic3d.io.calib import load_cameras, save_cameras
from frolic3d.io.detections import Detections
from frolic3d.io.tracks import Tracks
from frolic3d.io.video import iter_frames, list_videos, probe_video
from frolic3d.io.workdir import STAGES, WorkDir
from frolic3d.skeleton import N_JOINTS
from frolic3d.synthetic import make_scene, ring_cameras


def test_cameras_roundtrip(tmp_path):
    cams = ring_cameras(3)
    p = tmp_path / "cameras.json"
    save_cameras(cams, p)
    back = load_cameras(p)
    assert back.names == cams.names
    for a, b in zip(cams, back):
        np.testing.assert_allclose(a.K, b.K)
        np.testing.assert_allclose(a.R, b.R)
        np.testing.assert_allclose(a.t, b.t)
        np.testing.assert_allclose(a.center, b.center)
    with open(p) as fh:
        assert json.load(fh)["schema"] == "frolic3d.cameras/1"


def test_cameras_bad_schema(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "other", "cameras": []}))
    with pytest.raises(ValueError):
        load_cameras(p)


def test_cameraset_transform_keeps_projection():
    cams = ring_cameras(2)
    R = Camera("", 1, 1, np.eye(3)).R
    from frolic3d.geometry import rodrigues

    R = rodrigues([0.1, -0.2, 0.3])
    t = np.array([0.5, -1.0, 0.2])
    X = np.array([[0.2, 0.1, 0.9], [-0.3, 0.4, 1.2]])
    moved = cams.transformed(R, t, 1.0)
    Xn = X @ R.T + t
    for a, b in zip(cams, moved):
        ua, _ = a.project(X)
        ub, _ = b.project(Xn)
        np.testing.assert_allclose(ua, ub, atol=1e-6)
    assert moved.subset(["cam2"]).names == ["cam2"]


def test_detections_index_and_io(tmp_path):
    M = 7
    d = Detections(
        "camA",
        [0, 0, 1, 3, 3, 3, 5],
        np.zeros((M, 4)),
        np.ones(M),
        np.zeros((M, 26, 2)),
        np.ones((M, 26)),
        np.full((M, 3), np.nan),
        {"width": 64, "height": 32},
    )
    assert list(d.frames) == [0, 1, 3, 5]
    assert list(d.at(3)) == [3, 4, 5]
    assert len(d.at(2)) == 0
    d.save(tmp_path / "camA.npz")
    back = Detections.load(tmp_path / "camA.npz")
    assert back.camera == "camA" and back.width == 64 and len(back) == M
    sub = back.subset(back.at(0))
    assert len(sub) == 2
    cat = Detections.concat([sub, sub])
    assert len(cat) == 4
    assert len(Detections.empty("z")) == 0


def test_tracks_io_and_subset(tmp_path):
    t = Tracks.empty(["p0", "p1"], 10, ["cam1", "cam2"], 15.0, 2, frame0=4)
    t.xyz[0, :5] = 0.0
    t.score[0, :5] = 0.5
    t.support[0, :5, 0] = 3
    t.labels = {"p0": "adult"}
    assert t.has_3d.sum() == 5
    assert list(t.n_views()[0, :6]) == [1, 1, 1, 1, 1, 0]
    assert t.source_frame(3) == 10
    t.save(tmp_path / "tracks.npz")
    b = Tracks.load(tmp_path / "tracks.npz")
    assert b.ids == ["p0", "p1"] and b.fps == 15.0 and b.stride == 2 and b.frame0 == 4
    assert b.labels == {"p0": "adult"}
    assert b.observed[0].sum() == 5 and not b.single_view.any()
    s = b.subset(["p1"])
    assert s.n_people == 1 and s.xyz.shape == (1, 10, N_JOINTS, 3)
    r = b.renamed({"p0": "A"})
    assert r.ids == ["A", "p1"] and r.labels == {"A": "adult"}


def test_workdir_markers(tmp_path):
    w = WorkDir(tmp_path)
    assert not w.is_done("detect")
    w.mark_done("detect", {"n": 1})
    w.mark_done("track")
    assert w.is_done("detect") and w.status()["detect"]["n"] == 1
    w.clear_from("track")
    assert w.is_done("detect") and not w.is_done("track")
    w.clear_from("sync")
    assert not any(w.is_done(s) for s in STAGES)
    assert w.stage_dir("calibrate").name == "calib"


def test_video_probe_and_frames(tmp_path):
    info = make_scene(tmp_path / "scene", n_cams=2, n_frames=6, videos=True)
    vids = list_videos(tmp_path / "scene" / "videos")
    assert list(vids) == ["cam1", "cam2"]
    v = probe_video(vids["cam1"])
    assert v.width == 640 and v.height == 360 and abs(v.fps - 15.0) < 1e-3 and v.n_frames == 6
    frames = list(iter_frames(vids["cam1"], start=1, stop=5, step=2))
    assert [f for f, _ in frames] == [1, 3]
    assert frames[0][1].shape == (360, 640, 3)
    assert info["cameras"] == ["cam1", "cam2"]
