import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from frolic3d.calib.stage import run as calib_run
from frolic3d.config import load_config
from frolic3d.geometry import rodrigues
from frolic3d.io.calib import CameraSet, from_gui_json, load_cameras, save_cameras, to_gui_json
from frolic3d.io.workdir import WorkDir
from frolic3d.sync.stage import run as sync_run
from frolic3d.synthetic import make_scene, ring_cameras

BOARD = {
    "squares_x": 8,
    "squares_y": 6,
    "square_length": 0.095,
    "marker_length": 0.071,
    "dictionary_name": "DICT_4X4_100",
    "legacy_pattern": True,
    "units": "m",
}


def _assert_same(a: CameraSet, b: CameraSet, atol=1e-9):
    assert a.names == b.names
    for x, y in zip(a, b):
        assert np.allclose(x.K, y.K, atol=atol) and np.allclose(x.dist, y.dist, atol=atol)
        assert np.allclose(x.R, y.R, atol=atol) and np.allclose(x.t, y.t, atol=atol)
        assert (x.width, x.height) == (y.width, y.height)


def test_cameras_json_roundtrip(tmp_path):
    cams = ring_cameras(3, dist=[-0.1, 0.02, 0, 0, 0])
    cams.floor["levelled"] = True
    save_cameras(cams, tmp_path / "c.json")
    back = load_cameras(tmp_path / "c.json")
    _assert_same(cams, back)
    assert back.floor["levelled"] and back.source["format"] == "synthetic"
    with pytest.raises(ValueError):
        CameraSet.from_dict({"schema": "other"})


def test_gui_json_roundtrip_reference_frame(tmp_path):
    cams = ring_cameras(4, dist=[-0.1, 0.02, 0.001, 0, 0])
    mpath = to_gui_json(cams, tmp_path / "gui", BOARD, reference="cam2")
    d = json.loads(Path(mpath).read_text())
    assert d["mode"] == "multi_camera_extrinsics" and d["reference_camera"] == "cam2"
    ref = next(c for c in d["cameras"] if c["camera_name"] == "cam2")
    assert np.allclose(ref["T_reference_to_camera"], np.eye(4), atol=1e-9)
    back = from_gui_json(mpath)
    # the GUI world is the reference camera: compare after moving our cameras into cam2's frame
    R2, t2 = cams["cam2"].R, cams["cam2"].t
    expect = cams.transformed(R2, t2)
    _assert_same(expect, back, atol=1e-8)
    intr = json.loads(Path(tmp_path / "gui" / "cam1_intrinsics.json").read_text())
    assert intr["mode"] == "intrinsics" and intr["image_size"] == {"width": 640, "height": 360}


def test_gui_json_millimetres(tmp_path):
    cams = ring_cameras(2)
    board = dict(BOARD, square_length=95.0, marker_length=71.0, units="mm")
    mpath = to_gui_json(cams, tmp_path / "gui", board)
    d = json.loads(Path(mpath).read_text())
    T = np.array(d["cameras"][1]["T_reference_to_camera"])
    assert np.linalg.norm(T[:3, 3]) > 1000  # written in millimetres
    back = from_gui_json(mpath)
    expect = cams.transformed(cams["cam1"].R, cams["cam1"].t)
    _assert_same(expect, back, atol=1e-6)


def test_opencv_yaml_and_anipose(tmp_path):
    cams = ring_cameras(2)
    d = {
        "units": "m",
        "cameras": [
            {
                "name": c.name,
                "width": c.width,
                "height": c.height,
                "K": c.K.tolist(),
                "dist": c.dist.tolist(),
                "R": c.R.tolist(),
                "t": c.t.tolist(),
            }
            for c in cams
        ],
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d))
    _assert_same(cams, load_cameras(tmp_path / "c.yaml", "opencv"))
    lines = []
    for i, c in enumerate(cams):
        rv = rodrigues_vec(c.R)
        lines += [
            f"[cam_{i}]",
            f'name = "{c.name}"',
            f"size = [{c.width}, {c.height}]",
            f"matrix = {c.K.tolist()}",
            f"distortions = {c.dist[:4].tolist()}",
            f"rotation = {rv.tolist()}",
            f"translation = {c.t.tolist()}",
            "",
        ]
    (tmp_path / "calibration.toml").write_text("\n".join(lines))
    try:
        back = load_cameras(tmp_path / "calibration.toml", "anipose")
    except ModuleNotFoundError:
        pytest.skip("no toml parser")
    _assert_same(cams, back, atol=1e-9)


def rodrigues_vec(R):
    from frolic3d.geometry import rotvec

    return rotvec(R)


def test_calibrate_stage_given(tmp_path):
    scene = tmp_path / "scene"
    make_scene(scene, n_cams=3, n_frames=6, videos=True)
    cfgp = tmp_path / "f.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "input": {"video_dir": str(scene / "videos"), "work_dir": str(tmp_path / "work")},
                "calib": {"given": str(scene / "calib" / "cameras.json")},
            }
        )
    )
    cfg = load_config(cfgp)
    work = WorkDir(cfg.work_dir)
    sync_run(cfg, work)
    info = calib_run(cfg, work)
    assert info["cameras"] == ["cam1", "cam2", "cam3"]
    _assert_same(load_cameras(scene / "calib" / "cameras.json"), load_cameras(work.cameras))
    rep = json.loads(Path(work.stage_dir("calibrate") / "report.json").read_text())
    assert rep["mode"] == "given" and "cam1-cam2" in rep["baselines_m"]


def test_calibrate_stage_given_maps_names_by_order(tmp_path):
    scene = tmp_path / "scene"
    make_scene(scene, n_cams=2, n_frames=4, videos=True)
    cams = load_cameras(scene / "calib" / "cameras.json")
    cams.cameras[0].name, cams.cameras[1].name = "front", "side"
    save_cameras(cams, scene / "calib" / "renamed.json")
    cfgp = tmp_path / "f.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "input": {"video_dir": str(scene / "videos"), "work_dir": str(tmp_path / "work")},
                "calib": {"given": str(scene / "calib" / "renamed.json")},
            }
        )
    )
    cfg = load_config(cfgp)
    work = WorkDir(cfg.work_dir)
    sync_run(cfg, work)
    calib_run(cfg, work)
    assert load_cameras(work.cameras).names == ["cam1", "cam2"]


def test_camera_transform_in_gui_matches_given_R_t():
    cams = ring_cameras(2)
    R = rodrigues([0.1, 0.2, 0.3])
    t = np.array([1.0, 2.0, 3.0])
    moved = cams.transformed(R, t)
    X = np.array([0.3, 0.2, 1.0])
    uv1, _ = cams[1].project(X)
    uv2, _ = moved[1].project(R @ X + t)
    assert np.allclose(uv1, uv2)


def test_calibrate_stage_bootstrap(tmp_path):
    import shutil

    from frolic3d.geometry import umeyama

    scene = tmp_path / "scene"
    info = make_scene(scene, n_cams=3, n_frames=90, people=[{"stature": 1.6, "start": (-1, -0.5), "end": (1, 0.5)}])
    truth = load_cameras(scene / "calib" / "cameras.json")
    cfgp = tmp_path / "f.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "input": {"video_dir": str(scene / "videos"), "work_dir": str(tmp_path / "work")},
                "calib": {
                    "mode": "bootstrap",
                    "reference_height_m": 1.6,
                    "bootstrap": {"focal_guess": float(truth["cam1"].K[0, 0]), "min_pair_frames": 30},
                },
            }
        )
    )
    cfg = load_config(cfgp)
    work = WorkDir(cfg.work_dir)
    sync_run(cfg, work)
    with pytest.raises(RuntimeError, match="detect stage first"):
        calib_run(cfg, work)
    for c in info["cameras"]:
        shutil.copy(scene / "gt" / f"{c}.npz", work.stage_dir("detect") / f"{c}.npz")
        shutil.copy(scene / "gt" / f"{c}.json", work.stage_dir("detect") / f"{c}.json")
    calib_run(cfg, work)
    got = load_cameras(work.cameras)
    s, _, _ = umeyama(got.centers, truth.centers, with_scale=True)
    assert abs(s - 1) < 0.1
    assert (work.stage_dir("calibrate") / "bootstrap_points.npz").exists()
    rep = json.loads((work.stage_dir("calibrate") / "report.json").read_text())
    assert rep["mode"] == "bootstrap"
