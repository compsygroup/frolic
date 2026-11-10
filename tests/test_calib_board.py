import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from frolic.calib import board as B
from frolic.config import BoardConfig
from frolic.geometry import Camera, rodrigues, rotation_angle_deg, umeyama
from frolic.io.calib import from_gui_json, to_gui_json
from frolic.synthetic import ring_cameras

SPEC = B.BoardSpec(8, 6, 0.095, 0.071, "DICT_4X4_100", True)
PPS = 70  # board image pixels per square


@pytest.fixture(scope="module")
def board_image():
    return SPEC.make_board().generateImage((SPEC.squares_x * PPS, SPEC.squares_y * PPS), marginSize=0)


def render(board_img: np.ndarray, cam: Camera, R: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Warp the planar board (board->camera pose R, t) into the camera image; None when it leaves the image."""
    w, h = SPEC.squares_x * SPEC.square_length_m, SPEC.squares_y * SPEC.square_length_m
    obj = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], float)
    uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, cam.K, cam.dist)
    uv = uv.reshape(-1, 2)
    if (obj @ R.T + t)[:, 2].min() < 0.2 or not cam.in_image(uv, margin=5).all():
        return None
    src = np.array(
        [[0, 0], [board_img.shape[1], 0], [board_img.shape[1], board_img.shape[0]], [0, board_img.shape[0]]], np.float32
    )
    H = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    img = cv2.warpPerspective(
        board_img, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    return cv2.GaussianBlur(img, (3, 3), 0)


def board_poses_world(rng, n: int, cams, centre=(0.0, 0.0, 0.9)) -> list[tuple[np.ndarray, np.ndarray]]:
    """Board -> world poses: placed around the room centre, each facing the gap between two neighbouring cameras with a random tilt."""
    out = []
    w, h = SPEC.squares_x * SPEC.square_length_m, SPEC.squares_y * SPEC.square_length_m
    C = cams.centers
    for i in range(n):
        k = i % len(C)
        c = np.asarray(centre) + rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.4])
        z = 0.5 * (C[k] + C[(k + 1) % len(C)]) - c
        z /= np.linalg.norm(z)
        z = rodrigues(rng.uniform(-0.35, 0.35, 3)) @ z
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        x = rodrigues(z * rng.uniform(-0.4, 0.4)) @ x
        R = np.stack([x, np.cross(z, x), z], 1)
        out.append((R, c - R @ np.array([w / 2, h / 2, 0])))
    return out


def detect_rig(cams, poses, board_img, det):
    dets = {c.name: [] for c in cams}
    for f, (Rb, tb) in enumerate(poses):
        for cam in cams:
            R = cam.R @ Rb
            t = cam.R @ tb + cam.t
            img = render(board_img, cam, R, t)
            if img is None:
                continue
            r = B.detect_in_image(det, img, 12)
            if r is not None:
                dets[cam.name].append(B.BoardDetection(f, *r))
    return dets


@pytest.fixture(scope="module")
def rig(board_image):
    rng = np.random.default_rng(0)
    cams = ring_cameras(4, radius=2.6, height=1.6, center=(0, 0, 0.9), width=1280, height_px=720, fov_deg=55)
    poses = board_poses_world(rng, 64, cams)
    dets = detect_rig(cams, poses, board_image, SPEC.make_detector())
    return cams, poses, dets


def test_detection_matches_projection(board_image, rig):
    cams, poses, dets = rig
    cam = cams[0]
    obj = SPEC.object_points()
    d = dets[cam.name][0]
    Rb, tb = poses[d.frame]
    uv, _ = cam.project(obj[d.ids] @ Rb.T + tb)
    assert np.abs(uv - d.corners).max() < 0.7
    assert all(len(v) >= 15 for v in dets.values()), {k: len(v) for k, v in dets.items()}


def test_select_diverse_spreads():
    rng = np.random.default_rng(1)
    dets = [
        B.BoardDetection(i, rng.uniform(0, 100, (10, 2)) + [i * 10 % 600, i * 7 % 300], np.arange(10))
        for i in range(80)
    ]
    sel = B.select_diverse(dets, (640, 360), 24)
    assert len(sel) == 24 and len({d.frame for d in sel}) == 24
    assert B.select_diverse(dets[:5], (640, 360), 24) == dets[:5]


def test_intrinsics_recovered(rig):
    cams, _poses, dets = rig
    for cam in cams:
        r = B.calibrate_intrinsics(dets[cam.name], (cam.width, cam.height), SPEC, 40, fix_aspect=True, fix_k3=True)
        assert abs(r.K[0, 0] / cam.K[0, 0] - 1) < 0.01, (cam.name, r.K[0, 0], cam.K[0, 0], r.issues)
        assert abs(r.K[0, 2] - cam.K[0, 2]) < 6 and abs(r.K[1, 2] - cam.K[1, 2]) < 6
        assert not r.degenerate and r.rms_px < 0.5


def test_degenerate_frontoparallel_flagged(board_image):
    rng = np.random.default_rng(2)
    cam = ring_cameras(1, radius=2.0, height=0.9, center=(0, 0, 0.9), width=1280, height_px=720, fov_deg=55)[0]
    w, h = SPEC.squares_x * SPEC.square_length_m, SPEC.squares_y * SPEC.square_length_m
    det = SPEC.make_detector()
    dets = []
    for f in range(20):
        R = cam.R.T  # board parallel to the image plane
        c = cam.center + 2.0 * cam.axis + rng.uniform(-0.3, 0.3, 3) * [1, 1, 0]
        t = cam.R @ (c - R @ np.array([w / 2, h / 2, 0])) + cam.t
        img = render(board_image, cam, cam.R @ R, t)
        r = B.detect_in_image(det, img, 12)
        dets.append(B.BoardDetection(f, *r))
    res = B.calibrate_intrinsics(dets, (cam.width, cam.height), SPEC, 20)
    assert res.degenerate and ("low_view_diversity" in res.issues or res.issues)


def test_check_intrinsics_rules():
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], float)
    assert B.check_intrinsics(K, np.zeros(5), 1280, 720) == []
    K2 = K.copy()
    K2[1, 1] = 1200
    assert "aspect" in B.check_intrinsics(K2, np.zeros(5), 1280, 720)
    K3 = K.copy()
    K3[0, 2] = 100
    assert "principal_point" in B.check_intrinsics(K3, np.zeros(5), 1280, 720)
    assert "focal" in B.check_intrinsics(K * [[0.1, 1, 1], [1, 0.1, 1], [1, 1, 1]], np.zeros(5), 1280, 720)
    assert "low_view_diversity" in B.check_intrinsics(K, np.zeros(5), 1280, 720, [10, 11, 12, 10.5])


def _align_to_truth(out, cams):
    s, R, t = umeyama(out.centers, cams.centers, with_scale=False)
    return {cam.name: out[cam.name].transformed(R, t, s) for cam in cams}


def test_multi_camera_extrinsics_known_intrinsics(rig):
    cams, _poses, dets = rig
    sizes = {c.name: (c.width, c.height) for c in cams}
    intr = {c.name: (c.K, c.dist) for c in cams}
    out, rep = B.calibrate_multi({}, SPEC, BoardConfig(max_ba_frames=64), sizes, intrinsics=intr, detections=dets)
    assert out.names == cams.names and rep["ba"]["median_px_after"] < 0.3
    aligned = _align_to_truth(out, cams)
    for cam in cams:
        assert np.linalg.norm(aligned[cam.name].center - cam.center) < 1e-3, (
            cam.name,
            np.linalg.norm(aligned[cam.name].center - cam.center),
        )
        assert rotation_angle_deg(aligned[cam.name].R, cam.R) < 0.1
    for k, v in rep["baselines_m"].items():
        a, b = k.split("-")
        assert abs(v - np.linalg.norm(cams[a].center - cams[b].center)) < 1.5e-3


def test_multi_camera_extrinsics_estimated_intrinsics(rig, tmp_path):
    cams, _poses, dets = rig
    sizes = {c.name: (c.width, c.height) for c in cams}
    cfg = BoardConfig(n_intrinsic_frames=40, max_ba_frames=40)
    out, rep = B.calibrate_multi({}, SPEC, cfg, sizes, detections=dets)
    assert out.names == cams.names and rep["ba"]["median_px_after"] < 0.4
    assert not rep["degenerate_cameras"]
    aligned = _align_to_truth(out, cams)
    for cam in cams:
        assert np.linalg.norm(aligned[cam.name].center - cam.center) < 0.01, (
            cam.name,
            np.linalg.norm(aligned[cam.name].center - cam.center),
        )
        assert rotation_angle_deg(aligned[cam.name].R, cam.R) < 0.3
    cmp = B.compare_to_reference(out, cams)
    assert max(v["centre_mm"] for v in cmp.values()) < 10.0
    mpath = to_gui_json(out, tmp_path / "gui", SPEC.to_gui_dict(), rms_px=out.source["rms_px"])
    back = from_gui_json(mpath)
    ref = out[out.names[0]]
    expect = out.transformed(ref.R, ref.t)
    for x, y in zip(expect, back):
        assert np.allclose(x.t, y.t, atol=1e-9) and np.allclose(x.R, y.R, atol=1e-9)
    assert json.loads(Path(mpath).read_text())["board"]["square_length"] == pytest.approx(0.095)


def test_floor_world_from_board():
    rng = np.random.default_rng(4)
    cams = ring_cameras(3, radius=2.5, height=1.7)
    root = cams[0]
    placed = {c.name: (c.R @ root.R.T, c.t - c.R @ root.R.T @ root.t) for c in cams}  # X_cam = R X_root + t
    poses = {c.name: {} for c in cams}
    for f in range(6):  # boards flat on the floor (z = 0), random yaw and position
        Rw = rodrigues([0, 0, rng.uniform(0, 6.28)])
        tw = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.0])
        for c in cams:
            poses[c.name][f] = (c.R @ Rw, c.R @ tw + c.t)
    R, t = B._floor_world(placed, poses, cams.names)
    for c in cams:  # cameras end up above a floor at z = 0, at their true heights
        Rr, tr = placed[c.name]
        z = (R @ (-Rr.T @ tr) + t)[2]
        assert abs(z - 1.7) < 1e-6
    corner_root = root.R @ np.array([0.3, 0.2, 0.0]) + root.t  # a floor point, in the root camera frame
    assert abs((R @ corner_root + t)[2]) < 1e-9


def test_board_spec_from_config_mm():
    spec = B.BoardSpec.from_config(BoardConfig(square_length=95, marker_length=71))
    assert spec.square_length_m == pytest.approx(0.095) and spec.marker_length_m == pytest.approx(0.071)
    assert spec.to_gui_dict()["units"] == "m"


@pytest.mark.slow
def test_real_board_videos():
    vdir = os.environ.get("FROLIC_TEST_BOARD_DIR")
    ref = os.environ.get("FROLIC_TEST_BOARD_REF")
    if not vdir or not ref:
        pytest.skip("set FROLIC_TEST_BOARD_DIR and FROLIC_TEST_BOARD_REF")
    from frolic.io.video import probe_video

    vids = {f"cam{k}": Path(vdir) / f"Room6 CAM {k} 01.mp4" for k in range(1, 7)}
    vids = {k: v for k, v in vids.items() if v.exists()}
    assert len(vids) >= 2
    sizes = {n: (probe_video(p).width, probe_video(p).height) for n, p in vids.items()}
    spec = B.BoardSpec(8, 6, 0.095, 0.071, "DICT_4X4_50", True)
    cfg = BoardConfig(
        step=int(os.environ.get("FROLIC_TEST_BOARD_STEP", "30")), n_intrinsic_frames=40, max_ba_frames=150
    )
    out, rep = B.calibrate_multi(vids, spec, cfg, sizes)
    with open(Path(ref) / "report.json") as fh:
        r = json.load(fh)
    for i, n in enumerate(out.names):
        rf = r["intrinsics"].get(f"view_{i}")
        if rf and n not in rep["degenerate_cameras"]:
            assert abs(out[n].K[0, 0] / rf["fx"] - 1) < 0.05, (n, out[n].K[0, 0], rf["fx"])
    assert rep["ba"]["median_px_after"] < 2.0
