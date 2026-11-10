from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from frolic.config import FrolicConfig
from frolic.export.stage import interpolate_joints
from frolic.export.stage import run as run_export
from frolic.export.visibility import fill_gaps, visible_mask
from frolic.io.calib import save_cameras
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.skeleton import JOINT_NAMES, N_JOINTS
from frolic.synthetic import halpe26_from_joints, person_track, project_people, ring_cameras

TETRA = np.array([[0.03, 0, 0], [-0.015, 0.026, 0], [-0.015, -0.026, 0], [0, 0, 0.03]])


def fake_mesh(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A tetrahedron around every Halpe-26 point of the body: (N, 104, 3) vertices and (104, 3) faces."""
    pts = halpe26_from_joints(joints)
    V = (pts[:, :, None, :] + TETRA[None, None]).reshape(len(joints), -1, 3)
    faces = []
    for j in range(pts.shape[1]):
        b = 4 * j
        faces += [[b, b + 1, b + 2], [b, b + 1, b + 3], [b, b + 2, b + 3], [b + 1, b + 2, b + 3]]
    return V.astype(np.float32), np.array(faces, np.int32)


def write_fake_fit(
    fit_dir: Path,
    pid: str,
    joints: np.ndarray,
    fps: float,
    have: np.ndarray | None = None,
    rootrot: np.ndarray | None = None,
    stature: float = 1.7,
) -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    n = len(joints)
    have = np.ones(n, bool) if have is None else have
    J = joints.copy()
    J[~have] = np.nan
    V, F = fake_mesh(np.nan_to_num(joints))
    V[~have] = np.nan
    heading = np.arctan2(joints[:, 1, 1] - joints[:, 2, 1], joints[:, 1, 0] - joints[:, 2, 0])
    rr = np.c_[np.zeros(n), np.zeros(n), heading] if rootrot is None else rootrot
    np.savez_compressed(
        fit_dir / f"{pid}.npz",
        phenotype=np.full(6, 0.5, np.float32),
        phenotype_labels=np.array(["gender", "age", "muscle", "weight", "height", "proportions"]),
        rootrot=rr.astype(np.float32),
        roottr=joints[:, 0].astype(np.float32),
        rotvecs=np.zeros((n, 19, 3), np.float32),
        posed_bones=np.array(["spine05"] * 19),
        joints=J.astype(np.float32),
        target=J.astype(np.float32),
        have=have,
        single_view=np.zeros(n, bool),
        bridged=np.zeros(n, bool),
        observed=have.copy(),
        frame0=0,
        fps=fps,
        stature=stature,
        seg_labels=np.array([]),
        seg_values=np.array([]),
        biomech_report=np.array(json.dumps({"lim_knee": 0, "cap_pen_gt1cm": 0})),
    )
    np.save(fit_dir / f"{pid}_verts.npy", V.astype(np.float16))
    np.save(fit_dir / "faces.npy", F)


def build_work(
    root: Path, n_frames: int = 30, fps: float = 15.0, with_detect: bool = True, videos: dict | None = None
) -> tuple[WorkDir, FrolicConfig, list[np.ndarray]]:
    """Work directory with fake fit outputs for two people, levelled cameras, selected tracks and gt detections."""
    work = WorkDir(root)
    cams = ring_cameras(4)
    adult = person_track(n_frames, fps, 1.72, (-1.0, -0.6), (1.0, 0.6))
    child = person_track(n_frames, fps, 0.95, (1.0, -0.6), (-1.0, 0.6), phase=0.3)
    people = [adult, child]
    save_cameras(cams, work.cameras_level)
    save_cameras(cams, work.cameras)
    tr = Tracks.empty(["p0", "p1"], n_frames, cams.names, fps, 2, 0)
    for k, X in enumerate(people):
        tr.xyz[k] = X
        tr.score[k] = 0.9
    tr.labels = {"p0": "adult", "p1": "child"}
    tr.save(work.tracks_selected)
    for k, X in enumerate(people):
        write_fake_fit(work.stage_dir("fit"), f"p{k}", X, fps, stature=[1.72, 0.95][k])
    if with_detect:
        dets = project_people(cams, [X[::1] for X in people], noise_px=0.5)
        # track frame i = aligned frame 2 i: re-index the detections onto the aligned timeline
        for name, d in dets.items():
            d.frame = d.frame * 2
            d._index = None
            d.meta.update({"fps": 2 * fps, "width": 640, "height": 360})
            d.save(work.stage_dir("detect") / f"{name}.npz")
    work.write_videos(
        {
            "cameras": cams.names,
            "videos": videos or {},
            "frame_offset": dict.fromkeys(cams.names, 0),
            "fps": 2 * fps,
            "n_frames": 2 * n_frames,
            "offsets_s": dict.fromkeys(cams.names, 0.0),
        }
    )
    cfg = FrolicConfig(calib={"mode": "given", "given": str(work.cameras)})
    cfg.input.work_dir = root
    cfg.people = {"p1": {"label": "kid"}}
    return work, cfg, people


def test_fill_gaps():
    have = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1], bool)
    assert fill_gaps(have, 3).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert fill_gaps(have, 10).all()


def test_visible_mask_observed_rule():
    n = 20
    have = np.ones(n, bool)
    have[5:12] = False
    obs = have.copy()
    obs[5:9] = True  # cameras see the person but no 3D
    show = visible_mask(have, None, obs, 15.0, max_gap_s=0.0, observed_rule=True)
    assert show[5:9].all() and not show[9:12].any()
    show_far = visible_mask(have, None, obs, 15.0, 0.0, True, bridge_s=0.1)
    assert not show_far[5:9].any()
    sv = np.zeros(n, bool)
    sv[:3] = True
    show_tri = visible_mask(have, sv, obs, 15.0, 0.0, False)
    assert not show_tri[:3].any() and show_tri[3:5].all()


def test_interpolate_joints():
    n = 10
    joints = np.zeros((n, N_JOINTS, 3), np.float32)
    joints[:, :, 0] = np.arange(n)[:, None]
    t = np.arange(n) / 15.0
    show = np.ones(n, bool)
    jf, tf = interpolate_joints(joints, t, 30.0, show)
    assert len(tf) == 19 and np.isclose(jf[1, 0, 0], 0.5)
    show[4] = False
    jf, _ = interpolate_joints(joints, t, 30.0, show)
    assert np.isnan(jf[8, 0, 0])


def test_export_stage_writes_schema(tmp_path):
    work, cfg, _people = build_work(tmp_path / "w")
    cfg.export.full_rate_joints = True
    info = run_export(cfg, work)
    assert info["people"] == ["p0", "p1"]
    out = work.stage_dir("export")
    z = np.load(out / "p1.npz")
    for k in [
        "vertices",
        "faces",
        "joints",
        "joint_names",
        "frame_times",
        "source_frames",
        "phenotype",
        "phenotype_labels",
        "stature_m",
        "single_view",
        "observed",
        "label",
    ]:
        assert k in z.files, k
    assert z["vertices"].dtype == np.float16 and z["vertices"].shape[:2] == (30, 4 * 26)
    assert list(z["joint_names"]) == JOINT_NAMES
    assert str(z["label"]) == "kid"
    assert z["source_frames"][3] == 6 and np.isclose(z["frame_times"][3], 0.2)
    with open(out / "meta.json") as fh:
        meta = json.load(fh)
    assert meta["schema"] == "frolic.export/1" and len(meta["people"]) == 2
    assert (out / "cameras.json").exists() and (out / "metrics.json").exists() and (out / "summary.csv").exists()
    assert (out / "p0_joints_full.npz").exists()
    rows = (out / "summary.csv").read_text().strip().splitlines()
    assert rows[0].startswith("id,label") and len(rows) == 3


def test_export_hides_unobserved_gap(tmp_path):
    work, cfg, people = build_work(tmp_path / "w", with_detect=False)
    have = np.ones(30, bool)
    have[10:25] = False
    write_fake_fit(work.stage_dir("fit"), "p0", people[0], 15.0, have=have)
    run_export(cfg, work)
    z = np.load(work.stage_dir("export") / "p0.npz")
    assert np.isnan(z["vertices"][15, 0, 0]) and np.isnan(z["joints"][15, 0, 0])
    assert np.isfinite(z["vertices"][5, 0, 0])
    assert z["observed"].sum() == 15


def test_export_needs_fit(tmp_path):
    work = WorkDir(tmp_path / "w")
    cfg = FrolicConfig(calib={"mode": "given", "given": "x.json"})
    with pytest.raises(FileNotFoundError):
        run_export(cfg, work)
