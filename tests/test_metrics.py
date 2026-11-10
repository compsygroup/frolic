from __future__ import annotations

import json

import numpy as np
from test_export import build_work, fake_mesh, write_fake_fit

from frolic.export.stage import run as run_export
from frolic.io.calib import load_cameras
from frolic.io.detections import Detections
from frolic.metrics.compute import (
    compute_metrics,
    flips,
    jitter,
    reprojection,
    root_spins,
    violations_from_fit,
    write_summary_csv,
)
from frolic.skeleton import N_JOINTS


def test_jitter_zero_for_linear_motion():
    c = np.c_[np.linspace(0, 3, 30), np.zeros(30), np.ones(30)]
    m = jitter(c, 15.0)
    assert m["jitter_acc_median_ms2"] < 1e-6
    c2 = c.copy()
    c2[10] += 0.1
    assert jitter(c2, 15.0)["jitter_acc_p90_ms2"] > 1.0


def test_flips_detects_jump():
    J = np.zeros((20, N_JOINTS, 3))
    J[:, :, 0] = np.linspace(0, 1, 20)[:, None]
    V, _ = fake_mesh(J)
    assert flips(V, None)["flip_frames"] == 0
    V[12:, :8] += 0.3  # two body parts jump, the rest moves on: a flip
    f = flips(V, None)
    assert f["flip_frames"] == 1 and f["_flip_mask"][12]
    ref = V.copy()  # the reference jumps as much: not a flip of the fit
    assert flips(V, ref)["flip_frames"] == 0
    V[12:] += 0.3  # the whole body steps: not a flip
    assert flips(V[11:14], None)["flip_frames"] == 0


def test_root_spins():
    n = 60
    smooth = np.c_[np.zeros(n), np.zeros(n), np.linspace(0, np.pi / 2, n)]
    assert root_spins(smooth, 15.0)["root_spin_events"] == 0
    spin = smooth.copy()
    spin[20:30, 2] += np.linspace(0, 2 * np.pi, 10)  # a full turn in 10 frames, no net heading change
    r = root_spins(spin, 15.0)
    assert r["root_spin_events"] == 1 and 1.2 <= r["root_spin_times_s"][0] <= 2.0


def test_violations_from_fit_report(tmp_path):
    J = np.zeros((5, N_JOINTS, 3))
    write_fake_fit(tmp_path, "p0", J, 15.0)
    v = violations_from_fit(tmp_path / "p0.npz")
    assert v["biomech_source"] == "fit report" and v["biomech"]["lim_knee"] == 0
    assert violations_from_fit(tmp_path / "missing.npz")["biomech_source"] == "none"


def test_reprojection_against_gt_detections(tmp_path):
    work, cfg, _people = build_work(tmp_path / "w")
    run_export(cfg, work)
    out = work.stage_dir("export")
    cams = load_cameras(out / "cameras.json")
    dets = {c: Detections.load(work.stage_dir("detect") / f"{c}.npz") for c in cams.names}
    ppl = {p: dict(np.load(out / f"{p}.npz")) for p in ("p0", "p1")}
    r = reprojection(ppl, cams, dets)
    for p in ("p0", "p1"):
        assert r[p]["box_iou_median"] > 0.6, r[p]
        assert r[p]["kp2d_err_median"] < 0.08, r[p]
        assert r[p]["n_box_pairs"] > 50


def test_compute_metrics_and_summary(tmp_path):
    work, cfg, _people = build_work(tmp_path / "w")
    run_export(cfg, work)
    out = work.stage_dir("export")
    with open(out / "metrics.json") as fh:
        m = json.load(fh)
    assert m["schema"] == "frolic.metrics/1"
    for p in ("p0", "p1"):
        pm = m["per_person"][p]
        assert pm["coverage_pct"] == 100.0
        assert pm["flip_frames"] == 0 and pm["root_spin_events"] == 0
        assert pm["lowest_vertex_median_cm"] > -5 and pm["floor_below_5cm_pct"] == 0.0
        assert pm["biomech_source"] == "fit report"
    assert m["global"]["n_people"] == 2 and m["global"]["n_cameras"] == 4
    csv_path = tmp_path / "s.csv"
    write_summary_csv(m, csv_path)
    assert len(csv_path.read_text().splitlines()) == 3


def test_metrics_without_detections(tmp_path):
    work, cfg, _ = build_work(tmp_path / "w", with_detect=False)
    run_export(cfg, work)
    m = compute_metrics(work, work.stage_dir("export"))
    assert "box_iou_median" not in m["per_person"]["p0"]
    assert m["global"]["box_iou_median_mean"] is None
