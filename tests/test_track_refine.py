from __future__ import annotations

import numpy as np
import pytest

from frolic.config import TrackConfig
from frolic.skeleton import N_JOINTS, J
from frolic.synthetic import person_track, project_people
from frolic.track import refine
from frolic.track.associate import associate
from frolic.track.refine import blank_outliers, one_euro, refine_tracks, smooth_runs


def test_one_euro_removes_jitter():
    t = np.arange(200) / 15.0
    clean = np.sin(0.5 * t)
    noisy = clean + np.random.default_rng(0).normal(0, 0.03, len(t))
    f = one_euro(noisy, 15.0, 0.5, 5.0)
    assert np.abs(f - clean).mean() < 0.75 * np.abs(noisy - clean).mean()


def test_smooth_runs_keeps_gaps():
    X = np.ones((20, N_JOINTS, 3))
    X[8:11] = np.nan
    out = smooth_runs(X, 15.0, 0.3, 5.0)
    assert np.isnan(out[8:11]).all() and np.isfinite(out[:8]).all()


def test_blank_outliers():
    P = np.zeros((30, N_JOINTS, 3))
    P[:, :, 0] = np.linspace(0, 1, 30)[:, None]
    P[15, J["pelvis"]] += [0, 0.6, 0]
    out = blank_outliers(P, 15.0)
    assert out[15] and out.sum() == 1


def test_refine_reduces_noise(cams):
    fps, n = 15.0, 60
    people = [
        person_track(n, fps, 1.7, (-1, -0.5), (1, 0.5)),
        person_track(n, fps, 1.0, (1, -0.5), (-1, 0.5), phase=0.2),
    ]
    dets = project_people(cams, people, noise_px=3.0)
    cfg = TrackConfig(fps=fps, reid="none", refine_smooth=20.0)
    tr = associate(dets, cams, cfg, fps, n)
    tr.ids = [f"p{k}" for k in range(tr.n_people)]
    ref = refine_tracks(tr, dets, cams, cfg)
    for p in range(tr.n_people):
        g = int(
            np.argmin(
                [np.nanmedian(np.linalg.norm(tr.xyz[p, :, J["pelvis"]] - q[:, J["pelvis"]], axis=1)) for q in people]
            )
        )
        have = tr.has_3d[p] & ref.has_3d[p]
        before = np.nanmedian(np.linalg.norm(tr.xyz[p][have] - people[g][have], axis=-1))
        after = np.nanmedian(np.linalg.norm(ref.xyz[p][have] - people[g][have], axis=-1))
        assert after < before * 0.95
        assert ref.has_3d[p].sum() >= 0.8 * tr.has_3d[p].sum()


def test_refine_numpy_fallback(cams, two_people, gt_detections, monkeypatch):
    people, fps = two_people
    tr = associate(gt_detections, cams, TrackConfig(fps=fps, reid="none"), fps, people[0].shape[0])
    monkeypatch.setattr(refine, "_anipose_group", lambda cams: None)
    ref = refine_tracks(tr, gt_detections, cams, TrackConfig(fps=fps, reid="none"))
    assert ref.has_3d.sum() >= 0.9 * tr.has_3d.sum()
    p = 0
    have = ref.has_3d[p]
    g = int(
        np.argmin(
            [np.nanmedian(np.linalg.norm(ref.xyz[p, have, J["pelvis"]] - q[have, J["pelvis"]], axis=1)) for q in people]
        )
    )
    assert np.nanmedian(np.linalg.norm(ref.xyz[p][have] - people[g][have], axis=-1)) < 0.03


@pytest.mark.skipif(not pytest.importorskip("aniposelib", reason="aniposelib"), reason="aniposelib")
def test_anipose_group_builds(cams):
    cg = refine._anipose_group(cams)
    assert cg is not None and len(cg.cameras) == len(cams)
