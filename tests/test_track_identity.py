from __future__ import annotations

import numpy as np

from frolic.config import TrackConfig
from frolic.io.tracks import Tracks
from frolic.skeleton import N_JOINTS, J
from frolic.track.associate import associate
from frolic.track.identity import assemble_identities, bridge_gaps


def _raw_split_tracks(cams, two_people, gt_detections):
    """Raw tracks where person 0 is split into two slots with a gap between them (as a lost/re-acquired track)."""
    people, fps = two_people
    tr = associate(gt_detections, cams, TrackConfig(fps=fps, reid="none"), fps, people[0].shape[0])
    N = tr.n_frames
    out = Tracks.empty(["t0", "t1", "t2"], N, tr.cameras, fps, 1, 0)
    k = np.argmin(
        [
            np.nanmedian(np.linalg.norm(tr.xyz[p, :, J["pelvis"]] - people[0][:, J["pelvis"]], axis=1))
            for p in range(tr.n_people)
        ]
    )
    o = 1 - k
    cut = N // 2
    out.xyz[0, :cut], out.score[0, :cut], out.support[0, :cut] = tr.xyz[k, :cut], tr.score[k, :cut], tr.support[k, :cut]
    out.xyz[2, cut + 3 :], out.score[2, cut + 3 :], out.support[2, cut + 3 :] = (
        tr.xyz[k, cut + 3 :],
        tr.score[k, cut + 3 :],
        tr.support[k, cut + 3 :],
    )
    out.xyz[1], out.score[1], out.support[1] = tr.xyz[o], tr.score[o], tr.support[o]
    return out, people


def test_identities_merge_split_slots(cams, two_people, gt_detections):
    raw, people = _raw_split_tracks(cams, two_people, gt_detections)
    cfg = TrackConfig(fps=raw.fps, reid="none", min_identity_frac=0.03, bridge_s=1.0)
    tr, info = assemble_identities(raw, gt_detections, cfg)
    assert tr.n_people == 2 and tr.ids == ["p0", "p1"]
    # the two fragments of person 0 (same size, same colour, never co-occurring) became one identity, gap bridged
    k = np.argmin(
        [np.nanmedian(np.linalg.norm(tr.xyz[p, :, J["pelvis"]] - people[0][:, J["pelvis"]], axis=1)) for p in range(2)]
    )
    assert tr.has_3d[k].mean() > 0.95
    assert info[k]["bridged_frames"] == 3
    assert 1.5 < info[k]["stature_m"] < 1.95
    assert all(d["median_views"] >= 2 for d in info)


def test_identity_overrides_drop_and_merge(cams, two_people, gt_detections):
    raw, _ = _raw_split_tracks(cams, two_people, gt_detections)
    cfg = TrackConfig(fps=raw.fps, reid="none", identity_merge=0.99, drop=["p1"])
    tr, _ = assemble_identities(raw, gt_detections, cfg)
    assert tr.n_people == 1
    cfg = TrackConfig(fps=raw.fps, reid="none", merge=[["p0", "p1"]])
    tr2, _ = assemble_identities(raw, gt_detections, cfg)
    assert tr2.n_people == 1 and tr2.has_3d[0].mean() > 0.95


def test_identity_split_override(cams, two_people, gt_detections):
    people, fps = two_people
    raw = associate(gt_detections, cams, TrackConfig(fps=fps, reid="none"), fps, people[0].shape[0])
    cfg = TrackConfig(fps=fps, reid="none", split={"p0": [1.0]})
    tr, _ = assemble_identities(raw, gt_detections, cfg)
    assert tr.n_people == 3
    assert not tr.has_3d[0, 15:].any() and not tr.has_3d[1, :15].any()


def test_floating_ghost_and_room_filter(cams, two_people, gt_detections):
    people, fps = two_people
    raw = associate(gt_detections, cams, TrackConfig(fps=fps, reid="none"), fps, people[0].shape[0])
    N = raw.n_frames
    ghost = Tracks.empty(["t0", "t1", "t2"], N, raw.cameras, fps, 1, 0)
    ghost.xyz[:2], ghost.score[:2], ghost.support[:2] = raw.xyz[:2], raw.score[:2], raw.support[:2]
    g = np.zeros((N, N_JOINTS, 3), np.float32)
    g[:, :, 2] = 2.0
    g[:, :, 0] = 0.5
    ghost.xyz[2] = g
    ghost.score[2] = 0.8
    tr, _ = assemble_identities(ghost, gt_detections, TrackConfig(fps=fps, reid="none", min_identity_frac=0.01))
    assert tr.n_people == 2
    tr2, _ = assemble_identities(
        ghost, gt_detections, TrackConfig(fps=fps, reid="none", room_bounds_m=[-3, 3, -3, -0.2], min_identity_frac=0.01)
    )
    assert all(tr2.xyz[:, :, J["pelvis"], 1][tr2.has_3d] <= -0.2)


def test_bridge_gaps():
    xyz = np.full((14, N_JOINTS, 3), np.nan, np.float32)
    sc = np.zeros((14, N_JOINTS), np.float32)
    xyz[[0, 1, 5, 6, 13]] = 1.0
    xyz[5:7] = 3.0
    sc[[0, 1, 5, 6, 13]] = 0.8
    n = bridge_gaps(xyz, sc, 3)
    assert n == 3 and np.isfinite(xyz[2:5]).all() and np.isnan(xyz[7:13]).all()
    assert np.isclose(xyz[3, 0, 0], 2.0)
