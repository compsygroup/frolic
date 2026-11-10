from __future__ import annotations

import numpy as np

from frolic.config import TrackConfig
from frolic.skeleton import J
from frolic.synthetic import person_track, project_people
from frolic.track.associate import associate, estimate_people


def _match(tr, people):
    """Map each raw slot to the gt person it follows (by median pelvis distance) and return the per-slot errors."""
    out = {}
    for p in range(tr.n_people):
        have = tr.has_3d[p]
        if have.sum() < 5:
            continue
        errs = [
            np.nanmedian(np.linalg.norm(tr.xyz[p, have, J["pelvis"]] - g[have, J["pelvis"]], axis=1)) for g in people
        ]
        out[p] = (int(np.argmin(errs)), float(min(errs)))
    return out


def test_associate_two_people_crossing(cams, two_people, gt_detections):
    people, fps = two_people
    cfg = TrackConfig(fps=fps, reid="none")
    tr = associate(gt_detections, cams, cfg, fps, people[0].shape[0])
    assert tr.stride == 1 and tr.n_frames == people[0].shape[0]
    m = _match(tr, people)
    assert len(m) >= 2 and {g for g, _ in m.values()} == {0, 1}
    for p, (g, _) in m.items():
        have = tr.has_3d[p]
        assert have.mean() > 0.9
        err = np.linalg.norm(tr.xyz[p][have] - people[g][have], axis=-1)
        assert np.nanmedian(err) < 0.02
        # the slot never swaps person through the crossing
        d_self = np.linalg.norm(tr.xyz[p, have, J["pelvis"]] - people[g][have, J["pelvis"]], axis=1)
        d_other = np.linalg.norm(tr.xyz[p, have, J["pelvis"]] - people[1 - g][have, J["pelvis"]], axis=1)
        assert (d_self < d_other).mean() > 0.95
        # support rows point at detections of the right frames
        for i in np.flatnonzero(have)[:5]:
            for v, cam in enumerate(tr.cameras):
                r = tr.support[p, i, v]
                if r >= 0:
                    assert gt_detections[cam].frame[r] == tr.source_frame(i)


def test_associate_with_occlusion_and_subsampling(cams):
    fps, n = 30.0, 60
    people = [
        person_track(n, fps, 1.7, (-1, -0.5), (1, 0.5)),
        person_track(n, fps, 1.0, (1, -0.5), (-1, 0.5), phase=0.2),
    ]
    occ = {"cam1": [(0, 20, 35)], "cam2": [(0, 20, 35)]}
    dets = project_people(cams, people, noise_px=0.5, occlude=occ)
    cfg = TrackConfig(fps=15.0, reid="none", lost_frames=30)
    tr = associate(dets, cams, cfg, fps, n)
    assert tr.stride == 2 and tr.n_frames == 30
    gt = [g[::2] for g in people]
    m = _match(tr, gt)
    assert {g for g, _ in m.values()} == {0, 1}
    for p, (g, _) in m.items():
        have = tr.has_3d[p]
        err = np.linalg.norm(tr.xyz[p][have] - gt[g][have], axis=-1)
        assert np.nanmedian(err) < 0.03
    # person 0 is still triangulated by the two remaining cameras during the occlusion
    p0 = next(p for p, (g, _) in m.items() if g == 0)
    assert tr.has_3d[p0, 10:17].mean() > 0.8
    assert (tr.n_views()[p0, 10:17] <= 2).all()


def test_estimate_people(gt_detections):
    frames = list(range(0, 45, 5))
    assert estimate_people(frames, gt_detections) == 2


def test_associate_min_views_three(cams, two_people, gt_detections):
    people, fps = two_people
    cfg = TrackConfig(fps=fps, reid="none", min_views=3)
    tr = associate(gt_detections, cams, cfg, fps, people[0].shape[0])
    assert (tr.n_views()[tr.has_3d] >= 3).all()
