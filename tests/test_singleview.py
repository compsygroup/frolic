from __future__ import annotations

import numpy as np

from frolic3d.config import SingleViewConfig, TrackConfig
from frolic3d.singleview.lift import interp_prior, lift_frames, rel_depth_from_keypoints, room_bounds, spans_of
from frolic3d.skeleton import J
from frolic3d.synthetic import person_track, project_people, ring_cameras
from frolic3d.track.associate import associate


def _scene(noise=0.5, gap=(18, 30), n=45, fps=15.0):
    cams = ring_cameras(4)
    people = [
        person_track(n, fps, 1.7, (-1.0, -0.6), (1.0, 0.6)),
        person_track(n, fps, 0.95, (1.0, -0.6), (-1.0, 0.6), phase=0.3),
    ]
    # person 1 hidden in every camera but cam1 during the gap
    occ = {c: [(1, gap[0], gap[1])] for c in ("cam2", "cam3", "cam4")}
    dets = project_people(cams, people, noise_px=noise, occlude=occ)
    tr = associate(dets, cams, TrackConfig(fps=fps, reid="none"), fps, n)
    tr.ids = [f"p{k}" for k in range(tr.n_people)]
    k = int(
        np.argmin(
            [
                np.nanmedian(np.linalg.norm(tr.xyz[p, :, J["pelvis"]] - people[1][:, J["pelvis"]], axis=1))
                for p in range(tr.n_people)
            ]
        )
    )
    return cams, people, dets, tr, k


def test_lift_recovers_occluded_span():
    cams, people, dets, tr, k = _scene()
    assert not tr.has_3d[k, 18:30].any()
    out, rep = lift_frames(tr, dets, cams, SingleViewConfig())
    pid = tr.ids[k]
    lifted = rep["people"][pid]["lifted_frames"]
    assert len(lifted) >= 9
    err = np.linalg.norm(out.xyz[k, lifted, J["pelvis"]] - people[1][lifted, J["pelvis"]], axis=1)
    assert np.median(err) < 0.15
    assert out.single_view[k, lifted].all() and out.observed[k, lifted].all()
    assert (out.score[k, lifted, J["pelvis"]] == SingleViewConfig().score).all()
    assert (out.support[k, lifted] >= 0).sum(1).max() == 1
    # the other person is untouched
    o = 1 - k
    assert np.array_equal(np.isfinite(out.xyz[o]), np.isfinite(tr.xyz[o]))
    assert rep["people"][pid]["spans"][0]["observed"] >= 9


def test_lift_with_relative_depths():
    cams, people, dets, tr, k = _scene()
    pid = tr.ids[k]
    rel = {}
    for i in range(18, 30):
        d = (people[1][i] - people[1][i, J["pelvis"]]) @ cams["cam1"].axis
        from frolic3d.skeleton import HALPE26

        rel[(pid, "cam1", i)] = {HALPE26[j]: float(d[j]) for j in range(len(HALPE26))}
    out, rep = lift_frames(tr, dets, cams, SingleViewConfig(), rel_depth=rel)
    lifted = rep["people"][pid]["lifted_frames"]
    assert lifted and all(rep["people"][pid]["frames"][str(i)]["rel"] == "depths" for i in lifted)
    err = np.linalg.norm(out.xyz[k, lifted, J["knee_l"]] - people[1][lifted, J["knee_l"]], axis=1)
    assert np.nanmedian(err) < 0.15


def test_rejection_room_and_claimed():
    cams, _, dets, tr, k = _scene()
    pid = tr.ids[k]
    # a room box that excludes the person's path -> every lift rejected for "room"
    _, rep = lift_frames(tr, dets, cams, SingleViewConfig(), bounds=(5.0, 6.0, 5.0, 6.0))
    assert rep["people"][pid]["lifted_frames"] == []
    assert set(rep["people"][pid]["rejected"].values()) == {"room"}
    # a gate too tight to accept any candidate -> nothing observed
    _, rep = lift_frames(tr, dets, cams, SingleViewConfig(gate_m=0.001, gate_max_m=0.001, vmax_mps=0.0))
    assert rep["people"][pid]["observed_frames"] == []


def test_helpers():
    P = np.full((10, 17, 3), np.nan)
    valid = np.zeros(10, bool)
    P[2], P[6] = 0.0, 4.0
    valid[[2, 6]] = True
    X, dt, tpl = interp_prior(P, valid, 4, 2.0)
    assert np.allclose(X, 2.0) and dt == 1.0 and tpl == 2
    assert interp_prior(P, np.zeros(10, bool), 4, 2.0)[0] is None
    assert spans_of([1, 2, 3, 7, 8, 10]) == [[1, 3], [7, 8], [10, 10]]
    K3 = np.zeros((70, 3))
    K3[9], K3[10] = [0, 0, 2.0], [0, 0, 2.0]
    K3[5], K3[6] = [0, 0.5, 2.0], [0, 0.5, 2.0]
    K3[11] = [0, 0, 2.3]
    rel = rel_depth_from_keypoints(K3, 0.25, {13: 11}, (9, 10), (5, 6))
    assert np.isclose(rel[13], 0.15)
    _, _, _, tr, _ = _scene()
    x0, x1, y0, y1 = room_bounds(tr)
    assert x0 < -1 and x1 > 1 and y0 < -0.6 and y1 > 0.6


def test_sam3d_depths_relift_single_view_frames(tmp_path):
    from frolic3d.config import FrolicConfig
    from frolic3d.io.workdir import WorkDir
    from frolic3d.sam3d.adapter import expand17to70
    from frolic3d.sam3d.run import ViewResults
    from frolic3d.sam3d.stage import refine_single_view

    cams, people, dets, tr, k = _scene()
    work = WorkDir(tmp_path / "work")
    tr.save(work.tracks_selected)
    sv, _ = lift_frames(tr, dets, cams, SingleViewConfig())
    sv.save(work.tracks_singleview)
    frames = np.flatnonzero(sv.single_view[k])
    cam = cams["cam1"]
    Xc = people[1][frames] @ cam.R.T + cam.t
    vr = ViewResults(np.full(len(frames), k), frames, {"pred_keypoints_3d": expand17to70(Xc.astype(np.float32))})
    out = refine_single_view(
        work,
        sv,
        {"cam1": vr},
        cams,
        dets,
        FrolicConfig.model_validate({"calib": {"mode": "bootstrap", "reference_height_m": 1.7}}),
    )
    assert out.single_view[k, frames].all()
    import json

    rep = json.loads((work.stage_dir("sam3d") / "svfill_sam3d.json").read_text())
    pid = tr.ids[k]
    lifted = rep["people"][pid]["lifted_frames"]
    assert lifted and all(rep["people"][pid]["frames"][str(i)]["rel"] == "depths" for i in lifted)
    err = np.linalg.norm(out.xyz[k, lifted, J["knee_l"]] - people[1][lifted, J["knee_l"]], axis=1)
    assert np.nanmedian(err) < 0.15
