import numpy as np
import pytest

from frolic.calib import bootstrap as BS
from frolic.config import BootstrapConfig
from frolic.geometry import rotation_angle_deg, umeyama
from frolic.synthetic import person_track, project_people, ring_cameras


@pytest.fixture(scope="module")
def scene():
    cams = ring_cameras(4, width=640, height_px=360, fov_deg=60)
    walker = person_track(120, 15.0, 1.72, (-1.2, -0.8), (1.2, 0.9))
    dets = project_people(cams, [walker], noise_px=0.5, seed=1)
    return cams, walker, dets


def test_pick_frames_single_person(scene):
    cams, _walker, dets = scene
    picked = BS.pick_frames(dets, max_frames=50)
    assert 40 <= len(picked) <= 50
    assert all(set(v) == set(cams.names) for v in picked.values())


def test_pick_frames_colour_consistency(scene):
    cams, walker, _ = scene
    other = person_track(120, 15.0, 1.60, (1.0, -1.0), (-1.0, 1.0), phase=0.4)
    dets = project_people(cams, [walker, other], noise_px=0.3)
    picked = BS.pick_frames(dets, max_frames=400)
    assert len(picked) > 50
    # every picked frame holds one consistent person (same colour index) in every camera
    for rows in picked.values():
        hues = {dets[c].colour[r][0] for c, r in rows.items()}
        assert len(hues) == 1


def test_bootstrap_recovers_rig(scene):
    cams, _walker, dets = scene
    cfg = BootstrapConfig(max_frames=120, min_pair_frames=30, ransac_px=3.0)
    res = BS.bootstrap(dets, cfg, reference_height_m=1.72)
    out = res.cameras
    assert out.names == cams.names
    for n in cams.names:
        assert abs(out[n].K[0, 0] / cams[n].K[0, 0] - 1) < 0.02, (n, out[n].K[0, 0], cams[n].K[0, 0])
    s, R, t = umeyama(out.centers, cams.centers, with_scale=True)
    assert abs(s - 1) < 0.1  # metric scale from the stature prior
    for n in cams.names:
        c = out[n].transformed(R, t, s)
        assert np.linalg.norm(c.center - cams[n].center) < 0.05, (n, np.linalg.norm(c.center - cams[n].center))
        assert rotation_angle_deg(c.R, cams[n].R) < 0.5
    assert res.report["ba"]["median_px_after"] < 1.0
    # the world is roughly z-up with the origin near the feet
    assert abs(out.centers[:, 2].mean() - cams.centers[:, 2].mean()) < 0.3


def test_bootstrap_needs_enough_frames(scene):
    _cams, _walker, dets = scene
    short = {c: d.subset(np.flatnonzero(d.frame < 5)) for c, d in dets.items()}
    for d in short.values():
        d.meta = dict(dets["cam1"].meta)
    with pytest.raises(RuntimeError):
        BS.bootstrap(short, BootstrapConfig(min_pair_frames=40), 1.7)


def test_stature_estimate():
    X = np.full((10, 26, 3), np.nan)
    X[:, BS.HEAD] = [0, 0, 1.6]
    X[:, BS.ANKLES[0]] = [0.1, 0, 0.0]
    X[:, BS.ANKLES[1]] = [-0.1, 0, 0.0]
    assert BS.stature_estimate(X) == pytest.approx(1.6 / BS.HEAD_ANKLE_FRACTION)


def test_bootstrap_scale_from_the_chosen_person():
    cams = ring_cameras(4, width=640, height_px=360, fov_deg=60)
    adult = person_track(120, 15.0, 1.75, (-1.2, -0.8), (1.2, 0.9))
    child = person_track(120, 15.0, 1.00, (1.0, -0.8), (-1.0, 0.8), phase=0.3)
    occ = {c: [(1, 0, 70), (0, 70, 120)] for c in cams.names}
    dets = project_people(cams, [adult, child], noise_px=0.5, seed=2, occlude=occ)
    cfg = BootstrapConfig(max_frames=120, min_pair_frames=30, ransac_px=3.0, focal_guess=float(cams["cam1"].K[0, 0]))
    for rule, height in (("shortest", 1.00), ("tallest", 1.75), ("most_seen", 1.75)):
        res = BS.bootstrap(dets, cfg.model_copy(update={"person": rule}), reference_height_m=height)
        assert res.report["people"]["groups"] == 2
        s, _, _ = umeyama(res.cameras.centers, cams.centers, with_scale=True)
        assert abs(s - 1) < 0.1, (rule, s)
