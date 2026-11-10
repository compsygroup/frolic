import numpy as np

from frolic.calib.floor import apply_rigid, floor_align, level_floor, standing_feet
from frolic.geometry import rodrigues
from frolic.io.tracks import Tracks
from frolic.skeleton import J
from frolic.synthetic import person_track, ring_cameras


def _tracks(people, fps=15.0):
    P, N = len(people), people[0].shape[0]
    t = Tracks.empty([f"p{i}" for i in range(P)], N, ["cam1", "cam2"], fps, 1)
    t.xyz[:] = np.stack(people)
    t.score[:] = 1.0
    return t


def _standing_scene(reach: float = 1.0):
    """Two people with straight legs gliding slowly over the floor (0.27 m/s x reach)."""
    a = person_track(60, 15.0, 1.72, (-0.5, -0.4), (-0.5 + 0.9 * reach, -0.4 + 0.7 * reach), heading=0.3, gait_hz=0.0)
    b = person_track(60, 15.0, 1.68, (1.0, 0.5), (1.0 - 0.8 * reach, 0.5 + 0.7 * reach), heading=2.0, gait_hz=0.0)
    return [a, b]


def test_standing_feet_detected():
    tr = _tracks(_standing_scene())
    F = standing_feet(tr)
    assert len(F) >= 100
    assert np.all(F[:, 2] < 0.12)


def test_level_floor_recovers_tilt():
    people = _standing_scene()
    cams = ring_cameras(3)
    R = rodrigues([np.radians(3.0), 0, 0])
    t = np.array([0.2, -0.1, 0.5])
    tilted = apply_rigid(_tracks(people), R, t)
    tcams = cams.transformed(R, t)
    X = tilted.xyz[0, 10, J["head"]]
    uv_before, _ = tcams[0].project(X)
    lev, lcams, info = level_floor(tilted, tcams, ankle_height_m=0.07, min_spread_m=0.0)  # 1 m synthetic scene
    assert info["source"] == "standing" and abs(info["tilt_deg"] - 3.0) < 0.3
    ank = lev.xyz[:, :, [J["ankle_l"], J["ankle_r"]], 2]
    assert abs(np.median(ank) - 0.07) < 0.03
    assert lcams.floor["levelled"]
    uv_after, _ = lcams[0].project(lev.xyz[0, 10, J["head"]])
    assert np.allclose(uv_before, uv_after, atol=1e-3)


def test_level_floor_stance_fallback():
    people = _standing_scene(reach=0.4)
    for X in (
        people
    ):  # one knee bent with the foot raised on every frame: no standing frames, the other ankle stays low and slow
        X[:30, J["ankle_l"]] += [0.3, 0.0, 0.3]
        X[30:, J["ankle_r"]] += [0.3, 0.0, 0.3]
    R = rodrigues([0, np.radians(2.0), 0])
    tr = apply_rigid(_tracks(people), R, np.zeros(3))
    _lev, _, info = level_floor(tr, ring_cameras(2), 0.07, min_spread_m=0.0)
    assert info["source"] == "stance_ankles"
    assert abs(info["tilt_deg"] - 2.0) < 0.3


def test_level_floor_too_few_points_identity():
    walker = person_track(8, 15.0, 1.70, (-1.5, 0.0), (1.5, 0.0))
    tr = _tracks([walker])
    _lev, _cams, info = level_floor(tr, ring_cameras(2), 0.07)
    assert info["source"] == "none" and np.allclose(info["R"], np.eye(3))


def test_floor_align_shift():
    people = _standing_scene()
    tr = apply_rigid(_tracks(people), np.eye(3), np.array([0, 0, 0.3]))
    cams = ring_cameras(2)
    out, ocams, info = floor_align(tr, cams, target_ankle_z=0.05)
    feet = out.xyz[:, :, [J["ankle_l"], J["ankle_r"], J["toe_l"], J["toe_r"]], 2]
    assert abs(np.percentile(feet[np.isfinite(feet)], 10) - 0.05) < 1e-4
    assert abs(info["z_shift_m"] + 0.3 - (0.05 - np.percentile(np.stack(people)[:, :, [5, 6, 7, 8], 2], 10))) < 1e-4
    X = tr.xyz[0, 0, 0]
    uv1, _ = cams[0].project(X)
    uv2, _ = ocams[0].project(X + [0, 0, info["z_shift_m"]])
    assert np.allclose(uv1, uv2, atol=1e-6)


def test_level_floor_collinear_feet_keeps_orientation():
    """Standing feet along one line (a walkway) do not constrain the tilt across the line: no rotation."""
    a = person_track(60, 15.0, 1.72, (-0.5, 0.0), (0.5, 0.0), heading=0.0, gait_hz=0.0)
    b = person_track(60, 15.0, 1.68, (0.5, 0.0), (-0.5, 0.0), heading=np.pi, gait_hz=0.0)
    R = rodrigues([np.radians(3.0), 0, 0])
    tr = apply_rigid(_tracks([a, b]), R, np.zeros(3))
    _lev, cams, info = level_floor(tr, ring_cameras(2), 0.07, min_spread_m=0.25)
    assert info["source"] == "none" and np.allclose(info["R"], np.eye(3)) and not cams.floor["levelled"]
    _lev, _cams, info = level_floor(tr, ring_cameras(2), 0.07, min_spread_m=0.0)  # ungated: the tilt is wrong
    assert info["source"] == "standing" and abs(info["tilt_deg"] - 3.0) > 1.0


def test_level_floor_given_levelled_is_shift_only():
    people = _standing_scene()
    R = rodrigues([np.radians(3.0), 0, 0])
    tr = apply_rigid(_tracks(people), R, np.array([0.0, 0.0, 0.4]))
    cams = ring_cameras(2)
    cams.floor.update({"levelled": True, "tilt_deg": 2.3})  # written by level_floor on another clip of the set
    lev, lcams, info = level_floor(tr, cams, 0.07)
    assert info["source"] == "given" and np.allclose(info["R"], np.eye(3)) and lcams.floor["levelled"]
    assert abs(info["z_shift_m"]) > 0.1


def test_given_levelled_puts_standing_ankles_at_ankle_height():
    """Given levelled calibration: the z shift brings standing ankles to ankle_height_m, not the toe-biased p10 rule."""
    people = _standing_scene()
    tr = apply_rigid(_tracks(people), np.eye(3), np.array([0.0, 0.0, 0.09]))  # scene 9 cm too high
    cams = ring_cameras(2)
    cams.floor.update({"levelled": True, "tilt_deg": 2.3})
    lev, _c, info = level_floor(tr, cams, 0.07)
    F = standing_feet(lev)
    assert info["height_source"] == "standing"
    assert abs(np.median(F[:, 2]) - 0.07) < 0.01
