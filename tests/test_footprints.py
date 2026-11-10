import numpy as np

from frolic3d.render.footprints import footprint_mesh, footsteps
from frolic3d.skeleton import J, JOINT_NAMES


def _walk(n=90, fps=15.0, step_s=0.5, stride=0.3):
    """Feet alternating stance (0.5 s on the floor) and swing (lifted 8 cm), walking along +x; seated-high feet for p2."""
    X = np.full((n, len(JOINT_NAMES), 3), np.nan, np.float32)
    t = np.arange(n) / fps
    per = int(step_s * fps)
    for k, side in enumerate(("l", "r")):
        y = 0.08 if side == "l" else -0.08
        for i in range(n):
            cycle, ph = divmod(i + k * per, 2 * per)
            if ph < per:  # stance: foot fixed
                x = cycle * 2 * stride + k * stride
                z = 0.0
            else:  # swing: foot moves forward, lifted
                x = cycle * 2 * stride + k * stride + stride * 2 * (ph - per) / per
                z = 0.08
            X[i, J[f"ankle_{side}"]] = (x, y, z + 0.07)
            X[i, J[f"toe_{side}"]] = (x + 0.12, y, z + 0.01)
    X[:, J["pelvis"]] = np.c_[t * stride * 2 / (2 * step_s), np.zeros(n), np.full(n, 0.6)]
    return X


def test_footsteps_one_per_stance_with_heading():
    fs = footsteps(_walk(), 15.0)
    assert 11 <= len(fs) <= 14, len(fs)  # 6 s of 0.5 s stances per foot, alternating
    assert all(abs(f.heading[0] - 1.0) < 1e-6 for f in fs)  # toes point +x
    assert all(0.1 <= f.length <= 0.32 for f in fs)
    assert sorted(f.frame for f in fs) == [f.frame for f in fs]
    sides = {f.side for f in fs}
    assert sides == {"l", "r"}


def test_footsteps_none_when_feet_raised():
    X = _walk()
    X[:, J["ankle_l"], 2] += 0.3
    X[:, J["toe_l"], 2] += 0.3
    X[:, J["ankle_r"], 2] += 0.3
    X[:, J["toe_r"], 2] += 0.3
    assert footsteps(X, 15.0) == []


def test_footprint_mesh_fades_and_expires():
    fs = footsteps(_walk(), 15.0)
    fade = 5.0 * 15.0
    first = fs[0].frame
    assert footprint_mesh(fs, first - 1, fade, (200, 80, 80)) is None  # nothing has landed yet
    m = footprint_mesh(fs, first + 1, fade, (200, 80, 80))
    assert m is not None
    assert footprint_mesh(fs[:1], first + fade + 1, fade, (200, 80, 80)) is None
