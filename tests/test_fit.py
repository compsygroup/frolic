import itertools

import numpy as np
import pytest
import roma
import torch

from frolic3d.config import FitConfig, PersonConfig
from frolic3d.fit import filters, shape
from frolic3d.fit.anny_model import make_body
from frolic3d.fit.biomech import ADULT, BiomechConstraints, blended_ranges, ranges_table
from frolic3d.fit.fitter import FitInputs, PersonFitter
from frolic3d.fit.root import RootParam, continuous_rotvecs, geodesic_acceleration, spin_events
from frolic3d.skeleton import J
from frolic3d.synthetic import person_track

FPS = 15.0


def heading_rotvecs(deg: np.ndarray) -> np.ndarray:
    return np.stack([[0.0, 0.0, np.radians(d)] for d in deg]).astype(np.float32)


def wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180) % 360 - 180


# ---------------------------------------------------------------- root rotation continuity


def test_continuous_rotvecs_across_heading_crossing():
    deg = wrap_deg(np.linspace(150, 210, 31))  # passes through +-180
    rr = heading_rotvecs(deg)
    assert np.abs(np.diff(rr[:, 2])).max() > np.pi  # the naive rotation vectors jump
    rr[10:13] = np.nan
    rc = continuous_rotvecs(rr)
    R = roma.rotvec_to_rotmat(torch.tensor(rc))
    step = roma.rotmat_to_rotvec(R[:-1].transpose(-1, -2) @ R[1:]).norm(dim=-1).numpy()
    assert np.degrees(step).max() < 2.5
    truth = roma.rotvec_to_rotmat(torch.tensor(heading_rotvecs(deg)))
    assert roma.rotmat_geodesic_distance(R, truth).max() < 1e-3


def test_root_param_delta_starts_at_init_and_geodesic_acc():
    deg = wrap_deg(np.linspace(160, 200, 21))
    rr = heading_rotvecs(deg)
    root = RootParam(rr, torch.device("cpu"))
    assert torch.allclose(root.delta, torch.zeros_like(root.delta))
    assert float(geodesic_acceleration(root.matrices())) < 1e-8  # constant turn rate across the crossing
    acc = geodesic_acceleration(root.matrices())
    acc.backward()
    assert root.delta.grad is not None and torch.isfinite(root.delta.grad).all()
    assert float(geodesic_acceleration(root.matrices()[:2])) == 0.0


def test_root_fit_does_not_spin_through_crossing():
    deg = wrap_deg(np.linspace(150, 210, 31))
    target = roma.rotvec_to_rotmat(torch.tensor(heading_rotvecs(deg)))
    root = RootParam(heading_rotvecs(deg + np.random.default_rng(0).normal(0, 3, len(deg))), torch.device("cpu"))
    opt = torch.optim.Adam(root.params, lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = roma.rotmat_geodesic_distance(root.matrices(), target).pow(2).mean() + geodesic_acceleration(
            root.matrices()
        )
        loss.backward()
        opt.step()
    assert spin_events(root.rotvecs().detach().numpy(), FPS) == 0
    assert float(roma.rotmat_geodesic_distance(root.matrices(), target).max()) < np.radians(5)


def test_spin_events_detects_back_and_forth_spin():
    steady = heading_rotvecs(np.linspace(0, 90, 45))
    assert spin_events(steady, FPS) == 0
    spin = wrap_deg(np.r_[np.linspace(0, 170, 8), np.linspace(170, 0, 8), np.zeros(29)])
    assert spin_events(heading_rotvecs(spin), FPS) > 0


# ---------------------------------------------------------------- biomechanics


@pytest.fixture(scope="module")
def body():
    return make_body("mock", torch.device("cpu"), dense_n=100)


@pytest.fixture(scope="module")
def bm(body):
    ph = torch.full((6,), 0.5)
    H, _, V = body.rest(ph, torch.zeros(len(body.segment_labels)))
    return BiomechConstraints(body.bone_labels, body.posed_bones, H, V, body.dominant_bone, age01=0.8, fps=FPS)


def test_ranges_and_child_blend():
    adult, w_adult = blended_ranges(0.9, 1 / 3, 0.45)
    _child, w_child = blended_ranges(0.1, 1 / 3, 0.45)
    assert w_adult == 0.0 and w_child == 1.0
    assert adult["knee"]["flex"] == ADULT["knee"]["flex"]
    rows = ranges_table()
    assert rows and all(r["source"] for r in rows)


def test_limits_zero_at_rest_and_penalise_hyperextension(body, bm):
    rv = torch.zeros(4, len(body.posed_bones), 3)
    assert float(bm.limit_loss(rv)) < 1e-6
    k = body.posed_bones.index("lowerleg01.L")
    bad = rv.clone()
    bad[:, k, 0] = -1.0  # ~57 deg knee hyperextension
    bad.requires_grad_(True)
    loss = bm.limit_loss(bad)
    assert float(loss) > 0
    loss.backward()
    assert torch.isfinite(bad.grad).all() and float(bad.grad[:, k, 0].mean()) < 0
    with torch.no_grad():
        stepped = bad - 0.5 * bad.grad / bad.grad.norm()
    assert float(bm.limit_loss(stepped)) < float(loss)
    H = body.skeleton(
        torch.full((6,), 0.5),
        torch.eye(3).expand(4, 3, 3),
        torch.zeros(4, 3),
        bad.detach(),
        torch.zeros(len(body.segment_labels)),
    ).heads
    assert bm.report(bad.detach(), H)["lim_knee"] == 4


def test_collision_penalises_crossed_legs(body, bm):
    seg = torch.zeros(len(body.segment_labels))
    ph = torch.full((6,), 0.5)
    rv = torch.zeros(1, len(body.posed_bones), 3)
    eye = torch.eye(3).expand(1, 3, 3)
    H0 = body.skeleton(ph, eye, torch.zeros(1, 3), rv, seg).heads
    p0 = float(bm.penetration(H0).max())
    k = body.posed_bones.index("upperleg01.L")
    crossed = rv.clone()
    crossed[0, k, 2] = 0.6  # adduct the left thigh across the right one
    crossed.requires_grad_(True)
    H1 = body.skeleton(ph, eye, torch.zeros(1, 3), crossed, seg).heads
    loss = bm.collision_loss(H1)
    assert float(bm.penetration(H1.detach()).max()) > p0 + 0.01
    loss.backward()
    assert torch.isfinite(crossed.grad).all() and float(crossed.grad.abs().sum()) > 0


def test_forward_is_finite_with_gradients(body, bm):
    N = 6
    rv = (0.1 * torch.randn(N, len(body.posed_bones), 3, generator=torch.Generator().manual_seed(0))).requires_grad_(
        True
    )
    o = body.skeleton(
        torch.full((6,), 0.5),
        torch.eye(3).expand(N, 3, 3),
        torch.zeros(N, 3),
        rv,
        torch.zeros(len(body.segment_labels)),
    )
    loss = bm(rv, o.heads)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(rv.grad).all()


# ---------------------------------------------------------------- shape and filters


def test_age_slider_and_fixed_phenotype():
    a = [shape.age_years_to_slider(y) for y in (1, 5, 12, 30)]
    assert all(x < y for x, y in itertools.pairwise(a)) and a[0] >= 0 and a[-1] <= 1
    labels = ["gender", "age", "muscle", "weight", "height", "proportions"]
    fp = shape.FixedPhenotype.from_person(PersonConfig(sex="female", age_years=4.0, height_m=1.02))
    assert fp.mask(labels, "cpu").tolist() == [0, 0, 1, 1, 1, 1] and fp.height_m == 1.02
    pl = torch.zeros(6)
    fp.apply(pl, labels)
    assert torch.sigmoid(pl)[0] > 0.99
    assert shape.FixedPhenotype.from_person(None).values == {}


def test_segment_length_targets_from_track():
    X = person_track(60, FPS, 1.7, (-1, 0), (1, 0))
    t = shape.segment_length_targets(X)
    assert t.ok.all() and len(t.target) == len(t.groups)
    thigh = np.nanmedian(np.linalg.norm(X[:, J["hip_l"]] - X[:, J["knee_l"]], axis=-1))
    assert t.target[t.groups.index("thigh")] == pytest.approx(thigh, rel=0.05)
    assert not shape.segment_length_targets(X[:10]).ok.any()


def test_filters():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 60)
    clean = np.sin(t)[:, None]
    noisy = clean + rng.normal(0, 0.05, clean.shape)
    sm = filters.one_euro(noisy, FPS, 1.0, 0.1)
    assert np.abs(sm - clean).mean() < np.abs(noisy - clean).mean()
    X = np.tile(np.arange(10, dtype=float)[:, None, None], (1, 2, 3))
    sv = np.zeros(10, bool)
    sv[2:4] = True
    X[4:6] = np.nan
    Xb, brg = filters.bridge_single_view(X, sv, 3)
    assert brg[4:6].all() and np.allclose(Xb[4:6, 0, 0], [4, 5])
    a = np.array([1.0, np.nan, 3.0])
    assert np.allclose(filters.fill_interp(a), [1, 2, 3])


# ---------------------------------------------------------------- the fitter on a synthetic body


@pytest.fixture(scope="module")
def turning_walker():
    """A child walking a semicircle so the heading sweeps through +-180 degrees."""
    n = 60
    X = person_track(n, FPS, 1.05, (0.0, -1.0), (0.0, 1.0))  # faces +y
    for i, th in enumerate(np.linspace(0, np.pi, n)):  # facing sweeps 90 -> 270 deg
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        p = X[i] - X[i, J["pelvis"]] * [1, 1, 0]
        X[i] = p @ R.T + [np.cos(th), np.sin(th), 0]
    return X.astype(np.float32)


def test_fit_recovers_body_and_keeps_root_continuous(body, turning_walker):
    X = turning_walker.copy()
    sv = np.zeros(len(X), bool)
    sv[20:24] = True
    X[24:26] = np.nan  # a short hole inside the single-view run
    sv[24:26] = True
    cfg = FitConfig(backend="mock", device="cpu", iters=200)
    res = PersonFitter(body, cfg, FPS).fit(FitInputs(X, single_view=sv, observed=np.ones(len(X), bool)), "kid")
    assert res is not None
    assert res.stature == pytest.approx(1.05, rel=0.1)
    err = np.linalg.norm(res.joints - turning_walker, axis=-1)
    assert np.nanmedian(err) < 0.04
    assert res.bridged[24:26].all() and res.single_view[20:26].all()
    assert spin_events(res.rootrot, FPS) == 0
    R = roma.rotvec_to_rotmat(torch.tensor(res.rootrot)).numpy()
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    assert np.ptp(yaw) == pytest.approx(np.pi, abs=np.radians(20))
    assert isinstance(res.biomech_report, dict)


def test_fit_skips_short_tracks(body):
    X = person_track(10, FPS, 1.7, (0, 0), (1, 0))
    assert PersonFitter(body, FitConfig(backend="mock", device="cpu", iters=5), FPS).fit(FitInputs(X)) is None


def test_fixed_height_is_honoured(body):
    X = person_track(40, FPS, 1.30, (-1, 0), (1, 0))
    cfg = FitConfig(backend="mock", device="cpu", iters=60)
    res = PersonFitter(body, cfg, FPS).fit(FitInputs(X, person=PersonConfig(height_m=1.30)), "p")
    assert res.stature == pytest.approx(1.30, abs=0.02)


@pytest.mark.slow
def test_real_anny_body_fits_synthetic_walker(tmp_path):
    pytest.importorskip("anny")
    body = make_body("anny", torch.device("cpu"), cache_dir=tmp_path, dense_n=100)
    X = person_track(30, FPS, 1.10, (-1, 0), (1, 0))
    res = PersonFitter(body, FitConfig(backend="anny", device="cpu", iters=60), FPS).fit(FitInputs(X), "p")
    assert res is not None and np.isfinite(res.joints).all()
    assert res.stature == pytest.approx(1.10, rel=0.15)
    assert np.nanmedian(np.linalg.norm(res.joints - X, axis=-1)) < 0.08
