import numpy as np
import pytest

from frolic3d import geometry as g
from frolic3d.synthetic import ring_cameras


def test_project_undistort_roundtrip():
    cams = ring_cameras(2, dist=[-0.2, 0.05, 0.001, -0.001, 0.0])
    cam = cams[0]
    X = cam.center + 3.0 * cam.axis + np.array([[1.2, 0.6, -0.8], [-1.4, 0.9, 0.7]])
    uv, z = cam.project(X)
    assert (z > 0).all()
    uvd = cam.distort_points(uv)
    assert np.abs(uvd - uv).max() > 0.5
    back = cam.undistort_points(uvd)
    assert np.abs(back - uv).max() < 1e-3
    uv2, _ = cam.project(X, distort=True)
    assert np.allclose(uv2, uvd, atol=1e-6)


def test_rays_through_pixels():
    cam = ring_cameras(1)[0]
    X = cam.center + 2.5 * cam.axis + np.array([0.2, -0.1, 0.3])
    uv, _ = cam.project(X[None])
    d = cam.rays(uv)[0]
    dist, t = g.ray_point_distance(cam.center, d, X)
    assert dist < 1e-6 and t > 0


def test_triangulate_and_batch():
    cams = ring_cameras(3)
    X = np.array([[0.1, 0.2, 1.0], [-0.3, 0.1, 0.5]])
    uv = [cam.project(X)[0] for cam in cams]
    for n in range(2):
        Xr = g.triangulate(list(cams), [u[n] for u in uv])
        assert np.allclose(Xr, X[n], atol=1e-6)
    P = np.stack([c.P for c in cams])
    pts = np.stack(uv)
    pts[0, 1] = np.nan
    Xb = g.triangulate_batch(P, pts)
    assert np.allclose(Xb, X, atol=1e-5)
    pts[1, 1] = np.nan
    assert np.isnan(g.triangulate_batch(P, pts)[1]).all()


def test_ray_ray_closest_and_intersection():
    c1, u1 = np.zeros(3), np.array([1.0, 0, 0])
    c2, u2 = np.array([0, 1.0, 0]), np.array([0, -1.0, 0.0])
    t1, t2, d = g.ray_ray_closest(c1, u1, c2, u2)
    assert abs(t1) < 1e-9 and abs(t2 - 1) < 1e-9 and d < 1e-9
    X = g.rays_intersection(np.stack([c1, c2]), np.stack([u1, u2]))
    assert np.allclose(X, 0, atol=1e-9)
    assert g.ray_ray_closest(c1, u1, c2, u1)[2] == np.inf


def test_kabsch_umeyama():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(20, 3))
    R = g.rodrigues(np.array([0.2, -0.4, 0.1]))
    Q = 1.7 * P @ R.T + np.array([1, 2, 3])
    s, Rr, t = g.umeyama(P, Q)
    assert abs(s - 1.7) < 1e-9 and np.allclose(Rr, R) and np.allclose(t, [1, 2, 3])
    assert np.allclose(g.kabsch(P, P @ R.T), R)
    assert g.rotation_angle_deg(R, R) < 1e-6
    assert np.allclose(g.rotvec(R), [0.2, -0.4, 0.1])


def test_plane_and_level():
    rng = np.random.default_rng(2)
    xy = rng.uniform(-2, 2, (200, 2))
    c_true = np.array([0.05, -0.02, 0.3])
    z = xy @ c_true[:2] + c_true[2] + rng.normal(0, 0.002, 200)
    z[:10] += 0.5  # outliers
    c, keep = g.plane_from_points(np.c_[xy, z])
    assert np.allclose(c, c_true, atol=5e-3) and keep.sum() >= 185
    R, t, tilt = g.level_transform(c, np.zeros(2), 0.07)
    res = np.c_[xy, z][keep] @ R.T + t
    assert abs(np.median(res[:, 2]) - 0.07) < 0.01 and 2.5 < tilt < 3.5


def test_camera_transformed_keeps_projection():
    cam = ring_cameras(1)[0]
    X = cam.center + 3 * cam.axis
    R = g.rodrigues([0.1, 0.2, 0.3])
    t = np.array([0.5, -1, 2])
    c2 = cam.transformed(R, t, 2.0)
    uv1, _ = cam.project(X)
    uv2, _ = c2.project(2.0 * R @ X + t)
    assert np.allclose(uv1, uv2, atol=1e-6)


def test_box_iou():
    assert g.box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)
    assert g.box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
