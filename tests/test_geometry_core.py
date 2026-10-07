from __future__ import annotations

import numpy as np

from frolic3d.geometry import (
    Camera,
    box_iou,
    kabsch,
    level_transform,
    plane_from_points,
    ray_ray_closest,
    rays_intersection,
    rodrigues,
    rotation_angle_deg,
    triangulate,
    triangulate_batch,
    umeyama,
)
from frolic3d.synthetic import ring_cameras


def test_project_undistort_roundtrip():
    cams = ring_cameras(2, dist=[-0.25, 0.1, 0.001, -0.002, 0.0])
    cam = cams[0]
    X = np.array([[1.2, -0.9, 0.9], [1.5, 1.3, 1.1], [-1.4, 1.2, 0.3]])
    uv, z = cam.project(X)
    assert (z > 0).all()
    uvd, _ = cam.project(X, distort=True)
    assert np.abs(uvd - uv).max() > 1.0
    np.testing.assert_allclose(cam.undistort_points(uvd), uv, atol=1e-3)
    rays = cam.rays(uv)
    for r, x in zip(rays, X):
        d = x - cam.center
        np.testing.assert_allclose(r, d / np.linalg.norm(d), atol=1e-9)


def test_triangulation_recovers_points():
    cams = ring_cameras(4)
    X = np.array([[0.2, -0.1, 1.0], [-0.4, 0.5, 0.3], [0.0, 0.0, 1.6]])
    uvs = [cam.project(X)[0] for cam in cams]
    for n in range(len(X)):
        Y = triangulate(list(cams), [u[n] for u in uvs])
        np.testing.assert_allclose(Y, X[n], atol=1e-6)
    P = np.stack([c.P for c in cams])
    uv = np.stack(uvs)
    uv[2, 1] = np.nan
    uv[3, 2] = np.nan
    uv[0, 2] = np.nan
    Y = triangulate_batch(P, uv)
    np.testing.assert_allclose(Y[:2], X[:2], atol=1e-6)
    np.testing.assert_allclose(Y[2], X[2], atol=1e-6)
    uv[1, 2] = np.nan
    assert np.isnan(triangulate_batch(P, uv)[2]).all()


def test_rays_and_closest_points():
    cams = ring_cameras(3)
    X = np.array([0.3, 0.2, 1.1])
    o = np.stack([c.center for c in cams])
    d = np.stack([c.rays(c.project(X)[0][None])[0] for c in cams])
    np.testing.assert_allclose(rays_intersection(o, d), X, atol=1e-6)
    t1, _, dist = ray_ray_closest(o[0], d[0], o[1], d[1])
    np.testing.assert_allclose(o[0] + t1 * d[0], X, atol=1e-6)
    assert dist < 1e-6


def test_kabsch_umeyama_rotation_angle():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(20, 3))
    R = rodrigues([0.3, -0.4, 0.5])
    Q = P @ R.T + 2.0
    np.testing.assert_allclose(kabsch(P, Q), R, atol=1e-9)
    s, Rs, t = umeyama(P, 1.7 * P @ R.T + np.array([1, 2, 3]))
    assert abs(s - 1.7) < 1e-9
    np.testing.assert_allclose(Rs, R, atol=1e-9)
    np.testing.assert_allclose(t, [1, 2, 3], atol=1e-8)
    assert abs(rotation_angle_deg(R, np.eye(3)) - np.degrees(np.linalg.norm([0.3, -0.4, 0.5]))) < 1e-9


def test_plane_and_level():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-2, 2, size=(200, 2))
    c_true = np.array([0.05, -0.02, 0.1])
    z = xy @ c_true[:2] + c_true[2] + rng.normal(0, 0.003, 200)
    z[:10] += 0.5
    F = np.c_[xy, z]
    c, keep = plane_from_points(F)
    np.testing.assert_allclose(c, c_true, atol=5e-3)
    assert keep.sum() >= 180 and not keep[:10].any()
    R, t, tilt = level_transform(c, np.zeros(2), 0.0)
    Fn = F[keep] @ R.T + t
    assert abs(np.median(Fn[:, 2])) < 5e-3
    assert 2.5 < tilt < 3.6


def test_box_iou_and_camera_frame_transform():
    assert abs(box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-9
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    cam = Camera("c", 10, 10, np.eye(3))
    moved = cam.transformed(np.eye(3), np.array([0, 0, 1.0]), 2.0)
    np.testing.assert_allclose(moved.center, 2.0 * cam.center + [0, 0, 1.0])
