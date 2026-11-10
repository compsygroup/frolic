"""Pinhole cameras, triangulation, rays and rigid/similarity alignment (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Array = np.ndarray


@dataclass
class Camera:
    """World -> camera: x_c = R X + t.

    K is the pinhole matrix of the undistorted image; dist is OpenCV (k1 k2 p1 p2 k3).
    """

    name: str
    width: int
    height: int
    K: Array
    dist: Array = field(default_factory=lambda: np.zeros(5))
    R: Array = field(default_factory=lambda: np.eye(3))
    t: Array = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, float).reshape(3, 3)
        self.dist = np.asarray(self.dist, float).ravel()
        if self.dist.size < 5:
            self.dist = np.pad(self.dist, (0, 5 - self.dist.size))
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.t = np.asarray(self.t, float).ravel()

    @property
    def center(self) -> Array:
        return -self.R.T @ self.t

    @property
    def P(self) -> Array:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def focal(self) -> float:
        return 0.5 * float(self.K[0, 0] + self.K[1, 1])

    @property
    def axis(self) -> Array:
        return self.R[2]

    def undistort_points(self, uv: Array) -> Array:
        """Distorted pixels -> undistorted pixels in the same K."""
        p = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if p.size == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(p, self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort_points(self, uv: Array) -> Array:
        """Undistorted pixels -> distorted pixels (where the lens puts them)."""
        p = np.asarray(uv, np.float64).reshape(-1, 2)
        if p.size == 0:
            return np.zeros((0, 2))
        n = np.c_[(p - self.K[:2, 2]) / np.diag(self.K)[:2], np.ones(len(p))]
        d, _ = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return d.reshape(-1, 2)

    def project(self, X: Array, distort: bool = False) -> tuple[Array, Array]:
        """World points (..., 3) -> pixels (..., 2) and depth (...)."""
        X = np.asarray(X, float)
        Xc = X @ self.R.T + self.t
        z = Xc[..., 2]
        uv = Xc[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None] * np.diag(self.K)[:2] + self.K[:2, 2]
        if distort:
            uv = self.distort_points(uv.reshape(-1, 2)).reshape(uv.shape)
        return uv, z

    def rays(self, uv: Array) -> Array:
        """Undistorted pixels (N, 2) -> unit direction vectors in world coordinates (N, 3)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(self.K).T @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def in_image(self, uv: Array, margin: float = 0.0) -> Array:
        uv = np.asarray(uv, float)
        return (
            (uv[..., 0] >= margin)
            & (uv[..., 0] <= self.width - margin)
            & (uv[..., 1] >= margin)
            & (uv[..., 1] <= self.height - margin)
        )

    def transformed(self, R: Array, t: Array, scale: float = 1.0) -> Camera:
        """Camera in a new world frame X' = scale * R X + t."""
        Rn = self.R @ R.T
        tn = scale * self.t - Rn @ t
        return Camera(self.name, self.width, self.height, self.K.copy(), self.dist.copy(), Rn, tn)


def triangulate(cameras: list[Camera], uv: list[Array], weights: list[float] | None = None) -> Array:
    """Linear DLT from undistorted pixels of >= 2 cameras -> (3,) world point (NaN if degenerate)."""
    rows = []
    for i, (cam, x) in enumerate(zip(cameras, uv)):
        w = 1.0 if weights is None else float(weights[i])
        P = cam.P
        rows += [w * (x[0] * P[2] - P[0]), w * (x[1] * P[2] - P[1])]
    if len(rows) < 4:
        return np.full(3, np.nan)
    vt = np.linalg.svd(np.array(rows))[2]
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_batch(P: Array, uv: Array) -> Array:
    """P (V, 3, 4) projection matrices, uv (V, N, 2) with NaN for missing -> (N, 3) points (NaN where < 2 views)."""
    N = uv.shape[1]
    out = np.full((N, 3), np.nan)
    ok = np.isfinite(uv[..., 0])
    for n in range(N):
        vs = np.flatnonzero(ok[:, n])
        if len(vs) < 2:
            continue
        A = np.concatenate([np.stack([uv[v, n, 0] * P[v, 2] - P[v, 0], uv[v, n, 1] * P[v, 2] - P[v, 1]]) for v in vs])
        X = np.linalg.svd(A)[2][-1]
        if abs(X[3]) > 1e-12:
            out[n] = X[:3] / X[3]
    return out


def reprojection_error(cam: Camera, X: Array, uv: Array) -> float:
    p, z = cam.project(X)
    if z <= 0:
        return np.inf
    return float(np.linalg.norm(p - uv))


def ray_point_distance(origin: Array, direction: Array, X: Array) -> tuple[float, float]:
    """Distance from X to the ray and the parameter of the closest point."""
    v = X - origin
    t = float(v @ direction)
    return float(np.linalg.norm(v - t * direction)), t


def ray_ray_closest(c1: Array, u1: Array, c2: Array, u2: Array) -> tuple[float, float, float]:
    """Parameters (t1, t2) of closest approach of two rays and their distance."""
    w = c1 - c2
    a, b, c = u1 @ u1, u1 @ u2, u2 @ u2
    d, e = u1 @ w, u2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return np.nan, np.nan, np.inf
    t1 = (b * e - c * d) / den
    t2 = (a * e - b * d) / den
    return float(t1), float(t2), float(np.linalg.norm(c1 + t1 * u1 - c2 - t2 * u2))


def rays_intersection(origins: Array, directions: Array) -> Array:
    """Least-squares point closest to several rays (N, 3) each."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, u in zip(origins, directions):
        Pm = np.eye(3) - np.outer(u, u)
        A += Pm
        b += Pm @ c
    return np.linalg.solve(A, b)


def kabsch(P: Array, Q: Array) -> Array:
    """Rotation R with R @ P_i ~ Q_i after centring (P, Q: k x 3)."""
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def umeyama(X: Array, Y: Array, with_scale: bool = True) -> tuple[float, Array, Array]:
    """Similarity (s, R, t) with Y ~ s R X + t."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / Xc.var(0).sum()) if with_scale else 1.0
    return s, R, my - s * R @ mx


def rotation_angle_deg(R1: Array, R2: Array) -> float:
    c = (np.trace(R1 @ R2.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def rodrigues(rvec: Array) -> Array:
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]


def rotvec(R: Array) -> Array:
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def box_iou(a: Array, b: Array) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def plane_from_points(F: Array, n_iter: int = 5) -> tuple[Array, Array]:
    """Robust plane z = c0 x + c1 y + c2 (trimmed least squares) -> (coefficients, inlier mask)."""
    keep = np.ones(len(F), bool)
    c = np.zeros(3)
    for _ in range(n_iter):
        M = np.c_[F[keep, :2], np.ones(keep.sum())]
        c, *_ = np.linalg.lstsq(M, F[keep, 2], rcond=None)
        r = F[:, 2] - np.c_[F[:, :2], np.ones(len(F))] @ c
        mad = np.median(np.abs(r[keep])) * 1.4826
        keep = np.abs(r) < max(2.5 * mad, 1e-4)
    return c, keep


def level_transform(c: Array, pivot_xy: Array, target_z: float = 0.0) -> tuple[Array, Array, float]:
    """Rigid (R, t) rotating the plane z = c0 x + c1 y + c2 onto z = target_z about pivot_xy, plus the tilt (deg)."""
    n = np.array([-c[0], -c[1], 1.0])
    n /= np.linalg.norm(n)
    ctr = np.array([pivot_xy[0], pivot_xy[1], c[0] * pivot_xy[0] + c[1] * pivot_xy[1] + c[2]])
    v = np.cross(n, [0, 0, 1])
    s = np.linalg.norm(v)
    cth = n[2]
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + Vx + Vx @ Vx * ((1 - cth) / s**2) if s > 1e-9 else np.eye(3)
    t = -R @ ctr + np.array([ctr[0], ctr[1], target_z])
    return R, t, float(np.degrees(np.arccos(np.clip(cth, -1, 1))))
