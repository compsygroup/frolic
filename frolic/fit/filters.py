"""Temporal filters and gates used by the fitter (NumPy)."""

from __future__ import annotations

import itertools

import numpy as np


def one_euro(x: np.ndarray, fps: float, fmin: float, beta: float, dcut: float = 1.0) -> np.ndarray:
    """One-Euro filter run forward and backward and averaged (no lag); x (T,) finite."""

    def one(x: np.ndarray) -> np.ndarray:
        dt = 1.0 / fps
        y = x.copy()
        dx = 0.0

        def al(fc: float) -> float:
            return 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * dt))

        for k in range(1, len(x)):
            d = (x[k] - y[k - 1]) / dt
            dx = dx + al(dcut) * (d - dx)
            y[k] = y[k - 1] + al(fmin + beta * abs(dx)) * (x[k] - y[k - 1])
        return y

    return 0.5 * (one(x) + one(x[::-1])[::-1])


def smooth_series(X: np.ndarray, fps: float, fmin: float, beta: float, cuts: list[int] | None = None) -> np.ndarray:
    """One-Euro along axis 0 of X (N, ...) applied to every component, within each cut segment; NaN rows are skipped."""
    X = np.asarray(X, np.float32).copy()
    if fmin <= 0 or len(X) < 4:
        return X
    bounds = [0] + sorted(c for c in (cuts or []) if 0 < c < len(X)) + [len(X)]
    flat = X.reshape(len(X), -1)
    for a, b in itertools.pairwise(bounds):
        seg = flat[a:b]
        fin = np.isfinite(seg).all(1)
        if fin.sum() < 4:
            continue
        for k in range(seg.shape[1]):
            seg[fin, k] = one_euro(seg[fin, k], fps, fmin, beta)
    return flat.reshape(X.shape)


def soft_fade(w: np.ndarray, n: int) -> np.ndarray:
    """Fade a [0, 1] weight (T, ...) over +-n frames: w times its running mean, so switches ramp instead of flipping."""
    w = np.asarray(w, np.float32)
    if n <= 0:
        return w
    k = 2 * n + 1
    pad = np.concatenate([np.repeat(w[:1], n, 0), w, np.repeat(w[-1:], n, 0)], 0)
    cs = np.cumsum(np.concatenate([np.zeros_like(w[:1]), pad], 0), 0)
    return (w * (cs[k:] - cs[:-k]) / k).astype(np.float32)


def jumpy(K: np.ndarray, thr: float, half: int = 3, cuts: list[int] | None = None) -> np.ndarray:
    """(T, J, 3) keypoints -> (T, J) True where a point is farther than thr from its own +-half-frame median."""
    out = np.zeros(K.shape[:2], bool)
    bounds = [0] + sorted(c for c in (cuts or []) if 0 < c < len(K)) + [len(K)]
    for a, b in itertools.pairwise(bounds):
        seg = K[a:b]
        if len(seg) == 0:
            continue
        pad = np.pad(seg, ((half, half), (0, 0), (0, 0)), mode="edge")
        med = np.median(np.stack([pad[k : k + len(seg)] for k in range(2 * half + 1)]), 0)
        out[a:b] = np.linalg.norm(seg - med, axis=-1) > thr
    return out


def bridge_single_view(X: np.ndarray, single_view: np.ndarray, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill holes of up to max_frames inside or next to single-view runs. Returns (filled X, bridged mask)."""
    X = np.asarray(X, np.float32).copy()
    sv = np.asarray(single_view, bool)
    bridged = np.zeros(len(X), bool)
    if max_frames <= 0 or not sv.any():
        return X, bridged
    have = np.isfinite(X[:, 0, 0])
    idx = np.flatnonzero(have)
    for l_, r_ in itertools.pairwise(idx):
        if 1 < r_ - l_ <= max_frames + 1 and (sv[l_] or sv[r_]):
            w = ((np.arange(l_ + 1, r_) - l_) / (r_ - l_))[:, None, None]
            X[l_ + 1 : r_] = (1 - w) * X[l_] + w * X[r_]
            bridged[l_ + 1 : r_] = True
    return X, bridged


def speed_weights(X: np.ndarray, fps: float, v0: float, floor: float) -> np.ndarray:
    """Per-frame smoothness weight (N - 2,) in [floor, 1] from the track's joint speed (90th percentile over joints)."""
    N = len(X)
    vel = np.full(N, np.nan)
    if N > 2:
        with np.errstate(all="ignore"):
            d = np.linalg.norm(X[2:] - X[:-2], axis=-1) * fps / 2
            vel[1:-1] = np.array([np.nanpercentile(r, 90) if np.isfinite(r).any() else np.nan for r in d])
    fill = np.nanmedian(vel) if np.isfinite(vel).any() else 0.0
    vel = np.nan_to_num(np.convolve(np.nan_to_num(vel, nan=fill), np.ones(5) / 5, mode="same"))
    w = floor + (1 - floor) / (1.0 + (vel / v0) ** 2)
    return w[1:-1].astype(np.float32)


def fill_interp(arr: np.ndarray) -> np.ndarray:
    """Fill NaN rows of (N,) or (N, D) by linear interpolation over frames (edge values extended)."""
    src = np.asarray(arr, np.float32)
    out = src.reshape(len(src), -1).copy()
    g = np.isfinite(out).all(1)
    if not g.any():
        return np.zeros_like(src)
    for d in range(out.shape[1]):
        out[:, d] = np.interp(np.arange(len(out)), np.flatnonzero(g), out[g, d])
    return out.reshape(src.shape)
