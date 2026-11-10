"""Spatiotemporal refinement of per-person tracks: robust multi-view reprojection + smoothness + limb lengths."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from frolic.config import TrackConfig
from frolic.geometry import Camera
from frolic.io.calib import CameraSet
from frolic.io.detections import Detections
from frolic.io.tracks import Tracks
from frolic.log import get_logger
from frolic.skeleton import HALPE26, LIMBS, LIMBS_WEAK, N_JOINTS, J

log = get_logger(__name__)


def one_euro(x: np.ndarray, fps: float, fmin: float, beta: float, dcut: float = 1.0) -> np.ndarray:
    """One-Euro filter run forward and backward and averaged (no lag). x (T,) finite."""

    def one(x: np.ndarray) -> np.ndarray:
        dt = 1.0 / fps
        y = x.copy()
        dx = 0.0
        al = lambda fc: 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * dt))  # noqa: E731
        for k in range(1, len(x)):
            d = (x[k] - y[k - 1]) / dt
            dx = dx + al(dcut) * (d - dx)
            y[k] = y[k - 1] + al(fmin + beta * abs(dx)) * (x[k] - y[k - 1])
        return y

    return 0.5 * (one(x) + one(x[::-1])[::-1])


def smooth_runs(X: np.ndarray, fps: float, fmin: float, beta: float) -> np.ndarray:
    """One-Euro per joint/axis inside contiguous runs of finite frames."""
    out = X.copy()
    have = np.isfinite(X[:, 0, 0])
    idx = np.flatnonzero(have)
    if len(idx) == 0 or fmin <= 0:
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) < 3:
            continue
        for j in range(X.shape[1]):
            seg = X[run, j]
            if not np.isfinite(seg).all():
                continue
            for d in range(3):
                out[run, j, d] = one_euro(seg[:, d], fps, fmin, beta)
    return out


def blank_outliers(P: np.ndarray, fps: float, win_s: float = 0.5, thr_m: float = 0.25) -> np.ndarray:
    """Frames whose pelvis is > thr from the running median of its neighbours."""
    good = np.isfinite(P[:, J["pelvis"], 0])
    gi = np.flatnonzero(good)
    hw = round(win_s * fps)
    out = np.zeros(len(P), bool)
    for k, i in enumerate(gi):
        nb = gi[max(0, k - hw) : k + hw + 1]
        nb = nb[np.abs(nb - i) <= hw]
        if len(nb) >= 5 and np.linalg.norm(P[i, J["pelvis"]] - np.median(P[nb, J["pelvis"]], axis=0)) > thr_m:
            out[i] = True
    return out


def gather_2d(
    tr: Tracks, p: int, dets: dict[str, Detections], cams: CameraSet, gate_px: float, min_score: float, edge_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted 2D (V, N, 17, 2) and raw scores (V, N, 17) of the detections that built the track."""
    V, N = len(cams), tr.n_frames
    pts = np.full((V, N, N_JOINTS, 2), np.nan)
    scs = np.zeros((V, N, N_JOINTS))
    for i in np.flatnonzero(tr.has_3d[p]):
        X0 = tr.xyz[p, i, J["pelvis"]]
        for v, cam in enumerate(cams):
            r = int(tr.support[p, i, v])
            if r < 0:
                continue
            d = dets[cam.name]
            kp = cam.undistort_points(d.kps[r])
            ks = d.kps_score[r]
            pr, z = cam.project(X0)
            if z <= 0 or np.linalg.norm(pr - 0.5 * (kp[11] + kp[12])) > gate_px:
                continue
            raw = d.kps[r]
            for j, h in enumerate(HALPE26):
                edge = (
                    raw[h, 0] < edge_px
                    or raw[h, 0] > cam.width - edge_px
                    or raw[h, 1] < edge_px
                    or raw[h, 1] > cam.height - edge_px
                )
                if ks[h] >= min_score and not edge:
                    pts[v, i, j] = kp[h]
                    scs[v, i, j] = ks[h]
    return pts, scs


def noise_weights(pts: np.ndarray) -> np.ndarray:
    """Per camera and joint: (best jitter / jitter)^2 from the 2D second differences normalised by torso length."""
    V, _, Jn, _ = pts.shape
    tor = np.linalg.norm(
        0.5 * (pts[:, :, J["hip_l"]] + pts[:, :, J["hip_r"]])
        - 0.5 * (pts[:, :, J["shoulder_l"]] + pts[:, :, J["shoulder_r"]]),
        axis=-1,
    )
    acc = np.linalg.norm(pts[:, 2:] - 2 * pts[:, 1:-1] + pts[:, :-2], axis=-1) / tor[:, 1:-1, None]
    jit = np.full((V, Jn), np.nan)
    for c in range(V):
        for j in range(Jn):
            x = acc[c, :, j]
            x = x[np.isfinite(x)]
            if len(x) >= 30:
                jit[c, j] = np.median(x)
    best = np.nanmin(jit, axis=0) if np.isfinite(jit).any() else np.ones(Jn)
    w = np.clip((best / jit) ** 2, 0.05, 1.0)
    w[~np.isfinite(w)] = 1.0
    return w


def consensus_weights(
    pts: np.ndarray,
    scs: np.ndarray,
    cams: CameraSet,
    fps: float,
    frac: float = 0.06,
    min_px: float = 15,
    win_s: float = 1.5,
    sigma: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Per camera-joint observation: weight from the residual to the most agreed camera pair; dropped when
    persistently off."""
    V, N, Jn, _ = pts.shape
    Pm = np.stack([c.P for c in cams])
    ok = np.isfinite(pts[..., 0]) & (scs > 0)
    tor = np.linalg.norm(
        0.5 * (pts[:, :, J["hip_l"]] + pts[:, :, J["hip_r"]])
        - 0.5 * (pts[:, :, J["shoulder_l"]] + pts[:, :, J["shoulder_r"]]),
        axis=-1,
    )
    tmed = np.array([np.nanmedian(tor[c]) if np.isfinite(tor[c]).any() else np.nan for c in range(V)])
    thr = np.where(np.isfinite(tmed), np.maximum(min_px, frac * tmed), min_px)[:, None, None]
    best_cnt = np.zeros((N, Jn))
    best_sc = np.zeros((N, Jn))
    best_err = np.full((V, N, Jn), np.nan)
    for c1, c2 in combinations(range(V), 2):
        m = ok[c1] & ok[c2]
        if not m.any():
            continue
        x1, x2 = np.nan_to_num(pts[c1]), np.nan_to_num(pts[c2])
        A = np.stack(
            [
                x1[..., 0, None] * Pm[c1, 2] - Pm[c1, 0],
                x1[..., 1, None] * Pm[c1, 2] - Pm[c1, 1],
                x2[..., 0, None] * Pm[c2, 2] - Pm[c2, 0],
                x2[..., 1, None] * Pm[c2, 2] - Pm[c2, 1],
            ],
            axis=-2,
        )
        Xh = np.linalg.svd(A[m])[2][:, -1]
        X = np.full((N, Jn, 3), np.nan)
        X[m] = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
        pr = np.einsum("vab,njb->vnja", Pm, np.concatenate([X, np.ones((N, Jn, 1))], -1))
        err = np.linalg.norm(pr[..., :2] / np.where(np.abs(pr[..., 2:]) < 1e-9, 1e-9, pr[..., 2:]) - pts, axis=-1)
        inl = ok & (err < thr) & (pr[..., 2] > 0) & m[None]
        cnt = inl.sum(0)
        ssum = np.where(inl, scs, 0).sum(0)
        better = m & ((cnt > best_cnt) | ((cnt == best_cnt) & (ssum > best_sc)))
        best_cnt = np.where(better, cnt, best_cnt)
        best_sc = np.where(better, ssum, best_sc)
        best_err = np.where(better[None], err, best_err)
    tref = np.where(np.isfinite(tmed), tmed, np.nanmedian(tmed) if np.isfinite(tmed).any() else 1.0)[:, None, None]
    r = np.where(ok & (best_cnt >= 2)[None], best_err / tref, np.nan)
    win = 2 * round(win_s * fps) + 1
    rm = np.stack([_rolling_median(r[c], win) for c in range(V)])
    w = np.where(np.isfinite(rm), 1.0 / (1.0 + (rm / sigma) ** 2), 1.0)
    rej = ok & np.isfinite(rm) & (rm > 2 * sigma)
    return w, rej


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling nanmedian over axis 0 of (N, J), min_periods = max(3, win // 5)."""
    N = len(x)
    out = np.full_like(x, np.nan, dtype=float)
    h = win // 2
    minp = max(3, win // 5)
    for i in range(N):
        seg = x[max(0, i - h) : i + h + 1]
        cnt = np.isfinite(seg).sum(0)
        med = np.nanmedian(np.where(np.isfinite(seg), seg, np.nan), axis=0) if len(seg) else np.full(x.shape[1], np.nan)
        out[i] = np.where(cnt >= minp, med, np.nan)
    return out


def support_ok(
    scs_raw: np.ndarray, scs: np.ndarray, X: np.ndarray, centres: np.ndarray, support_score: float, support_deg: float
) -> bool:
    sup = np.flatnonzero((scs_raw >= support_score) & (scs > 0))
    if len(sup) < 2:
        return False
    rays = centres[sup] - X
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1)))) >= support_deg


def _anipose_group(cams: CameraSet):
    try:
        from aniposelib.cameras import Camera as ACam
        from aniposelib.cameras import CameraGroup
    except ImportError:
        return None
    from frolic.geometry import rotvec

    return CameraGroup([ACam(matrix=c.K, dist=np.zeros(5), rvec=rotvec(c.R), tvec=c.t, name=c.name) for c in cams])


def _optimise_chunk(
    cg, sub: np.ndarray, init: np.ndarray, scs: np.ndarray, smooth_abs: float, scale_length: float
) -> np.ndarray | None:
    from scipy.signal import medfilt

    init = init.copy()
    for j in range(init.shape[1]):
        ok = np.isfinite(init[:, j, 0])
        if ok.sum() == 0:
            init[:, j] = np.nan_to_num(init[:, 0])
            continue
        for k in range(3):
            init[:, j, k] = np.interp(np.arange(len(init)), np.flatnonzero(ok), init[ok, j, k])
    init = np.nan_to_num(init)
    med = np.stack([np.stack([medfilt(init[:, j, k], 7) for k in range(3)], 1) for j in range(init.shape[1])], 1)
    ss = smooth_abs * max(float(np.mean(np.abs(np.diff(med, axis=0)))), 1e-4)
    try:
        return cg.optim_points(
            sub,
            init,
            constraints=LIMBS,
            constraints_weak=LIMBS_WEAK,
            scale_smooth=ss,
            scale_length=scale_length,
            reproj_error_threshold=15,
            scores=scs,
            n_deriv_smooth=2,
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        log.warning("chunk not refined (%s)", e)
        return None


def refine_person(
    tr: Tracks, p: int, dets: dict[str, Detections], cams: CameraSet, cfg: TrackConfig, chunk: int = 450
) -> np.ndarray:
    """Refined (N, 17, 3) joints for one person (NaN where no 3D)."""
    fps = tr.fps
    P3 = tr.xyz[p].astype(float).copy()
    pts, scs_raw = gather_2d(tr, p, dets, cams, 80.0, cfg.min_score, 40.0)
    if np.isfinite(P3[..., 0]).sum() < N_JOINTS * 10:
        return P3
    scs = np.where(scs_raw > 0, scs_raw**4, 0.0)
    bridged = (tr.score[p, :, 0] > 0) & (tr.score[p, :, 0] < 0.5) & (tr.support[p] < 0).all(1)
    drop = bridged | blank_outliers(P3, fps)
    P3[drop] = np.nan
    pts[:, drop] = np.nan
    scs[:, drop] = 0
    if len(cams) >= 2:
        scs *= noise_weights(pts)[:, None, :]
        w, rej = consensus_weights(pts, scs, cams, fps)
        pts[rej] = np.nan
        scs[rej] = 0
        scs *= w
    out = P3.copy()
    cg = _anipose_group(cams)
    valid = np.isfinite(P3[:, J["pelvis"], 0])
    idx = np.flatnonzero(valid)
    runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > int(fps)) + 1) if len(r) >= 10] if len(idx) else []
    spans = [(c0, min(r[-1] + 1, c0 + chunk)) for r in runs for c0 in range(r[0], r[-1] + 1, chunk)]
    if cg is not None:
        for c0, c1 in spans:
            if c1 - c0 < 10:
                continue
            res = _optimise_chunk(cg, pts[:, c0:c1], P3[c0:c1], scs[:, c0:c1], cfg.refine_smooth, cfg.refine_length)
            if res is not None:
                out[c0:c1] = res
    out = smooth_runs(out, fps, cfg.euro_min_hz, cfg.euro_beta)
    # blanked frames: fill between refined neighbours (<= 3 s) else leave empty
    have = np.isfinite(out[:, J["pelvis"], 0])
    hi = np.flatnonzero(have)
    for i in np.flatnonzero(drop & ~have):
        L, R = hi[hi < i], hi[hi > i]
        if len(L) and len(R) and R[0] - L[-1] <= 3 * fps:
            w_ = (i - L[-1]) / (R[0] - L[-1])
            out[i] = (1 - w_) * out[L[-1]] + w_ * out[R[0]]
    centres = cams.centers
    limb = [j for j in range(N_JOINTS) if j not in (J["pelvis"], J["hip_l"], J["hip_r"])]
    unsupported = 0
    for i in np.flatnonzero(np.isfinite(out[:, J["pelvis"], 0])):
        if drop[i] or cg is None:
            continue
        for j in limb:
            if not np.isfinite(out[i, j]).all():
                continue
            if not support_ok(scs_raw[:, i, j], scs[:, i, j], out[i, j], centres, 0.5, 15.0):
                out[i, j] = np.nan
                unsupported += 1
    log.info(
        "%s: refined %d spans, %d limb joint-frames without two distinct supporting views",
        tr.ids[p],
        len(spans),
        unsupported,
    )
    return out.astype(np.float32)


def refine_tracks(tr: Tracks, dets: dict[str, Detections], cams: CameraSet, cfg: TrackConfig) -> Tracks:
    out = tr.subset(tr.ids)
    for p in range(tr.n_people):
        new = refine_person(tr, p, dets, cams, cfg)
        had = tr.has_3d[p]
        fin = np.isfinite(new[..., 0])
        out.xyz[p] = np.where(fin[..., None], new, np.nan)
        out.score[p] = np.where(fin, np.maximum(tr.score[p], 0.3), 0.0)
        gone = had & ~fin[:, J["pelvis"]]
        out.score[p, gone] = 0
        out.support[p, gone] = -1
    return out


def project_camera(cam: Camera, X: np.ndarray) -> np.ndarray:
    uv, _ = cam.project(X)
    return uv
