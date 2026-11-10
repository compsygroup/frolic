"""Calibration-free bootstrap: camera poses and focal lengths from one person's 2D keypoints across time (incremental
SfM + bundle adjustment), metric scale from that person's known stature."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from frolic.config import BootstrapConfig
from frolic.geometry import Camera, triangulate_batch
from frolic.io.calib import CameraSet
from frolic.io.detections import N_KP, Detections
from frolic.log import get_logger
from frolic.skeleton import HALPE26_TORSO

log = get_logger(__name__)

HEAD, ANKLES = 17, (15, 16)
HEAD_ANKLE_FRACTION = 0.93  # head-top to ankle distance as a fraction of stature


@dataclass
class BootstrapResult:
    cameras: CameraSet
    points: np.ndarray  # (F, 26, 3) world points (NaN where not triangulated)
    frames: list[int]
    report: dict[str, Any]


def _colour_distance(a: np.ndarray, b: np.ndarray) -> float:
    dh = abs(a[0] - b[0])
    dh = min(dh, 180 - dh) / 90.0
    return float(dh + abs(a[1] - b[1]) / 255.0 + 0.5 * abs(a[2] - b[2]) / 255.0)


def _candidates(d: Detections, frame: int, min_torso: float) -> list[int]:
    rows = d.at(frame)
    out = []
    for r in rows:
        if d.kps_score[r, HALPE26_TORSO].mean() < min_torso:
            continue
        x1, y1, x2, y2 = d.box[r]
        if (y2 - y1) < 0.1 * d.height:
            continue
        if x1 < 0.02 * d.width or y1 < 0.02 * d.height or x2 > 0.98 * d.width or y2 > 0.98 * d.height:
            continue
        out.append(int(r))
    return out


def _pick_consistent(d: dict[str, Detections], cand: dict[str, list[int]]) -> dict[str, int] | None:
    cols = {c: [d[c].colour[r] for r in rows] for c, rows in cand.items()}
    if any(not np.isfinite(v).all() for vs in cols.values() for v in vs):
        return None
    names = list(cand)
    best = None
    for combo in itertools.product(*[range(len(cand[c])) for c in names]):
        worst = max(
            _colour_distance(cols[a][i], cols[b][j]) for (a, i), (b, j) in itertools.combinations(zip(names, combo), 2)
        )
        if best is None or worst < best[0]:
            best = (worst, combo)
    if best is None or best[0] > 0.25:
        return None
    return {c: cand[c][i] for c, i in zip(names, best[1])}


def pick_frames(
    dets: dict[str, Detections], max_frames: int = 400, min_torso: float = 0.5
) -> dict[int, dict[str, int]]:
    """Frames where the same person is confidently seen in every camera -> camera -> detection row."""
    common = sorted(set.intersection(*[set(d.frames.tolist()) for d in dets.values()]))
    picked: dict[int, dict[str, int]] = {}
    for f in common:
        cand = {c: _candidates(d, f, min_torso) for c, d in dets.items()}
        if any(not v for v in cand.values()):
            continue
        if all(len(v) == 1 for v in cand.values()):
            picked[f] = {c: v[0] for c, v in cand.items()}
        else:
            sel = _pick_consistent(dets, cand)
            if sel is not None:
                picked[f] = sel
    if len(picked) > max_frames:
        keys = list(picked)
        keep = [keys[i] for i in np.linspace(0, len(keys) - 1, max_frames).astype(int)]
        picked = {k: picked[k] for k in keep}
    return picked


def gather(
    dets: dict[str, Detections], picked: dict[int, dict[str, int]], min_kp: float = 0.5
) -> tuple[list[int], dict[str, np.ndarray], dict[str, np.ndarray]]:
    frames = sorted(picked)
    uv = {c: np.full((len(frames), N_KP, 2), np.nan) for c in dets}
    ok = {c: np.zeros((len(frames), N_KP), bool) for c in dets}
    for i, f in enumerate(frames):
        for c, r in picked[f].items():
            uv[c][i] = dets[c].kps[r]
            ok[c][i] = dets[c].kps_score[r] >= min_kp
    return frames, uv, ok


def initial_K(width: int, height: int, focal: float) -> np.ndarray:
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def _normalised(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    return (uv - K[:2, 2]) / np.diag(K)[:2]


def pair_pose(xa: np.ndarray, xb: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Essential-matrix pose of camera b relative to a from normalised points: X_b = R X_a + t (|t| = 1), inlier mask."""
    E, mask = cv2.findEssentialMat(xa, xb, 1.0, (0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=thr)
    if E is None or E.shape != (3, 3):
        return np.eye(3), np.zeros(3), np.zeros(len(xa), bool)
    _, R, t, mask2 = cv2.recoverPose(E, xa, xb, mask=mask.copy())
    return R, t.ravel(), mask2.ravel() > 0


def _project_all(cams: dict[str, Camera], X: np.ndarray) -> dict[str, np.ndarray]:
    return {c: cam.project(X)[0] for c, cam in cams.items()}


def triangulate_points(
    cams: dict[str, Camera], uvn: dict[str, np.ndarray], ok: dict[str, np.ndarray], max_err: float
) -> np.ndarray:
    """(F, 26, 3) from every camera that sees a point, keeping points whose reprojection stays within max_err (normalised)."""
    names = list(cams)
    F = next(iter(uvn.values())).shape[0]
    P = np.stack([np.hstack([cams[c].R, cams[c].t[:, None]]) for c in names])
    pts = np.stack([np.where(ok[c][..., None], uvn[c], np.nan) for c in names]).reshape(len(names), -1, 2)
    X = triangulate_batch(P, pts).reshape(F, N_KP, 3)
    for c in names:
        Xc = X @ cams[c].R.T + cams[c].t
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = Xc[..., :2] / Xc[..., 2:]
            bad = ok[c] & ((Xc[..., 2] < 0.05) | (np.linalg.norm(pr - uvn[c], axis=-1) > max_err))
        X[bad] = np.nan
    return X


def incremental_sfm(
    uvn: dict[str, np.ndarray], ok: dict[str, np.ndarray], thr: float, min_pair: int
) -> tuple[dict[str, Camera], np.ndarray, dict[str, Any]]:
    """Normalised keypoints per camera (F, 26, 2) -> cameras (root at identity, unit first baseline) and points."""
    names = list(uvn)
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        m = ok[a] & ok[b]
        if m.sum() < min_pair:
            continue
        R, t, inl = pair_pose(uvn[a][m], uvn[b][m], thr)
        pairs[(a, b)] = (R, t, int(inl.sum()), int(m.sum()))
    if not pairs:
        raise RuntimeError("no camera pair shares enough confident keypoints")
    (a, b), (R, t, n_inl, n_all) = max(pairs.items(), key=lambda kv: kv[1][2])
    log.info("bootstrap: seed pair %s-%s (%d/%d inliers)", a, b, n_inl, n_all)
    cams = {a: Camera(a, 1, 1, np.eye(3)), b: Camera(b, 1, 1, np.eye(3), R=R, t=t)}
    X = triangulate_points(cams, {c: uvn[c] for c in cams}, {c: ok[c] for c in cams}, thr * 3)
    todo = [c for c in names if c not in cams]
    while todo:
        vis = {c: int((ok[c] & np.isfinite(X[..., 0])).sum()) for c in todo}
        c = max(vis, key=vis.get)
        m = ok[c] & np.isfinite(X[..., 0])
        if m.sum() < 12:
            raise RuntimeError(f"camera {c} sees too few triangulated points ({m.sum()})")
        succ, rv, tv, inl = cv2.solvePnPRansac(
            X[m].astype(np.float64),
            uvn[c][m].astype(np.float64),
            np.eye(3),
            np.zeros(5),
            reprojectionError=thr,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not succ or inl is None or len(inl) < 12:
            raise RuntimeError(f"PnP failed for camera {c}")
        rv, tv = cv2.solvePnPRefineLM(X[m][inl.ravel()], uvn[c][m][inl.ravel()], np.eye(3), np.zeros(5), rv, tv)
        cams[c] = Camera(c, 1, 1, np.eye(3), R=cv2.Rodrigues(rv)[0], t=tv.ravel())
        todo.remove(c)
        X = triangulate_points(cams, {k: uvn[k] for k in cams}, {k: ok[k] for k in cams}, thr * 3)
        log.info("bootstrap: placed %s from %d PnP inliers", c, len(inl))
    return (
        cams,
        X,
        {"pairs": {f"{a}-{b}": {"inliers": v[2], "matches": v[3]} for (a, b), v in pairs.items()}, "seed": [a, b]},
    )


def _distort_project(Xc: np.ndarray, f: float, k1: float, pp: np.ndarray) -> np.ndarray:
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    return np.stack([f * x * (1 + k1 * r2) + pp[0], f * y * (1 + k1 * r2) + pp[1]], 1)


def bundle_adjust(
    cams: dict[str, Camera],
    X: np.ndarray,
    uv: dict[str, np.ndarray],
    ok: dict[str, np.ndarray],
    Ks: dict[str, np.ndarray],
    refine_intrinsics: bool,
    px_scale: float,
) -> tuple[dict[str, Camera], np.ndarray, dict[str, Any]]:
    """Joint refinement of poses, points and (optionally) focal + k1 per camera; the root pose and the first baseline fix the gauge."""
    names = list(cams)
    vis = np.isfinite(X[..., 0])
    pid = np.flatnonzero(vis.ravel())
    P0 = X.reshape(-1, 3)[pid]
    pmap = {p: i for i, p in enumerate(pid)}
    oc, op, ouv = [], [], []
    for ci, c in enumerate(names):
        rows = np.flatnonzero((ok[c] & vis).ravel())
        oc.append(np.full(len(rows), ci))
        op.append(np.array([pmap[r] for r in rows]))
        ouv.append(uv[c].reshape(-1, 2)[rows])
    oc, op, ouv = np.concatenate(oc), np.concatenate(op), np.concatenate(ouv)
    nc, npts = len(names), len(P0)
    pp = {c: Ks[c][:2, 2] for c in names}
    x0 = np.concatenate(
        [np.r_[Rotation.from_matrix(cams[c].R).as_rotvec(), cams[c].t] for c in names]
        + [np.array([Ks[c][0, 0], 0.0]) for c in names]
        + [P0.ravel()]
    )
    ni = 2 * nc

    def unpack(x):
        cp = x[: 6 * nc].reshape(nc, 6)
        ip = x[6 * nc : 6 * nc + ni].reshape(nc, 2)
        pts = x[6 * nc + ni :].reshape(npts, 3)
        return cp, ip, pts

    def resid(x):
        cp, ip, pts = unpack(x)
        Rc = Rotation.from_rotvec(cp[:, :3]).as_matrix()
        Xc = np.einsum("nij,nj->ni", Rc[oc], pts[op]) + cp[oc, 3:]
        out = np.empty_like(ouv)
        for ci, c in enumerate(names):
            m = oc == ci
            out[m] = _distort_project(Xc[m], ip[ci, 0], ip[ci, 1], pp[c])
        r = (out - ouv).ravel()
        gauge = 100.0 * px_scale * (np.linalg.norm(cp[1, 3:]) - 1.0)
        return np.r_[r, gauge]

    n_res = 2 * len(oc) + 1
    S = lil_matrix((n_res, len(x0)), dtype=int)
    rows = np.repeat(np.arange(len(oc)), 2) * 2 + np.tile([0, 1], len(oc))
    for k in range(6):
        S[rows, np.repeat(6 * oc + k, 2)] = 1
    for k in range(2):
        S[rows, np.repeat(6 * nc + 2 * oc + k, 2)] = 1
    for k in range(3):
        S[rows, np.repeat(6 * nc + ni + 3 * op + k, 2)] = 1
    S[:, 0:6] = 0
    S[n_res - 1, 6 + 3 : 6 + 6] = 1
    if not refine_intrinsics:
        S[:, 6 * nc : 6 * nc + ni] = 0
    r0 = resid(x0)
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=px_scale, max_nfev=100, x_scale="jac")
    cp, ip, pts = unpack(res.x)
    out = {}
    for ci, c in enumerate(names):
        K = Ks[c].copy()
        K[0, 0] = K[1, 1] = ip[ci, 0]
        out[c] = Camera(
            c,
            cams[c].width,
            cams[c].height,
            K,
            np.array([ip[ci, 1], 0, 0, 0, 0]),
            Rotation.from_rotvec(cp[ci, :3]).as_matrix(),
            cp[ci, 3:].copy(),
        )
    Xo = np.full_like(X, np.nan)
    Xo.reshape(-1, 3)[pid] = pts
    e0 = np.linalg.norm(r0[:-1].reshape(-1, 2), axis=1)
    e1 = np.linalg.norm(res.fun[:-1].reshape(-1, 2), axis=1)
    rep = {
        "observations": len(oc),
        "points": int(npts),
        "median_px_before": float(np.median(e0)),
        "median_px_after": float(np.median(e1)),
        "per_camera_median_px": {c: float(np.median(e1[oc == i])) for i, c in enumerate(names)},
    }
    return out, Xo, rep


def stature_estimate(X: np.ndarray) -> float:
    """Median head-top to ankle-midpoint distance / HEAD_ANKLE_FRACTION over frames where both are known."""
    head = X[:, HEAD]
    ank = 0.5 * (X[:, ANKLES[0]] + X[:, ANKLES[1]])
    d = np.linalg.norm(head - ank, axis=-1)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        raise RuntimeError("too few frames with head and both ankles for the metric scale")
    return float(np.median(d) / HEAD_ANKLE_FRACTION)


def group_people(dets: dict[str, Detections], picked: dict[int, dict[str, int]], frames: list[int]) -> np.ndarray:
    """Group label per bootstrap frame: frames whose cross-camera median clothing colour agrees share a person."""
    cols = []
    for f in frames:
        c = np.array([dets[cam].colour[r] for cam, r in picked[f].items()], float)
        cols.append(np.nanmedian(c, 0) if np.isfinite(c).any() else np.full(3, np.nan))
    labels = np.full(len(frames), -1)
    centres: list[np.ndarray] = []
    for i, c in enumerate(cols):
        if not np.isfinite(c).all():
            continue
        d = [_colour_distance(c, m) for m in centres]
        if d and min(d) < 0.25:
            labels[i] = int(np.argmin(d))
        else:
            labels[i] = len(centres)
            centres.append(c)
    return labels


def reference_frames(X: np.ndarray, labels: np.ndarray, rule: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Frames of the person whose stature is known: the most seen colour group, or the tallest / shortest one."""
    head = X[:, HEAD]
    ank = 0.5 * (X[:, ANKLES[0]] + X[:, ANKLES[1]])
    size = np.linalg.norm(head - ank, axis=-1)
    groups = {}
    for g in np.unique(labels[labels >= 0]):
        m = (labels == g) & np.isfinite(size)
        if m.sum() >= 5:
            groups[int(g)] = (int(m.sum()), float(np.median(size[m])))
    if len(groups) <= 1:
        return np.ones(len(X), bool), {"groups": len(groups)}
    if rule == "tallest":
        pick = max(groups, key=lambda g: groups[g][1])
    elif rule == "shortest":
        pick = min(groups, key=lambda g: groups[g][1])
    else:
        pick = max(groups, key=lambda g: groups[g][0])
    log.info(
        "bootstrap: %d colour groups, scale from group %d (%s, %d frames)", len(groups), pick, rule, groups[pick][0]
    )
    rep = {"groups": len(groups), "group_frames": {str(g): v[0] for g, v in groups.items()}, "reference_group": pick}
    return labels == pick, rep


def world_frame(X: np.ndarray, root: Camera) -> tuple[np.ndarray, np.ndarray]:
    """(R, origin) with z along the mean head-to-ankle direction and the origin at the mean ankle midpoint."""
    head = X[:, HEAD]
    ank = 0.5 * (X[:, ANKLES[0]] + X[:, ANKLES[1]])
    m = np.isfinite(head[:, 0]) & np.isfinite(ank[:, 0])
    up = (head[m] - ank[m]).mean(0)
    up /= np.linalg.norm(up)
    x = root.R[0] - (root.R[0] @ up) * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up]), ank[m].mean(0)


def _solve(
    uv: dict[str, np.ndarray],
    ok: dict[str, np.ndarray],
    dets: dict[str, Detections],
    cfg: BootstrapConfig,
    focal: float,
    refine: bool,
) -> tuple[dict[str, Camera], np.ndarray, dict[str, Any], dict[str, Any]]:
    Ks = {c: initial_K(d.width, d.height, focal) for c, d in dets.items()}
    uvn = {c: _normalised(uv[c], Ks[c]) for c in dets}
    cams, X, rep = incremental_sfm(uvn, ok, cfg.ransac_px / focal, cfg.min_pair_frames)
    cams = {c: Camera(c, dets[c].width, dets[c].height, np.eye(3), R=cams[c].R, t=cams[c].t) for c in cams}
    cams, X, ba = bundle_adjust(cams, X, uv, ok, Ks, refine, cfg.ransac_px)
    return cams, X, rep, ba


def search_focal(
    uv: dict[str, np.ndarray], ok: dict[str, np.ndarray], dets: dict[str, Detections], cfg: BootstrapConfig
) -> tuple[float, list[dict[str, float]]]:
    """Shared focal with the lowest fixed-intrinsics reprojection error over a horizontal-FOV grid, refined by a parabola."""
    width = float(np.median([d.width for d in dets.values()]))
    fovs = np.linspace(cfg.fov_range_deg[0], cfg.fov_range_deg[1], cfg.fov_steps)
    focals = width / 2 / np.tan(np.radians(fovs) / 2)
    errs = []
    for f in focals:
        try:
            errs.append(_solve(uv, ok, dets, cfg, float(f), False)[3]["median_px_after"])
        except RuntimeError:
            errs.append(np.inf)
    errs = np.asarray(errs)
    if not np.isfinite(errs).any():
        raise RuntimeError("bootstrap failed for every focal length in the search range")
    i = int(np.argmin(errs))
    best = float(focals[i])
    if 0 < i < len(focals) - 1 and np.isfinite(errs[i - 1 : i + 2]).all():
        x, y = np.log(focals[i - 1 : i + 2]), errs[i - 1 : i + 2]
        a, b, _ = np.polyfit(x, y, 2)
        if a > 0:
            best = float(np.exp(np.clip(-b / (2 * a), x.min(), x.max())))
    return best, [{"focal_px": float(f), "median_px": float(e)} for f, e in zip(focals, errs)]


def bootstrap(dets: dict[str, Detections], cfg: BootstrapConfig, reference_height_m: float) -> BootstrapResult:
    picked = pick_frames(dets, cfg.max_frames)
    if len(picked) < cfg.min_pair_frames:
        raise RuntimeError(f"only {len(picked)} frames with one person in every camera (need {cfg.min_pair_frames})")
    frames, uv, ok = gather(dets, picked)
    if cfg.focal_guess is None:
        f0, sweep = search_focal(uv, ok, dets, cfg)
        log.info("bootstrap: focal search picked %.0f px", f0)
    else:
        f0, sweep = cfg.focal_guess, []
    cams, X, rep, ba = _solve(uv, ok, dets, cfg, f0, cfg.refine_intrinsics)
    rep["focal_search"] = sweep
    rep["ba"] = ba
    ref, rep["people"] = reference_frames(X, group_people(dets, picked, frames), cfg.person)
    stat = stature_estimate(X[ref])
    scale = reference_height_m / stat
    root = cams[next(iter(cams))]
    Rw, origin = world_frame(X, root)
    t = -scale * Rw @ origin
    out = CameraSet([cams[c].transformed(Rw, t, scale) for c in dets], "m", {}, {"format": "bootstrap"})
    Xw = scale * (X - origin) @ Rw.T
    rep.update(
        {
            "frames": len(frames),
            "stature_unscaled": stat,
            "scale": scale,
            "focal_px": {c: float(cams[c].K[0, 0]) for c in cams},
            "k1": {c: float(cams[c].dist[0]) for c in cams},
        }
    )
    log.info(
        "bootstrap: %d frames, BA median %.2f px, scale %.3f (stature %.2f m)",
        len(frames),
        ba["median_px_after"],
        scale,
        reference_height_m,
    )
    return BootstrapResult(out, Xw.astype(np.float32), frames, rep)
