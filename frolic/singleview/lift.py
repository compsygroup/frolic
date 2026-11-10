"""Single-view fallback: lift frames where a person has no triangulation but one camera sees them confidently."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from frolic.config import SingleViewConfig
from frolic.geometry import Camera, ray_point_distance, ray_ray_closest, rays_intersection
from frolic.io.calib import CameraSet
from frolic.io.detections import Detections
from frolic.io.tracks import Tracks
from frolic.log import get_logger
from frolic.skeleton import HALPE26, HALPE26_HIP, HALPE26_TORSO, N_JOINTS, SEGMENTS, J

log = get_logger(__name__)
RelDepth = dict[
    tuple[str, str, int], dict[int, float]
]  # (person id, camera, track frame) -> {halpe index: depth offset to the hip}
LIFT_SEGS = ["trunk", "shoulders", "thigh_l", "thigh_r", "shin_l", "shin_r", "upperarm_l", "upperarm_r"]
HIP_PAIR = (11, 12)


@dataclass
class PersonModel:
    pid: str
    P: np.ndarray  # (N, 17, 3)
    valid: np.ndarray
    seglen: dict[str, float]
    pelvis_stand: float
    ankle_h: float
    colour: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # camera -> (median, scale)


@dataclass
class Cand:
    view: int
    row: int
    frame: int
    tconf: float
    dp: float
    rad: float
    dt: float
    tpl: int
    X0: np.ndarray
    box: np.ndarray
    hip_in: bool
    dcol: dict[str, float] = field(default_factory=dict)
    reid: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    score: float = 0.0


def colour_feat(hsv: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(hsv).all():
        return None
    h, s, v = hsv
    return np.array([np.cos(h * np.pi / 90) * s / 255, np.sin(h * np.pi / 90) * s / 255, v / 255])


def person_models(tr: Tracks, dets: dict[str, Detections]) -> dict[str, PersonModel]:
    out = {}
    for p, pid in enumerate(tr.ids):
        P = tr.xyz[p].astype(float)
        valid = tr.has_3d[p] & ~tr.single_view[p]
        feats: dict[str, list[np.ndarray]] = {c: [] for c in tr.cameras}
        for i in np.flatnonzero(valid):
            for v, cam in enumerate(tr.cameras):
                r = int(tr.support[p, i, v])
                if r >= 0:
                    x = colour_feat(dets[cam].colour[r])
                    if x is not None:
                        feats[cam].append(x)
        colour = {}
        for cam, F in feats.items():
            if len(F) >= 20:
                F = np.array(F)
                med = np.median(F, 0)
                colour[cam] = (med, np.median(np.abs(F - med), 0) * 1.4826 + 0.03)
        seglen = {k: float(np.nanmedian(np.linalg.norm(P[:, a] - P[:, b], axis=-1))) for k, (a, b) in SEGMENTS.items()}
        z = P[:, J["pelvis"], 2]
        ank = np.nanmin(P[:, [J["ankle_l"], J["ankle_r"]], 2], 1)
        ank_h = float(np.nanpercentile(ank[np.isfinite(ank)], 20)) if np.isfinite(ank).any() else 0.07
        out[pid] = PersonModel(
            pid,
            P,
            valid,
            seglen,
            float(np.nanpercentile(z, 90)) if np.isfinite(z).any() else 0.9,
            float(np.clip(ank_h, 0.03, 0.12)),
            colour,
        )
    return out


def interp_prior(P: np.ndarray, valid: np.ndarray, i: int, fps: float) -> tuple[np.ndarray | None, float, int]:
    vi = np.flatnonzero(valid)
    L, R = vi[vi < i], vi[vi > i]
    if len(L) and len(R):
        l_, r_ = L[-1], R[0]
        w = (i - l_) / (r_ - l_)
        return (1 - w) * P[l_, 0] + w * P[r_, 0], min(i - l_, r_ - i) / fps, int(l_ if i - l_ <= r_ - i else r_)
    if len(L):
        return P[L[-1], 0], (i - L[-1]) / fps, int(L[-1])
    if len(R):
        return P[R[0], 0], (R[0] - i) / fps, int(R[0])
    return None, np.inf, -1


def _hip_px(cam: Camera, kp_raw: np.ndarray, ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp = cam.undistort_points(kp_raw)
    hip = kp[HALPE26_HIP] if ks[HALPE26_HIP] >= 0.3 else 0.5 * (kp[HIP_PAIR[0]] + kp[HIP_PAIR[1]])
    return kp, hip


def find_candidates(
    tr: Tracks, dets: dict[str, Detections], cams: CameraSet, models: dict[str, PersonModel], cfg: SingleViewConfig
) -> dict[tuple[str, int], list[Cand]]:
    cand: dict[tuple[str, int], list[Cand]] = {}
    ids = tr.ids
    for pid in ids:
        m = models[pid]
        for i in np.flatnonzero(~m.valid):
            X0, dt, tpl = interp_prior(m.P, m.valid, int(i), tr.fps)
            if X0 is None or dt > cfg.max_prior_s:
                continue
            rad = min(cfg.gate_m + cfg.vmax_mps * dt, cfg.gate_max_m)
            f = tr.source_frame(int(i))
            for v, cam in enumerate(cams):
                d = dets[cam.name]
                for r in d.at(f):
                    ks = d.kps_score[r]
                    tconf = float(np.mean(ks[HALPE26_TORSO]))
                    if tconf < cfg.min_torso or d.box_score[r] < cfg.min_box:
                        continue
                    _, hip = _hip_px(cam, d.kps[r], ks)
                    u = cam.rays(hip[None])[0]
                    dp, tp = ray_point_distance(cam.center, u, X0)
                    if tp <= 0.3 or dp > rad:
                        continue
                    if _claimed(tr, models, pid, int(i), cam, u, dp, v, r, cfg.claim_m):
                        continue
                    cf = colour_feat(d.colour[r])
                    dcol = {}
                    for q in ids:
                        cm = models[q].colour.get(cam.name)
                        if cf is not None and cm is not None:
                            dcol[q] = float(np.sqrt(np.mean(((cf - cm[0]) / cm[1]) ** 2)))
                    hr = d.kps[r][HALPE26_HIP]
                    hip_in = bool(
                        cfg.edge_px <= hr[0] <= cam.width - cfg.edge_px
                        and cfg.edge_px <= hr[1] <= cam.height - cfg.edge_px
                    )
                    cand.setdefault((pid, int(i)), []).append(
                        Cand(
                            v,
                            int(r),
                            f,
                            tconf,
                            float(dp),
                            float(rad),
                            float(dt),
                            tpl,
                            X0,
                            d.box[r].astype(float),
                            hip_in,
                            dcol,
                        )
                    )
    return cand


def _claimed(
    tr: Tracks,
    models: dict[str, PersonModel],
    pid: str,
    i: int,
    cam: Camera,
    u: np.ndarray,
    dp: float,
    v: int,
    r: int,
    claim_m: float,
) -> bool:
    for q in tr.ids:
        if q == pid or not models[q].valid[i]:
            continue
        dq, tq = ray_point_distance(cam.center, u, models[q].P[i, 0])
        if tq > 0 and dq < claim_m and dq < dp:
            return True
        if int(tr.support[tr.index(q), i, v]) == r:
            return True
    return False


def score_candidates(cand: dict[tuple[str, int], list[Cand]], ids: list[str], cfg: SingleViewConfig) -> None:
    for (pid, _), L in cand.items():
        others = [q for q in ids if q != pid]
        for c in L:
            col_ok = pid not in c.dcol or (
                c.dcol[pid] <= cfg.colour_abs
                and all(c.dcol[pid] <= c.dcol.get(q, np.inf) + cfg.colour_margin for q in others)
            )
            reid_ok = pid not in c.reid or all(c.reid[pid] >= c.reid.get(q, -1) - cfg.reid_margin for q in others)
            c.ok = bool(col_ok and reid_ok)
            c.score = (
                c.tconf
                * float(np.exp(-((c.dp / max(c.rad, 1e-3)) ** 2)))
                * (1.0 if col_ok else 0.3)
                * (1.0 if reid_ok else 0.3)
            )


def resolve(cand: dict[tuple[str, int], list[Cand]]) -> dict[tuple[str, int], list[Cand]]:
    """One detection per (person, frame) per camera, each detection used by at most one person (greedy by score)."""
    used: set[tuple[int, int, int]] = set()
    match: dict[tuple[str, int], list[Cand]] = {}
    flat = sorted(((c.score, pid, i, c) for (pid, i), L in cand.items() for c in L if c.ok), key=lambda x: -x[0])
    for _, pid, i, c in flat:
        key = (c.view, c.frame, c.row)
        if key in used or any(m.view == c.view for m in match.get((pid, i), [])):
            continue
        used.add(key)
        match.setdefault((pid, i), []).append(c)
    return match


def _inside(cam: Camera, q: np.ndarray, edge: float) -> bool:
    return bool(edge <= q[0] <= cam.width - edge and edge <= q[1] <= cam.height - edge)


def lift_one(
    pid: str,
    i: int,
    L: list[Cand],
    m: PersonModel,
    dets: dict[str, Detections],
    cams: CameraSet,
    cfg: SingleViewConfig,
    rel: dict[int, float] | None,
) -> dict[str, Any] | None:
    L = sorted(L, key=lambda c: (-c.hip_in, -c.score))
    c = L[0]
    cam = cams[c.view]
    d = dets[cam.name]
    kp, hip = _hip_px(cam, d.kps[c.row], d.kps_score[c.row])
    ks = d.kps_score[c.row]
    u = cam.rays(hip[None])[0]
    C0, ax = cam.center, cam.axis
    cosu = float(u @ ax)
    X0, dt = c.X0, c.dt
    H: list[tuple[float, float, str]] = [((X0 - C0) @ u, cfg.sigma_prior_m + 1.0 * dt, "prior")]
    sz = 0.05 + 0.20 * min(dt / 1.5, 1.0)
    if abs(u[2]) > 0.05:
        H.append(((X0[2] - C0[2]) / u[2], sz / abs(u[2]), "height"))
    dl = []
    for name in LIFT_SEGS:
        a, b = SEGMENTS[name]
        ha, hb = HALPE26[a], HALPE26[b]
        Lm = m.seglen.get(name, np.nan)
        if ks[ha] >= 0.5 and ks[hb] >= 0.5 and np.isfinite(Lm):
            lp = np.linalg.norm(kp[ha] - kp[hb])
            if lp > 8:
                dl.append(cam.focal * Lm / lp)
    if len(dl) >= 2:
        zc = float(np.mean(sorted(dl)[:2]))
        H.append((zc / cosu, 0.15 * zc / cosu, "limbs"))
    standing = X0[2] > 0.75 * m.pelvis_stand
    fe = [h for h in (HALPE26[J["ankle_l"]], HALPE26[J["ankle_r"]]) if ks[h] >= 0.5]
    if fe:
        ua = cam.rays(kp[fe])
        tt = (m.ankle_h - C0[2]) / ua[:, 2]
        if (tt > 0).all() and (ua[:, 2] < -0.05).all():
            F = (C0 + tt[:, None] * ua).mean(0)
            if standing:
                uh = u[:2]
                H.append((((F[:2] - C0[:2]) @ uh) / (uh @ uh), 0.10 / max(np.linalg.norm(uh), 0.3), "feet"))
            else:
                H.append((((F - C0) @ ax) / cosu, 0.35, "feet_seated"))
    cons = [(c, cam, kp, ks)]
    for c2 in L[1:]:
        if not c2.hip_in:
            continue
        cam2 = cams[c2.view]
        d2 = dets[cam2.name]
        kp2, h2 = _hip_px(cam2, d2.kps[c2.row], d2.kps_score[c2.row])
        u2 = cam2.rays(h2[None])[0]
        t1, t2, dd = ray_ray_closest(C0, u, cam2.center, u2)
        if dd < cfg.multi_m and t1 > 0 and t2 > 0:
            H.append((t1, 0.08, f"view2:{cam2.name}"))
            cons.append((c2, cam2, kp2, d2.kps_score[c2.row]))
    while True:
        t_ = np.array([h[0] for h in H])
        s_ = np.array([h[1] for h in H])
        w = 1 / s_**2
        tf = (w * t_).sum() / w.sum()
        r = np.abs(t_ - tf) / s_
        if len(H) <= 2 or r.max() < 3:
            break
        H.pop(int(r.argmax()))
    sigma = float(1 / np.sqrt(w.sum()))
    Xp = C0 + tf * u
    X = np.full((N_JOINTS, 3), np.nan)
    vis = np.zeros(N_JOINTS, bool)
    T = m.P[c.tpl]
    ntri = 0
    for j, h in enumerate(HALPE26):
        obs = [
            (cm.C if hasattr(cm, "C") else cm.center, cm.rays(kq[h][None])[0])
            for cc, cm, kq, kqs in cons
            if kqs[h] >= cfg.min_kp and _inside(cm, dets[cm.name].kps[cc.row][h], cfg.edge_px)
        ]
        if len(obs) >= 2:
            X[j] = rays_intersection(np.array([o for o, _ in obs]), np.array([u_ for _, u_ in obs]))
            vis[j] = True
            ntri += 1
            continue
        if j == J["pelvis"]:
            X[j] = Xp
            vis[j] = True
            continue
        if ks[h] < cfg.min_kp or not _inside(cam, d.kps[c.row][h], cfg.edge_px):
            continue
        if rel is not None and h in rel:
            dz = rel[h]
        elif np.isfinite(T[j]).all() and np.isfinite(T[0]).all():
            dz = float((T[j] - T[0]) @ ax)
        else:
            continue
        zp = (X[0] - C0) @ ax
        uj = cam.rays(kp[h][None])[0]
        X[j] = C0 + (zp + dz) / (uj @ ax) * uj
        vis[j] = True
    if not c.hip_in and not (len(cons) >= 2 and ntri > 0):
        return None
    return {
        "X": X, "vis": vis, "sigma": sigma, "view": cam.name, "det": c.row, "frame": c.frame, "box": c.box.tolist(),
        "hyp": [(h[2], round(float(h[0]), 3), round(float(h[1]), 3)) for h in H],
        "views": [cams[x.view].name for x in L],
        "dets": [x.row for x in L],
        "rel": "depths" if rel is not None else "template",
        "cons_views": [cm.name for _, cm, _, _ in cons],
        "mode": "multi" if len(cons) >= 2 and ntri > 0 else "single",
        "ntri": ntri,
    }  # fmt: skip


def room_bounds(tr: Tracks, margin: float = 0.5) -> tuple[float, float, float, float]:
    pel = tr.xyz[:, :, J["pelvis"], :2].reshape(-1, 2)
    pel = pel[np.isfinite(pel[:, 0])]
    if len(pel) == 0:
        return -np.inf, np.inf, -np.inf, np.inf
    lo, hi = pel.min(0) - margin, pel.max(0) + margin
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def smooth_and_reject(
    models: dict[str, PersonModel],
    lifted: dict[str, dict[int, dict]],
    fps: float,
    cfg: SingleViewConfig,
    bounds: tuple[float, float, float, float],
) -> dict[str, dict[int, str]]:
    rx0, rx1, ry0, ry1 = bounds
    rej: dict[str, dict[int, str]] = {}
    for pid, D in lifted.items():
        m = models[pid]
        N = len(m.P)
        P = m.P
        for _ in range(5):
            sv = np.zeros(N, bool)
            sv[list(D)] = True
            pel = P[:, 0].copy()
            if D:
                pel[sv] = np.array([D[i]["X"][0] for i in sorted(D)])
            have = m.valid | sv
            idx = np.flatnonzero(have)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
            sm = pel.copy()
            for r in runs:
                if not sv[r].any():
                    continue
                lo, hi = (
                    max(r[0], np.flatnonzero(sv[r])[0] + r[0] - 3),
                    min(r[-1], np.flatnonzero(sv[r])[-1] + r[0] + 3),
                )
                seg = pel[lo : hi + 1].copy()
                g = gaussian_filter1d(seg, cfg.smooth_frames, axis=0, mode="nearest")
                fix = ~sv[lo : hi + 1]
                g[fix] = seg[fix]
                g = gaussian_filter1d(g, cfg.smooth_frames * 0.5, axis=0, mode="nearest")
                g[fix] = seg[fix]
                sm[lo : hi + 1] = g
            bad = []
            for i in sorted(D):
                d = D[i]
                dX = sm[i] - d["X"][0]
                d["Xs"] = d["X"] + dX * d["vis"][:, None]
                Xp = d["Xs"][0]
                why = None
                if not (
                    rx0 - cfg.room_margin_m <= Xp[0] <= rx1 + cfg.room_margin_m
                    and ry0 - cfg.room_margin_m <= Xp[1] <= ry1 + cfg.room_margin_m
                ):
                    why = "room"
                elif not (0.03 <= Xp[2] <= 1.25 * m.pelvis_stand + 0.1):
                    why = "height"
                else:
                    for q in (i - 1, i + 1):
                        if (
                            0 <= q < N
                            and have[q]
                            and np.isfinite(sm[q]).all()
                            and np.linalg.norm(sm[q, :2] - Xp[:2]) * fps > cfg.vmax_mps
                        ):
                            why = "speed"
                    tor = np.linalg.norm(d["Xs"][J["neck"]] - Xp) if d["vis"][J["neck"]] else np.nan
                    if (
                        why is None
                        and np.isfinite(tor)
                        and np.isfinite(m.seglen["trunk"])
                        and not (0.6 < tor / m.seglen["trunk"] < 1.5)
                    ):
                        why = "limbs"
                if why:
                    bad.append((i, why))
            if not bad:
                break
            for i, why in bad:
                rej.setdefault(pid, {})[i] = why
                D.pop(i, None)
    return rej


def spans_of(frames: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def lift_frames(
    tracks: Tracks,
    dets: dict[str, Detections],
    cams: CameraSet,
    cfg: SingleViewConfig,
    rel_depth: RelDepth | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    reid: dict[tuple[str, int, int], dict[str, float]] | None = None,
) -> tuple[Tracks, dict[str, Any]]:
    """Lift single-view frames into `tracks` (copy). rel_depth: (pid, camera, frame) -> {halpe index: depth offset
    to the hip} (e.g. from SAM 3D Body); without it the nearest triangulated frame's pose supplies the offsets.
    reid: (pid, frame, view) ->
    {other pid: similarity} from an appearance model, optional. Returns the new tracks and the svfill report."""
    models = person_models(tracks, dets)
    cand = find_candidates(tracks, dets, cams, models, cfg)
    if reid:
        for (pid, i), L in cand.items():
            for c in L:
                c.reid = reid.get((pid, i, c.view), {})
    score_candidates(cand, tracks.ids, cfg)
    match = resolve(cand)
    lifted: dict[str, dict[int, dict]] = {}
    for (pid, i), L in sorted(match.items()):
        rel = None
        if rel_depth:
            for c in sorted(L, key=lambda c: (-c.hip_in, -c.score)):
                rel = rel_depth.get((pid, cams[c.view].name, i))
                if rel is not None:
                    break
        d = lift_one(pid, i, L, models[pid], dets, cams, cfg, rel)
        if d is not None:
            lifted.setdefault(pid, {})[i] = d
    rej = smooth_and_reject(models, lifted, tracks.fps, cfg, bounds or room_bounds(tracks))
    out = tracks.subset(tracks.ids)
    report: dict[str, Any] = {"people": {}}
    for p, pid in enumerate(tracks.ids):
        D = lifted.get(pid, {})
        obs = sorted(i for (q, i) in match if q == pid)
        for i, d in D.items():
            x = np.full((N_JOINTS, 3), np.nan, np.float32)
            keep = d["vis"] & np.isfinite(d["Xs"]).all(1)
            x[keep] = d["Xs"][keep]
            out.xyz[p, i] = x
            out.score[p, i] = np.where(keep, cfg.score, 0.0)
            out.support[p, i] = -1
            for v_name, r in zip(d["cons_views"], [d["det"]] + d["dets"][1:]):
                out.support[p, i, tracks.cameras.index(v_name)] = r
            out.single_view[p, i] = True
            out.observed[p, i] = True
        for i in obs:
            out.observed[p, i] = True
        missing = np.flatnonzero(~models[pid].valid)
        spans = []
        for s0, s1 in spans_of(missing.tolist()):
            fr = range(s0, s1 + 1)
            o = [i for i in fr if (pid, i) in match]
            if not o:
                continue
            spans.append(
                {
                    "t0_s": s0 / tracks.fps,
                    "t1_s": (s1 + 1) / tracks.fps,
                    "frames": s1 - s0 + 1,
                    "observed": len(o),
                    "lifted": sum(i in D for i in fr),
                    "rejected": sum(i in rej.get(pid, {}) for i in fr),
                }
            )
        report["people"][pid] = {
            "lifted_frames": sorted(int(i) for i in D),
            "observed_frames": [int(i) for i in obs],
            "rejected": {str(k): v for k, v in rej.get(pid, {}).items()},
            "frames": {str(i): {k: v for k, v in d.items() if k not in ("X", "Xs", "vis")} for i, d in D.items()},
            "spans": spans,
            "ankle_h": models[pid].ankle_h, "pelvis_stand": models[pid].pelvis_stand, "seglen": models[pid].seglen,
        }  # fmt: skip
        log.info(
            "%s: %d frames observed by one camera, %d lifted (%d multi), %d rejected",
            pid,
            len(obs),
            len(D),
            sum(d["mode"] == "multi" for d in D.values()),
            len(rej.get(pid, {})),
        )
    return out, report


def rel_depth_from_keypoints(
    K3: np.ndarray, torso_m: float, halpe_to_kp: dict[int, int], hips: tuple[int, int], shoulders: tuple[int, int]
) -> dict[int, float]:
    """Camera-frame keypoints (K, 3) of a single-view estimate -> {halpe index: depth offset to the hip centre},
    scaled to the person's torso length."""
    hipc = 0.5 * (K3[hips[0]] + K3[hips[1]])
    tor = np.linalg.norm(0.5 * (K3[shoulders[0]] + K3[shoulders[1]]) - hipc)
    sc = torso_m / tor if np.isfinite(torso_m) and tor > 0.05 else 1.0
    return {h: float(sc * (K3[k][2] - hipc[2])) for h, k in halpe_to_kp.items()}
