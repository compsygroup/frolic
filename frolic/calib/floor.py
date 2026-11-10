"""Floor alignment (z shift) and levelling (rigid rotation onto the standing-feet plane) of tracks and cameras."""

from __future__ import annotations

from typing import Any

import numpy as np

from frolic3d.geometry import level_transform, plane_from_points
from frolic3d.io.calib import CameraSet
from frolic3d.io.tracks import Tracks
from frolic3d.log import get_logger
from frolic3d.skeleton import J

log = get_logger(__name__)

FEET = [J["ankle_l"], J["ankle_r"], J["toe_l"], J["toe_r"]]
MIN_POINTS = 30


def apply_rigid(tracks: Tracks, R: np.ndarray, t: np.ndarray) -> Tracks:
    """X' = R X + t on every finite joint (copy)."""
    out = tracks.subset(tracks.ids)
    ok = np.isfinite(out.xyz).all(-1)
    xyz = out.xyz.astype(np.float64)
    xyz[ok] = xyz[ok] @ R.T + t
    out.xyz = xyz.astype(np.float32)
    return out


def _translate(tracks: Tracks, cams: CameraSet, dz: float) -> tuple[Tracks, CameraSet]:
    t = np.array([0.0, 0.0, dz])
    return apply_rigid(tracks, np.eye(3), t), cams.transformed(np.eye(3), t)


def floor_align(
    tracks: Tracks, cams: CameraSet, target_ankle_z: float = 0.05, min_score: float = 0.3
) -> tuple[Tracks, CameraSet, dict[str, Any]]:
    """Pure z shift so the pooled 10th percentile of foot heights sits at `target_ankle_z` (feet-on-floor invariant)."""
    z = tracks.xyz[:, :, FEET, 2]
    s = tracks.score[:, :, FEET]
    fz = z[np.isfinite(z) & (s > min_score)]
    if len(fz) < 5:
        log.warning("floor_align: only %d foot points, no shift", len(fz))
        return tracks, cams, {"z_shift_m": 0.0, "n_points": len(fz)}
    med = np.median(fz)
    fz = fz[np.abs(fz - med) < 1.0]
    shift = float(target_ankle_z - np.percentile(fz, 10))
    log.info("floor_align: feet p10 %.3f m -> shifting z by %+.3f m", target_ankle_z - shift, shift)
    tr, cs = _translate(tracks, cams, shift)
    return tr, cs, {"z_shift_m": shift, "n_points": len(fz)}


def standing_feet(tracks: Tracks, speed_max: float = 0.3) -> np.ndarray:
    """Lower-ankle positions on frames where a person stands still with straight, vertical legs."""
    X, S, fps = tracks.xyz.astype(np.float64), tracks.score, tracks.fps
    need = [J["pelvis"], J["hip_l"], J["hip_r"], J["knee_l"], J["knee_r"], J["ankle_l"], J["ankle_r"]]
    st = (S[:, :, need] > 0).all(-1) & np.isfinite(X[:, :, need]).all((-1, -2))
    for h, k, a in ((J["hip_l"], J["knee_l"], J["ankle_l"]), (J["hip_r"], J["knee_r"], J["ankle_r"])):
        th = np.linalg.norm(X[:, :, h] - X[:, :, k], axis=-1)
        sh = np.linalg.norm(X[:, :, k] - X[:, :, a], axis=-1)
        ha = np.linalg.norm(X[:, :, h] - X[:, :, a], axis=-1)
        st &= (ha > 0.95 * (th + sh)) & ((X[:, :, h, 2] - X[:, :, a, 2]) > 0.9 * ha)
    v = np.full(st.shape, np.inf)
    v[:, 1:] = np.linalg.norm(np.diff(X[:, :, J["pelvis"]], axis=1), axis=-1) * fps
    st &= v < speed_max
    lo = np.where(X[:, :, J["ankle_l"], 2] < X[:, :, J["ankle_r"], 2], J["ankle_l"], J["ankle_r"])
    p, f = np.nonzero(st)
    return X[p, f, lo[p, f]]


def stance_ankles(tracks: Tracks, speed_max: float = 0.25) -> np.ndarray:
    """Ankles that are the lower of the two and nearly static (stance phase of walking)."""
    X, S, fps = tracks.xyz.astype(np.float64), tracks.score, tracks.fps
    out = []
    for a, b in ((J["ankle_l"], J["ankle_r"]), (J["ankle_r"], J["ankle_l"])):
        ok = (S[:, :, a] > 0) & np.isfinite(X[:, :, a]).all(-1)
        v = np.full(ok.shape, np.inf)
        if X.shape[1] > 2:
            v[:, 1:-1] = np.linalg.norm(X[:, 2:, a] - X[:, :-2, a], axis=-1) * fps / 2
        both = ok & (S[:, :, b] > 0) & np.isfinite(X[:, :, b]).all(-1)
        lower = ~both | (X[:, :, a, 2] <= X[:, :, b, 2])
        sel = ok & (v < speed_max) & lower
        out.append(X[sel][:, a])
    return np.concatenate(out) if out else np.zeros((0, 3))


def plane_spread(F: np.ndarray, keep: np.ndarray) -> float:
    """Smallest horizontal std (m) of the inlier points: a plane through points on a line is free to tilt across it."""
    Q = F[keep, :2]
    if len(Q) < 3:
        return 0.0
    return float(np.sqrt(max(np.linalg.eigvalsh(np.cov(Q.T)).min(), 0.0)))


def given_floor_shift(
    tracks: Tracks, cams: CameraSet, ankle_height_m: float = 0.07
) -> tuple[Tracks, CameraSet, dict[str, Any]]:
    """Z shift for an already levelled calibration: the median standing-feet (else stance) ankle goes to `ankle_height_m`.

    Same height rule as the full levelling. floor_align (p10 of all ankle + toe points of everyone) sits a few cm too
    high because toes are lower than ankles, and the full levelling overwrites that; a given calibration has to redo it
    (Untitled 6 take 03: +8.8 cm, the subject's stance ankles at 0.17 m instead of 0.07 m, 2026-11-03).
    """
    F, src = standing_feet(tracks), "standing"
    if len(F) < MIN_POINTS:
        F, src = stance_ankles(tracks), "stance_ankles"
    if len(F) < MIN_POINTS:
        log.warning("given floor: %d standing/stance ankles, falling back to floor_align", len(F))
        tr, cs, info = floor_align(tracks, cams, ankle_height_m)
        info["height_source"] = "floor_align"
        return tr, cs, info
    z = F[:, 2]
    z = z[np.abs(z - np.median(z)) < 0.15]
    shift = float(ankle_height_m - np.median(z))
    log.info("given floor: %d %s ankles, median z %.3f m -> shifting z by %+.3f m", len(z), src, ankle_height_m - shift, shift)
    tr, cs = _translate(tracks, cams, shift)
    return tr, cs, {"z_shift_m": shift, "n_points": int(len(z)), "height_source": src}


def level_floor(
    tracks: Tracks, cams: CameraSet, ankle_height_m: float = 0.07, min_spread_m: float = 0.25
) -> tuple[Tracks, CameraSet, dict[str, Any]]:
    """Rotate the world so the standing-feet plane is z = 0 (ankles at `ankle_height_m`); cameras move with it.

    A calibration that level_floor already levelled (static cameras, another clip of the same set: `tilt_deg` in its
    floor record) is only z-shifted. The plane
    needs `min_spread_m` of horizontal spread across its narrowest direction: standing feet on a walkway line do not
    constrain the tilt across the line; stance ankles are tried next, then the orientation is kept.
    """
    if cams.floor.get("levelled") and "tilt_deg" in cams.floor:
        log.info("level_floor: the given calibration was levelled on another clip (tilt %.2f deg), z shift only", cams.floor["tilt_deg"])
        tr, cs, info = given_floor_shift(tracks, cams, ankle_height_m)
        info.update(source="given", tilt_deg=0.0, R=np.eye(3).tolist(), t=[0.0, 0.0, info["z_shift_m"]])
        return tr, cs, info
    F, source = standing_feet(tracks), "standing"
    c, keep = (plane_from_points(F) if len(F) >= MIN_POINTS else (np.zeros(3), np.zeros(len(F), bool)))
    spread = plane_spread(F, keep)
    if len(F) < MIN_POINTS or spread < min_spread_m:
        G = stance_ankles(tracks)
        if len(G) >= MIN_POINTS:
            cg, kg = plane_from_points(G)
            sg = plane_spread(G, kg)
            if sg >= min_spread_m or (spread < min_spread_m and sg > spread):
                log.info("level_floor: standing feet %d points, spread %.2f m: using stance ankles", len(F), spread)
                F, source, c, keep, spread = G, "stance_ankles", cg, kg, sg
    if len(F) < MIN_POINTS or spread < min_spread_m:
        log.warning(
            "level_floor: %d %s points with %.2f m spread (< %.2f m), keeping the orientation (z shift only)",
            len(F),
            source,
            spread,
            min_spread_m,
        )
        tr, cs, info = floor_align(tracks, cams, ankle_height_m)
        info.update(source="none", tilt_deg=0.0, R=np.eye(3).tolist(), t=[0.0, 0.0, info["z_shift_m"]])
        cs.floor.update({"levelled": False, "floor_z": 0.0})
        return tr, cs, info
    R, t, tilt = level_transform(c, np.median(F[keep, :2], 0), ankle_height_m)
    res = F[keep] @ R.T + t
    log.info(
        "level_floor: %d %s points (kept %d, spread %.2f m), tilt %.2f deg, ankle z p10/p50/p90 %.3f/%.3f/%.3f",
        len(F),
        source,
        keep.sum(),
        spread,
        tilt,
        *np.percentile(res[:, 2], [10, 50, 90]),
    )
    tr = apply_rigid(tracks, R, t)
    cs = cams.transformed(R, t)
    cs.floor.update({"levelled": True, "floor_z": 0.0, "tilt_deg": tilt})
    return (
        tr,
        cs,
        {
            "source": source,
            "n_points": len(F),
            "n_inliers": int(keep.sum()),
            "spread_m": spread,
            "tilt_deg": tilt,
            "R": R.tolist(),
            "t": t.tolist(),
        },
    )
