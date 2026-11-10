"""Footprints on the floor: one flat mark where a foot lands, fading out over `footprints_s` seconds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frolic3d.skeleton import J


@dataclass
class Footstep:
    frame: int  # track frame the foot lands on
    xy: np.ndarray  # foot centre on the floor
    heading: np.ndarray  # unit vector ankle -> toe
    length: float  # foot length (m)
    side: str


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def footsteps(
    joints: np.ndarray,
    fps: float,
    toe_max_z: float = 0.12,  # fitted toe/ankle joint centres sit 3-10 cm above the floor in stance
    ankle_max_z: float = 0.20,
    speed_max: float = 0.5,
    min_s: float = 0.13,
) -> list[Footstep]:
    """Stance phases of each foot: on the floor (toe/ankle low) and nearly static for `min_s`, one footstep per phase.

    Thresholds set on the DBP gait pilot (15 Hz fitted joints): a running 3-year-old (1.5 m/s, ~3 steps/s) keeps each
    foot still for only ~0.15 s and below 0.5 m/s, while walkers are unchanged by them (T08: 88 contacts either way).

    A standing person gets one mark per foot when the stance begins; a seated person with raised feet gets none.
    """
    out: list[Footstep] = []
    n = len(joints)
    for side in ("l", "r"):
        A, T = joints[:, J[f"ankle_{side}"]].astype(np.float64), joints[:, J[f"toe_{side}"]].astype(np.float64)
        C = (A + T) / 2
        ok = np.isfinite(A).all(-1) & np.isfinite(T).all(-1)
        v = np.full(n, np.inf)
        if n > 2:
            v[1:-1] = np.linalg.norm(C[2:, :2] - C[:-2, :2], axis=-1) * fps / 2
        ground = ok & (T[:, 2] < toe_max_z) & (A[:, 2] < ankle_max_z)
        stance = ground & (v < speed_max)
        prev: Footstep | None = None
        for s, e in _runs(stance, max(2, round(min_s * fps))):
            xy = C[s:e, :2].mean(0)
            d = (T[s:e, :2] - A[s:e, :2]).mean(0)
            if np.linalg.norm(d) < 1e-3:
                d = np.array([1.0, 0.0])
            d = d / np.linalg.norm(d)
            length = float(np.clip(1.6 * np.median(np.linalg.norm(T[s:e] - A[s:e], axis=-1)), 0.10, 0.32))
            if prev is not None and np.linalg.norm(xy - prev.xy) < 0.08 and (s - prev.frame) / fps < 0.6:
                continue  # the same stance split by a short glitch
            prev = Footstep(int(s), xy, d, length, side)
            out.append(prev)
    out.sort(key=lambda f: f.frame)
    return out


# Bare-foot sole outline in foot units: x along the foot (0 = heel, 1 = toe tip), y = medial (big-toe) side positive.
_HEEL = [(0.13 - 0.13 * np.cos(a), 0.13 * np.sin(a)) for a in np.linspace(-np.pi / 2, np.pi / 2, 9)]  # lateral->medial
_SOLE = np.array(
    [*reversed(_HEEL), (0.30, -0.16), (0.50, -0.18), (0.70, -0.20), (0.80, -0.18), (0.84, -0.08), (0.84, 0.10)]
    + [(0.78, 0.18), (0.62, 0.16), (0.48, 0.09), (0.30, 0.11)]
)
_TOES = [((0.92, 0.11), 0.065), ((0.95, 0.00), 0.045), ((0.94, -0.08), 0.04), ((0.92, -0.15), 0.035), ((0.88, -0.21), 0.03)]


def _fan(pts: np.ndarray, z: float, rgb: tuple[int, int, int], alpha: float):
    """Flat star-shaped polygon triangulated from its centroid, with one RGBA for every vertex."""
    import trimesh

    c = len(pts)
    V = np.c_[np.r_[pts, pts.mean(0, keepdims=True)], np.full(c + 1, z)]
    x, y = pts[:, 0], pts[:, 1]
    ccw = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0  # a mirrored (left) foot would face down
    F = np.array([[c, k, (k + 1) % c] if ccw else [c, (k + 1) % c, k] for k in range(c)])
    m = trimesh.Trimesh(V, F, process=False)
    m.visual.vertex_colors = np.tile(np.array([*rgb, int(255 * alpha)], np.uint8), (len(V), 1))
    return m


def footprint_mesh(
    steps: list[Footstep],
    t: float,
    fade_frames: float,
    rgb: tuple[int, int, int],
    z: float = 0.006,
    hold_frames: float = 0.0,
):
    """Bare-foot prints (sole + five toes, left/right mirrored, the left foot a darker shade) for the footsteps
    landed before `t`: full alpha for `hold_frames`, then fading to 0 at `fade_frames`; None if none is visible."""
    import pyrender
    import trimesh

    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    disc = np.c_[np.cos(ang), np.sin(ang)]
    hold = min(hold_frames, fade_frames)
    parts = []
    for f in steps:
        age = t - f.frame
        if age < 0 or age >= fade_frames:
            continue
        a = 1.0 if age < hold else 1.0 - (age - hold) / max(fade_frames - hold, 1e-6)
        rgb_s = rgb if f.side == "r" else tuple(int(c * 0.5) for c in rgb)
        h = f.heading
        n = np.array([-h[1], h[0]]) * (1.0 if f.side == "r" else -1.0)  # medial side: left of heading for the right foot
        heel = f.xy - h * f.length / 2

        def place(uv: np.ndarray) -> np.ndarray:
            return heel + uv[:, :1] * h * f.length + uv[:, 1:] * n * f.length

        parts.append(_fan(place(_SOLE), z, rgb_s, a))
        for (cx, cy), r in _TOES:
            parts.append(_fan(place(np.array([cx, cy]) + disc * r), z, rgb_s, a))
    if not parts:
        return None
    return pyrender.Mesh.from_trimesh(trimesh.util.concatenate(parts), smooth=False)
