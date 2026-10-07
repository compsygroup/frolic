"""Synthetic multi-camera scenes (stick-figure people) for tests and the example configuration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from frolic3d.geometry import Camera
from frolic3d.io.calib import CameraSet, save_cameras
from frolic3d.io.detections import N_KP, Detections
from frolic3d.io.video import write_video
from frolic3d.skeleton import HALPE26, N_JOINTS, J

# segment lengths as a fraction of stature (rough anthropometry)
PROPORTIONS = {
    "hip_half": 0.05,
    "thigh": 0.245,
    "shin": 0.246,
    "foot": 0.10,
    "trunk": 0.30,
    "head": 0.13,
    "shoulder_half": 0.11,
    "upperarm": 0.17,
    "forearm": 0.16,
}


def ring_cameras(
    n: int = 4,
    radius: float = 4.0,
    height: float = 2.2,
    center=(0.0, 0.0, 0.8),
    width: int = 640,
    height_px: int = 360,
    fov_deg: float = 60.0,
    dist=None,
) -> CameraSet:
    """n cameras on a circle looking at `center` (world z up, floor z = 0)."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height_px / 2], [0, 0, 1.0]])
    cams = []
    for k in range(n):
        th = 2 * np.pi * k / n + 0.3
        C = np.array([center[0] + radius * np.cos(th), center[1] + radius * np.sin(th), height])
        z = np.asarray(center, float) - C
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])  # rows = camera axes in world coordinates (x right, y down, z forward)
        cams.append(
            Camera(f"cam{k + 1}", width, height_px, K, np.zeros(5) if dist is None else np.asarray(dist), R, -R @ C)
        )
    return CameraSet(cams, "m", {"floor_z": 0.0, "levelled": True}, {"format": "synthetic"})


def pose_at(stature: float, t: float, pos_xy, heading: float, gait_hz: float = 1.0, sit: bool = False) -> np.ndarray:
    """A walking (or seated) stick figure (17, 3) in the world frame, feet on z = 0."""
    p = {k: v * stature for k, v in PROPORTIONS.items()}
    fwd = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up = np.array([0, 0, 1.0])
    ph = 2 * np.pi * gait_hz * t
    X = np.zeros((N_JOINTS, 3))
    leg = p["thigh"] + p["shin"]
    if sit:
        pelvis = np.array([pos_xy[0], pos_xy[1], 0.45 * stature])
        X[J["pelvis"]] = pelvis
        for side, sgn in (("l", 1), ("r", -1)):
            hip = pelvis + sgn * p["hip_half"] * left
            knee = hip + p["thigh"] * fwd - 0.02 * up
            ankle = knee - p["shin"] * up
            X[J[f"hip_{side}"]], X[J[f"knee_{side}"]], X[J[f"ankle_{side}"]] = hip, knee, ankle
            X[J[f"toe_{side}"]] = ankle + p["foot"] * fwd - 0.03 * up * stature
        swing = {"l": 0.0, "r": 0.0}
    else:
        bob = 0.015 * stature * np.cos(2 * ph)
        pelvis = np.array([pos_xy[0], pos_xy[1], leg + p["hip_half"] * 0.5 + bob])
        X[J["pelvis"]] = pelvis
        swing = {"l": 0.35 * np.sin(ph), "r": -0.35 * np.sin(ph)}
        for side, sgn in (("l", 1), ("r", -1)):
            hip = pelvis + sgn * p["hip_half"] * left
            a = swing[side]
            knee_bend = 0.4 * max(0.0, -np.sin(ph if side == "l" else ph + np.pi))
            knee = hip + p["thigh"] * (np.sin(a) * fwd - np.cos(a) * up)
            b = a - knee_bend
            ankle = knee + p["shin"] * (np.sin(b) * fwd - np.cos(b) * up)
            ankle[2] = max(ankle[2], 0.05 * stature)
            X[J[f"hip_{side}"]], X[J[f"knee_{side}"]], X[J[f"ankle_{side}"]] = hip, knee, ankle
            X[J[f"toe_{side}"]] = ankle + p["foot"] * fwd - 0.04 * stature * up
            X[J[f"toe_{side}"], 2] = max(X[J[f"toe_{side}"], 2], 0.0)
    neck = pelvis + p["trunk"] * up
    X[J["neck"]] = neck
    X[J["head"]] = neck + p["head"] * up
    for side, sgn in (("l", 1), ("r", -1)):
        sh = neck + sgn * p["shoulder_half"] * left
        a = -swing[side] * 0.8
        elbow = sh + p["upperarm"] * (np.sin(a) * fwd - np.cos(a) * up)
        wrist = elbow + p["forearm"] * (np.sin(a + 0.4) * fwd - np.cos(a + 0.4) * up)
        X[J[f"shoulder_{side}"]], X[J[f"elbow_{side}"]], X[J[f"wrist_{side}"]] = sh, elbow, wrist
    return X


def person_track(
    n_frames: int,
    fps: float,
    stature: float,
    start_xy,
    end_xy,
    heading: float | None = None,
    sit: bool = False,
    gait_hz: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """(N, 17, 3) joints of one person walking from start to end (or sitting at start)."""
    start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
    d = end - start
    h = heading if heading is not None else (np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0)
    out = np.zeros((n_frames, N_JOINTS, 3))
    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        out[i] = pose_at(stature, i / fps + phase, start + u * d, h, gait_hz, sit)
    return out


def halpe26_from_joints(X: np.ndarray) -> np.ndarray:
    """(…, 17, 3) joints -> (…, 26, 3) Halpe-26 points (face/foot points derived from head and toes)."""
    H = np.zeros(X.shape[:-2] + (N_KP, 3))
    for j, h in enumerate(HALPE26):
        H[..., h, :] = X[..., j, :]
    head, neck = X[..., J["head"], :], X[..., J["neck"], :]
    up = head - neck
    for h, off in ((0, 0.55), (1, 0.62), (2, 0.62), (3, 0.5), (4, 0.5)):
        H[..., h, :] = neck + off * up
    for h in (1, 3):
        H[..., h, 0] -= 0.03
    for h in (2, 4):
        H[..., h, 0] += 0.03
    for side, toe, small, heel in (("l", 20, 22, 24), ("r", 21, 23, 25)):
        ank, to = X[..., J[f"ankle_{side}"], :], X[..., J[f"toe_{side}"], :]
        H[..., toe, :] = to
        H[..., small, :] = to + 0.3 * (ank - to)
        H[..., heel, :] = ank - 0.4 * (to - ank)
        H[..., heel, 2] = np.minimum(H[..., heel, 2], ank[..., 2])
    H[..., 23, :] = X[..., J["toe_r"], :]  # the layout maps toe_r to halpe 23
    H[..., 20, :] = X[..., J["toe_l"], :]
    return H


def project_people(
    cams: CameraSet,
    people: list[np.ndarray],
    noise_px: float = 0.0,
    occlude: dict[str, list[tuple[int, int, int]]] | None = None,
    seed: int = 0,
) -> dict[str, Detections]:
    """Ground-truth Halpe-26 detections per camera; occlude: camera -> [(person, start, stop)] hidden spans."""
    rng = np.random.default_rng(seed)
    out = {}
    n_frames = people[0].shape[0]
    for cam in cams:
        rows = []
        for i in range(n_frames):
            for p, X in enumerate(people):
                if occlude and any(pp == p and a <= i < b for pp, a, b in occlude.get(cam.name, [])):
                    continue
                H = halpe26_from_joints(X[i])
                uv, z = cam.project(H, distort=True)
                if (z <= 0.1).any():
                    continue
                vis = cam.in_image(uv, margin=1)
                if vis.sum() < 8:
                    continue
                uv = uv + rng.normal(0, noise_px, uv.shape) if noise_px > 0 else uv
                box = np.r_[uv[vis].min(0) - 5, uv[vis].max(0) + 5]
                score = np.where(vis, 0.9, 0.1).astype(np.float32)
                rows.append((i, box, 0.95, uv, score, np.array([20 + 40 * p, 150, 120 + 30 * p], np.float32)))
        if rows:
            det = Detections(
                cam.name,
                [r[0] for r in rows],
                [r[1] for r in rows],
                [r[2] for r in rows],
                [r[3] for r in rows],
                [r[4] for r in rows],
                [r[5] for r in rows],
            )
        else:
            det = Detections.empty(cam.name)
        det.meta = {"width": cam.width, "height": cam.height, "fps": 0.0, "step": 1, "n_frames": n_frames, "video": ""}
        out[cam.name] = det
    return out


_BONES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (0, 9),
    (9, 10),
    (9, 11),
    (9, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


def draw_person(img: np.ndarray, cam: Camera, X: np.ndarray, colour: tuple[int, int, int], stature: float) -> None:
    uv, z = cam.project(X, distort=True)
    if (z <= 0.1).any():
        return
    px_per_m = cam.focal / max(float(np.median(z)), 0.1)
    w = max(2, int(0.06 * stature * px_per_m))
    for a, b in _BONES:
        cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), colour, w, cv2.LINE_AA)
    cv2.circle(
        img, tuple(np.round(uv[J["head"]]).astype(int)), max(3, int(0.07 * stature * px_per_m)), colour, -1, cv2.LINE_AA
    )


def render_videos(
    cams: CameraSet, people: list[np.ndarray], statures: list[float], out_dir: Path, fps: float, crf: int = 23
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    colours = [(60, 90, 220), (40, 170, 90), (200, 120, 40), (150, 60, 170), (90, 200, 200)]
    for cam in cams:
        path = out_dir / f"{cam.name}.mp4"
        proc = write_video(path, (cam.width, cam.height), fps, crf)
        for i in range(people[0].shape[0]):
            img = np.full((cam.height, cam.width, 3), 235, np.uint8)
            for p, X in enumerate(people):
                draw_person(img, cam, X[i], colours[p % len(colours)], statures[p])
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait()
        paths[cam.name] = path
    return paths


def make_scene(
    out_dir: Path,
    n_cams: int = 4,
    n_frames: int = 60,
    fps: float = 15.0,
    people: list[dict] | None = None,
    videos: bool = True,
    noise_px: float = 0.5,
    seed: int = 0,
) -> dict:
    """Write cameras.json, ground-truth detections (gt/<cam>.npz), joints (gt/joints.npz) and optionally videos."""
    out_dir = Path(out_dir)
    cams = ring_cameras(n_cams)
    people = people or [
        {"stature": 1.70, "start": (-1.0, -0.5), "end": (1.0, 0.5)},
        {"stature": 0.95, "start": (0.8, -0.8), "end": (-0.6, 0.6), "phase": 0.3},
    ]
    tracks = [
        person_track(
            n_frames,
            fps,
            p["stature"],
            p["start"],
            p["end"],
            p.get("heading"),
            p.get("sit", False),
            phase=p.get("phase", 0.0),
        )
        for p in people
    ]
    save_cameras(cams, out_dir / "calib" / "cameras.json")
    dets = project_people(cams, tracks, noise_px=noise_px, seed=seed)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    for name, d in dets.items():
        d.meta["fps"] = fps
        d.save(out_dir / "gt" / f"{name}.npz")
    np.savez_compressed(
        out_dir / "gt" / "joints.npz",
        xyz=np.stack(tracks).astype(np.float32),
        statures=np.array([p["stature"] for p in people]),
        fps=fps,
    )
    info = {"n_cams": n_cams, "n_frames": n_frames, "fps": fps, "people": people, "cameras": cams.names}
    if videos:
        paths = render_videos(cams, tracks, [p["stature"] for p in people], out_dir / "videos", fps)
        info["videos"] = {k: str(v) for k, v in paths.items()}
    with open(out_dir / "scene.json", "w") as fh:
        json.dump(info, fh, indent=1)
    return info
