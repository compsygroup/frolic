"""Shared rendering pieces: export loading, mesh interpolation, floor and frustum geometry, cameras, text."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frolic3d.config import RenderConfig
from frolic3d.geometry import Camera
from frolic3d.io.calib import CameraSet, load_cameras

PALETTE: list[tuple[int, int, int]] = [
    (236, 116, 36), (48, 104, 204), (46, 160, 98), (160, 78, 204),
    (220, 60, 90), (40, 170, 190), (190, 160, 30), (120, 120, 120),
]  # fmt: skip
CAMERA_RGB = (0.16, 0.18, 0.22)


@dataclass
class Person:
    pid: str
    label: str
    vertices: np.ndarray  # (N, V, 3) float32, NaN rows hidden
    faces: np.ndarray
    joints: np.ndarray
    colour: tuple[int, int, int]
    stature_m: float | None = None  # fitted standing height, shown in the legend

    @property
    def legend(self) -> str:
        return f"{self.label}  {self.stature_m:.2f} m" if self.stature_m else self.label

    def visible(self, i: int) -> bool:
        return 0 <= i < len(self.vertices) and bool(np.isfinite(self.vertices[i, 0, 0]))


@dataclass
class Scene:
    people: list[Person]
    cams: CameraSet
    fps: float
    n_frames: int
    source_frames: np.ndarray
    videos: dict[str, Any] = field(default_factory=dict)

    @property
    def floor_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Floor grid extent: the people's paths with a margin (cameras stay outside, as frustums)."""
        pts = []
        for p in self.people:
            c = np.nanmean(p.vertices[:, ::40, :2], axis=1)
            pts.append(c[np.isfinite(c[:, 0])])
        allp = np.vstack([q for q in pts if len(q)]) if any(len(q) for q in pts) else self.cams.centers[:, :2]
        lo = np.floor((allp.min(0) - 1.0) * 2) / 2
        hi = np.ceil((allp.max(0) + 1.0) * 2) / 2
        return lo, hi

    @property
    def people_center(self) -> np.ndarray:
        cs = []
        for p in self.people:
            c = np.nanmean(p.vertices[:, ::40], axis=1)
            cs.append(c[np.isfinite(c[:, 0])])
        allc = np.vstack([c for c in cs if len(c)]) if any(len(c) for c in cs) else np.zeros((1, 3))
        return np.r_[np.median(allc[:, :2], 0), 0.6]


def load_scene(export_dir: Path, videos_json: Path | None, colours: dict[str, list[int]] | None = None) -> Scene:
    with open(export_dir / "meta.json") as fh:
        meta = json.load(fh)
    cams = load_cameras(export_dir / "cameras.json")
    people = []
    src = None
    for k, info in enumerate(meta["people"]):
        pid = info["id"]
        z = np.load(export_dir / f"{pid}.npz", allow_pickle=False)
        col = tuple(int(c) for c in (colours or {}).get(pid, PALETTE[k % len(PALETTE)]))
        people.append(
            Person(
                pid,
                str(z["label"]),
                np.asarray(z["vertices"], np.float32),
                np.asarray(z["faces"]),
                np.asarray(z["joints"], np.float32),
                col,
                float(info["stature_m"]) if info.get("stature_m") else None,
            )
        )
        if src is None:
            src = np.asarray(z["source_frames"], int)
    videos = {}
    if videos_json is not None and Path(videos_json).exists():
        with open(videos_json) as fh:
            videos = json.load(fh)
    n = max(len(p.vertices) for p in people)
    return Scene(people, cams, float(meta["fps"]), n, src if src is not None else np.arange(n), videos)


def catmull_rom(X: np.ndarray, t: float) -> np.ndarray | None:
    """C1 interpolation of a frame sequence at fractional index t; None where the frame is hidden."""
    i = int(np.floor(t))
    u = t - i
    if i < 0 or i >= len(X):
        return None
    P1 = X[i]
    if not np.isfinite(P1.flat[0]):
        return None
    P2 = X[min(i + 1, len(X) - 1)]
    if u == 0 or not np.isfinite(P2.flat[0]):
        return P1
    P0 = X[i - 1] if i - 1 >= 0 and np.isfinite(X[i - 1].flat[0]) else 2 * P1 - P2
    P3 = X[i + 2] if i + 2 < len(X) and np.isfinite(X[i + 2].flat[0]) else 2 * P2 - P1
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * P1) + (-P0 + P2) * u + (2 * P0 - 5 * P1 + 4 * P2 - P3) * u2 + (-P0 + 3 * P1 - 3 * P2 + P3) * u3)


def ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.eye(4)
    P[:3, :3] = np.stack([x, y, z], 1)
    P[:3, 3] = eye
    return P


def camera_pose(cam: Camera) -> np.ndarray:
    """pyrender camera pose (OpenGL convention) of a calibrated camera."""
    T = np.eye(4)
    T[:3, :3] = cam.R.T
    T[:3, 3] = cam.center
    return T @ np.diag([1, -1, -1, 1])


def material(rgb: tuple[int, int, int], alpha: float = 1.0, rough: float = 0.55):
    pyrender = __import__("pyrender")
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*(np.array(rgb) / 255.0), alpha],
        metallicFactor=0.0,
        roughnessFactor=rough,
        alphaMode="BLEND" if alpha < 1 else "OPAQUE",
    )


def floor_meshes(lo: np.ndarray, hi: np.ndarray, step: float = 0.5) -> list:
    import pyrender
    import trimesh

    fv = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
    ff = np.array([[0, 1, 2], [0, 2, 3]])
    out = [
        pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(fv, ff, process=False),
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.78, 0.79, 0.81, 1], metallicFactor=0, roughnessFactor=0.95
            ),
        )
    ]
    boxes = []
    for x in np.arange(lo[0], hi[0] + 1e-6, step):
        w = 0.012 if abs(x - round(x)) < 1e-6 else 0.006
        b = trimesh.creation.box([w, hi[1] - lo[1], 0.002])
        b.apply_translation([x, (lo[1] + hi[1]) / 2, 0.001])
        boxes.append(b)
    for y in np.arange(lo[1], hi[1] + 1e-6, step):
        w = 0.012 if abs(y - round(y)) < 1e-6 else 0.006
        b = trimesh.creation.box([hi[0] - lo[0], w, 0.002])
        b.apply_translation([(lo[0] + hi[0]) / 2, y, 0.001])
        boxes.append(b)
    out.append(
        pyrender.Mesh.from_trimesh(
            trimesh.util.concatenate(boxes),
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.5, 0.52, 0.56, 1], metallicFactor=0, roughnessFactor=1
            ),
        )
    )
    return out


def frustum_mesh(cam: Camera, depth: float = 0.3, radius: float = 0.008):
    import pyrender
    import trimesh

    c = cam.center
    Ki = np.linalg.inv(cam.K)
    corners = [
        c + cam.R.T @ (Ki @ np.array([x, y, 1.0]) * depth)
        for x, y in [(0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)]
    ]
    segs = [(c, q) for q in corners] + [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    ms = [trimesh.creation.cylinder(radius=radius, segment=[p, q], sections=8) for p, q in segs]
    body = trimesh.creation.box([depth * 0.3, depth * 0.22, depth * 0.22])
    T = np.eye(4)
    T[:3, :3] = cam.R.T
    T[:3, 3] = c - cam.R.T @ np.array([0, 0, depth * 0.12])
    body.apply_transform(T)
    ms.append(body)
    return pyrender.Mesh.from_trimesh(
        trimesh.util.concatenate(ms),
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*CAMERA_RGB, 1], metallicFactor=0.2, roughnessFactor=0.5
        ),
    )


def body_mesh(V: np.ndarray, F: np.ndarray, rgb: tuple[int, int, int], alpha: float = 1.0):
    import pyrender
    import trimesh

    return pyrender.Mesh.from_trimesh(trimesh.Trimesh(V, F, process=False), material=material(rgb, alpha), smooth=True)


def shadow_mesh(V: np.ndarray, F: np.ndarray):
    import pyrender
    import trimesh

    S = V.copy()
    S[:, 2] = 0.003
    return pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(S, F, process=False), material=material((0, 0, 0), 0.28, 1.0), smooth=False
    )


def font_path() -> str:
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


def draw_text(img_bgr: np.ndarray, items: list[tuple[int, int, str, int, tuple[int, int, int]]]) -> np.ndarray:
    """items: (x, y, text, size, rgb) drawn with a dark outline onto a BGR image."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img_bgr[..., ::-1])
    d = ImageDraw.Draw(im)
    for x, y, s, size, rgb in items:
        f = ImageFont.truetype(font_path(), size)
        d.text((x + 1, y + 1), s, font=f, fill=(0, 0, 0))
        d.text((x, y), s, font=f, fill=tuple(int(v) for v in rgb))
    return np.asarray(im)[..., ::-1].copy()


def draw_legend(img_bgr: np.ndarray, people: list[Person], size: int = 18, x: int = 12, y: int = 12) -> np.ndarray:
    import cv2

    out = img_bgr.copy()
    items = []
    for k, p in enumerate(people):
        yy = y + k * int(size * 1.6)
        cv2.rectangle(out, (x, yy + 2), (x + size - 2, yy + size - 2), tuple(int(c) for c in p.colour[::-1]), -1)
        items.append((x + size + 6, yy - 2, p.legend, size, (245, 245, 245)))
    return draw_text(out, items)


def lights(pyrender, pose: np.ndarray, fill_pose: np.ndarray | None = None) -> list:
    out = [(pyrender.DirectionalLight(intensity=2.6), pose)]
    if fill_pose is not None:
        out.append((pyrender.DirectionalLight(intensity=1.2), fill_pose))
    return out


def render_window(scene: Scene, cfg: RenderConfig) -> tuple[int, int]:
    i0 = round(cfg.start_s * scene.fps)
    i1 = scene.n_frames if cfg.duration_s <= 0 else min(scene.n_frames, i0 + round(cfg.duration_s * scene.fps))
    return max(0, i0), max(i0, i1)


def output_times(i0: int, i1: int, track_fps: float, out_fps: float) -> np.ndarray:
    """Fractional track-frame indices of the output frames."""
    n = round((i1 - i0 - 1) * out_fps / track_fps) + 1 if i1 > i0 else 0
    return i0 + np.arange(n) * track_fps / out_fps
