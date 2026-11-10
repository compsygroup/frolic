"""Mesh overlay on the raw camera frames: the pinhole render is warped through the lens model, not the video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frolic3d.config import RenderConfig
from frolic3d.geometry import Camera
from frolic3d.io.video import FrameReader, write_video
from frolic3d.log import get_logger
from frolic3d.render.scene import Scene, camera_pose, catmull_rom, draw_text, material, output_times, render_window

log = get_logger(__name__)


class ViewRenderer:
    """Renders meshes seen by one calibrated camera onto a tile of width tile_w, warped into the distorted image."""

    def __init__(self, cam: Camera, tile_w: int, pad: float = 0.25, supersample: int = 1, renderer=None) -> None:
        import pyrender

        self.cam = cam
        s = tile_w / cam.width
        self.tw, self.th = tile_w, round(cam.height * s)
        self.ss = max(1, supersample)
        px, py = round(self.tw * pad), round(self.th * pad)
        self.cw, self.ch = self.tw + 2 * px, self.th + 2 * py
        Kc = cam.K * np.array([[s], [s], [1.0]])
        Kc[0, 2] += px
        Kc[1, 2] += py
        K2 = Kc * np.array([[self.ss], [self.ss], [1.0]])
        gx, gy = np.meshgrid(np.arange(self.tw, dtype=np.float32), np.arange(self.th, dtype=np.float32))
        pts = np.stack([(gx + 0.5) / s - 0.5, (gy + 0.5) / s - 0.5], -1).reshape(-1, 1, 2).astype(np.float64)
        u = cv2.undistortPoints(pts, cam.K, cam.dist, P=Kc)
        u = u.reshape(self.th, self.tw, 2).astype(np.float32)
        self.map_x, self.map_y = u[..., 0], u[..., 1]
        self.owns_renderer = renderer is None
        self.renderer = renderer or pyrender.OffscreenRenderer(self.cw * self.ss, self.ch * self.ss)
        self.camera = pyrender.IntrinsicsCamera(K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2], znear=0.05, zfar=80)
        self.pose = camera_pose(cam)

    def overlay(
        self, img: np.ndarray, meshes: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]], alpha: float
    ) -> np.ndarray:
        import pyrender
        import trimesh

        img = (
            cv2.resize(img, (self.tw, self.th), interpolation=cv2.INTER_AREA)
            if img.shape[:2] != (self.th, self.tw)
            else img
        )
        if not meshes:
            return img
        sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
        for V, F, rgb in meshes:
            sc.add(
                pyrender.Mesh.from_trimesh(trimesh.Trimesh(V, F, process=False), material=material(rgb), smooth=True)
            )
        sc.add(self.camera, pose=self.pose)
        sc.add(pyrender.DirectionalLight(intensity=2.6), pose=self.pose)
        self.renderer.viewport_width, self.renderer.viewport_height = self.cw * self.ss, self.ch * self.ss
        col, dep = self.renderer.render(sc, flags=pyrender.RenderFlags.RGBA)
        m = (dep > 0).astype(np.float32)
        edge = cv2.morphologyEx((dep > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(
            np.float32
        )
        col = cv2.resize(col[..., [2, 1, 0]].astype(np.float32), (self.cw, self.ch), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (self.cw, self.ch), interpolation=cv2.INTER_AREA)
        edge = cv2.resize(edge, (self.cw, self.ch), interpolation=cv2.INTER_AREA)
        col, m, edge = (
            cv2.remap(x, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for x in (col, m, edge)
        )
        m = m[..., None]
        edge = edge[..., None]
        out = img.astype(np.float32) * (1 - alpha * m) + alpha * m * col
        out = out * (1 - 0.8 * edge) + 0.8 * edge * np.array([255, 245, 235], np.float32)
        return out.clip(0, 255).astype(np.uint8)

    def close(self) -> None:
        if self.owns_renderer:
            self.renderer.delete()


def meshes_at(scene: Scene, t: float) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]]:
    out = []
    for p in scene.people:
        V = catmull_rom(p.vertices, t)
        if V is not None and p.faces.size:
            out.append((np.asarray(V, np.float32), p.faces, p.colour))
    return out


def tile_label(img: np.ndarray, text: str, size: int) -> np.ndarray:
    return draw_text(img, [(int(0.015 * img.shape[1]), img.shape[0] - int(size * 1.5), text, size, (245, 245, 245))])


class CameraFrames:
    """Aligned frames of one camera at fractional track-frame times (nearest source frame)."""

    def __init__(self, scene: Scene, cam: Camera) -> None:
        vids = scene.videos
        self.path = Path(vids["videos"][cam.name]) if vids and cam.name in vids.get("videos", {}) else None
        self.offset = int(vids.get("frame_offset", {}).get(cam.name, 0)) if vids else 0
        self.reader = FrameReader(self.path) if self.path is not None and self.path.exists() else None
        self.scene = scene
        self.blank = np.full((cam.height, cam.width, 3), 40, np.uint8)
        self.cam = cam

    def at(self, t: float) -> np.ndarray:
        if self.reader is None:
            return self.blank
        i = int(np.clip(np.round(t), 0, len(self.scene.source_frames) - 1))
        lo = int(np.floor(t))
        hi = min(lo + 1, len(self.scene.source_frames) - 1)
        f0, f1 = int(self.scene.source_frames[lo]), int(self.scene.source_frames[hi])
        f = round(f0 + (t - lo) * (f1 - f0)) if hi > lo else int(self.scene.source_frames[i])
        img = self.reader.read(f + self.offset)
        return img if img is not None else self.blank

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()


def render_overlay(
    scene: Scene, cfg: RenderConfig, out_path: Path, cam_names: list[str] | None = None, grid: bool = False
) -> Path:
    """One video per camera (overlay) or all cameras tiled (grid)."""
    import pyrender

    names = cam_names or scene.cams.names
    cams = [scene.cams[n] for n in names]
    shared = pyrender.OffscreenRenderer(8, 8)  # one GL context for every view; viewports are resized per view
    tw = (cfg.grid_tile_width or cfg.tile_width) if grid else cfg.tile_width
    views = [ViewRenderer(c, tw, supersample=cfg.supersample, renderer=shared) for c in cams]
    frames = [CameraFrames(scene, c) for c in cams]
    i0, i1 = render_window(scene, cfg)
    ts = output_times(i0, i1, scene.fps, cfg.fps)
    label_size = max(12, int(tw / 40))
    if grid:
        ncol = cfg.grid_cols or int(np.ceil(np.sqrt(len(cams))))
        nrow = int(np.ceil(len(cams) / ncol))
        W, H = ncol * views[0].tw, nrow * views[0].th
    else:
        W, H = views[0].tw, views[0].th
    proc = write_video(out_path, (W, H), cfg.fps, cfg.crf)
    try:
        for k, t in enumerate(ts):
            meshes = meshes_at(scene, float(t))
            tiles = [
                tile_label(v.overlay(f.at(float(t)), meshes, cfg.alpha), c.name, label_size)
                for v, f, c in zip(views, frames, cams)
            ]
            if grid:
                blank = np.zeros_like(tiles[0])
                tiles += [blank] * (ncol * nrow - len(tiles))
                frame = np.vstack([np.hstack(tiles[r * ncol : (r + 1) * ncol]) for r in range(nrow)])
            else:
                frame = tiles[0]
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if k % 200 == 0:
                log.info("%s: %d/%d", out_path.name, k, len(ts))
    finally:
        proc.stdin.close()
        proc.wait()
        shared.delete()
        for f in frames:
            f.close()
    return out_path
