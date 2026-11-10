"""3D room view: floor grid, camera frustums, meshes with contact shadows, a slow eased pan, legend."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frolic3d.config import RenderConfig
from frolic3d.io.video import write_video
from frolic3d.log import get_logger
from frolic3d.render.footprints import footprint_mesh, footsteps
from frolic3d.render.scene import (
    Scene,
    body_mesh,
    catmull_rom,
    draw_legend,
    draw_text,
    ease,
    floor_meshes,
    frustum_mesh,
    look_at,
    output_times,
    render_window,
    shadow_mesh,
)

log = get_logger(__name__)


class RoomView:
    def __init__(self, scene: Scene, cfg: RenderConfig, fov_deg: float = 38.0) -> None:
        import pyrender

        self.scene = scene
        self.cfg = cfg
        self.W, self.H = cfg.room_width, cfg.room_height
        self.ss = max(1, cfg.supersample)
        self.lo, self.hi = scene.floor_bounds
        self.target = scene.people_center
        self.yfov = np.radians(fov_deg)
        self.asp = self.W / self.H
        mc = scene.cams.centers.mean(0)[:2] - self.target[:2]
        self.az0 = np.degrees(np.arctan2(mc[1], mc[0])) - cfg.pan_deg / 2
        self.dist = self._fit_distance() / max(cfg.room_zoom, 1e-3)
        self.renderer = pyrender.OffscreenRenderer(self.W * self.ss, self.H * self.ss)
        self.camera = pyrender.PerspectiveCamera(yfov=self.yfov, aspectRatio=self.asp, znear=0.05, zfar=200)
        depth = 0.25 if self.dist < 8 else 0.45
        self.static = floor_meshes(self.lo, self.hi) + [frustum_mesh(c, depth) for c in scene.cams]

    def eye(self, az: float, dist: float) -> np.ndarray:
        th, el = np.radians(az), np.radians(self.cfg.elevation_deg)
        return self.target + dist * np.array([np.cos(el) * np.cos(th), np.cos(el) * np.sin(th), np.sin(el)])

    def _keypoints(self) -> np.ndarray:
        corners = np.array(
            [[self.lo[0], self.lo[1]], [self.hi[0], self.lo[1]], [self.hi[0], self.hi[1]], [self.lo[0], self.hi[1]]]
        )
        return np.vstack([np.c_[corners, np.zeros(4)], np.c_[corners, np.full(4, 1.6)], self.scene.cams.centers])

    def _fits(self, dist: float, keyp: np.ndarray) -> bool:
        for az in np.linspace(self.az0, self.az0 + self.cfg.pan_deg, 5):
            P = look_at(self.eye(az, dist), self.target)
            Xc = (keyp - P[:3, 3]) @ P[:3, :3]
            if (Xc[:, 2] > -0.3).any():
                return False
            u = Xc[:, 0] / -Xc[:, 2] / np.tan(self.yfov / 2) / self.asp
            w = Xc[:, 1] / -Xc[:, 2] / np.tan(self.yfov / 2)
            if np.abs(u).max() > 0.92 or np.abs(w).max() > 0.88:
                return False
        return True

    def _fit_distance(self) -> float:
        keyp = self._keypoints()
        dist = 1.5
        while not self._fits(dist, keyp) and dist < 80:
            dist *= 1.04
        return dist

    def pose_at(self, u: float) -> np.ndarray:
        az = self.az0 + self.cfg.pan_deg * ease(u)
        return look_at(self.eye(az, self.dist), self.target)

    def render(self, t: float, u: float, extra_nodes: list | None = None, alpha: float = 1.0) -> np.ndarray:
        import pyrender

        sc = pyrender.Scene(bg_color=[0.11, 0.12, 0.14, 1.0], ambient_light=[0.35, 0.35, 0.37])
        for m in self.static:
            sc.add(m)
        for p in self.scene.people:
            V = catmull_rom(p.vertices, t)
            if V is None or not p.faces.size:
                continue
            V = np.asarray(V, np.float32)
            sc.add(body_mesh(V, p.faces, p.colour, alpha))
            sc.add(shadow_mesh(V, p.faces))
        for m in extra_nodes or []:
            sc.add(m)
        P = self.pose_at(u)
        sc.add(self.camera, pose=P)
        sc.add(
            pyrender.DirectionalLight(intensity=2.6),
            pose=look_at(self.target + np.array([1.5, -2.0, 4.0]), self.target),
        )
        sc.add(pyrender.DirectionalLight(intensity=1.4), pose=P)
        col, _ = self.renderer.render(sc)
        img = np.ascontiguousarray(col[..., ::-1])
        if self.ss > 1:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return self._labels(img, P)

    def _labels(self, img: np.ndarray, P: np.ndarray) -> np.ndarray:
        items = []
        size = max(10, int(self.W / 110))
        for cam in self.scene.cams:
            Xc = (cam.center - P[:3, 3]) @ P[:3, :3]
            if Xc[2] < -0.1:
                x = self.W / 2 + Xc[0] / -Xc[2] / np.tan(self.yfov / 2) / self.asp * self.W / 2
                y = self.H / 2 - Xc[1] / -Xc[2] / np.tan(self.yfov / 2) * self.H / 2
                if 0 <= x < self.W and 0 <= y < self.H:
                    items.append((int(x), int(y) - size, cam.name, size, (200, 215, 235)))
        img = draw_text(img, items)
        return draw_legend(img, self.scene.people, size=max(12, int(self.W / 90)))

    def close(self) -> None:
        self.renderer.delete()


def render_room(scene: Scene, cfg: RenderConfig, out_path: Path) -> Path:
    view = RoomView(scene, cfg)
    i0, i1 = render_window(scene, cfg)
    ts = output_times(i0, i1, scene.fps, cfg.fps)
    steps = {p.pid: footsteps(p.joints, scene.fps) for p in scene.people} if cfg.footprints_s > 0 else {}
    for pid, st in steps.items():
        log.info("%s: %d footsteps", pid, len(st))
    fade, hold = cfg.footprints_s * scene.fps, cfg.footprints_hold_s * scene.fps
    proc = write_video(out_path, (view.W, view.H), cfg.fps, cfg.crf)
    try:
        for k, t in enumerate(ts):
            u = k / max(1, len(ts) - 1)
            extra = [
                footprint_mesh(steps[p.pid], float(t), fade, p.colour, hold_frames=hold)
                for p in scene.people
                if p.pid in steps
            ]
            proc.stdin.write(np.ascontiguousarray(view.render(float(t), u, [m for m in extra if m is not None])).tobytes())
            if k % 200 == 0:
                log.info("%s: %d/%d", out_path.name, k, len(ts))
    finally:
        proc.stdin.close()
        proc.wait()
        view.close()
    return out_path
