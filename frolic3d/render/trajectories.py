"""Room view with fading pelvis trails per person."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frolic3d.config import RenderConfig
from frolic3d.io.video import write_video
from frolic3d.log import get_logger
from frolic3d.render.room import RoomView
from frolic3d.render.scene import Scene, output_times, render_window
from frolic3d.skeleton import J

log = get_logger(__name__)


def trail_mesh(points: np.ndarray, rgb: tuple[int, int, int], radius: float):
    """Tube segments with alpha fading toward the oldest point."""
    import pyrender
    import trimesh

    n = len(points)
    if n < 2:
        return None
    parts = []
    for k in range(n - 1):
        p, q = points[k], points[k + 1]
        if np.linalg.norm(q - p) < 1e-4:
            continue
        seg = trimesh.creation.cylinder(radius=radius, segment=[p, q], sections=6)
        a = 0.15 + 0.85 * (k + 1) / (n - 1)
        col = np.array([*rgb, int(255 * a)], np.uint8)
        seg.visual.vertex_colors = np.tile(col, (len(seg.vertices), 1))
        parts.append(seg)
    if not parts:
        return None
    m = trimesh.util.concatenate(parts)
    return pyrender.Mesh.from_trimesh(m, smooth=False)


def render_trajectories(scene: Scene, cfg: RenderConfig, out_path: Path) -> Path:
    view = RoomView(scene, cfg)
    i0, i1 = render_window(scene, cfg)
    ts = output_times(i0, i1, scene.fps, cfg.fps)
    n_trail = max(2, round(cfg.trail_s * scene.fps))
    radius = 0.01 * max(1.0, view.dist / 4.5)
    proc = write_video(out_path, (view.W, view.H), cfg.fps, cfg.crf)
    try:
        for k, t in enumerate(ts):
            u = k / max(1, len(ts) - 1)
            i = int(np.floor(t))
            extra = []
            for p in scene.people:
                lo = max(0, i - n_trail + 1)
                pel = p.joints[lo : i + 1, J["pelvis"]].copy()
                pel = pel[np.isfinite(pel[:, 0])]
                pel[:, 2] = 0.02
                m = trail_mesh(pel, p.colour, radius)
                if m is not None:
                    extra.append(m)
            proc.stdin.write(np.ascontiguousarray(view.render(float(t), u, extra, alpha=0.55)).tobytes())
            if k % 200 == 0:
                log.info("%s: %d/%d", out_path.name, k, len(ts))
    finally:
        proc.stdin.close()
        proc.wait()
        view.close()
    return out_path
