"""All camera views with mesh overlays tiled into a near-square grid."""

from __future__ import annotations

from pathlib import Path

from frolic.config import RenderConfig
from frolic.render.overlay import render_overlay
from frolic.render.scene import Scene


def render_grid(scene: Scene, cfg: RenderConfig, out_path: Path) -> Path:
    return render_overlay(scene, cfg, out_path, grid=True)
