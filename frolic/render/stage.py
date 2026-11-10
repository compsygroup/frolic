"""Render stage: overlay (per camera), grid, room and trajectory videos from the export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frolic.config import FrolicConfig
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.render.backend import select_backend
from frolic.render.scene import load_scene

log = get_logger(__name__)


def render_what(what: str, scene, cfg: FrolicConfig, out_dir: Path) -> list[Path]:
    rc = cfg.render
    if what == "overlay":
        from frolic.render.overlay import render_overlay

        return [render_overlay(scene, rc, out_dir / f"overlay_{c}.mp4", [c]) for c in scene.cams.names]
    if what == "grid":
        from frolic.render.grid import render_grid

        return [render_grid(scene, rc, out_dir / "grid.mp4")]
    if what == "room":
        from frolic.render.room import render_room

        return [render_room(scene, rc, out_dir / "room.mp4")]
    if what == "trajectories":
        from frolic.render.trajectories import render_trajectories

        return [render_trajectories(scene, rc, out_dir / "trajectories.mp4")]
    raise ValueError(f"unknown render target {what!r}")


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    select_backend(cfg.render.backend)
    export_dir = work.stage_dir("export")
    if not (export_dir / "meta.json").exists():
        raise FileNotFoundError(f"no export in {export_dir}; run `frolic export` first")
    scene = load_scene(export_dir, work.videos_json, cfg.render.colours)
    out_dir = work.stage_dir("render")
    written = []
    for what in cfg.render.what:
        log.info("rendering %s", what)
        written += [str(p) for p in render_what(what, scene, cfg, out_dir)]
    return {"videos": written}
