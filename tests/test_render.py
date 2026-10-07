from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest
from conftest import can_render
from test_export import build_work

from frolic3d.config import RenderConfig
from frolic3d.export.stage import run as run_export
from frolic3d.render.scene import catmull_rom, ease, look_at, output_times
from frolic3d.synthetic import make_scene

pytestmark = pytest.mark.skipif(not can_render(), reason="no offscreen OpenGL backend (pyrender EGL/OSMesa) available")


def probe(path):
    q = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    s = json.loads(q.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_frames"])


def test_catmull_rom_and_helpers():
    X = np.arange(5, dtype=np.float32)[:, None, None] * np.ones((5, 2, 3), np.float32)
    assert np.allclose(catmull_rom(X, 1.5), 1.5)
    assert catmull_rom(X, 7.0) is None
    X[3] = np.nan
    assert np.allclose(catmull_rom(X, 2.5), 2.0) and catmull_rom(X, 3.2) is None
    assert ease(0) == 0 and ease(1) == 1 and 0 < ease(0.5) < 1
    P = look_at(np.array([3.0, 0, 2]), np.zeros(3))
    assert np.allclose(np.linalg.det(P[:3, :3]), 1)
    assert len(output_times(0, 3, 15.0, 30)) == 5 and len(output_times(0, 3, 15.0, 15)) == 3


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    root = tmp_path_factory.mktemp("render")
    scene = make_scene(root / "scene", n_cams=4, n_frames=6, fps=30.0, videos=True)  # track stride 2
    work, cfg, _ = build_work(root / "w", n_frames=3, videos=scene["videos"])
    vids = work.read_videos()
    vids["fps"] = 15.0
    work.write_videos(vids)
    cfg.render = RenderConfig(
        fps=15, tile_width=160, room_width=160, room_height=90, what=["overlay", "grid", "room", "trajectories"]
    )
    run_export(cfg, work)
    from frolic3d.render.backend import select_backend

    select_backend("auto")
    return work, cfg


def test_render_stage_all_targets(tiny_export):
    work, cfg = tiny_export
    from frolic3d.render.stage import run

    info = run(cfg, work)
    out = work.stage_dir("render")
    assert len(info["videos"]) == 4 + 3
    for c in ("cam1", "cam4"):
        w, h, n = probe(out / f"overlay_{c}.mp4")
        assert (w, h, n) == (160, 90, 3)
    assert probe(out / "grid.mp4") == (320, 180, 3)
    assert probe(out / "room.mp4") == (160, 90, 3)
    assert probe(out / "trajectories.mp4") == (160, 90, 3)


def test_render_window_and_fps(tiny_export):
    work, cfg = tiny_export
    from frolic3d.render.room import render_room
    from frolic3d.render.scene import load_scene

    scene = load_scene(work.stage_dir("export"), work.videos_json)
    assert len(scene.people) == 2 and len(scene.cams) == 4
    rc = cfg.render.model_copy(update={"fps": 30, "start_s": 0.0, "duration_s": 2 / 15})
    p = render_room(scene, rc, work.stage_dir("render") / "room30.mp4")
    assert probe(p) == (160, 90, 3)


def test_render_overlay_with_distortion(tiny_export):
    work, _cfg = tiny_export
    from frolic3d.io.calib import load_cameras
    from frolic3d.render.overlay import ViewRenderer

    cams = load_cameras(work.stage_dir("export") / "cameras.json")
    cam = cams[0]
    cam.dist = np.array([-0.2, 0.05, 0, 0, 0])
    v = ViewRenderer(cam, 160)
    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    tri = np.array([[0, 0, 0.5], [0.3, 0, 0.5], [0, 0.3, 0.9]], np.float32) + cams.centers.mean(0) * 0
    out = v.overlay(img, [(tri, np.array([[0, 1, 2]]), (255, 0, 0))], 0.6)
    v.close()
    assert out.shape == (90, 160, 3) and out.max() > 0
