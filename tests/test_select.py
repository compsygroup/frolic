from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from frolic.config import FrolicConfig, SelectConfig
from frolic.io.detections import Detections
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.select import stage
from frolic.select.filters import auto_select
from frolic.select.picker import pick_browser, pick_file, tunnel_hint
from frolic.select.sheet import contact_sheet
from frolic.select.stage import visibility_spans
from frolic.synthetic import make_scene
from frolic.track.associate import associate

PEOPLE = [
    {"id": "p0", "frames": 300, "duration_s": 20.0, "stature_m": 1.72, "median_views": 3.0},
    {"id": "p1", "frames": 280, "duration_s": 18.0, "stature_m": 0.95, "median_views": 2.0},
    {"id": "p2", "frames": 20, "duration_s": 1.3, "stature_m": 1.60, "median_views": 2.0},
    {"id": "p3", "frames": 100, "duration_s": 6.0, "stature_m": 1.80, "median_views": 1.0},
]


def test_auto_filters():
    ids, rej = auto_select(PEOPLE, SelectConfig())
    assert ids == ["p0", "p1"] and rej == {"min_views": ["p3"], "min_track_s": ["p2"]}
    ids, rej = auto_select(PEOPLE, SelectConfig(children_only=True, child_height_m=1.3))
    assert ids == ["p1"]
    ids, rej = auto_select(PEOPLE, SelectConfig(max_people=1, min_track_s=0, min_views=1))
    assert ids == ["p0"] and rej["max_people"] == ["p1", "p3", "p2"]
    ids, _ = auto_select(PEOPLE, SelectConfig(min_height_m=1.0, max_height_m=1.75, min_views=1, min_track_s=0))
    assert ids == ["p0", "p2"]


def _submit(ids):
    def go(url):
        page = urllib.request.urlopen(url).read().decode()
        assert "type='checkbox'" in page
        req = urllib.request.Request(url + "select", json.dumps({"ids": []}).encode(), method="POST")
        with pytest.raises(urllib.error.HTTPError):  # empty selection refused
            urllib.request.urlopen(req)
        req = urllib.request.Request(url + "select", json.dumps({"ids": ids}).encode(), method="POST")
        assert urllib.request.urlopen(req).status == 200

    return go


def test_pick_file_and_browser(tmp_path):
    f = tmp_path / "sel.txt"
    f.write_text("p1  # the child\n0\nbogus\n")
    assert pick_file(f, ["p0", "p1", "p2"]) == ["p1", "p0"]
    got = pick_browser(["p0", "p1", "p2"], PEOPLE, {}, port=0, open_browser=False, on_ready=_submit(["p2", "p0", "zz"]))
    assert got == ["p0", "p2"]


def test_tunnel_hint():
    slurm = tunnel_hint("127.0.0.1", 8770, {"USER": "u", "SLURM_JOB_ID": "9", "SLURMD_NODENAME": "gpu-1", "SLURM_SUBMIT_HOST": "login1"})
    assert "ssh -N -L 8770:127.0.0.1:8770 -J u@login1 u@gpu-1" in slurm[1]
    wide = tunnel_hint("0.0.0.0", 8770, {"USER": "u", "SLURM_JOB_ID": "9", "SLURMD_NODENAME": "gpu-1", "SLURM_SUBMIT_HOST": "login1"})
    assert "ssh -N -L 8770:gpu-1:8770 u@login1" in wide[1]
    ws = tunnel_hint("127.0.0.1", 8771, {"USER": "u", "TERM_PROGRAM": "vscode"})
    assert ws[0].startswith("VS Code") and ws[1].startswith("  ssh -N -L 8771:127.0.0.1:8771 u@")
    assert tunnel_hint("10.0.0.5", 8770, {"USER": "u"}) == []


def test_pick_browser_timeout():
    with pytest.raises(TimeoutError):
        pick_browser(["p0"], PEOPLE, {}, port=0, open_browser=False, timeout_s=0.2)


def test_visibility_spans():
    nv = np.array([3, 3, 3, 1, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3])
    spans = visibility_spans(nv, 15.0, 3, min_s=0.3, bridge_s=0.1)
    assert spans == [(0, 8), (12, 18)]
    assert visibility_spans(nv, 15.0, 3, min_s=0.35, bridge_s=0.0) == [(12, 18)]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    info = make_scene(d, n_cams=3, n_frames=30, videos=True)
    return d, info


def _prepare(scene, tmp_path, select: dict, people_cfg=None):
    scene_dir, info = scene
    cfg = FrolicConfig.model_validate(
        {
            "input": {"video_dir": str(scene_dir / "videos"), "work_dir": str(tmp_path / "work")},
            "calib": {"mode": "given", "given": str(scene_dir / "calib" / "cameras.json")},
            "track": {"reid": "none"},
            "select": select,
            "people": people_cfg or {},
        }
    )
    work = WorkDir(cfg.work_dir)
    work.root.mkdir(parents=True)
    cams = info["cameras"]
    work.write_videos(
        {
            "cameras": cams,
            "videos": info["videos"],
            "frame_offset": {c: 0 for c in cams},
            "fps": info["fps"],
            "width": {c: 640 for c in cams},
            "height": {c: 360 for c in cams},
            "n_frames": info["n_frames"],
            "offsets_s": {c: 0.0 for c in cams},
        }
    )
    dets = {}
    for c in cams:
        d = Detections.load(scene_dir / "gt" / f"{c}.npz")
        d.save(work.stage_dir("detect") / f"{c}.npz")
        dets[c] = d
    from frolic.io.calib import load_cameras

    cs = load_cameras(scene_dir / "calib" / "cameras.json")
    tr = associate(dets, cs, cfg.track, info["fps"], info["n_frames"])
    tr.ids = [f"p{k}" for k in range(tr.n_people)]
    tr.save(work.tracks)
    people = [
        {
            "id": pid,
            "frames": int(tr.has_3d[k].sum()),
            "duration_s": float(tr.has_3d[k].sum() / tr.fps),
            "stature_m": 1.7 if k == 0 else 0.95,
            "median_views": 3.0,
        }
        for k, pid in enumerate(tr.ids)
    ]
    with open(work.stage_dir("track") / "identities.json", "w") as fh:
        json.dump({"people": people, "floor": {}}, fh)
    return cfg, work, tr, dets, people


def test_sheet_generation(scene, tmp_path):
    _, work, tr, dets, people = _prepare(scene, tmp_path, {"mode": "auto"})
    out = work.stage_dir("select")
    tiles = contact_sheet(tr, dets, work.read_videos(), people, out / "sheet.jpg", out / "sheet.html", n=4)
    assert (out / "sheet.jpg").exists() and (out / "sheet.html").exists()
    assert len(tiles) == tr.n_people and all(len(t) == 4 for t in tiles.values())
    assert any(t.std() > 5 for ts in tiles.values() for t in ts)
    assert "data:image/jpeg;base64" in (out / "sheet.html").read_text()


def test_stage_config_mode_and_labels(scene, tmp_path):
    cfg, work, _, _, _ = _prepare(scene, tmp_path, {"mode": "config", "ids": ["p1", "zz"], "labels": {"p1": "child"}})
    out = stage.run(cfg, work)
    assert out["ids"] == ["p1"] and out["mode"] == "config"
    sel = Tracks.load(work.tracks_selected)
    assert sel.ids == ["p1"] and sel.labels == {"p1": "child"}
    with open(work.stage_dir("select") / "selection.json") as fh:
        assert json.load(fh)["labels"] == {"p1": "child"}


def test_stage_file_mode_and_spans(scene, tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("p0\n")
    cfg, work, _, _, _ = _prepare(
        scene,
        tmp_path,
        {"mode": "interactive", "selection_file": str(f), "spans_min_views": 3, "spans_min_s": 0.2},
        {"p0": {"label": "adult"}},
    )
    out = stage.run(cfg, work)
    assert out["ids"] == ["p0"] and out["mode"] == "file"
    sel = Tracks.load(work.tracks_selected)
    assert sel.labels == {"p0": "adult"}
    assert (sel.n_views()[0][sel.has_3d[0]] >= 3).all()


def test_stage_interactive_browser(scene, tmp_path):
    cfg, work, _, _, _ = _prepare(scene, tmp_path, {"mode": "interactive", "port": 0, "open_browser": False})
    out = stage.run(cfg, work, on_ready=_submit(["p1"]))
    assert out["ids"] == ["p1"] and out["mode"] == "browser"
    assert (work.stage_dir("select") / "picked.txt").read_text().split() == ["p1"]


def test_stage_auto_mode(scene, tmp_path):
    cfg, work, _, _, _ = _prepare(scene, tmp_path, {"mode": "auto", "children_only": True})
    out = stage.run(cfg, work)
    assert out["ids"] == ["p1"]
