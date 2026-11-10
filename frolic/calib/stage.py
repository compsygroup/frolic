"""Stage `calibrate`: write work/calib/cameras.json from a given file, ChArUco board videos or the person bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frolic.config import FrolicConfig
from frolic.io.calib import CameraSet, load_cameras, save_cameras, to_gui_json
from frolic.io.detections import Detections
from frolic.io.video import list_videos, probe_video
from frolic.io.workdir import WorkDir
from frolic.log import get_logger

log = get_logger(__name__)


def _match_names(cams: CameraSet, names: list[str]) -> CameraSet:
    """Reorder to the video cameras; map by order (with a warning) when the names differ but the counts match."""
    if set(names) <= set(cams.names):
        return cams.subset(names)
    if len(cams) != len(names):
        raise RuntimeError(f"calibration cameras {cams.names} do not match the videos {names}")
    log.warning("calibration camera names %s differ from the videos %s: mapping by order", cams.names, names)
    out = cams.subset(cams.names)
    for c, n in zip(out.cameras, names):
        c.name = n
    return out


def _fill_sizes(cams: CameraSet, videos: dict[str, Any]) -> None:
    for c in cams:
        if c.width <= 0 or c.height <= 0:
            c.width, c.height = int(videos["width"][c.name]), int(videos["height"][c.name])


def _given(cfg: FrolicConfig, names: list[str]) -> tuple[CameraSet, dict[str, Any]]:
    cams = load_cameras(cfg.calib.given, cfg.calib.given_format)
    return _match_names(cams, names), {"mode": "given", "path": str(cfg.calib.given), "format": cfg.calib.given_format}


def _load_intrinsics_json(paths: dict[str, Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, p in paths.items():
        with open(p) as fh:
            d = json.load(fh)
        out[name] = (np.array(d["camera_matrix"], float), np.array(d["distortion_coefficients"], float).ravel())
    return out


def _board(cfg: FrolicConfig, names: list[str], work: WorkDir) -> tuple[CameraSet, dict[str, Any]]:
    from frolic.calib.board import BoardSpec, calibrate_multi

    b = cfg.calib.board
    if b.video_dir is None:
        raise RuntimeError("calib.mode = board needs calib.board.video_dir")
    vids = list_videos(b.video_dir, b.pattern)
    if set(names) <= set(vids):
        vids = {n: vids[n] for n in names}
    elif len(vids) == len(names):
        log.warning("board video names %s differ from the cameras %s: mapping by order", list(vids), names)
        vids = dict(zip(names, vids.values()))
    else:
        raise RuntimeError(f"board videos {list(vids)} do not match the cameras {names}")
    sizes = {n: (probe_video(p).width, probe_video(p).height) for n, p in vids.items()}
    spec = BoardSpec.from_config(b)
    intr = _load_intrinsics_json(b.intrinsics_json) if b.intrinsics_json else None
    cams, report = calibrate_multi(vids, spec, b, sizes, intr)
    to_gui_json(cams, work.stage_dir("calibrate") / "gui", spec.to_gui_dict(), rms_px=cams.source.get("rms_px"))
    report["mode"] = "board"
    return cams, report


def _bootstrap(cfg: FrolicConfig, names: list[str], work: WorkDir) -> tuple[CameraSet, dict[str, Any]]:
    from frolic.calib.bootstrap import bootstrap

    det_dir = work.stage_dir("detect")
    missing = [n for n in names if not (det_dir / f"{n}.npz").exists()]
    if missing:
        raise RuntimeError(f"calib.mode = bootstrap needs the detect stage first (missing {missing})")
    dets = {n: Detections.load(det_dir / f"{n}.npz") for n in names}
    res = bootstrap(dets, cfg.calib.bootstrap, float(cfg.calib.reference_height_m))
    np.savez_compressed(
        work.stage_dir("calibrate") / "bootstrap_points.npz", xyz=res.points, frames=np.array(res.frames)
    )
    res.report["mode"] = "bootstrap"
    return res.cameras, res.report


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    videos = work.read_videos()
    names = list(videos["cameras"])
    if cfg.calib.mode == "given":
        cams, report = _given(cfg, names)
    elif cfg.calib.mode == "board":
        cams, report = _board(cfg, names, work)
    else:
        cams, report = _bootstrap(cfg, names, work)
    _fill_sizes(cams, videos)
    for c in cams:
        if (c.width, c.height) != (int(videos["width"][c.name]), int(videos["height"][c.name])):
            log.warning(
                "%s: calibration size %dx%d differs from the video %sx%s",
                c.name,
                c.width,
                c.height,
                videos["width"][c.name],
                videos["height"][c.name],
            )
    save_cameras(cams, work.cameras)
    report["cameras"] = cams.names
    report["baselines_m"] = report.get("baselines_m") or {
        f"{a}-{b}": float(np.linalg.norm(cams[a].center - cams[b].center))
        for i, a in enumerate(cams.names)
        for b in cams.names[i + 1 :]
    }
    with open(work.stage_dir("calibrate") / "report.json", "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    return {"mode": cfg.calib.mode, "cameras": cams.names, "degenerate": report.get("degenerate_cameras", [])}
