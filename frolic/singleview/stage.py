"""Single-view stage: recover frames where a selected person is seen by only one camera."""

from __future__ import annotations

import json
from typing import Any

from frolic.config import FrolicConfig
from frolic.io.calib import load_cameras
from frolic.io.detections import load_all
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.singleview.lift import lift_frames

log = get_logger(__name__)


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    tr = Tracks.load(work.tracks_selected)
    out_dir = work.stage_dir("singleview")
    if not cfg.single_view.enabled or tr.n_people == 0:
        tr.save(work.tracks_singleview)
        with open(out_dir / "svfill.json", "w") as fh:
            json.dump({"people": {}, "enabled": cfg.single_view.enabled}, fh)
        return {"enabled": False, "lifted": 0}
    cams = load_cameras(work.cameras_level).subset(tr.cameras)
    dets = load_all(work.stage_dir("detect"), tr.cameras)
    bounds = tuple(cfg.track.room_bounds_m) if cfg.track.room_bounds_m else None
    out, report = lift_frames(tr, dets, cams, cfg.single_view, None, bounds)
    out.save(work.tracks_singleview)
    with open(out_dir / "svfill.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    lifted = {pid: len(v["lifted_frames"]) for pid, v in report["people"].items()}
    return {"enabled": True, "lifted": lifted}
