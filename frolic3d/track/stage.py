"""Track stage: association -> identities -> floor align -> refinement -> floor levelling."""

from __future__ import annotations

import json
from typing import Any

from frolic3d.config import FrolicConfig
from frolic3d.io.calib import load_cameras, save_cameras
from frolic3d.io.detections import load_all
from frolic3d.io.tracks import Tracks
from frolic3d.io.workdir import WorkDir
from frolic3d.log import get_logger
from frolic3d.track.associate import associate
from frolic3d.track.identity import assemble_identities
from frolic3d.track.refine import refine_tracks
from frolic3d.track.reid import OsnetEmbedder, centre_embeddings, embed_tracks, weights_path

log = get_logger(__name__)


def _embeddings(cfg: FrolicConfig, tr: Tracks, dets, videos: dict[str, Any]):
    if cfg.track.reid == "none":
        return None
    w = weights_path(cfg.track)
    if w is None:
        log.warning(
            "re-ID weights not found (track.reid_weights / FROLIC_OSNET_WEIGHTS): identities use colour and size only"
        )
        return None
    try:
        emb = OsnetEmbedder(w, cfg.compute.device, threads=cfg.compute.threads)
    except ImportError as e:
        log.warning("%s: identities use colour and size only", e)
        return None
    E = embed_tracks(tr, dets, videos["videos"], videos.get("frame_offset", {}), emb, cfg.track.reid_every_s)
    return centre_embeddings(E)


def _floor(cfg: FrolicConfig, tr: Tracks, cams, step: str):
    try:
        from frolic3d.calib import floor
    except ImportError:
        log.warning("floor module unavailable: %s skipped", step)
        return tr, cams, {}
    fn = floor.level_floor if step == "level" else floor.floor_align
    try:
        if step == "level":
            return fn(tr, cams, cfg.calib.ankle_height_m, cfg.calib.level_min_spread_m)
        return fn(tr, cams, cfg.calib.ankle_height_m)
    except Exception as e:
        log.warning("%s failed (%s): kept the unlevelled frame", step, e)
        return tr, cams, {"error": str(e)}


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    videos = work.read_videos()
    cams = load_cameras(work.cameras)
    cams = cams.subset([c for c in videos["cameras"] if c in cams.names])
    dets = load_all(work.stage_dir("detect"), cams.names)
    n_frames = int(videos["n_frames"])
    # frame indices are already relative to the sync window (sync applies input.start_s by trimming or frame_offset)
    raw = associate(dets, cams, cfg.track, float(videos["fps"]), n_frames, frame0=0)
    raw.save(work.tracks_raw)
    E = _embeddings(cfg, raw, dets, videos)
    tr, info = assemble_identities(raw, dets, cfg.track, E)
    floor_info: dict[str, Any] = {}
    if cfg.calib.level_floor and tr.n_people:
        tr, cams, floor_info["align"] = _floor(cfg, tr, cams, "align")
    if cfg.track.refine and tr.n_people:
        tr = refine_tracks(tr, dets, cams, cfg.track)
    if cfg.calib.level_floor and tr.n_people:
        tr, cams, floor_info["level"] = _floor(cfg, tr, cams, "level")
        if floor_info["level"].get("source") != "none":
            cams.floor.update({"levelled": True, "floor_z": 0.0})
    tr.meta["floor"] = floor_info
    tr.save(work.tracks)
    save_cameras(cams, work.cameras_level)
    with open(work.stage_dir("track") / "identities.json", "w") as fh:
        json.dump({"people": info, "floor": floor_info}, fh, indent=1, default=float)
    log.info("track: %d people, %d frames at %.2f Hz", tr.n_people, tr.n_frames, tr.fps)
    return {"n_people": tr.n_people, "n_frames": tr.n_frames, "fps": tr.fps, "ids": tr.ids, "reid": E is not None}
