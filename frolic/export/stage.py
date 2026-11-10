"""Export stage: per-person npz meshes, meta.json, cameras.json, metrics.json and summary.csv."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from frolic.config import FrolicConfig
from frolic.export.visibility import visible_mask
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.metrics.compute import compute_metrics, write_summary_csv
from frolic.skeleton import JOINT_NAMES

log = get_logger(__name__)

SCHEMA = "frolic.export/1"


def fitted_people(fit_dir: Path) -> list[str]:
    """Ids fitted by the last fit run (fit.json), so files of an earlier selection are not exported."""
    rep = fit_dir / "fit.json"
    if rep.exists():
        with open(rep) as fh:
            ids = list(json.load(fh).get("people", {}))
        if ids:
            return sorted(i for i in ids if (fit_dir / f"{i}.npz").exists())
    return sorted(p.stem for p in fit_dir.glob("*.npz") if not p.stem.endswith("_joints_full"))


def interpolate_joints(
    joints: np.ndarray, frame_times: np.ndarray, src_fps: float, show: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (N, J, 3) joints to the source frame rate; NaN outside visible runs."""
    if len(frame_times) < 2:
        return joints.copy(), frame_times.copy()
    t_full = np.arange(frame_times[0], frame_times[-1] + 0.5 / src_fps, 1.0 / src_fps)
    out = np.full((len(t_full), *joints.shape[1:]), np.nan, np.float32)
    idx = np.flatnonzero(show)
    if len(idx) == 0:
        return out, t_full
    flat = joints.reshape(len(joints), -1)
    for k in range(flat.shape[1]):
        out[:, k // 3, k % 3] = np.interp(t_full, frame_times[idx], flat[idx, k], left=np.nan, right=np.nan)
    # keep NaN inside hidden spans longer than one track step
    hidden = ~show
    if hidden.any():
        step = frame_times[1] - frame_times[0]
        for i in np.flatnonzero(hidden):
            out[(t_full > frame_times[i] - 0.5 * step) & (t_full < frame_times[i] + 0.5 * step)] = np.nan
    return out, t_full


def export_person(
    pid: str, fit_dir: Path, out_dir: Path, tracks: Tracks | None, label: str, cfg: FrolicConfig, src_fps: float | None
) -> dict[str, Any]:
    z = np.load(fit_dir / f"{pid}.npz", allow_pickle=False)
    fps = float(z["fps"])
    have = np.asarray(z["have"], bool)
    n = len(have)
    sv = z["single_view"] if "single_view" in z.files else None
    obs = z["observed"] if "observed" in z.files else None
    show = visible_mask(have, sv, obs, fps, cfg.export.max_gap_s, cfg.export.observed_rule)
    joints = np.asarray(z["joints"], np.float32)[:n].copy()
    joints[~show] = np.nan
    faces = np.load(fit_dir / "faces.npy").astype(np.int32)
    verts_path = fit_dir / f"{pid}_verts.npy"
    dtype = np.float16 if cfg.export.vertices_dtype == "float16" else np.float32
    if verts_path.exists():
        verts = np.load(verts_path, mmap_mode="r")
        vertices = np.asarray(verts[:n], np.float32)
        vertices[~show] = np.nan
        vertices = vertices.astype(dtype)
    else:
        vertices = np.full((n, 0, 3), np.nan, dtype)
    stride = tracks.stride if tracks is not None else 1
    # fit frame0 = source (sync-window) frame of track frame 0, copied from the tracks; add it once
    src0 = int(z["frame0"]) if "frame0" in z.files else (tracks.frame0 if tracks is not None else 0)
    source_frames = (src0 + np.arange(n) * stride).astype(np.int32)
    frame_times = source_frames / (src_fps or fps * stride)
    np.savez_compressed(
        out_dir / f"{pid}.npz",
        vertices=vertices,
        faces=faces,
        joints=joints,
        joint_names=np.array(JOINT_NAMES),
        frame_times=frame_times.astype(np.float64),
        source_frames=source_frames,
        phenotype=np.asarray(z["phenotype"], np.float32),
        phenotype_labels=np.asarray(z["phenotype_labels"]),
        stature_m=np.float32(z["stature"]),
        single_view=np.zeros(n, bool) if sv is None else np.asarray(sv, bool)[:n],
        observed=show,
        label=np.array(label),
    )
    if cfg.export.full_rate_joints and src_fps:
        jf, tf = interpolate_joints(joints, frame_times, src_fps, show)
        np.savez_compressed(
            out_dir / f"{pid}_joints_full.npz",
            joints=jf,
            frame_times=tf,
            joint_names=np.array(JOINT_NAMES),
            fps=src_fps,
        )
    log.info("%s: %d/%d frames visible", pid, int(show.sum()), n)
    return {
        "id": pid,
        "label": label,
        "stature_m": float(z["stature"]),
        "frames_visible": int(show.sum()),
        "n_frames": n,
        "fps": fps,
    }


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    fit_dir = work.stage_dir("fit")
    out_dir = work.stage_dir("export")
    people = fitted_people(fit_dir)
    if not people:
        raise FileNotFoundError(f"no fitted people in {fit_dir}")
    tracks_path = work.fit_input_tracks
    tracks = Tracks.load(tracks_path) if tracks_path.exists() else None
    videos = work.read_videos() if work.videos_json.exists() else {}
    src_fps = float(videos["fps"]) if "fps" in videos else None
    infos = []
    for pid in people:
        label = (
            cfg.people.get(pid).label
            if pid in cfg.people and cfg.people[pid].label
            else (tracks.labels.get(pid, pid) if tracks is not None else pid)
        )
        infos.append(export_person(pid, fit_dir, out_dir, tracks, label, cfg, src_fps))
    cam_src = work.cameras_level if work.cameras_level.exists() else work.cameras
    if cam_src.exists():
        shutil.copy(cam_src, out_dir / "cameras.json")
    n_frames = max(i["n_frames"] for i in infos)
    meta = {
        "schema": SCHEMA,
        "fps": infos[0]["fps"],
        "n_frames": n_frames,
        "source_fps": src_fps,
        "people": [{k: v for k, v in i.items() if k in ("id", "label", "stature_m", "frames_visible")} for i in infos],
        "body_model": "Anny (SMPL-X topology)" if cfg.fit.backend == "anny" else cfg.fit.backend,
        "joint_names": JOINT_NAMES,
        "world": {"up": "z", "floor_z": 0.0, "units": "m"},
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    metrics = compute_metrics(work, out_dir, fit_dir)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=1)
    write_summary_csv(metrics, out_dir / "summary.csv")
    return {"people": [i["id"] for i in infos], "n_frames": n_frames}
