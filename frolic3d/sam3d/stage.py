"""Stage `sam3d`: prompts -> SAM 3D Body per camera -> single-view depth refinement -> multi-view MHR fit per person."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from frolic3d.config import FrolicConfig
from frolic3d.io.calib import CameraSet, load_cameras
from frolic3d.io.detections import load_all
from frolic3d.io.tracks import Tracks
from frolic3d.io.workdir import WorkDir
from frolic3d.log import get_logger
from frolic3d.sam3d.adapter import MockMhr, expand17to70, make_estimator, mock_mesh_from_joints
from frolic3d.sam3d.mhr_fit import MhrFitter, PersonObservations
from frolic3d.sam3d.prompts import Prompts, gather_prompts
from frolic3d.sam3d.run import ViewResults, run_view
from frolic3d.singleview.lift import lift_frames, rel_depth_from_keypoints
from frolic3d.skeleton import HALPE26_TO_MHR70, MHR70_HIPS, MHR70_SHOULDERS, J

log = get_logger(__name__)


def _cameras(work: WorkDir):
    return load_cameras(work.cameras_level if work.cameras_level.exists() else work.cameras)


def mock_outputs(tracks: Tracks, mhr_dir: Path) -> dict[str, Any]:
    """MHR-style outputs straight from the tracks (keypoints through MHR70_BODY, a tube mesh), for tests."""
    info = {}
    faces = MockMhr().faces
    np.save(mhr_dir / "faces.npy", faces)
    for k, pid in enumerate(tracks.ids):
        X = tracks.xyz[k]
        kp = expand17to70(X)
        have = np.isfinite(X[:, 0, 0])
        verts = mock_mesh_from_joints(X)
        N = len(X)
        np.savez_compressed(
            mhr_dir / f"{pid}.npz",
            kp=kp,
            have=have,
            rot=np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)),
            trans=np.nan_to_num(X[:, 0]),
            stature=float(np.nanmedian(np.nanmax(verts[..., 2], 1) - np.nanmin(verts[..., 2], 1)))
            if have.any()
            else 0.0,
            backend="mock",
        )
        np.save(mhr_dir / f"{pid}_verts.npy", verts.astype(np.float16))
        info[pid] = {"frames": int(have.sum())}
    return info


def _torso_m(X: np.ndarray) -> float:
    hip = 0.5 * (X[:, J["hip_l"]] + X[:, J["hip_r"]])
    sho = 0.5 * (X[:, J["shoulder_l"]] + X[:, J["shoulder_r"]])
    d = np.linalg.norm(sho - hip, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("nan")


def refine_single_view(
    work: WorkDir, tracks: Tracks, views: dict[str, ViewResults], cams: CameraSet, dets, cfg: FrolicConfig
) -> Tracks:
    """Re-lift the single-view frames with SAM 3D Body's camera-frame joint depths in place of the template pose."""
    if not cfg.single_view.enabled or not tracks.single_view.any():
        return tracks
    selected = Tracks.load(work.tracks_selected)
    rel: dict[tuple[str, str, int], dict[int, float]] = {}
    for p, pid in enumerate(tracks.ids):
        multi = selected.xyz[p][~selected.single_view[p]]
        torso = _torso_m(multi)
        for cam, vr in views.items():
            for i, o in vr.by_person(p).items():
                if tracks.single_view[p, i]:
                    rel[(pid, cam, int(i))] = rel_depth_from_keypoints(
                        o["pred_keypoints_3d"], torso, HALPE26_TO_MHR70, MHR70_HIPS, MHR70_SHOULDERS
                    )
    if not rel:
        return tracks
    bounds = tuple(cfg.track.room_bounds_m) if cfg.track.room_bounds_m else None
    out, report = lift_frames(
        selected, {c: dets[c] for c in selected.cameras}, cams.subset(selected.cameras), cfg.single_view, rel, bounds
    )
    out.save(work.tracks_singleview)
    with open(work.stage_dir("sam3d") / "svfill_sam3d.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    log.info("single-view frames re-lifted with SAM 3D Body depths (%d estimates)", len(rel))
    return out


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    if not cfg.sam3d.enabled:
        log.info("sam3d disabled in the configuration")
        return {"skipped": True}
    out = work.stage_dir("sam3d")
    mhr_dir = out / "mhr"
    mhr_dir.mkdir(exist_ok=True)
    tracks = Tracks.load(work.fit_input_tracks)
    if cfg.sam3d.backend == "mock":
        return {"backend": "mock", "people": mock_outputs(tracks, mhr_dir)}
    cams = _cameras(work)
    dets = load_all(work.stage_dir("detect"), cams.names)
    pp = out / "prompts.npz"
    if pp.exists():
        prompts = Prompts.load(pp)
    else:
        prompts = gather_prompts(tracks, dets, cams, cfg.sam3d.gate_px)
        prompts.save(pp)
    videos = work.read_videos()
    est = make_estimator(cfg.sam3d.backend, cfg.sam3d.root, cfg.sam3d.checkpoint, cfg.compute.device)
    views: dict[str, ViewResults] = {}
    for cam in cams:
        vp = out / "views" / f"{cam.name}.npz"
        if vp.exists():
            views[cam.name] = ViewResults.load(vp)
            continue
        views[cam.name] = run_view(
            cam, videos["videos"][cam.name], int(videos.get("frame_offset", {}).get(cam.name, 0)), prompts, est, vp
        )
    tracks = refine_single_view(work, tracks, views, cams, dets, cfg)
    if tracks.single_view.any():
        prompts = gather_prompts(tracks, dets, cams, cfg.sam3d.gate_px)
    fitter = MhrFitter(est.mhr, cams, cfg.sam3d, tracks.fps)
    np.save(mhr_dir / "faces.npy", np.asarray(est.mhr.faces, np.int32))
    info: dict[str, Any] = {"people": {}}
    cuts = [int(c) for c in tracks.meta.get("cuts", [])]
    for p, pid in enumerate(tracks.ids):
        if (mhr_dir / f"{pid}.npz").exists():
            info["people"][pid] = {"reused": True}
            continue
        obs = PersonObservations(
            prompts.box[p],
            prompts.kps[p],
            prompts.ks[p],
            prompts.P3[p],
            {c: views[cam].by_person(p) for c, cam in enumerate(cams.names) if cam in views},
            prompts.sizes,
        )
        obs.others = {
            (c, i): [
                prompts.box[q, c, i] for q in range(len(tracks.ids)) if q != p and np.isfinite(prompts.box[q, c, i, 0])
            ]
            for c in range(len(cams))
            for i in range(prompts.n_frames)
        }
        fit = fitter.fit(obs, pid, cuts=cuts)
        if fit is None:
            info["people"][pid] = {"skipped": "too few observations"}
            continue
        fit.save(mhr_dir / f"{pid}.npz", fitter.vertices(fit))
        info["people"][pid] = {
            "frames": int(fit.have.sum()),
            "stature_m": round(fit.stature, 3),
            "size_factor": round(fit.gscale, 3),
        }
    return info
