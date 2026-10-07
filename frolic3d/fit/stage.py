"""Stage `fit`: Anny body per selected person from the tracks (+ SAM 3D Body when present)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from frolic3d.config import FrolicConfig
from frolic3d.fit.anny_model import make_body
from frolic3d.fit.fitter import FitInputs, PersonFitter
from frolic3d.io.tracks import Tracks
from frolic3d.io.workdir import WorkDir
from frolic3d.log import get_logger

log = get_logger(__name__)


def pick_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, fitting on the CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_sam_inputs(mhr_dir: Path, pid: str, n_frames: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    f = mhr_dir / f"{pid}.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    kp = np.asarray(z["kp"], np.float32)
    if "have" in z.files:
        kp = kp.copy()
        kp[~np.asarray(z["have"], bool)] = np.nan
    if len(kp) < n_frames:
        kp = np.concatenate([kp, np.full((n_frames - len(kp), *kp.shape[1:]), np.nan, np.float32)])
    vf = mhr_dir / f"{pid}_verts.npy"
    verts = np.load(vf, mmap_mode="r") if vf.exists() else None
    return kp[:n_frames], verts


def run(cfg: FrolicConfig, work: WorkDir) -> dict[str, Any]:
    tracks = Tracks.load(work.fit_input_tracks)
    out_dir = work.stage_dir("fit")
    dev = pick_device(cfg.fit.device)
    fps = cfg.fit_fps
    body = make_body(cfg.fit.backend, dev, cfg.fit.anny_cache, cfg.fit.dense_n, cfg.fit.segments)
    np.save(out_dir / "faces.npy", np.asarray(body.faces, np.int32))
    fitter = PersonFitter(body, cfg.fit, fps)
    mhr_dir = work.stage_dir("sam3d") / "mhr"
    cuts = [int(c) for c in tracks.meta.get("cuts", [])]
    info: dict[str, Any] = {"people": {}}
    for k, pid in enumerate(tracks.ids):
        sam_kp, sam_verts = load_sam_inputs(mhr_dir, pid, tracks.n_frames) if cfg.sam3d.enabled else (None, None)
        inp = FitInputs(
            tracks.xyz[k], tracks.single_view[k], tracks.observed[k], sam_kp, sam_verts, cuts, cfg.people.get(pid)
        )
        res = fitter.fit(inp, pid)
        if res is None:
            info["people"][pid] = {"skipped": "too few frames"}
            continue
        res.frame0 = tracks.frame0
        verts = fitter.vertices(res) if cfg.fit.write_vertices else None
        res.save(out_dir / f"{pid}.npz", verts)
        info["people"][pid] = {
            "stature_m": round(res.stature, 3),
            "frames": int(res.have.sum()),
            **{k_: v for k_, v in res.diagnostics.items() if isinstance(v, (int, float)) or v is None},
        }
    with open(out_dir / "fit.json", "w") as fh:
        json.dump(info, fh, indent=1)
    return info
