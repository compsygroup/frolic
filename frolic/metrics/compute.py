"""Export quality metrics: coverage, reprojection vs the detector, jitter, flips, root spins, biomechanics."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from frolic.geometry import Camera, box_iou
from frolic.io.calib import CameraSet, load_cameras
from frolic.io.detections import Detections
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.skeleton import HALPE26, J

log = get_logger(__name__)

SCHEMA = "frolic.metrics/1"
VERTEX_STRIDE = 10
KP_JOINTS = [j for j in range(len(HALPE26)) if j not in (J["pelvis"], J["neck"], J["head"], J["toe_l"], J["toe_r"])]


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    return x[np.isfinite(x)]


def _stat(x: np.ndarray, q: float | None = None) -> float | None:
    x = _finite(x)
    if len(x) == 0:
        return None
    return float(np.median(x) if q is None else np.percentile(x, q))


def load_export(out_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    people = {}
    for p in sorted(out_dir.glob("*.npz")):
        if p.stem.endswith("_joints_full"):
            continue
        z = np.load(p, allow_pickle=False)
        people[p.stem] = {k: z[k] for k in z.files}
    return people


def coverage(vertices: np.ndarray, joints: np.ndarray) -> dict[str, Any]:
    have = np.isfinite(joints[:, 0, 0]) if joints.shape[1] else np.isfinite(vertices[:, 0, 0])
    return {
        "n_frames": len(have),
        "frames_visible": int(have.sum()),
        "coverage_pct": round(100 * float(have.mean()), 1) if len(have) else 0.0,
    }


def jitter(centroids: np.ndarray, fps: float) -> dict[str, Any]:
    acc = np.linalg.norm(centroids[2:] - 2 * centroids[1:-1] + centroids[:-2], axis=1) * fps**2
    return {"jitter_acc_median_ms2": _stat(acc), "jitter_acc_p90_ms2": _stat(acc, 90)}


def vertex_jumps(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max and median vertex displacement vs the previous frame (cm), NaN where either frame is hidden."""
    mx = np.full(len(V), np.nan)
    md = np.full(len(V), np.nan)
    for i in range(1, len(V)):
        a, b = V[i - 1], V[i]
        if np.isfinite(a[0, 0]) and np.isfinite(b[0, 0]):
            d = np.linalg.norm(a - b, axis=1) * 100
            mx[i], md[i] = float(d.max()), float(np.median(d))
    return mx, md


def flips(
    V: np.ndarray, ref_V: np.ndarray | None, jump_cm: float = 10.0, ratio: float = 2.0, body_ratio: float = 4.0
) -> dict[str, Any]:
    """Frames where part of the mesh jumps: max vertex step > jump_cm and > ratio x the reference model's step
    (when a SAM 3D Body surface exists), else > body_ratio x the median vertex step (whole-body motion is no flip)."""
    jm, md = vertex_jumps(V)
    if ref_V is not None and len(ref_V) >= len(V):
        js, _ = vertex_jumps(np.asarray(ref_V[: len(V), ::VERTEX_STRIDE], np.float32))
        ok = np.isfinite(jm) & np.isfinite(js)
        fl = ok & (jm > jump_cm) & (jm > ratio * js)
    else:
        fl = np.isfinite(jm) & (jm > jump_cm) & (jm > body_ratio * md)
    return {"flip_frames": int(fl.sum()), "flip_times_s": [], "vertex_jump_max_cm": _stat(jm, 100), "_flip_mask": fl}


def root_spins(
    rootrot: np.ndarray, fps: float, window_s: float = 1.5, min_deg: float = 150.0, net_frac: float = 0.4
) -> dict[str, Any]:
    """Root turns larger than min_deg within window_s whose net heading change is small (a spin, not a turn)."""
    R = Rotation.from_rotvec(np.nan_to_num(rootrot))
    n = len(rootrot)
    w = max(2, round(window_s * fps))
    rel = np.degrees((R[:-1].inv() * R[1:]).magnitude()) if n > 1 else np.zeros(0)
    yaw = R.as_euler("zyx")[:, 0] if n else np.zeros(0)
    events = []
    i = 0
    while i + w <= n:
        total = float(rel[i : i + w - 1].sum())
        if total > min_deg:
            net = abs(float(np.degrees(np.angle(np.exp(1j * (yaw[i + w - 1] - yaw[i]))))))
            if net < net_frac * total:
                events.append(round((i + int(np.argmax(rel[i : i + w - 1]))) / fps, 2))
                i += w
                continue
        i += 1
    return {"root_spin_events": len(events), "root_spin_times_s": events}


def floor_stats(V: np.ndarray) -> dict[str, Any]:
    have = np.isfinite(V[:, 0, 0])
    if not have.any():
        return {"lowest_vertex_median_cm": None, "floor_below_5cm_pct": None, "stature_p95_m": None}
    zmin = V[have][..., 2].min(1)
    ext = V[have][..., 2].max(1) - zmin
    return {
        "lowest_vertex_median_cm": round(100 * float(np.median(zmin)), 1),
        "floor_below_5cm_pct": round(100 * float((zmin < -0.05).mean()), 1),
        "stature_p95_m": round(float(np.percentile(ext, 95)), 3),
    }


def violations_from_fit(fit_npz: Path) -> dict[str, Any]:
    """Joint-limit / collision counts from the fit, through the biomech module when it is available."""
    try:
        z = np.load(fit_npz, allow_pickle=False)
        if "biomech_report" in z.files:
            rep = json.loads(str(z["biomech_report"]))
            if isinstance(rep, dict) and rep:
                return {"biomech": rep, "biomech_source": "fit report"}
    except Exception as exc:
        log.debug("no biomech report in %s: %s", fit_npz, exc)
    try:
        import torch

        from frolic.fit.biomech import BiomechConstraints  # noqa: F401

        z = np.load(fit_npz, allow_pickle=False)
        if "limit_excess_deg" in z.files:
            E = np.asarray(z["limit_excess_deg"])
            return {
                "biomech": {"limit_frames_gt5deg": int((E.max(axis=tuple(range(1, E.ndim))) > 5).sum())},
                "biomech_source": "fit arrays",
            }
        _ = torch
    except Exception as exc:
        log.debug("biomech module unavailable: %s", exc)
    return {"biomech": {}, "biomech_source": "none"}


def _project_box(cam: Camera, X: np.ndarray) -> tuple[list[float] | None, np.ndarray | None]:
    uv, z = cam.project(X, distort=True)
    if (z <= 0.05).any():
        return None, None
    b = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    bc = [max(0.0, b[0]), max(0.0, b[1]), min(float(cam.width), b[2]), min(float(cam.height), b[3])]
    if bc[2] <= bc[0] or bc[3] <= bc[1]:
        return None, None
    if (bc[2] - bc[0]) * (bc[3] - bc[1]) < 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
        return None, None
    return bc, uv


def reprojection(
    people: dict[str, dict[str, np.ndarray]], cams: CameraSet, dets: dict[str, Detections], stride: int = 1
) -> dict[str, dict[str, Any]]:
    """Mesh bbox IoU against the detector and 2D joint error, one-to-one across people per camera and frame."""
    ids = list(people)
    n = max(len(people[p]["joints"]) for p in ids)
    ious: dict[str, list[float]] = {p: [] for p in ids}
    kp: dict[str, list[float]] = {p: [] for p in ids}
    for i in range(0, n, stride):
        for cam in cams:
            det = dets.get(cam.name)
            if det is None:
                continue
            rows = (
                det.at(int(people[ids[0]]["source_frames"][i]))
                if i < len(people[ids[0]]["source_frames"])
                else np.zeros(0, int)
            )
            boxes, joints2d, who = [], [], []
            for p in ids:
                d = people[p]
                if i >= len(d["joints"]) or not np.isfinite(d["joints"][i, 0, 0]):
                    continue
                V = d["vertices"][i, ::VERTEX_STRIDE].astype(np.float32) if d["vertices"].shape[1] else d["joints"][i]
                b, _ = _project_box(cam, V)
                if b is None:
                    continue
                boxes.append(b)
                joints2d.append(cam.project(d["joints"][i], distort=True)[0])
                who.append(p)
            if not boxes:
                continue
            if len(rows) == 0:
                for p in who:
                    ious[p].append(0.0)
                continue
            dboxes = det.box[rows]
            M = np.array([[box_iou(b, q) for q in dboxes] for b in boxes])
            r, c = linear_sum_assignment(-M)
            assigned = dict(zip(r, c))
            for k, p in enumerate(who):
                if k not in assigned:
                    ious[p].append(0.0)
                    continue
                j = assigned[k]
                ious[p].append(float(M[k, j]))
                if M[k, j] < 0.3:
                    continue
                row = rows[j]
                bh = max(float(det.box[row, 3] - det.box[row, 1]), 1.0)
                sc = det.kps_score[row]
                m = sc[[HALPE26[q] for q in KP_JOINTS]] > 0.5
                if m.sum() < 6:
                    continue
                e = np.linalg.norm(joints2d[k][KP_JOINTS] - det.kps[row][[HALPE26[q] for q in KP_JOINTS]], axis=1)[m]
                kp[p].append(float(np.median(e) / bh))
    out = {}
    for p in ids:
        io = np.array(ious[p])
        out[p] = {
            "box_iou_median": _stat(io),
            "box_iou_ge_0.5_pct": round(100 * float((io >= 0.5).mean()), 1) if len(io) else None,
            "n_box_pairs": len(io),
            "kp2d_err_median": _stat(np.array(kp[p])),
        }
    return out


def compute_metrics(work: WorkDir, out_dir: Path, fit_dir: Path | None = None) -> dict[str, Any]:
    people = load_export(out_dir)
    with open(out_dir / "meta.json") as fh:
        meta = json.load(fh)
    fps = float(meta["fps"])
    cams = load_cameras(out_dir / "cameras.json") if (out_dir / "cameras.json").exists() else None
    det_dir = work.stage_dir("detect")
    dets = {}
    if cams is not None:
        for c in cams.names:
            p = det_dir / f"{c}.npz"
            if p.exists():
                dets[c] = Detections.load(p)
    per: dict[str, dict[str, Any]] = {}
    for pid, d in people.items():
        V = np.asarray(d["vertices"], np.float32)
        Vs = V[:, ::VERTEX_STRIDE] if V.shape[1] else d["joints"]
        m: dict[str, Any] = coverage(V, d["joints"])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cen = np.nanmean(Vs, axis=1) if Vs.shape[1] else np.full((len(V), 3), np.nan)
        m.update(jitter(cen, fps))
        ref = None
        mhr_v = work.stage_dir("sam3d") / "mhr" / f"{pid}_verts.npy"
        if mhr_v.exists():
            ref = np.load(mhr_v, mmap_mode="r")
        fl = flips(Vs, ref)
        mask = fl.pop("_flip_mask")
        fl["flip_times_s"] = [round(float(t), 2) for t in (np.flatnonzero(mask) / fps)[:50]]
        m.update(fl)
        m.update(floor_stats(Vs))
        fit_npz = (fit_dir or work.stage_dir("fit")) / f"{pid}.npz"
        if fit_npz.exists():
            z = np.load(fit_npz, allow_pickle=False)
            if "rootrot" in z.files:
                m.update(root_spins(np.asarray(z["rootrot"]), fps))
            m.update(violations_from_fit(fit_npz))
        m["stature_m"] = round(float(d["stature_m"]), 3)
        m["label"] = str(d["label"])
        per[pid] = m
    if cams is not None and dets and people:
        for pid, r in reprojection(people, cams, dets).items():
            per[pid].update(r)
    ids = list(per)
    glob = {
        "n_people": len(ids),
        "n_cameras": len(cams) if cams is not None else None,
        "coverage_min_pct": min(per[p]["coverage_pct"] for p in ids) if ids else None,
        "box_iou_median_mean": float(
            np.mean([per[p]["box_iou_median"] for p in ids if per[p].get("box_iou_median") is not None])
        )
        if any(per[p].get("box_iou_median") is not None for p in ids)
        else None,
        "flip_frames_total": int(sum(per[p]["flip_frames"] for p in ids)),
    }
    return {"schema": SCHEMA, "fps": fps, "per_person": per, "global": glob}


SUMMARY_COLUMNS = [
    "id", "label", "stature_m", "coverage_pct", "box_iou_median", "box_iou_ge_0.5_pct", "kp2d_err_median",
    "jitter_acc_median_ms2", "flip_frames", "root_spin_events", "lowest_vertex_median_cm", "floor_below_5cm_pct",
]  # fmt: skip


def write_summary_csv(metrics: dict[str, Any], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLUMNS)
        for pid, m in metrics["per_person"].items():
            w.writerow([pid] + [m.get(k, "") if m.get(k) is not None else "" for k in SUMMARY_COLUMNS[1:]])
