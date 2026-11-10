"""ChArUco multi-camera calibration from videos: diverse-frame intrinsics with sanity checks, IPPE board poses, chained
extrinsics and a multi-frame bundle adjustment. Board model and JSON files follow the ChArUco calibrator GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from frolic3d.config import BoardConfig
from frolic3d.geometry import Camera, rotation_angle_deg
from frolic3d.io.calib import CameraSet
from frolic3d.io.video import iter_frames
from frolic3d.log import get_logger

log = get_logger(__name__)

MM_THRESHOLD = 5.0  # square lengths above this are millimetres


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 8
    squares_y: int = 6
    square_length_m: float = 0.095
    marker_length_m: float = 0.071
    dictionary: str = "DICT_4X4_100"
    legacy_pattern: bool = True

    @classmethod
    def from_config(cls, c: BoardConfig) -> BoardSpec:
        k = 1e-3 if c.square_length > MM_THRESHOLD else 1.0
        return cls(c.squares_x, c.squares_y, c.square_length * k, c.marker_length * k, c.dictionary, c.legacy_pattern)

    def to_gui_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length": self.square_length_m,
            "marker_length": self.marker_length_m,
            "dictionary_name": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
            "units": "m",
        }

    def make_board(self) -> Any:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length_m, self.marker_length_m, d)
        if self.legacy_pattern:
            board.setLegacyPattern(True)
        return board

    def object_points(self) -> np.ndarray:
        return np.asarray(self.make_board().getChessboardCorners(), np.float64).reshape(-1, 3)

    def make_detector(self) -> Any:
        return cv2.aruco.CharucoDetector(self.make_board())


@dataclass
class BoardDetection:
    frame: int
    corners: np.ndarray  # (n, 2) float32 pixels
    ids: np.ndarray  # (n,) int32

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    n_frames: int
    issues: list[str] = field(default_factory=list)
    degenerate: bool = False
    tilt_deg: list[float] = field(default_factory=list)


def detect_in_image(detector: Any, gray: np.ndarray, min_corners: int) -> tuple[np.ndarray, np.ndarray] | None:
    cc, ci, _, _ = detector.detectBoard(gray)
    if ci is None or len(ci) < min_corners:
        return None
    return cc.reshape(-1, 2).astype(np.float32), ci.ravel().astype(np.int32)


def detect_in_video(
    path: Path | str, spec: BoardSpec, step: int = 3, min_corners: int = 12, max_frames: int = 0
) -> list[BoardDetection]:
    det = spec.make_detector()
    out = []
    for f, img in iter_frames(path, step=step):
        if max_frames and f >= max_frames:
            break
        r = detect_in_image(det, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), min_corners)
        if r is not None:
            out.append(BoardDetection(f, *r))
    log.info("%s: board in %d frames", Path(path).name, len(out))
    return out


def _orientation_deg(corners: np.ndarray) -> float:
    c = corners - corners.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180.0)


def select_diverse(dets: list[BoardDetection], size: tuple[int, int], n: int) -> list[BoardDetection]:
    """Spread the picks over image cells (4x3), board size (3 bins) and orientation (4 bins); most corners first."""
    if len(dets) <= n:
        return list(dets)
    w, h = size
    area = np.array([cv2.contourArea(cv2.convexHull(d.corners.astype(np.float32))) for d in dets])
    abin = np.digitize(area, np.quantile(area, [1 / 3, 2 / 3]))
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for i, d in enumerate(dets):
        cx, cy = d.corners.mean(0)
        key = (min(int(cx / w * 4), 3), min(int(cy / h * 3), 2), int(abin[i]), int(_orientation_deg(d.corners) // 45))
        groups.setdefault(key, []).append(i)
    for g in groups.values():
        g.sort(key=lambda i: -dets[i].n)
    picked: list[int] = []
    while len(picked) < n and any(groups.values()):
        for g in list(groups.values()):
            if g and len(picked) < n:
                picked.append(g.pop(0))
    return [dets[i] for i in sorted(picked)]


def check_intrinsics(
    K: np.ndarray, dist: np.ndarray, width: int, height: int, tilt_deg: list[float] | None = None
) -> list[str]:
    issues = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(fx / fy - 1) > 0.05:
        issues.append("aspect")
    if abs(cx - width / 2) > 0.15 * width or abs(cy - height / 2) > 0.15 * height:
        issues.append("principal_point")
    if not 0.3 * width <= fx <= 3.0 * width or not 0.3 * width <= fy <= 3.0 * width:
        issues.append("focal")
    if np.abs(dist[:2]).max() > 1.5:
        issues.append("distortion")
    if tilt_deg is not None and len(tilt_deg) >= 3:
        t = np.asarray(tilt_deg)
        if t.std() < 5.0 and (t.max() - t.min()) < 15.0:
            issues.append("low_view_diversity")
    return issues


def _calib_flags(fix_aspect: bool, fix_k3: bool, strict: bool = False) -> int:
    flags = 0
    if fix_aspect or strict:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if fix_k3 or strict:
        flags |= cv2.CALIB_FIX_K3
    if strict:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2
    return flags


def _run_calibrate(
    obj: np.ndarray, dets: list[BoardDetection], size: tuple[int, int], flags: int
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    op = [obj[d.ids].astype(np.float32) for d in dets]
    ip = [d.corners.reshape(-1, 1, 2) for d in dets]
    w, h = size
    K0 = np.array([[1.2 * w, 0, w / 2], [0, 1.2 * w, h / 2], [0, 0, 1]], float)
    rms, K, dist, rvecs, _ = cv2.calibrateCamera(
        op, ip, size, K0, np.zeros(5), flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS
    )
    tilt = [float(np.degrees(np.arccos(np.clip(cv2.Rodrigues(r)[0][2, 2], -1, 1)))) for r in rvecs]
    return float(rms), K, dist.ravel()[:5], tilt


def calibrate_intrinsics(
    dets: list[BoardDetection],
    size: tuple[int, int],
    spec: BoardSpec,
    n_frames: int = 60,
    fix_aspect: bool = True,
    fix_k3: bool = True,
    min_frames: int = 8,
) -> Intrinsics:
    obj = spec.object_points()
    use = select_diverse(dets, size, n_frames)
    if len(use) < min_frames:
        raise RuntimeError(f"only {len(use)} usable board frames (need {min_frames})")
    rms, K, dist, tilt = _run_calibrate(obj, use, size, _calib_flags(fix_aspect, fix_k3))
    issues = check_intrinsics(K, dist, *size, tilt)
    degenerate = False
    if any(i in issues for i in ("aspect", "principal_point", "focal", "distortion")):
        log.warning(
            "implausible intrinsics (%s, rms %.2f px): refitting with a constrained model", ",".join(issues), rms
        )
        rms, K, dist, tilt = _run_calibrate(obj, use, size, _calib_flags(fix_aspect, fix_k3, strict=True))
        issues = check_intrinsics(K, dist, *size, tilt)
        degenerate = True
    elif "low_view_diversity" in issues:
        log.warning(
            "board poses lack view-angle diversity (tilt std %.1f deg): focal length is poorly constrained",
            np.std(tilt),
        )
        degenerate = True
    return Intrinsics(K, dist, rms, len(use), issues, degenerate, [round(t, 1) for t in tilt])


def board_pose(
    det: BoardDetection, K: np.ndarray, dist: np.ndarray, obj: np.ndarray, max_err_px: float = 3.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Board -> camera (R, t) by IPPE; None when PnP fails or reprojects badly."""
    if det.n < 6:
        return None
    ok, rv, tv = cv2.solvePnP(obj[det.ids], det.corners.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    pr, _ = cv2.projectPoints(obj[det.ids], rv, tv, K, dist)
    if np.linalg.norm(pr.reshape(-1, 2) - det.corners, axis=1).mean() > max_err_px:
        return None
    return cv2.Rodrigues(rv)[0], tv.ravel()


def _relative(pa: dict[int, tuple], pb: dict[int, tuple], frames: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """T_b<-a from shared board poses (chordal-mean rotation, median translation)."""
    Rs, ts = [], []
    for f in frames:
        Ra, ta = pa[f]
        Rb, tb = pb[f]
        R = Rb @ Ra.T
        Rs.append(R)
        ts.append(tb - R @ ta)
    U, _, Vt = np.linalg.svd(sum(Rs))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, np.median(np.array(ts), 0)


def chain_poses(
    poses: dict[str, dict[int, tuple]], min_shared: int = 3
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str, list[str]]:
    """Camera poses in the frame of the best-connected camera (X_cam = R X_root + t), chained over shared frames."""
    views = list(poses)
    shared = {(p, q): sorted(set(poses[p]) & set(poses[q])) for p in views for q in views if p != q}
    root = max(views, key=lambda v: sum(len(shared[(v, w)]) for w in views if w != v))
    placed: dict[str, tuple[np.ndarray, np.ndarray]] = {root: (np.eye(3), np.zeros(3))}
    todo = [v for v in views if v != root]
    links = []
    while todo:
        p, q, n = max(((p, q, len(shared[(p, q)])) for p in placed for q in todo), key=lambda x: x[2])
        if n < min_shared:
            raise RuntimeError(f"cameras {todo} share < {min_shared} board frames with the calibrated ones")
        R, t = _relative(poses[p], poses[q], shared[(p, q)])
        Rp, tp = placed[p]
        placed[q] = (R @ Rp, R @ tp + t)
        todo.remove(q)
        links.append(f"{p}->{q}:{n}")
    return placed, root, links


def _distort_project(Xc: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = dist[:5]
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def bundle_adjust(
    cams: list[str],
    placed: dict[str, tuple[np.ndarray, np.ndarray]],
    root: str,
    poses: dict[str, dict[int, tuple]],
    dets: dict[str, dict[int, BoardDetection]],
    intr: dict[str, tuple[np.ndarray, np.ndarray]],
    obj: np.ndarray,
    max_frames: int = 600,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Refine camera and board poses jointly (root fixed) with a Huber loss on corner reprojection."""
    frames = sorted({f for v in cams for f in poses[v] if sum(f in poses[w] for w in cams) >= 2})
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    fidx = {f: i for i, f in enumerate(frames)}
    cidx = {v: i for i, v in enumerate(cams)}
    board0 = {}
    for f in frames:
        v = next(w for w in cams if f in poses[w])
        R, t = poses[v][f]
        Rw, tw = placed[v]
        board0[f] = (Rw.T @ R, Rw.T @ (t - tw))
    pc, pf, pts, uv = [], [], [], []
    for v in cams:
        for f in frames:
            if f in poses[v] and f in dets[v]:
                d = dets[v][f]
                pc.append(np.full(d.n, cidx[v]))
                pf.append(np.full(d.n, fidx[f]))
                pts.append(obj[d.ids])
                uv.append(d.corners.astype(np.float64))
    pc, pf, pts, uv = np.concatenate(pc), np.concatenate(pf), np.concatenate(pts), np.concatenate(uv)
    Ks = [intr[v][0] for v in cams]
    Ds = [intr[v][1] for v in cams]
    nc, nf = len(cams), len(frames)
    x0 = np.concatenate(
        [np.r_[Rotation.from_matrix(placed[v][0]).as_rotvec(), placed[v][1]] for v in cams]
        + [np.r_[Rotation.from_matrix(board0[f][0]).as_rotvec(), board0[f][1]] for f in frames]
    )
    ri = cidx[root]

    def resid(x: np.ndarray) -> np.ndarray:
        cp = x[: 6 * nc].reshape(nc, 6)
        bp = x[6 * nc :].reshape(nf, 6)
        Rc = Rotation.from_rotvec(cp[:, :3]).as_matrix()
        Rb = Rotation.from_rotvec(bp[:, :3]).as_matrix()
        Xr = np.einsum("nij,nj->ni", Rb[pf], pts) + bp[pf, 3:]
        Xc = np.einsum("nij,nj->ni", Rc[pc], Xr) + cp[pc, 3:]
        out = np.empty_like(uv)
        for c in range(nc):
            m = pc == c
            out[m] = _distort_project(Xc[m], Ks[c], Ds[c])
        return (out - uv).ravel()

    S = lil_matrix((2 * len(pc), len(x0)), dtype=int)
    rows = np.repeat(np.arange(len(pc)), 2) * 2 + np.tile([0, 1], len(pc))
    for k in range(6):
        S[rows, np.repeat(6 * pc + k, 2)] = 1
        S[rows, np.repeat(6 * nc + 6 * pf + k, 2)] = 1
    S[:, 6 * ri : 6 * ri + 6] = 0
    r0 = np.linalg.norm(resid(x0).reshape(-1, 2), axis=1)
    res = least_squares(resid, x0, jac_sparsity=S, loss="huber", f_scale=2.0, max_nfev=80, x_scale="jac")
    r1 = np.linalg.norm(res.fun.reshape(-1, 2), axis=1)
    cp = res.x[: 6 * nc].reshape(nc, 6)
    out = {v: (Rotation.from_rotvec(cp[cidx[v], :3]).as_matrix(), cp[cidx[v], 3:].copy()) for v in cams}
    Rr, tr = out[root]
    out = {v: (R @ Rr.T, t - R @ Rr.T @ tr) for v, (R, t) in out.items()}  # keep the root at the identity
    report = {
        "frames": len(frames),
        "points": len(pc),
        "median_px_before": float(np.median(r0)),
        "median_px_after": float(np.median(r1)),
        "per_camera_median_px": {v: float(np.median(r1[pc == cidx[v]])) for v in cams},
    }
    return out, report


def _board_in_root(
    placed: dict[str, tuple], poses: dict[str, dict[int, tuple]], v: str, f: int
) -> tuple[np.ndarray, np.ndarray]:
    Rb, tb = poses[v][f]
    Rw, tw = placed[v]
    return Rw.T @ Rb, Rw.T @ (tb - tw)


def _floor_world(
    placed: dict[str, tuple], poses: dict[str, dict[int, tuple]], cams: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """World = plane of the boards lying on the floor, z towards the cameras: (R, t) with X_new = R X_root + t.

    The frame is the most-observed board frame among those whose normal agrees with the median board normal."""
    frames: dict[int, tuple[str, np.ndarray]] = {}
    counts: dict[int, int] = {}
    for v in cams:
        for f in poses[v]:
            counts[f] = counts.get(f, 0) + 1
            frames.setdefault(f, (v, _board_in_root(placed, poses, v, f)[0][:, 2]))
    normals = np.array([n for _, n in frames.values()])
    ref = normals[np.argmax(normals @ normals.T @ np.ones(len(normals)))]
    agree = [f for f, (_, n) in frames.items() if abs(n @ ref) > np.cos(np.radians(10))]
    f = max(agree, key=lambda k: (counts[k], -k))
    Rbw, tbw = _board_in_root(placed, poses, frames[f][0], f)
    R, t = Rbw.T, -Rbw.T @ tbw
    centres = np.array([R @ (-placed[w][0].T @ placed[w][1]) + t for w in cams])
    if np.median(centres[:, 2]) < 0:
        F = np.diag([1.0, -1.0, -1.0])
        R, t = F @ R, F @ t
    return R, t


def calibrate_multi(
    videos: dict[str, Path],
    spec: BoardSpec,
    cfg: BoardConfig,
    sizes: dict[str, tuple[int, int]],
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    on_floor: bool = False,
    detections: dict[str, list[BoardDetection]] | None = None,
) -> tuple[CameraSet, dict[str, Any]]:
    """Detect the board in every video, calibrate each camera, chain and bundle-adjust the rig (metric from the board)."""
    obj = spec.object_points()
    dets = detections or {v: detect_in_video(p, spec, cfg.step, cfg.min_corners) for v, p in videos.items()}
    intrinsics = dict(intrinsics or {})
    report: dict[str, Any] = {
        "n_detections": {v: len(d) for v, d in dets.items()},
        "intrinsics": {},
        "board": spec.to_gui_dict(),
    }
    for v, d in dets.items():
        if v in intrinsics:
            report["intrinsics"][v] = {"source": "given"}
            continue
        r = calibrate_intrinsics(d, sizes[v], spec, cfg.n_intrinsic_frames, cfg.fix_aspect_ratio, cfg.fix_k3)
        intrinsics[v] = (r.K, r.dist)
        report["intrinsics"][v] = {
            "n_frames": r.n_frames,
            "rms_px": r.rms_px,
            "fx": float(r.K[0, 0]),
            "fy": float(r.K[1, 1]),
            "cx": float(r.K[0, 2]),
            "cy": float(r.K[1, 2]),
            "dist": np.round(r.dist, 5).tolist(),
            "issues": r.issues,
            "degenerate": r.degenerate,
            "tilt_deg_range": [min(r.tilt_deg), max(r.tilt_deg)] if r.tilt_deg else None,
        }
        log.info(
            "%s: %d frames, rms %.2f px, fx %.1f%s",
            v,
            r.n_frames,
            r.rms_px,
            r.K[0, 0],
            " DEGENERATE" if r.degenerate else "",
        )
    poses = {v: {d.frame: p for d in dets[v] if (p := board_pose(d, *intrinsics[v], obj)) is not None} for v in dets}
    bydet = {v: {d.frame: d for d in dets[v]} for v in dets}
    placed, root, links = chain_poses(poses)
    cams = [v for v in dets if v in placed]
    report["chain"] = {"root": root, "links": links}
    placed, ba = bundle_adjust(cams, placed, root, poses, bydet, intrinsics, obj, cfg.max_ba_frames)
    report["ba"] = ba
    if on_floor:
        R, t = _floor_world(placed, poses, cams)
        placed = {v: (Rw @ R.T, tw - Rw @ R.T @ t) for v, (Rw, tw) in placed.items()}
        report["world"] = "board_plane"
    else:
        report["world"] = f"camera:{root}"
    out = [Camera(v, sizes[v][0], sizes[v][1], intrinsics[v][0], intrinsics[v][1], *placed[v]) for v in cams]
    cs = CameraSet(
        out, "m", {"levelled": on_floor, "floor_z": 0.0} if on_floor else {}, {"format": "board", "root": root}
    )
    cs.source["rms_px"] = {v: report["intrinsics"][v].get("rms_px", 0.0) for v in cams}
    C = cs.centers
    report["baselines_m"] = {
        f"{a}-{b}": float(np.linalg.norm(C[i] - C[j])) for i, a in enumerate(cams) for j, b in enumerate(cams) if i < j
    }
    report["degenerate_cameras"] = [v for v in cams if report["intrinsics"][v].get("degenerate")]
    return cs, report


def compare_to_reference(cams: CameraSet, ref: CameraSet) -> dict[str, Any]:
    """Centre and rotation differences after a rigid alignment on the common cameras (diagnostics)."""
    from frolic3d.geometry import umeyama

    common = [n for n in cams.names if n in ref.names]
    A = np.array([cams[n].center for n in common])
    B = np.array([ref[n].center for n in common])
    s, R, t = umeyama(A, B, with_scale=False)
    out = {}
    for n in common:
        c = cams[n].transformed(R, t, s)
        out[n] = {
            "centre_mm": 1000 * float(np.linalg.norm(c.center - ref[n].center)),
            "rot_deg": rotation_angle_deg(c.R, ref[n].R),
        }
    return out
