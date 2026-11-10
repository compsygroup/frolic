"""Cross-view association + triangulation of multi-person 2D poses into per-frame clusters and raw tracks."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from frolic3d.config import TrackConfig
from frolic3d.geometry import Camera, triangulate
from frolic3d.io.calib import CameraSet
from frolic3d.io.detections import Detections
from frolic3d.io.tracks import Tracks
from frolic3d.log import get_logger
from frolic3d.skeleton import HALPE26, HALPE26_HIP, HALPE26_NECK, HALPE26_SHOULDERS, N_JOINTS, J

log = get_logger(__name__)
ANCHORS = (HALPE26_HIP, HALPE26_NECK)


@dataclass
class Candidate:
    view: int
    row: int
    kps: np.ndarray  # (26, 2) undistorted pixels
    score: np.ndarray  # (26,)
    box: np.ndarray
    colour: np.ndarray | None  # (3,) feature or None


def colour_feature(hsv: np.ndarray, vmean: float, vstd: float) -> np.ndarray | None:
    if not np.isfinite(hsv).all():
        return None
    h, _, v = hsv
    return np.array([np.cos(h * np.pi / 90), np.sin(h * np.pi / 90), (v - vmean) / vstd])


def colour_stats(dets: dict[str, Detections]) -> dict[str, tuple[float, float]]:
    out = {}
    for c, d in dets.items():
        v = d.colour[:, 2]
        v = v[np.isfinite(v)]
        out[c] = (float(v.mean()), float(v.std()) + 1e-6) if len(v) else (0.0, 1.0)
    return out


def candidates_at(
    frame: int, dets: dict[str, Detections], cams: CameraSet, stats: dict[str, tuple[float, float]], min_score: float
) -> list[Candidate]:
    out = []
    for vi, cam in enumerate(cams):
        d = dets[cam.name]
        for r in d.at(frame):
            sc = d.kps_score[r]
            if min(sc[a] for a in ANCHORS) < min_score:
                continue
            kp = cam.undistort_points(d.kps[r])
            out.append(Candidate(vi, int(r), kp, sc, d.box[r], colour_feature(d.colour[r], *stats[cam.name])))
    return out


def estimate_people(frames: list[int], dets: dict[str, Detections], cap: int = 8) -> int:
    """75th percentile over frames of the max concurrent confident detections in one camera; counted on pelvis+neck
    and on neck+both shoulders (a seated person behind a desk never shows the pelvis), the larger wins."""
    per_frame, per_frame_sh = [], []
    for f in frames:
        best = best_sh = 0
        for d in dets.values():
            rows = d.at(f)
            if not len(rows):
                continue
            ks = d.kps_score[rows]
            best = max(best, int(((ks[:, HALPE26_HIP] > 0.5) & (ks[:, HALPE26_NECK] > 0.5)).sum()))
            best_sh = max(best_sh, int(((ks[:, HALPE26_NECK] > 0.5) & (ks[:, HALPE26_SHOULDERS] > 0.5).all(1)).sum()))
        per_frame.append(best)
        per_frame_sh.append(best_sh)
    if not per_frame:
        return 1
    n_pel, n_sh = round(np.percentile(per_frame, 75)), round(np.percentile(per_frame_sh, 75))
    log.info("people estimate: pelvis+neck %d, neck+shoulders %d", n_pel, n_sh)
    return int(min(cap, max(1, n_pel, n_sh)))


def reproj_err(cam: Camera, X: np.ndarray, uv: np.ndarray) -> float:
    p, z = cam.project(X)
    return np.inf if z <= 0 else float(np.linalg.norm(p - uv))


def cluster_frame(
    C: list[Candidate], cams: CameraSet, n_people: int, min_views: int, reproj_px: float
) -> list[tuple[list[int], np.ndarray]]:
    """Greedy RANSAC on pelvis+neck pair triangulations -> [(member indices, anchors (2, 3))]."""
    remaining = list(range(len(C)))
    clusters: list[tuple[list[int], np.ndarray]] = []
    pair_cache: dict[tuple[int, int], np.ndarray] = {}
    while len(clusters) < n_people and len(remaining) >= 2:
        best = None
        for i, j in itertools.combinations(remaining, 2):
            if C[i].view == C[j].view:
                continue
            if (i, j) not in pair_cache:
                X = np.stack(
                    [triangulate([cams[C[i].view], cams[C[j].view]], [C[i].kps[a], C[j].kps[a]]) for a in ANCHORS]
                )
                pair_cache[(i, j)] = X
            X = pair_cache[(i, j)]
            if not np.isfinite(X).all():
                continue
            inl: dict[int, tuple[int, float]] = {}
            for m in remaining:
                v = C[m].view
                e = np.mean([reproj_err(cams[v], X[q], C[m].kps[a]) for q, a in enumerate(ANCHORS)])
                if e < reproj_px and (v not in inl or e < inl[v][1]):
                    inl[v] = (m, e)
            if len(inl) >= min_views:
                sc = (len(inl), -sum(e for _, e in inl.values()))
                if best is None or sc > best[0]:
                    best = (sc, inl, X)
        if best is None:
            break
        mem = [m for m, _ in best[1].values()]
        clusters.append((mem, best[2]))
        remaining = [m for m in remaining if m not in mem]
    return merge_duplicates(clusters, C)


def merge_duplicates(
    clusters: list[tuple[list[int], np.ndarray]], C: list[Candidate], pelvis_m: float = 0.30, neck_m: float = 0.25
) -> list[tuple[list[int], np.ndarray]]:
    merged: list[tuple[list[int], np.ndarray]] = []
    for mem, X in clusters:
        hit = next(
            (
                k
                for k, (_, X2) in enumerate(merged)
                if np.linalg.norm(X[0] - X2[0]) < pelvis_m and np.linalg.norm(X[1] - X2[1]) < neck_m
            ),
            None,
        )
        if hit is None:
            merged.append((list(mem), X))
        else:
            have = {C[m].view for m in merged[hit][0]}
            merged[hit] = (merged[hit][0] + [m for m in mem if C[m].view not in have], merged[hit][1])
    return merged


def triangulate_cluster(
    mem: list[int], C: list[Candidate], cams: CameraSet, min_score: float, reproj_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """(17, 3) joints and (17,) scores from the cluster members (>= 2 cameras per joint, reprojection-checked)."""
    xyz = np.full((N_JOINTS, 3), np.nan)
    sc = np.zeros(N_JOINTS)
    for j, h in enumerate(HALPE26):
        use = [m for m in mem if C[m].score[h] > min_score]
        if len(use) < 2:
            continue
        X = triangulate([cams[C[m].view] for m in use], [C[m].kps[h] for m in use], [float(C[m].score[h]) for m in use])
        if not np.isfinite(X).all():
            continue
        if np.median([reproj_err(cams[C[m].view], X, C[m].kps[h]) for m in use]) < reproj_px * 1.5:
            xyz[j] = X
            sc[j] = float(np.mean([C[m].score[h] for m in use]))
    return xyz, sc


class TrackState:
    def __init__(self, n: int) -> None:
        self.prev: list[np.ndarray | None] = [None] * n
        self.colour: list[np.ndarray | None] = [None] * n
        self.size: list[float | None] = [None] * n
        self.last_seen = [-1] * n
        self.started = [False] * n


def _colour_dist(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.35
    return float(np.linalg.norm(a - b))


def assign(
    clusters: list[tuple[list[int], np.ndarray]],
    C: list[Candidate],
    st: TrackState,
    cfg: TrackConfig,
    appearance: bool,
    i: int = 0,
    fps: float = 15.0,
) -> list[tuple[int, int]]:
    """Hungarian assignment of clusters to track slots -> [(cluster index, slot)]."""
    n = len(st.prev)
    ccols = [
        np.mean([C[m].colour for m in mem if C[m].colour is not None], 0)
        if any(C[m].colour is not None for m in mem)
        else None
        for mem, _ in clusters
    ]
    csize = [float(np.linalg.norm(X[0] - X[1])) for _, X in clusters]
    cost = np.full((len(clusters), n), 9.0)
    for ci, (_, X) in enumerate(clusters):
        for t in range(n):
            app = 0.0
            if appearance:
                sd = abs(csize[ci] - st.size[t]) if st.size[t] is not None else 0.1
                app = 0.8 * _colour_dist(ccols[ci], st.colour[t]) + 2.5 * sd
            if st.prev[t] is not None:
                cost[ci, t] = np.linalg.norm(X[0] - st.prev[t]) + app
            elif not st.started[t]:
                cost[ci, t] = 3.5 + app
            else:
                cost[ci, t] = 1.0 + 1.5 * app
    def accept(ci: int, t: int) -> bool:
        # jump gate grows with the time since the slot last saw its person (a slot keeps its last position for
        # lost_frames; with a fixed gate a person who moved on while unseen was rejected every frame until the slot
        # was freed: DBP gait pilot T08, 3-7 s, 113 rejections, kids untracked)
        if st.prev[t] is not None:
            dt = max(0, i - st.last_seen[t]) / fps
            if np.linalg.norm(clusters[ci][1][0] - st.prev[t]) > cfg.max_jump_m + 1.0 + cfg.max_speed_mps * dt:
                return False
        # appearance gates only while the slot is being tracked (continuity); a free slot takes any cluster and the
        # identity stage decides who it was (hard gates on a free slot's stale memory left people untracked for whole
        # takes: DBP gait pilot T08, 6-year-old 0 % of 4-12 s)
        tracked = st.prev[t] is not None
        if (
            appearance
            and tracked
            and st.colour[t] is not None
            and ccols[ci] is not None
            and _colour_dist(ccols[ci], st.colour[t]) > 1.4
        ):
            return False
        if appearance and tracked and st.size[t] is not None and abs(csize[ci] - st.size[t]) > 0.22:
            return False
        return True

    # Hungarian assignment, re-run on the leftovers after every rejection: a cluster the optimal pairing sent to a
    # tracked slot too far away (rejected) still gets a free slot instead of being dropped for the frame
    out: list[tuple[int, int]] = []
    banned: set[tuple[int, int]] = set()
    free_c, free_t = list(range(len(clusters))), list(range(n))
    while free_c and free_t:
        sub = cost[np.ix_(free_c, free_t)].copy()
        for ci, t in banned:
            if ci in free_c and t in free_t:
                sub[free_c.index(ci), free_t.index(t)] = 1e6
        progress = False
        for ri, cj in zip(*linear_sum_assignment(sub)):
            ci, t = free_c[ri], free_t[cj]
            if sub[ri, cj] >= 1e6:
                continue
            if not accept(ci, t):
                # last resort: the slot has not seen its person for >= 1 s and no other slot is left for this
                # (well-triangulated) cluster -> the slot gives way and restarts on it; cut_segments splits the
                # slot there (position jump) and the identity stage links the pieces by ReID
                stale = st.prev[t] is None or (i - st.last_seen[t]) >= fps
                if stale and len(free_t) == 1 and (ci, t) not in banned:
                    st.prev[t] = None
                    st.colour[t] = None
                    st.size[t] = None
                else:
                    banned.add((ci, t))
                    progress = True
                    continue
            out.append((int(ci), int(t)))
            if ccols[ci] is not None:
                st.colour[t] = ccols[ci] if st.colour[t] is None else 0.98 * st.colour[t] + 0.02 * ccols[ci]
            st.size[t] = csize[ci] if st.size[t] is None else 0.98 * st.size[t] + 0.02 * csize[ci]
            progress = True
        free_c = [ci for ci in free_c if ci not in {c for c, _ in out}]
        free_t = [t for t in free_t if t not in {t_ for _, t_ in out}]
        if not progress or all((ci, t) in banned for ci in free_c for t in free_t):
            break
    return out


def associate(
    dets: dict[str, Detections], cams: CameraSet, cfg: TrackConfig, source_fps: float, n_frames: int, frame0: int = 0
) -> Tracks:
    """Raw per-slot tracks (ids t0..tK) on the track grid."""
    stride = max(1, round(source_fps / cfg.fps))
    fps = source_fps / stride
    frames = list(range(frame0, n_frames, stride))
    N = len(frames)
    stats = colour_stats(dets)
    n_people = cfg.max_people if cfg.max_people > 0 else estimate_people(frames[:: max(1, N // 400)], dets)
    log.info("association: %d frames at %.2f Hz (stride %d), %d people slots", N, fps, stride, n_people)
    tr = Tracks.empty([f"t{k}" for k in range(n_people)], N, cams.names, fps, stride, frame0)
    st = TrackState(n_people)
    lost_frames = max(1, round(cfg.lost_frames))
    for i, f in enumerate(frames):
        for t in range(n_people):
            if st.prev[t] is not None and i - st.last_seen[t] > lost_frames:
                # the slot lost its person: forget position AND appearance memory; a slot whose colour/size came from a
                # swap otherwise rejects the real person at the gates below for the rest of the take (the identity
                # stage re-links segments by ReID, so short-term memory is all a slot needs)
                st.prev[t] = None
                st.colour[t] = None
                st.size[t] = None
        C = candidates_at(f, dets, cams, stats, cfg.min_score)
        clusters = cluster_frame(C, cams, n_people, cfg.min_views, cfg.reproj_px)
        if not clusters:
            continue
        for ci, t in assign(clusters, C, st, cfg, cfg.appearance, i, fps):
            mem = clusters[ci][0]
            xyz, sc = triangulate_cluster(mem, C, cams, cfg.min_score, cfg.reproj_px)
            if sc[J["pelvis"]] == 0:
                continue
            tr.xyz[t, i] = xyz
            tr.score[t, i] = sc
            for m in mem:
                tr.support[t, i, C[m].view] = C[m].row
            st.prev[t] = xyz[J["pelvis"]]
            st.last_seen[t] = i
            st.started[t] = True
    counts = Counter({tid: int(tr.has_3d[k].sum()) for k, tid in enumerate(tr.ids)})
    log.info("association: frames per slot %s", dict(counts))
    tr.meta["source_fps"] = source_fps
    return tr
