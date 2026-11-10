"""Segment-level identity assembly: raw track slots -> stable people (appearance + geometry, no roles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from frolic3d.config import TrackConfig
from frolic3d.io.detections import Detections
from frolic3d.io.tracks import Tracks
from frolic3d.log import get_logger
from frolic3d.skeleton import J
from frolic3d.track.reid import Embedding

log = get_logger(__name__)
THR = 1.6  # z-scored feature distance to join an identity


@dataclass
class Segment:
    slot: int
    frames: np.ndarray  # track frames, contiguous-ish
    pelvis: np.ndarray  # (n, 3)
    torso: float
    colour: np.ndarray  # (3,) hue cos/sin, brightness z (NaN if unknown)
    support: float
    emb: np.ndarray | None
    feat: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def t0(self) -> int:
        return int(self.frames[0])

    @property
    def t1(self) -> int:
        return int(self.frames[-1])

    def __len__(self) -> int:
        return len(self.frames)


def _segment_colour(
    tr: Tracks, dets: dict[str, Detections], p: int, frames: np.ndarray, stats: dict[str, tuple[float, float]]
) -> np.ndarray:
    vals = []
    for i in frames:
        for v, cam in enumerate(tr.cameras):
            r = int(tr.support[p, i, v])
            if r < 0:
                continue
            hsv = dets[cam].colour[r]
            if not np.isfinite(hsv).all():
                continue
            m, s = stats[cam]
            vals.append([np.cos(hsv[0] * np.pi / 90), np.sin(hsv[0] * np.pi / 90), (hsv[2] - m) / s])
    return np.median(np.array(vals), 0) if vals else np.full(3, np.nan)


def cut_segments(tr: Tracks, dets: dict[str, Detections], cfg: TrackConfig, E: Embedding | None) -> list[Segment]:
    from frolic3d.track.associate import colour_stats

    stats = colour_stats(dets)
    fps = tr.fps
    segs: list[Segment] = []
    for p, pid in enumerate(tr.ids):
        idx = np.flatnonzero(tr.has_3d[p])
        if len(idx) == 0:
            continue
        pel = tr.xyz[p, :, J["pelvis"]]
        runs: list[list[int]] = [[int(idx[0])]]
        cur_e: list[np.ndarray] = [E[(pid, int(idx[0]))]] if E and (pid, int(idx[0])) in E else []
        for f in idx[1:]:
            f = int(f)
            last = runs[-1][-1]
            cut = f - last > 2 * fps or np.linalg.norm(pel[f] - pel[last]) > 0.6
            if not cut and E and cfg.identity_cut > 0 and (pid, f) in E and len(cur_e) >= 3:
                m = np.mean(cur_e, 0)
                m /= np.linalg.norm(m) + 1e-9
                if float(E[(pid, f)] @ m) < cfg.identity_cut:
                    cut = True
            if cut:
                runs.append([f])
                cur_e = [E[(pid, f)]] if E and (pid, f) in E else []
            else:
                runs[-1].append(f)
                if E and (pid, f) in E:
                    cur_e.append(E[(pid, f)])
        for run in runs:
            if len(run) < 3:
                continue
            fr = np.array(run)
            tor = np.linalg.norm(tr.xyz[p, fr, J["neck"]] - tr.xyz[p, fr, J["pelvis"]], axis=1)
            tor = tor[np.isfinite(tor)]
            ev = [E[(pid, int(f))] for f in fr if (pid, int(f)) in E] if E else []
            emb = None
            if ev:
                emb = np.mean(ev, 0)
                emb /= np.linalg.norm(emb) + 1e-9
            segs.append(
                Segment(
                    p,
                    fr,
                    pel[fr],
                    float(np.median(tor)) if len(tor) else np.nan,
                    _segment_colour(tr, dets, p, fr, stats),
                    float((tr.support[p, fr] >= 0).sum(1).mean()),
                    emb,
                )
            )
    X = np.array([[s.torso, *s.colour] for s in segs]) if segs else np.zeros((0, 4))
    ok = np.isfinite(X).all(1) if len(X) else np.zeros(0, bool)
    mu = np.nanmean(X, 0) if len(X) else np.zeros(4)
    sd = (np.nanstd(X, 0) + 1e-6) if len(X) else np.ones(4)
    for s, row, o in zip(segs, X, ok):
        s.feat = (row - mu) / sd if o else np.zeros(4)
        if not o:
            s.feat[0] = (row[0] - mu[0]) / sd[0] if np.isfinite(row[0]) else 0.0
    return segs


def _overlap(a: Segment, b: Segment, fps: float) -> bool:
    return min(a.t1, b.t1) - max(a.t0, b.t0) > int(0.5 * fps)


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class Identities:
    def __init__(self, segs: list[Segment], fps: float, cfg: TrackConfig, use_emb: bool) -> None:
        self.segs, self.fps, self.cfg, self.use_emb = segs, fps, cfg, use_emb
        self.W = np.array([len(s) for s in segs], float)
        self.members: list[list[int]] = []

    def proto(self, k: int) -> np.ndarray:
        m = self.members[k]
        return np.average(np.stack([self.segs[i].feat for i in m]), axis=0, weights=self.W[m])

    def emb(self, k: int) -> np.ndarray | None:
        es = [(self.segs[i].emb, self.W[i]) for i in self.members[k] if self.segs[i].emb is not None]
        if not es:
            return None
        return np.average(np.stack([e for e, _ in es]), axis=0, weights=[w for _, w in es])

    def overlaps(self, k: int, si: int) -> bool:
        return any(_overlap(self.segs[si], self.segs[m], self.fps) for m in self.members[k])

    def cluster(self) -> None:
        order = np.argsort(-self.W)
        for si in order:
            s = self.segs[si]
            best, bd = None, THR
            for k in range(len(self.members)):
                if self.overlaps(k, si):
                    continue
                c = _cos(s.emb, self.emb(k)) if self.use_emb else None
                if c is not None:
                    d = (1 - c) / max(1e-6, 1 - self.cfg.identity_join) * THR + 0.3 * abs(s.feat[0] - self.proto(k)[0])
                else:
                    d = float(np.linalg.norm(s.feat - self.proto(k)))
                if d < bd:
                    best, bd = k, d
            if best is None:
                self.members.append([int(si)])
            else:
                self.members[best].append(int(si))

    def can_merge(self, i: int, k: int) -> bool:
        c = _cos(self.emb(i), self.emb(k)) if self.use_emb else None
        if any(_overlap(self.segs[a], self.segs[b], self.fps) for a in self.members[i] for b in self.members[k]):
            return False
        if c is not None:
            return c > self.cfg.identity_merge
        return float(np.linalg.norm(self.proto(i) - self.proto(k))) < 2.6

    def merge_disjoint(self) -> None:
        changed = True
        while changed:
            changed = False
            n = len(self.members)
            pairs = sorted(
                ((np.linalg.norm(self.proto(i) - self.proto(j)), i, j) for i in range(n) for j in range(i + 1, n)),
                key=lambda x: x[0],
            )
            for _, i, j in pairs:
                if self.can_merge(i, j):
                    self.members[i] += self.members[j]
                    del self.members[j]
                    changed = True
                    break

    def frames_of(self, k: int) -> set[int]:
        return {int(f) for si in self.members[k] for f in self.segs[si].frames}

    def absorb_fragments(self, dup_dist: float = 0.5) -> int:
        """Small identities join the nearest big one unless they overlap it heavily (then duplicate -> dropped)."""
        self.members.sort(key=lambda m: -self.W[m].sum())
        tot = self.W.sum()
        big = [i for i, m in enumerate(self.members) if self.W[m].sum() / tot >= 0.10]
        dropped = 0
        for i in range(len(self.members)):
            if i in big or not self.members[i]:
                continue

            def dist(k: int, i: int = i) -> float:
                c = _cos(self.emb(i), self.emb(k)) if self.use_emb else None
                return 1 - c if c is not None else float(np.linalg.norm(self.proto(i) - self.proto(k)))

            cands = sorted(big, key=dist)
            if not cands:
                continue
            fi = self.frames_of(i)
            for k in cands:
                if len(fi & self.frames_of(k)) / max(1, len(fi)) < 0.30:
                    self.members[k] += self.members[i]
                    self.members[i] = []
                    break
            if self.members[i]:
                k = cands[0]
                c = _cos(self.emb(i), self.emb(k)) if self.use_emb else None
                dup = (
                    (c > self.cfg.identity_merge)
                    if c is not None
                    else float(np.linalg.norm(self.proto(i) - self.proto(k))) < 1.2
                )
                if dup:
                    pi = {
                        int(f): p for si in self.members[i] for f, p in zip(self.segs[si].frames, self.segs[si].pelvis)
                    }
                    pk = {
                        int(f): p for si in self.members[k] for f, p in zip(self.segs[si].frames, self.segs[si].pelvis)
                    }
                    common = sorted(set(pi) & set(pk))
                    if common and np.median([np.linalg.norm(pi[f][:2] - pk[f][:2]) for f in common]) > dup_dist:
                        dup = False
                if dup:
                    self.members[i] = []
                    dropped += 1
        self.members = [m for m in self.members if m and self.W[m].sum() / tot >= self.cfg.min_identity_frac]
        return dropped


def _fill_identity(
    segs: list[Segment], members: list[int], tr: Tracks, N: int, anchor_m: float, anchor_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Anchored fill of one identity -> xyz (N, 17, 3), score (N, 17), support (N, V), ghost frames."""
    xyz = np.full((N, *tr.xyz.shape[2:]), np.nan, np.float32)
    sc = np.zeros((N, tr.score.shape[2]), np.float32)
    sup = np.full((N, tr.support.shape[2]), -1, np.int32)
    filled: list[int] = []
    ghost = 0
    for si in sorted(members, key=lambda i: (-min(segs[i].support, 3), -len(segs[i]))):
        s = segs[si]
        if filled and anchor_m > 0:
            arr = np.array(sorted(filled))
            bad = tested = 0
            for f_end, p_end in ((s.t0, s.pelvis[0]), (s.t1, s.pelvis[-1])):
                k = np.searchsorted(arr, f_end)
                near = [g for g in arr[max(0, k - 1) : k + 1] if abs(int(g) - f_end) <= int(anchor_s * tr.fps)]
                if near:
                    tested += 1
                    g = min(near, key=lambda g: abs(int(g) - f_end))
                    if np.linalg.norm(xyz[g, J["pelvis"], :2] - p_end[:2]) > anchor_m:
                        bad += 1
            if tested and bad == tested:
                ghost += len(s)
                continue
        for f in s.frames:
            if np.isfinite(xyz[f, 0, 0]):
                continue
            xyz[f] = tr.xyz[s.slot, f]
            sc[f] = tr.score[s.slot, f]
            sup[f] = tr.support[s.slot, f]
            filled.append(int(f))
    return xyz, sc, sup, ghost


def _teleport_filter(xyz: np.ndarray, sc: np.ndarray, sup: np.ndarray, fps: float, vmax: float) -> int:
    valid = sc[:, 0] > 0
    idx = np.flatnonzero(valid)
    if vmax <= 0 or len(idx) == 0:
        return 0
    pel = xyz[:, J["pelvis"], :2]
    cuts = np.flatnonzero((np.diff(idx) > 1) | (np.linalg.norm(np.diff(pel[idx], axis=0), axis=1) > 0.5)) + 1
    runs = np.split(idx, cuts)
    prev = None
    dropped = 0
    for ri, run in enumerate(runs):
        nxt = runs[ri + 1] if ri + 1 < len(runs) else None
        v_in = (
            np.linalg.norm(pel[run[0]] - pel[prev[-1]]) / max(1, run[0] - prev[-1]) * fps
            if prev is not None
            else np.nan
        )
        v_out = (
            np.linalg.norm(pel[nxt[0]] - pel[run[-1]]) / max(1, nxt[0] - run[-1]) * fps if nxt is not None else np.nan
        )
        short = len(run) < 10 * fps
        bad_in = np.isfinite(v_in) and v_in > vmax
        bad_out = np.isfinite(v_out) and v_out > vmax
        if short and ((bad_in and (bad_out or nxt is None)) or (prev is None and bad_out)):
            xyz[run] = np.nan
            sc[run] = 0
            sup[run] = -1
            dropped += len(run)
            continue
        prev = run
    return dropped


def bridge_gaps(xyz: np.ndarray, sc: np.ndarray, max_gap: int) -> int:
    """Linear interpolation of gaps <= max_gap frames (score halved)."""
    valid = sc[:, 0] > 0
    gaps = np.flatnonzero(~valid)
    n = 0
    if len(gaps) == 0 or max_gap <= 0:
        return 0
    for run in np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1):
        a, b = run[0] - 1, run[-1] + 1
        if a < 0 or b >= len(valid) or len(run) > max_gap or not (valid[a] and valid[b]):
            continue
        w = ((run - a) / (b - a))[:, None, None]
        xyz[run] = xyz[a][None] * (1 - w) + xyz[b][None] * w
        sc[run] = np.minimum(sc[a], sc[b])[None] * 0.5
        n += len(run)
    return n


def apply_overrides(members: list[list[int]], ids: list[str], cfg: TrackConfig) -> list[list[int]]:
    name = {pid: k for k, pid in enumerate(ids)}
    for pair in cfg.merge:
        ks = sorted({name[p] for p in pair if p in name})
        if len(ks) < 2:
            continue
        for k in ks[1:]:
            members[ks[0]] += members[k]
            members[k] = []
    for pid in cfg.drop:
        if pid in name:
            members[name[pid]] = []
    return members


def _summary(pid: str, xyz: np.ndarray, sup: np.ndarray, fps: float, colour: list[np.ndarray | None]) -> dict[str, Any]:
    have = np.isfinite(xyz[:, 0, 0])
    idx = np.flatnonzero(have)
    stat = np.linalg.norm(xyz[:, J["head"]] - 0.5 * (xyz[:, J["ankle_l"]] + xyz[:, J["ankle_r"]]), axis=1)
    stat = stat[np.isfinite(stat)]
    spans = []
    if len(idx):
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            spans.append([float(run[0] / fps), float((run[-1] + 1) / fps)])
    return {
        "id": pid,
        "frames": int(have.sum()),
        "duration_s": float(have.sum() / fps),
        "stature_m": float(np.median(stat) * 1.08) if len(stat) else None,
        "median_views": float(np.median((sup[have] >= 0).sum(1))) if have.any() else 0.0,
        "spans_s": spans,
        "colour_hsv": {},
    }


def assemble_identities(
    tr: Tracks, dets: dict[str, Detections], cfg: TrackConfig, E: Embedding | None = None
) -> tuple[Tracks, list[dict[str, Any]]]:
    """Raw slot tracks -> people tracks (ids p0..pK) and a per-identity summary."""
    N = tr.n_frames
    if cfg.room_bounds_m is not None:
        x0, x1, y0, y1 = cfg.room_bounds_m
        pel = tr.xyz[:, :, J["pelvis"]]
        out = tr.has_3d & ~((pel[..., 0] >= x0) & (pel[..., 0] <= x1) & (pel[..., 1] >= y0) & (pel[..., 1] <= y1))
        if out.any():
            log.info("room filter: dropped %d frames outside %s", int(out.sum()), cfg.room_bounds_m)
            tr.xyz[out] = np.nan
            tr.score[out] = 0
            tr.support[out] = -1
    segs = cut_segments(tr, dets, cfg, E)
    use_emb = bool(E) and any(s.emb is not None for s in segs)
    ident = Identities(segs, tr.fps, cfg, use_emb)
    if not segs:
        return Tracks.empty([], N, tr.cameras, tr.fps, tr.stride, tr.frame0), []
    ident.cluster()
    ident.merge_disjoint()
    dropped = ident.absorb_fragments()
    log.info("identities: %d segments -> %d identities (%d duplicates dropped)", len(segs), len(ident.members), dropped)
    members = apply_overrides(ident.members, [f"p{k}" for k in range(len(ident.members))], cfg)
    members = [m for m in members if m]
    people, summary = [], []
    for m in members:
        xyz, sc, sup, ghost = _fill_identity(segs, m, tr, N, 1.5, 3.0)
        pz = xyz[np.isfinite(xyz[:, 0, 0]), J["pelvis"], 2]
        if len(pz) and np.median(pz) > cfg.max_pelvis_z_m:
            log.info("identity dropped as a floating ghost (median pelvis z %.2f m)", float(np.median(pz)))
            continue
        tp = _teleport_filter(xyz, sc, sup, tr.fps, cfg.teleport_mps)
        br = bridge_gaps(xyz, sc, int(cfg.bridge_s * tr.fps))
        people.append((xyz, sc, sup))
        summary.append({"ghost_frames": ghost, "teleport_frames": tp, "bridged_frames": br})
    people, summary = _apply_splits(people, summary, cfg, tr.fps)
    ids = [f"p{k}" for k in range(len(people))]
    out = Tracks.empty(ids, N, tr.cameras, tr.fps, tr.stride, tr.frame0)
    for k, (xyz, sc, sup) in enumerate(people):
        out.xyz[k], out.score[k], out.support[k] = xyz, sc, sup
    out.meta = dict(tr.meta)
    info = []
    for k, pid in enumerate(ids):
        d = _summary(pid, out.xyz[k], out.support[k], out.fps, [])
        d.update(summary[k])
        info.append(d)
    return out, info


def _apply_splits(
    people: list[tuple], summary: list[dict], cfg: TrackConfig, fps: float
) -> tuple[list[tuple], list[dict]]:
    if not cfg.split:
        return people, summary
    out_p, out_s = [], []
    for k, (xyz, sc, sup) in enumerate(people):
        cuts = sorted(round(t * fps) for t in cfg.split.get(f"p{k}", []))
        if not cuts:
            out_p.append((xyz, sc, sup))
            out_s.append(summary[k])
            continue
        bounds = [0, *cuts, len(xyz)]
        for a, b in pairwise(bounds):
            x2, s2, u2 = np.full_like(xyz, np.nan), np.zeros_like(sc), np.full_like(sup, -1)
            x2[a:b], s2[a:b], u2[a:b] = xyz[a:b], sc[a:b], sup[a:b]
            if (s2[:, 0] > 0).any():
                out_p.append((x2, s2, u2))
                out_s.append(dict(summary[k]))
    return out_p, out_s
