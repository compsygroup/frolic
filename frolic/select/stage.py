"""Select stage: which tracked people go on to reconstruction, with labels and optional visibility spans."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np

from frolic.config import FrolicConfig
from frolic.io.detections import load_all
from frolic.io.tracks import Tracks
from frolic.io.workdir import WorkDir
from frolic.log import get_logger
from frolic.select.filters import auto_select
from frolic.select.picker import pick
from frolic.select.sheet import contact_sheet, img_tags

log = get_logger(__name__)


def visibility_spans(
    n_views: np.ndarray, fps: float, need: int, min_s: float, bridge_s: float
) -> list[tuple[int, int]]:
    """[start, stop) track frames seen by >= need cameras; gaps <= bridge_s merged, spans >= min_s kept."""
    ok = n_views >= need
    b = round(bridge_s * fps)
    idx = np.flatnonzero(ok)
    for l_, r_ in pairwise(idx):
        if 1 < r_ - l_ <= b + 1:
            ok[l_:r_] = True
    spans, f = [], 0
    while f < len(ok):
        if ok[f]:
            g = f
            while g < len(ok) and ok[g]:
                g += 1
            if g - f >= min_s * fps:
                spans.append((int(f), int(g)))
            f = g
        else:
            f += 1
    return spans


def apply_spans(tr: Tracks, cfg: FrolicConfig) -> dict[str, list[tuple[int, int]]]:
    out = {}
    if cfg.select.spans_min_views <= 0:
        return out
    nv = tr.n_views()
    for p, pid in enumerate(tr.ids):
        spans = visibility_spans(
            nv[p], tr.fps, cfg.select.spans_min_views, cfg.select.spans_min_s, cfg.select.spans_bridge_s
        )
        keep = np.zeros(tr.n_frames, bool)
        for a, b in spans:
            keep[a:b] = True
        tr.xyz[p, ~keep] = np.nan
        tr.score[p, ~keep] = 0
        tr.support[p, ~keep] = -1
        out[pid] = spans
    return out


def run(cfg: FrolicConfig, work: WorkDir, on_ready: Callable[[str], None] | None = None) -> dict[str, Any]:
    videos = work.read_videos()
    tr = Tracks.load(work.tracks)
    with open(work.stage_dir("track") / "identities.json") as fh:
        people = json.load(fh)["people"]
    dets = load_all(work.stage_dir("detect"), tr.cameras)
    out_dir = work.stage_dir("select")
    sheet_jpg, sheet_html = out_dir / "sheet.jpg", out_dir / "sheet.html"
    tiles: dict[str, list[np.ndarray]] = {}
    if tr.n_people:
        tiles = contact_sheet(tr, dets, videos, people, sheet_jpg, sheet_html, cfg.select.sheet_tiles)
    sc = cfg.select
    rejected: dict[str, list[str]] = {}
    if sc.mode == "config":
        ids, mode = [i for i in sc.ids if i in tr.ids], "config"
        missing = [i for i in sc.ids if i not in tr.ids]
        if missing:
            log.warning("select.ids not found in the tracks: %s", missing)
    elif sc.mode == "interactive":
        ids, mode = pick(tr.ids, people, {k: img_tags(v) for k, v in tiles.items()}, sc, on_ready)
        (out_dir / "picked.txt").write_text("\n".join(ids) + "\n")
    else:
        ids, rejected = auto_select(people, sc)
        mode = "auto"
    sel = tr.subset(ids)
    labels = {}
    for pid in ids:
        lab = sc.labels.get(pid) or (cfg.people[pid].label if pid in cfg.people else None)
        labels[pid] = lab or pid
    sel.labels = labels
    spans = apply_spans(sel, cfg)
    sel.save(work.tracks_selected)
    selection = {
        "ids": ids,
        "labels": labels,
        "mode": mode,
        "rejected": rejected,
        "spans": {k: [list(s) for s in v] for k, v in spans.items()},
    }
    with open(out_dir / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=1)
    log.info("select (%s): kept %s", mode, ids)
    return {"ids": ids, "mode": mode, "n_rejected": sum(len(v) for v in rejected.values())}
