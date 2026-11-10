"""Automatic person selection from the identity summary."""

from __future__ import annotations

from typing import Any

from frolic.config import SelectConfig


def auto_select(people: list[dict[str, Any]], cfg: SelectConfig) -> tuple[list[str], dict[str, list[str]]]:
    """Ids that pass the filters, and the ids rejected per filter."""
    rejected: dict[str, list[str]] = {}
    keep = []
    for p in people:
        pid = p["id"]
        why = None
        stat = p.get("stature_m")
        if p.get("median_views", 0) < cfg.min_views:
            why = "min_views"
        elif p.get("duration_s", 0) < cfg.min_track_s:
            why = "min_track_s"
        elif cfg.min_height_m is not None and stat is not None and stat < cfg.min_height_m:
            why = "min_height_m"
        elif cfg.max_height_m is not None and stat is not None and stat > cfg.max_height_m:
            why = "max_height_m"
        elif cfg.children_only and (stat is None or stat > cfg.child_height_m):
            why = "children_only"
        if why:
            rejected.setdefault(why, []).append(pid)
        else:
            keep.append(p)
    keep.sort(key=lambda p: -p.get("frames", 0))
    if cfg.max_people > 0 and len(keep) > cfg.max_people:
        rejected["max_people"] = [p["id"] for p in keep[cfg.max_people :]]
        keep = keep[: cfg.max_people]
    return [p["id"] for p in keep], rejected
