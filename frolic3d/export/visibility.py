"""Which frames of a fitted person are shown in the export."""

from __future__ import annotations

import itertools

import numpy as np


def fill_gaps(have: np.ndarray, max_gap: int) -> np.ndarray:
    """Frames to show: observed frames plus gaps no longer than max_gap between observations."""
    have = np.asarray(have, bool)
    show = have.copy()
    idx = np.flatnonzero(have)
    for i0, i1 in itertools.pairwise(idx):
        if 1 < i1 - i0 <= max_gap:
            show[i0:i1] = True
    return show


def visible_mask(
    have: np.ndarray,
    single_view: np.ndarray | None,
    observed: np.ndarray | None,
    fps: float,
    max_gap_s: float,
    observed_rule: bool,
    bridge_s: float = 3.0,
) -> np.ndarray:
    """Visibility of a fitted person.

    observed_rule: triangulated, single-view and merely observed frames all count as observations; a frame is
    hidden only when no camera observes the person. Observed frames without a 3D estimate are shown only between
    3D frames at most `bridge_s` apart. Without the rule only triangulated frames count.
    """
    have = np.asarray(have, bool)
    n = len(have)
    sv = np.zeros(n, bool) if single_view is None else np.asarray(single_view, bool)[:n]
    max_gap = round(max_gap_s * fps)
    if not observed_rule:
        return fill_gaps(have & ~sv, max_gap)
    obs = have | (np.asarray(observed, bool)[:n] if observed is not None else False)
    show = fill_gaps(obs, max_gap)
    if observed is not None:
        hi = np.flatnonzero(have)
        for i in np.flatnonzero(show & ~have):
            left = hi[hi < i]
            right = hi[hi > i]
            if not (len(left) and len(right) and right[0] - left[-1] <= bridge_s * fps):
                show[i] = False
    return show
