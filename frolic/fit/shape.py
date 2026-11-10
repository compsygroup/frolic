"""Body-shape helpers: phenotype from person attributes, segment-length targets and the ankle-keypoint bias."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from frolic.skeleton import SEGMENTS, J

PHENOTYPE_LABELS = ["gender", "age", "muscle", "weight", "height", "proportions"]

# Anny morphological-age anchors (years -> slider) used when the anny package is unavailable; the real mapping
# (anny.shape_distribution.SimpleShapeDistribution) gives 2 y -> 0.105, 6 y -> 0.272, 30 y -> 0.786.
_AGE_YEARS = np.array([0.0, 1.0, 2.0, 6.0, 8.0, 12.0, 18.0, 30.0, 60.0, 90.0])
_AGE_SLIDER = np.array([0.0, 0.06, 0.105, 0.272, 0.329, 0.43, 0.6, 0.786, 0.92, 1.0])


def age_years_to_slider(years: float) -> float:
    """Piecewise-linear fallback of Anny's morphological age mapping (monotone, clamped to [0, 1])."""
    return float(np.interp(float(years), _AGE_YEARS, _AGE_SLIDER))


def initial_age_from_leg(X: np.ndarray) -> float:
    """Age slider guess from the tracked left leg length (hip-knee + knee-ankle)."""
    leg = np.nanmedian(
        np.linalg.norm(X[:, J["hip_l"]] - X[:, J["knee_l"]], axis=1)
        + np.linalg.norm(X[:, J["knee_l"]] - X[:, J["ankle_l"]], axis=1)
    )
    if not np.isfinite(leg):
        return 0.5
    return 0.5 if leg > 0.62 else 0.25 if leg > 0.42 else 0.12


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


@dataclass
class FixedPhenotype:
    """Sliders fixed from the person's attributes (values in [0, 1]) and the ones still to be fitted."""

    values: dict[str, float] = field(default_factory=dict)
    height_m: float | None = None
    weight_kg: float | None = None

    @classmethod
    def from_person(cls, person, years_to_slider=age_years_to_slider) -> FixedPhenotype:
        if person is None:
            return cls()
        vals: dict[str, float] = {}
        if person.sex is not None:
            vals["gender"] = 0.0 if person.sex == "male" else 1.0
        if person.age_years is not None:
            vals["age"] = years_to_slider(person.age_years)
        return cls(vals, person.height_m, person.weight_kg)

    def mask(self, labels: list[str], device) -> torch.Tensor:
        """1 where the slider is free, 0 where fixed (hook on the logit gradient)."""
        return torch.tensor([0.0 if lab in self.values else 1.0 for lab in labels], device=device)

    def apply(self, pl: torch.Tensor, labels: list[str]) -> None:
        with torch.no_grad():
            for i, lab in enumerate(labels):
                if lab in self.values:
                    pl[i] = logit(self.values[lab])


LEN_GROUPS: dict[str, list[str]] = {
    "thigh": ["thigh_l", "thigh_r"],
    "shin": ["shin_l", "shin_r"],
    "upperarm": ["upperarm_l", "upperarm_r"],
    "forearm": ["forearm_l", "forearm_r"],
    "trunk": ["trunk"],
}
GROUP_SEGMENT_LABELS: dict[str, list[str]] = {
    "thigh": ["measure-upperleg-height-incr"],
    "shin": ["measure-lowerleg-height-incr"],
    "upperarm": ["measure-upperarm-length-incr"],
    "forearm": ["measure-lowerarm-length-incr"],
    "trunk": ["measure-napetowaist-dist-incr", "measure-waisttohip-dist-incr", "measure-neck-height-incr"],
}


@dataclass
class LengthTargets:
    groups: list[str]
    target: np.ndarray  # (G,) metres, NaN when unknown
    weight: np.ndarray  # (G,) consistency weight in [0, 1]

    @property
    def ok(self) -> np.ndarray:
        return np.isfinite(self.target) & (self.weight > 0)


def segment_length_targets(X: np.ndarray, min_frames: int = 30) -> LengthTargets:
    """One robust target per symmetric segment group from the tracked joints (N, 17, 3).

    Each side is weighted by its consistency (frames x 1 / IQR^2) so a hidden or crossed limb barely counts, and the
    group weight drops when even its best side is inconsistent.
    """
    N = len(X)
    stats: dict[str, tuple[float, float, float]] = {}
    for name, (a, b) in SEGMENTS.items():
        l_ = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
        l_ = l_[np.isfinite(l_)]
        if len(l_) >= min_frames:
            med = float(np.median(l_))
            stats[name] = (med, float((np.percentile(l_, 75) - np.percentile(l_, 25)) / max(med, 1e-6)), len(l_) / N)
        else:
            stats[name] = (np.nan, np.inf, 0.0)
    groups = list(LEN_GROUPS)
    target = np.full(len(groups), np.nan)
    weight = np.zeros(len(groups))
    for gi, g in enumerate(groups):
        members = LEN_GROUPS[g]
        w = np.array([stats[m][2] / (stats[m][1] ** 2 + 0.03**2) if np.isfinite(stats[m][0]) else 0.0 for m in members])
        if w.sum() == 0:
            continue
        target[gi] = float(
            np.sum(w * np.array([stats[m][0] if np.isfinite(stats[m][0]) else 0.0 for m in members])) / w.sum()
        )
        weight[gi] = 1.0 / (1.0 + (min(stats[m][1] for m in members) / 0.10) ** 2)
    return LengthTargets(groups, target, weight)


def standing_ankle_height(X: np.ndarray, fps: float, min_frames: int = 20) -> float:
    """Median lowest-ankle height on still, upright frames (NaN when too few)."""
    N = len(X)
    pz = X[:, J["pelvis"], 2]
    av = np.full(N, np.inf)
    ank = X[:, [J["ankle_l"], J["ankle_r"]]]
    if N > 2:
        with np.errstate(all="ignore"):
            av[1:-1] = np.nanmax(np.linalg.norm(ank[2:] - ank[:-2], axis=-1), 1) * fps / 2
    if not np.isfinite(pz).any():
        return np.nan
    stand = np.isfinite(pz) & (pz > np.nanpercentile(pz, 60)) & (np.nan_to_num(av, nan=np.inf) < 0.1)
    # standing gate: the pelvis must also be above 0.8 x the tracked leg length (thigh + shin); a person who never
    # stands (seated child, kneeling parent) otherwise defines the ankle height from sitting frames
    with np.errstate(all="ignore"):
        leg = np.nanmedian(
            np.r_[
                np.linalg.norm(X[:, J["hip_l"]] - X[:, J["knee_l"]], axis=-1)
                + np.linalg.norm(X[:, J["knee_l"]] - X[:, J["ankle_l"]], axis=-1),
                np.linalg.norm(X[:, J["hip_r"]] - X[:, J["knee_r"]], axis=-1)
                + np.linalg.norm(X[:, J["knee_r"]] - X[:, J["ankle_r"]], axis=-1),
            ]
        )
    if np.isfinite(leg):
        stand &= pz > 0.8 * leg
    if stand.sum() < min_frames:
        return np.nan
    return float(np.nanmedian(np.nanmin(ank[stand][:, :, 2], 1)))


def weight_muscle_for_mass(
    mesh_mass_fn, ph0: torch.Tensor, labels: list[str], mass_kg: float
) -> tuple[float, float, float]:
    """Grid search of the weight and muscle sliders so that the rest-mesh mass matches mass_kg (mild prior to 0.7 / 0.5)."""
    gi_w, gi_m = labels.index("weight"), labels.index("muscle")
    best = None
    for wv in np.linspace(0.02, 0.98, 25):
        for mv in np.linspace(0.02, 0.98, 25):
            php = ph0.clone()
            php[gi_w] = float(wv)
            php[gi_m] = float(mv)
            ms = mesh_mass_fn(php)
            cost = abs(ms - mass_kg) / mass_kg * 20 + (wv - 0.7) ** 2 + (mv - 0.5) ** 2
            if best is None or cost < best[0]:
                best = (cost, float(wv), float(mv), float(ms))
    return best[1], best[2], best[3]


def height_slider_for_stature(stature_fn, ph0: torch.Tensor, labels: list[str], height_m: float) -> float:
    """1-D search of the height slider so that the rest-pose stature matches height_m."""
    gi = labels.index("height")
    best = None
    for hv in np.linspace(0.0, 1.0, 41):
        php = ph0.clone()
        php[gi] = float(hv)
        err = abs(stature_fn(php) - height_m)
        if best is None or err < best[0]:
            best = (err, float(hv))
    lo, hi = max(0.0, best[1] - 0.025), min(1.0, best[1] + 0.025)
    for hv in np.linspace(lo, hi, 11):
        php = ph0.clone()
        php[gi] = float(hv)
        err = abs(stature_fn(php) - height_m)
        if err < best[0]:
            best = (err, float(hv))
    return best[1]
