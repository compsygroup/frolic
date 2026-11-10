"""Root rotation handling: a continuous quaternion initialisation and a delta parameterisation around it.

A rotation-vector root crosses the pi singularity when the heading sweeps through +-180 degrees and the optimiser then
spins the body by 2 pi; the root is therefore R(f) = R_init(f) exp(delta(f)) with delta starting at 0 and staying small,
and the root smoothness is the geodesic angular acceleration of R itself.
"""

from __future__ import annotations

import numpy as np
import roma
import torch


def continuous_rotvecs(rr: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors with NaN rows for missing frames -> continuous (N, 3): hemisphere-aligned quaternions,
    gaps filled along the short arc, converted back to rotation vectors."""
    rr = np.asarray(rr, np.float32)
    N = len(rr)
    g = np.isfinite(rr).all(1)
    if not g.any():
        return np.zeros((N, 3), np.float32)
    qv = roma.rotvec_to_unitquat(torch.tensor(rr[g])).numpy()
    for k in range(1, len(qv)):
        if np.dot(qv[k], qv[k - 1]) < 0:
            qv[k] = -qv[k]
    gi = np.flatnonzero(g)
    qf = np.stack([np.interp(np.arange(N), gi, qv[:, d]) for d in range(4)], 1)
    qf /= np.linalg.norm(qf, axis=1, keepdims=True)
    return roma.unitquat_to_rotvec(torch.tensor(qf, dtype=torch.float32)).numpy().astype(np.float32)


class RootParam:
    """R(f) = R_init(f) @ exp(delta(f)); `delta` is the optimised tensor."""

    def __init__(self, rr_init: np.ndarray, device: torch.device, continuous: bool = True) -> None:
        rr = continuous_rotvecs(rr_init) if continuous else np.nan_to_num(np.asarray(rr_init, np.float32))
        self.continuous = continuous
        self.R_init = roma.rotvec_to_rotmat(torch.tensor(rr, device=device))
        if continuous:
            self.delta = torch.zeros(len(rr), 3, device=device, requires_grad=True)
        else:
            self.delta = torch.tensor(rr, device=device, requires_grad=True)

    def matrices(self) -> torch.Tensor:
        if self.continuous:
            return self.R_init @ roma.rotvec_to_rotmat(self.delta)
        return roma.rotvec_to_rotmat(self.delta)

    def rotvecs(self) -> torch.Tensor:
        return roma.rotmat_to_rotvec(self.matrices())

    @property
    def params(self) -> list[torch.Tensor]:
        return [self.delta]


def geodesic_acceleration(R: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared second difference of the relative rotations of R (N, 3, 3); representation-free."""
    if R.shape[0] < 3:
        return R.sum() * 0
    w = roma.rotmat_to_rotvec(R[:-1].transpose(-1, -2) @ R[1:])
    d = (w[1:] - w[:-1]).pow(2)
    if weights is not None:
        d = weights.view(-1, 1) * d
    return d.mean()


def spin_events(rr: np.ndarray, fps: float, min_deg: float = 150.0, window_s: float = 1.5, ratio: float = 2.5) -> int:
    """Count windows in which the root turns by more than min_deg while the net heading change is ratio times smaller."""
    rr = np.asarray(rr, np.float32)
    N = len(rr)
    w = max(2, round(window_s * fps))
    if w + 1 > N:
        return 0
    R = roma.rotvec_to_rotmat(torch.tensor(rr))
    rel = roma.rotmat_to_rotvec(R[:-1].transpose(-1, -2) @ R[1:]).norm(dim=-1).numpy()
    count = 0
    for i in range(0, N - w):
        travelled = float(np.degrees(rel[i : i + w].sum()))
        net = float(np.degrees(roma.rotmat_to_rotvec(R[i].T @ R[i + w]).norm()))
        if travelled > min_deg and travelled > ratio * max(net, 1e-3):
            count += 1
    return count
