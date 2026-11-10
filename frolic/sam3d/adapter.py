"""SAM 3D Body (Meta) and its MHR body model behind two small protocols, plus deterministic mocks for tests."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from frolic.skeleton import HALPE26_TO_MHR70, MHR70_BODY, N_JOINTS, J

KEEP = [
    "bbox", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_cam_t", "pred_pose_raw", "global_rot", "body_pose_params",
    "hand_pose_params", "scale_params", "shape_params", "expr_params", "pred_joint_coords", "pred_global_rots",
]  # fmt: skip
N_KP70 = 70
N_BODY, N_HAND, N_SHAPE, N_SCALE, N_EXPR = 133, 108, 10, 1, 72


class Sam3dBody(Protocol):
    def estimate(self, img_rgb: np.ndarray, boxes: np.ndarray, K: np.ndarray) -> list[dict[str, np.ndarray]]: ...

    @property
    def mhr(self) -> MhrModel: ...


class MhrModel(Protocol):
    faces: np.ndarray
    device: torch.device

    def forward(
        self, body: torch.Tensor, hand: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 133), (B, 108), (B, S), (B, Sh) with the global rotation at zero -> vertices (B, V, 3), keypoints (B, 70, 3)."""
        ...


def sam3d_root(root: Path | str | None) -> Path:
    r = root or os.environ.get("FROLIC_SAM3D_ROOT")
    if not r:
        raise RuntimeError("SAM 3D Body root not set (sam3d.root or FROLIC_SAM3D_ROOT); see THIRD_PARTY.md")
    return Path(r)


class _RealMhr:
    def __init__(self, head, device: torch.device) -> None:
        self.head = head.eval()
        self.device = device
        f = head.faces
        self.faces = f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f)

    def forward(self, body, hand, scale, shape):
        B = body.shape[0]
        z = torch.zeros(B, 3, device=self.device)
        verts, kp = self.head.mhr_forward(
            z, z, body, hand, scale, shape, torch.zeros(B, N_EXPR, device=self.device), return_keypoints=True
        )
        return verts, kp[:, :N_KP70]


class Sam3dBodyReal:
    """Lazy wrapper around the sam_3d_body package (imports on first use)."""

    def __init__(
        self, root: Path | str | None, checkpoint: str = "checkpoints/sam-3d-body-dinov3", device: str = "cuda"
    ) -> None:
        self.root = sam3d_root(root)
        self.ck = self.root / checkpoint
        self.device = torch.device(device)
        self._model = self._cfg = self._est = self._mhr = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        self._model, self._cfg = load_sam_3d_body(
            str(self.ck / "model.ckpt"), device=str(self.device), mhr_path=str(self.ck / "assets" / "mhr_model.pt")
        )
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._est = SAM3DBodyEstimator(sam_3d_body_model=self._model, model_cfg=self._cfg)
        self._mhr = _RealMhr(self._model.head_pose, self.device)

    def estimate(self, img_rgb: np.ndarray, boxes: np.ndarray, K: np.ndarray) -> list[dict[str, np.ndarray]]:
        self._load()
        h, w = img_rgb.shape[:2]
        b = np.asarray(boxes, np.float32).copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
        with contextlib.redirect_stdout(io.StringIO()):
            outs = self._est.process_one_image(
                img_rgb, bboxes=b, cam_int=torch.tensor(K, dtype=torch.float32)[None], inference_type="full"
            )
        return [{k: np.asarray(o[k]).astype(np.float32) for k in KEEP if k in o} for o in outs]

    @property
    def mhr(self) -> MhrModel:
        self._load()
        return self._mhr


# ------------------------------------------------------------------------------------------------------ mocks
_BONES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (0, 9),
    (9, 10),
    (9, 11),
    (9, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]
_RADII = [0.08, 0.08, 0.07, 0.07, 0.05, 0.05, 0.03, 0.03, 0.11, 0.08, 0.06, 0.06, 0.04, 0.04, 0.035, 0.035]
_RING, _ALONG = 4, 3
_FACE = [
    (0, 0.55, 0.0),
    (1, 0.62, -0.03),
    (2, 0.62, 0.03),
    (3, 0.5, -0.03),
    (4, 0.5, 0.03),
]  # (halpe, fraction neck->head, x offset)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def expand17to70_t(X: torch.Tensor) -> torch.Tensor:
    """Joints (B, 17, 3) -> MHR-70 keypoints (B, 70, 3), differentiable: body joints through MHR70_BODY, face points
    along neck->head, toes/heels from ankle and toe, finger points fanned out from the wrists, the rest at the pelvis."""
    B = X.shape[0]
    K = [X[:, J["pelvis"]]] * N_KP70
    head, neck = X[:, J["head"]], X[:, J["neck"]]
    up = head - neck
    xoff = torch.tensor([1.0, 0, 0], device=X.device)
    for h, frac, dx in _FACE:
        K[HALPE26_TO_MHR70[h]] = neck + frac * up + dx * xoff
    for side, toe_h, small_h, heel_h in (("l", 20, 22, 24), ("r", 21, 23, 25)):
        ank, to = X[:, J[f"ankle_{side}"]], X[:, J[f"toe_{side}"]]
        K[HALPE26_TO_MHR70[toe_h]] = to
        K[HALPE26_TO_MHR70[small_h]] = to + 0.3 * (ank - to)
        heel = ank - 0.4 * (to - ank)
        K[HALPE26_TO_MHR70[heel_h]] = torch.stack([heel[:, 0], heel[:, 1], torch.minimum(heel[:, 2], ank[:, 2])], -1)
    for j, m in MHR70_BODY.items():
        K[m] = X[:, j]
    zup = torch.tensor([0, 0, 1.0], device=X.device)
    for base, wrist, elbow in ((21, J["wrist_r"], J["elbow_r"]), (42, J["wrist_l"], J["elbow_l"])):
        d = _unit(X[:, wrist] - X[:, elbow])
        side = torch.cross(d, zup.expand(B, 3), dim=-1)
        for f in range(5):
            for k in range(4):
                K[base + 4 * f + k] = X[:, wrist] + d * (0.04 + 0.02 * (3 - k)) + side * 0.015 * (f - 2)
        K[base + 20] = X[:, wrist]
    return torch.stack(K, 1)


def expand17to70(X: np.ndarray) -> np.ndarray:
    """NumPy front end of expand17to70_t for (..., 17, 3) joints (NaN rows stay NaN)."""
    X = np.asarray(X, np.float32)
    flat = X.reshape(-1, N_JOINTS, 3)
    out = expand17to70_t(torch.tensor(np.nan_to_num(flat))).numpy()
    out[~np.isfinite(flat[:, 0, 0])] = np.nan
    return out.reshape(X.shape[:-2] + (N_KP70, 3))


def mock_mesh_t(X: torch.Tensor) -> torch.Tensor:
    """Tube points around every bone of (B, 17, 3) joints -> (B, 192, 3), rigidly attached, differentiable."""
    B = X.shape[0]
    zup = torch.tensor([0, 0, 1.0], device=X.device).expand(B, 3)
    xax = torch.tensor([1.0, 0, 0], device=X.device).expand(B, 3)
    out = []
    for (a, b), r in zip(_BONES, _RADII):
        p, q = X[:, a], X[:, b]
        d = _unit(q - p)
        u = torch.cross(d, zup, dim=-1)
        u = torch.where((u.norm(dim=-1, keepdim=True) < 1e-3), torch.cross(d, xax, dim=-1), u)
        u = _unit(u)
        v = torch.cross(d, u, dim=-1)
        for ti in range(_ALONG):
            t = 0.2 + 0.6 * ti / (_ALONG - 1)
            for ai in range(_RING):
                th = 2 * np.pi * ai / _RING
                out.append(p + t * (q - p) + r * (np.cos(th) * u + np.sin(th) * v))
    return torch.stack(out, 1)


def mock_mesh_from_joints(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32)
    flat = X.reshape(-1, N_JOINTS, 3)
    out = mock_mesh_t(torch.tensor(np.nan_to_num(flat))).numpy()
    out[~np.isfinite(flat[:, 0, 0])] = np.nan
    return out.reshape(X.shape[:-2] + (out.shape[1], 3))


def mock_faces() -> np.ndarray:
    faces = []
    for b in range(len(_BONES)):
        base = b * _ALONG * _RING
        for ti in range(_ALONG - 1):
            for ai in range(_RING):
                a0 = base + ti * _RING + ai
                a1 = base + ti * _RING + (ai + 1) % _RING
                faces += [[a0, a1, a1 + _RING], [a0, a1 + _RING, a0 + _RING]]
    return np.array(faces, np.int32)


def mock_template17() -> np.ndarray:
    """Canonical standing pose, pelvis at the origin, z up, stature 1 m."""
    from frolic.synthetic import pose_at

    X = pose_at(1.0, 0.0, (0.0, 0.0), 0.0)
    return (X - X[J["pelvis"]]).astype(np.float32)


class MockMhr:
    """Body (133): the first 51 values are per-joint offsets (canonical frame) on the template; scale (1): log size."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.template = torch.tensor(mock_template17(), device=self.device)
        self.faces = mock_faces()

    def joints(self, body: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (self.template[None] + body[:, : 3 * N_JOINTS].reshape(-1, N_JOINTS, 3)) * torch.exp(scale[:, :1])[
            :, None
        ]

    def forward(self, body, hand, scale, shape):
        Jt = self.joints(body, scale)
        return mock_mesh_t(Jt), expand17to70_t(Jt)

    @staticmethod
    def encode(J17: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Body params (133,) that reproduce J17 (canonical frame, pelvis at the origin) at the given size."""
        body = np.zeros(N_BODY, np.float32)
        body[: 3 * N_JOINTS] = (np.asarray(J17, np.float32) / scale - mock_template17()).ravel()
        return body


class MockSam3d:
    """Returns the canonical template at a depth consistent with the box height; for stage tests only."""

    def __init__(self, device: torch.device | None = None) -> None:
        self._mhr = MockMhr(device)

    @property
    def mhr(self) -> MhrModel:
        return self._mhr

    def estimate(self, img_rgb: np.ndarray, boxes: np.ndarray, K: np.ndarray) -> list[dict[str, np.ndarray]]:
        outs = []
        T = mock_template17()
        for b in np.asarray(boxes, np.float32):
            h = max(float(b[3] - b[1]), 1.0)
            z = float(K[1, 1]) * 1.6 / h
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            Xc = np.stack([T[:, 0], -T[:, 2], T[:, 1]], -1) * 1.6
            Xc = Xc + np.array([(cx - K[0, 2]) / K[0, 0] * z, (cy - K[1, 2]) / K[1, 1] * z, z])
            kp = expand17to70(Xc)
            uv = kp[:, :2] / kp[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
            outs.append(
                {
                    "bbox": b,
                    "pred_keypoints_3d": kp.astype(np.float32),
                    "pred_keypoints_2d": uv.astype(np.float32),
                    "pred_cam_t": Xc[J["pelvis"]].astype(np.float32),
                    "body_pose_params": MockMhr.encode(Xc - Xc[J["pelvis"]], 1.6),
                    "hand_pose_params": np.zeros(N_HAND, np.float32),
                    "scale_params": np.array([np.log(1.6)], np.float32),
                    "shape_params": np.zeros(N_SHAPE, np.float32),
                    "focal_length": np.array([K[0, 0]], np.float32),
                }
            )
        return outs


def make_estimator(backend: str, root: Path | str | None, checkpoint: str, device: str) -> Sam3dBody:
    if backend == "mock":
        return MockSam3d(torch.device("cpu"))
    return Sam3dBodyReal(root, checkpoint, device)
