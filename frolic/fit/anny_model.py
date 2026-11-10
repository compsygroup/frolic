"""Body-model adapters with one interface: the Anny model (NAVER) and a stick-figure mock for tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import roma
import torch

from frolic.skeleton import ANNY_POINTS, ANNY_POSED_BONES, ANNY_SEGMENT_LABELS, MHR70_SURFACE, N_JOINTS

HAND_POINTS: list[tuple[int, str, str]] = []  # (MHR keypoint, Anny bone, head|tail): tip -> finger-3 tail, ...
for _sd, _base in (("R", 21), ("L", 42)):
    for _f in range(5):
        _k0 = _base + 4 * _f
        _b = f"finger{_f + 1}"
        HAND_POINTS += [
            (_k0 + 3, f"{_b}-1.{_sd}", "h"),
            (_k0 + 2, f"{_b}-2.{_sd}", "h"),
            (_k0 + 1, f"{_b}-3.{_sd}", "h"),
            (_k0, f"{_b}-3.{_sd}", "t"),
        ]

_FACE_PREFIXES = (
    "finger",
    "metacarpal",
    "toe",
    "eye",
    "jaw",
    "tongue",
    "teeth",
    "oris",
    "levator",
    "special",
    "temporalis",
    "orbicularis",
    "risorius",
)


@dataclass
class SkeletonOut:
    points: torch.Tensor  # (N, 17, 3) joint points in the shared layout
    surf: torch.Tensor  # (N, 11, 3) face / foot surface landmarks (MHR70_SURFACE order)
    dense: torch.Tensor  # (N, D, 3) dense surface sample
    heads: torch.Tensor  # (N, B, 3) world bone heads
    bone_rot: torch.Tensor  # (N, B, 3, 3) world bone rotations
    hand: torch.Tensor  # (N, H, 3) finger points
    hand_wrist: torch.Tensor  # (N, H, 3) wrist of the same side
    rest_heads: torch.Tensor  # (B, 3)
    rest_tails: torch.Tensor  # (B, 3)


class Body(Protocol):
    bone_labels: list[str]
    posed_bones: list[str]
    phenotype_labels: list[str]
    segment_labels: list[str]
    hand_points: list[tuple[int, str, str]]
    device: torch.device
    faces: np.ndarray
    dominant_bone: torch.Tensor

    def skeleton(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> SkeletonOut: ...

    def rest(self, pheno: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def skin(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor: ...

    def age_from_years(self, years: float) -> float: ...


def _point(heads: torch.Tensor, tails: torch.Tensor, idx: dict[str, int], bone: str, end: str) -> torch.Tensor:
    return (heads if end == "h" else tails)[..., idx[bone], :]


def joint_points(heads: torch.Tensor, tails: torch.Tensor, idx: dict[str, int]) -> torch.Tensor:
    """Bone heads/tails (N, B, 3) -> (N, 17, 3) joints of the shared layout (pelvis = hip midpoint)."""
    pts = [0.5 * (heads[:, idx["upperleg01.L"]] + heads[:, idx["upperleg01.R"]])]
    for j in range(1, N_JOINTS):
        b, e, _ = ANNY_POINTS[j]
        pts.append(_point(heads, tails, idx, b, e))
    return torch.stack(pts, 1)


def joint_weights(device) -> torch.Tensor:
    return torch.tensor([1.0] + [ANNY_POINTS[j][2] for j in range(1, N_JOINTS)], device=device)


def farthest_points(P: np.ndarray, n: int) -> np.ndarray:
    sel = [int(np.argmin(P[:, 2]))]
    dmin = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(min(n, len(P)) - 1):
        k = int(np.argmax(dmin))
        sel.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(P - P[k], axis=1))
    return np.array(sel)


class AnnyBody:
    """Anny (rig default, SMPL-X topology, root_relative_world) with per-person local segment changes."""

    def __init__(
        self, device: torch.device, cache_dir: Path | str | None = None, dense_n: int = 400, segments: bool = True
    ) -> None:
        cache = str(cache_dir or os.environ.get("ANNY_CACHE_DIR", ""))
        if cache:
            os.environ.setdefault("ANNY_CACHE_DIR", cache)
        import anny

        kw = {"cache_dirname": cache} if cache else {}
        self.m = anny.create_fullbody_model(
            rig="default",
            topology="smplx",
            local_changes=segments,
            default_pose_parameterization="root_relative_world",
            **kw,
        ).to(dtype=torch.float32, device=device)
        m = self.m
        self.device = device
        self.bone_labels = list(m.bone_labels)
        self.idx = {b: i for i, b in enumerate(self.bone_labels)}
        self.n_bones = len(self.bone_labels)
        self.posed_bones = [b for b in ANNY_POSED_BONES if b in self.idx]
        self.posed_idx = torch.tensor([self.idx[b] for b in self.posed_bones], device=device)
        self.phenotype_labels = list(m.phenotype_labels)
        local = list(getattr(m, "local_change_labels", [])) if segments else []
        self.segment_labels = [lab for lab in ANNY_SEGMENT_LABELS if lab in local]
        self.hand_points = [h for h in HAND_POINTS if h[1] in self.idx]
        self.faces = m.faces.cpu().numpy() if torch.is_tensor(m.faces) else np.asarray(m.faces)
        self.n_vertices = m.vertex_bone_weights.shape[0]
        self.dominant_bone = m.vertex_bone_indices[
            torch.arange(self.n_vertices, device=device), m.vertex_bone_weights.argmax(1)
        ]
        self.surf_ids = torch.tensor([v for _, v in MHR70_SURFACE], device=device)
        self.dense_ids = torch.tensor(self._dense_ids(dense_n), device=device)
        self._vw = {
            k: (m.vertex_bone_weights[ids], m.vertex_bone_indices[ids])
            for k, ids in (("surf", self.surf_ids), ("dense", self.dense_ids))
        }
        self._age_map = None

    def _dense_ids(self, n: int) -> np.ndarray:
        dom = self.dominant_bone.cpu().numpy()
        bad = {i for i, b in enumerate(self.bone_labels) if b.startswith(_FACE_PREFIXES)}
        _, _, R0 = self.rest(
            torch.full((len(self.phenotype_labels),), 0.5, device=self.device),
            torch.zeros(len(self.segment_labels), device=self.device),
        )
        cand = np.flatnonzero([d not in bad for d in dom])
        return cand[farthest_points(R0[cand].cpu().numpy(), n)]

    def _coeffs(self, pheno: torch.Tensor, seg: torch.Tensor):
        m = self.m
        local = {lab: torch.tanh(seg[i : i + 1]) for i, lab in enumerate(self.segment_labels)}
        return m.get_phenotype_blendshape_coefficients(
            **m.parse_phenotype_kwargs({k: pheno[i : i + 1] for i, k in enumerate(self.phenotype_labels)}),
            local_changes=local,
        )

    def rest(self, pheno: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self._coeffs(pheno, seg)
        H0, T0, _ = self.m.get_rest_bone_poses(c)
        return H0[0], T0[0], self.m.get_rest_vertices(c)[0]

    def _skinned(self, key: str, RV: torch.Tensor, bt: torch.Tensor) -> torch.Tensor:
        W, I = self._vw[key]
        ids = self.surf_ids if key == "surf" else self.dense_ids
        vh = torch.cat([RV[ids], torch.ones(len(ids), 1, device=self.device)], 1)
        return (W[None, :, :, None] * (bt[:, I] @ vh[None, :, None, :, None]).squeeze(-1)[..., :3]).sum(2)

    def skeleton(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> SkeletonOut:
        m, dev, N, B = self.m, self.device, rv.shape[0], self.n_bones
        c = self._coeffs(pheno, seg)
        H0, T0, P0 = m.get_rest_bone_poses(c)
        H0, T0, P0 = H0.expand(N, -1, -1), T0.expand(N, -1, -1), P0.expand(N, -1, -1, -1)
        Dm = torch.eye(4, device=dev).expand(N, B, 4, 4).clone()
        Dm[:, self.posed_idx, :3, :3] = roma.rotvec_to_rotmat(rv)
        rest_root = roma.Rigid.from_homogeneous(P0[:, 0])
        base = rest_root.inverse().to_homogeneous()
        root = torch.eye(4, device=dev).expand(N, 4, 4).clone()
        root[:, :3, :3] = rootR
        root[:, :3, 3] = roottr
        Dm[:, 0] = root @ roma.Rigid(rest_root.linear, translation=None).to_homogeneous()
        bp, bt = m.get_bone_poses(rest_bone_poses=P0, delta_transforms=Dm, base_transform=base)
        heads, tails = m.get_bone_ends(H0, T0, P0, bp)
        RV = m.get_rest_vertices(c)[0]
        surf = self._skinned("surf", RV, bt)
        dense = self._skinned("dense", RV, bt)
        if self.hand_points:
            hand = torch.stack([_point(heads, tails, self.idx, b, e) for _, b, e in self.hand_points], 1)
            hand_wrist = torch.stack([heads[:, self.idx["wrist." + b[-1]]] for _, b, _ in self.hand_points], 1)
        else:
            hand = hand_wrist = torch.zeros(N, 0, 3, device=dev)
        return SkeletonOut(
            joint_points(heads, tails, self.idx), surf, dense, heads, bp[:, :, :3, :3], hand, hand_wrist, H0[0], T0[0]
        )

    def skin(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor:
        m, dev, n, B = self.m, self.device, rv.shape[0], self.n_bones
        Dm = torch.eye(4, device=dev).expand(n, B, 4, 4).clone()
        Dm[:, self.posed_idx, :3, :3] = roma.rotvec_to_rotmat(rv)
        Dm[:, 0, :3, :3] = rootR
        Dm[:, 0, :3, 3] = roottr
        ph = {k: pheno[i].expand(n) for i, k in enumerate(self.phenotype_labels)}
        local = {lab: torch.tanh(seg[i : i + 1]).expand(n) for i, lab in enumerate(self.segment_labels)}
        o = m(
            pose_parameters=Dm,
            phenotype_kwargs=ph,
            local_changes_kwargs=local,
            pose_parameterization="root_relative_world",
        )
        return o["vertices"]

    def age_from_years(self, years: float) -> float:
        if self._age_map is None:
            from anny.shape_distribution import SimpleShapeDistribution

            self._age_map = SimpleShapeDistribution(self.m).morphological_age_mapping
        a = self._age_map.morphological_age_anchors
        return float(self._age_map.morphological_to_anny_age(torch.tensor([float(years)], dtype=a.dtype, device=a.device)))


# ---------------------------------------------------------------------------------------------------- mock body
_MOCK_TREE: list[tuple[str, str | None]] = [
    ("root", None), ("spine05", "root"), ("spine03", "spine05"), ("spine01", "spine03"), ("neck01", "spine01"), ("head", "neck01"),
    ("clavicle.L", "spine01"), ("upperarm01.L", "clavicle.L"), ("lowerarm01.L", "upperarm01.L"), ("wrist.L", "lowerarm01.L"),
    ("clavicle.R", "spine01"), ("upperarm01.R", "clavicle.R"), ("lowerarm01.R", "upperarm01.R"), ("wrist.R", "lowerarm01.R"),
    ("upperleg01.L", "root"), ("lowerleg01.L", "upperleg01.L"), ("foot.L", "lowerleg01.L"), ("toe1-1.L", "foot.L"), ("toe5-1.L", "foot.L"),
    ("upperleg01.R", "root"), ("lowerleg01.R", "upperleg01.R"), ("foot.R", "lowerleg01.R"), ("toe1-1.R", "foot.R"), ("toe5-1.R", "foot.R"),
]  # fmt: skip
_MOCK_RADIUS = {
    "root": 0.10,
    "spine": 0.11,
    "neck01": 0.05,
    "head": 0.08,
    "clavicle": 0.04,
    "upperarm": 0.045,
    "lowerarm": 0.035,
    "wrist": 0.03,
    "upperleg": 0.07,
    "lowerleg": 0.05,
    "foot": 0.035,
    "toe": 0.02,
}
_SEG_SCALE = {
    "measure-upperarm-length-incr": ["upperarm01"],
    "measure-lowerarm-length-incr": ["lowerarm01"],
    "measure-upperleg-height-incr": ["upperleg01"],
    "measure-lowerleg-height-incr": ["lowerleg01"],
    "measure-napetowaist-dist-incr": ["spine03", "spine01"],
    "measure-waisttohip-dist-incr": ["spine05"],
    "measure-neck-height-incr": ["neck01"],
}


class MockBody:
    """A stick-figure body with Anny's bone names and conventions (y along the bone, x flexion axis) for tests."""

    def __init__(self, device: torch.device | None = None, dense_n: int = 300, segments: bool = True) -> None:
        self.device = device or torch.device("cpu")
        self.bone_labels = [b for b, _ in _MOCK_TREE]
        self.idx = {b: i for i, b in enumerate(self.bone_labels)}
        self.parent = [self.idx[p] if p else -1 for _, p in _MOCK_TREE]
        self.n_bones = len(self.bone_labels)
        self.posed_bones = [b for b in ANNY_POSED_BONES if b in self.idx]
        self.posed_idx = torch.tensor([self.idx[b] for b in self.posed_bones], device=self.device)
        self.phenotype_labels = ["gender", "age", "muscle", "weight", "height", "proportions"]
        self.segment_labels = list(ANNY_SEGMENT_LABELS) if segments else []
        self.hand_points: list[tuple[int, str, str]] = []
        self._build_vertices()
        self.n_vertices = len(self._v_t)
        self.dense_ids = torch.tensor(farthest_points(self._rest_vertices_np(0.5), dense_n), device=self.device)

    # rest geometry ---------------------------------------------------------------------------------------------------
    def _rest_geometry(self, pheno: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rest heads and tails (B, 3) as differentiable functions of the sliders; stature grows with height and age."""
        age, height, prop = pheno[1], pheno[4], pheno[5]
        s = (0.9 + 1.0 * height) * (0.4 + 0.6 * age)
        leg = 1.0 + 0.2 * (prop - 0.5)
        trunk = 1.0 - 0.1 * (prop - 0.5)
        sc = {b: torch.ones((), device=self.device) for b in self.bone_labels}
        for lab, bones in _SEG_SCALE.items():
            if lab in self.segment_labels:
                k = self.segment_labels.index(lab)
                for b in bones:
                    for sd in ("L", "R", ""):
                        name = f"{b}.{sd}" if sd else b
                        if name in sc:
                            sc[name] = sc[name] * (1 + 0.3 * torch.tanh(seg[k]))
        dev = self.device

        def vec(x, y, zz) -> torch.Tensor:
            return torch.stack(
                [
                    torch.as_tensor(v, dtype=torch.float32, device=dev) if not torch.is_tensor(v) else v
                    for v in (x, y, zz)
                ]
            )

        z = torch.zeros((), device=dev)
        heads: dict[str, torch.Tensor] = {}
        tails: dict[str, torch.Tensor] = {}
        leg_len = (0.245 * sc["upperleg01.L"] + 0.246 * sc["lowerleg01.L"]) * leg * s
        hip_c = vec(z, z, leg_len + 0.04 * s)
        heads["root"] = hip_c
        heads["spine05"] = hip_c + vec(z, z, 0.05 * s * trunk * sc["spine05"])
        heads["spine03"] = heads["spine05"] + vec(z, z, 0.10 * s * trunk * sc["spine03"])
        heads["spine01"] = heads["spine03"] + vec(z, z, 0.08 * s * trunk * sc["spine01"])
        heads["neck01"] = heads["spine01"] + vec(z, z, 0.08 * s * trunk * sc["neck01"])
        heads["head"] = heads["neck01"] + vec(z, z, 0.05 * s)
        tails["head"] = heads["head"] + vec(z, z, 0.11 * s)
        for sd, lat in (("L", 1.0), ("R", -1.0)):
            heads[f"clavicle.{sd}"] = heads["spine01"] + vec(lat * 0.02 * s, z, 0.06 * s)
            heads[f"upperarm01.{sd}"] = heads["spine01"] + vec(lat * 0.11 * s, z, 0.06 * s)
            ua = vec(lat * 0.67, 0.0, -0.74) * 0.17 * s * sc[f"upperarm01.{sd}"]
            heads[f"lowerarm01.{sd}"] = heads[f"upperarm01.{sd}"] + ua
            fa = vec(lat * 0.45, -0.55, -0.70) * 0.16 * s * sc[f"lowerarm01.{sd}"]
            heads[f"wrist.{sd}"] = heads[f"lowerarm01.{sd}"] + fa
            tails[f"wrist.{sd}"] = heads[f"wrist.{sd}"] + fa / 2
            heads[f"upperleg01.{sd}"] = hip_c + vec(lat * 0.05 * s, z, z)
            heads[f"lowerleg01.{sd}"] = heads[f"upperleg01.{sd}"] + vec(
                z, -0.03 * s, -0.245 * leg * s * sc[f"upperleg01.{sd}"]
            )
            heads[f"foot.{sd}"] = heads[f"lowerleg01.{sd}"] + vec(
                z, 0.03 * s, -0.246 * leg * s * sc[f"lowerleg01.{sd}"]
            )
            heads[f"toe1-1.{sd}"] = heads[f"foot.{sd}"] + vec(lat * 0.01 * s, -0.12 * s, -0.04 * s)
            heads[f"toe5-1.{sd}"] = heads[f"foot.{sd}"] + vec(lat * 0.04 * s, -0.10 * s, -0.04 * s)
            tails[f"toe1-1.{sd}"] = heads[f"toe1-1.{sd}"] + vec(z, -0.03 * s, z)
            tails[f"toe5-1.{sd}"] = heads[f"toe5-1.{sd}"] + vec(z, -0.025 * s, z)
            tails[f"foot.{sd}"] = 0.5 * (heads[f"toe1-1.{sd}"] + heads[f"toe5-1.{sd}"])
        for b, _p in _MOCK_TREE:
            if b in tails:
                continue
            kids = [c for c, pp in _MOCK_TREE if pp == b]
            tails[b] = heads[kids[0]] if kids else heads[b] + vec(z, z, 0.05 * s)
        tails["root"] = heads["spine05"]
        tails["spine01"] = heads["neck01"]
        H = torch.stack([heads[b] for b in self.bone_labels])
        T = torch.stack([tails[b] for b in self.bone_labels])
        return H, T

    @staticmethod
    def _frames(H: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Rest world rotation per bone (B, 3, 3): y along the bone, x the world-lateral flexion axis."""
        y = T - H
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        xw = torch.tensor([1.0, 0.0, 0.0], device=H.device).expand_as(y)
        x = xw - (xw * y).sum(-1, keepdim=True) * y
        bad = x.norm(dim=-1) < 1e-3
        x = torch.where(bad[:, None], torch.tensor([0.0, -1.0, 0.0], device=H.device).expand_as(y), x)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z = torch.cross(x, y, dim=-1)
        return torch.stack([x, y, z], -1)

    def _rest_vertices_np(self, h: float) -> np.ndarray:
        ph = torch.full((6,), 0.5)
        ph[4] = h
        return self.rest(ph, torch.zeros(len(self.segment_labels)))[2].numpy()

    def _build_vertices(self) -> None:
        """Tube of points around each bone, rigidly attached: (bone, t along the bone, angle, radius key)."""
        rows, faces = [], []
        for bi, b in enumerate(self.bone_labels):
            if b == "root":
                continue
            key = next((k for k in _MOCK_RADIUS if b.startswith(k)), "wrist")
            n_t, n_a = 5, 6
            base = len(rows)
            for ti in range(n_t):
                for ai in range(n_a):
                    rows.append((bi, 0.1 + 0.8 * ti / (n_t - 1), 2 * np.pi * ai / n_a, _MOCK_RADIUS[key]))
            for ti in range(n_t - 1):
                for ai in range(n_a):
                    a0, a1 = base + ti * n_a + ai, base + ti * n_a + (ai + 1) % n_a
                    b0, b1 = a0 + n_a, a1 + n_a
                    faces += [[a0, a1, b1], [a0, b1, b0]]
        self._v_bone = torch.tensor([r[0] for r in rows], device=self.device)
        self._v_t = torch.tensor([r[1] for r in rows], device=self.device)
        self._v_ang = torch.tensor([r[2] for r in rows], device=self.device)
        self._v_r = torch.tensor([r[3] for r in rows], device=self.device)
        self.dominant_bone = self._v_bone.clone()
        self.faces = np.array(faces, np.int32)

    def _local_vertices(self, H: torch.Tensor, T: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Vertices in each bone's local frame (origin at the head, y along the bone)."""
        L = (T - H).norm(dim=-1)[self._v_bone]
        r = self._v_r * s
        return torch.stack([r * torch.cos(self._v_ang), self._v_t * L, r * torch.sin(self._v_ang)], -1)

    def _stature_scale(self, pheno: torch.Tensor) -> torch.Tensor:
        return (0.9 + 1.0 * pheno[4]) * (0.4 + 0.6 * pheno[1])

    def rest(self, pheno: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, T = self._rest_geometry(pheno, seg)
        W0 = self._frames(H, T)
        V = H[self._v_bone] + torch.einsum(
            "vij,vj->vi", W0[self._v_bone], self._local_vertices(H, T, self._stature_scale(pheno))
        )
        return H, T, V

    def _fk(self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor):
        H, T = self._rest_geometry(pheno, seg)
        W0 = self._frames(H, T)
        N = rv.shape[0]
        Rl = torch.eye(3, device=self.device).expand(N, self.n_bones, 3, 3).clone()
        Rl[:, self.posed_idx] = roma.rotvec_to_rotmat(rv)
        R = [None] * self.n_bones
        t = [None] * self.n_bones
        for b in range(self.n_bones):
            p = self.parent[b]
            if p < 0:
                R[b] = rootR @ W0[b]
                t[b] = roottr + rootR @ H[b]
            else:
                rel = W0[p].T @ W0[b]
                R_pre = R[p] @ rel
                t[b] = t[p] + torch.einsum("nij,j->ni", R[p], W0[p].T @ (H[b] - H[p]))
                R[b] = R_pre @ Rl[:, b]
        return torch.stack(R, 1), torch.stack(t, 1), H, T, W0

    def skeleton(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> SkeletonOut:
        R, heads, H, T, W0 = self._fk(pheno, rootR, roottr, rv, seg)
        L = (T - H).norm(dim=-1)
        tails = heads + R[:, :, :, 1] * L[None, :, None]
        s = self._stature_scale(pheno)
        surf = self._surface(R, heads, H, T, W0, s)
        Vl = self._local_vertices(H, T, s)[self.dense_ids]
        bd = self._v_bone[self.dense_ids]
        dense = heads[:, bd] + torch.einsum("nvij,vj->nvi", R[:, bd], Vl)
        N = rv.shape[0]
        return SkeletonOut(
            joint_points(heads, tails, self.idx),
            surf,
            dense,
            heads,
            R,
            torch.zeros(N, 0, 3, device=self.device),
            torch.zeros(N, 0, 3, device=self.device),
            H,
            T,
        )

    def _surface(self, R, heads, H, T, W0, s) -> torch.Tensor:
        """The 11 MHR surface landmarks: nose, eyes, ears on the head; big toe, small toe, heel per foot."""
        specs = [
            ("head", (0.0, -0.09, 0.07)),
            ("head", (0.03, -0.08, 0.09)),
            ("head", (-0.03, -0.08, 0.09)),
            ("head", (0.07, 0.0, 0.08)),
            ("head", (-0.07, 0.0, 0.08)),
        ]
        for sd in ("L", "R"):
            specs += [
                (f"toe1-1.{sd}", (0.0, -0.03, 0.0)),
                (f"toe5-1.{sd}", (0.0, -0.025, 0.0)),
                (f"foot.{sd}", (0.0, 0.04, -0.04)),
            ]
        out = []
        for b, off in specs:
            bi = self.idx[b]
            loc = W0[bi].T @ (torch.tensor(off, device=self.device) * s)
            out.append(heads[:, bi] + torch.einsum("nij,j->ni", R[:, bi], loc))
        return torch.stack(out, 1)

    def skin(
        self, pheno: torch.Tensor, rootR: torch.Tensor, roottr: torch.Tensor, rv: torch.Tensor, seg: torch.Tensor
    ) -> torch.Tensor:
        R, heads, H, T, _ = self._fk(pheno, rootR, roottr, rv, seg)
        Vl = self._local_vertices(H, T, self._stature_scale(pheno))
        return heads[:, self._v_bone] + torch.einsum("nvij,vj->nvi", R[:, self._v_bone], Vl)

    def age_from_years(self, years: float) -> float:
        from frolic.fit.shape import age_years_to_slider

        return age_years_to_slider(years)


def make_body(
    backend: str, device: torch.device, cache_dir: Path | str | None = None, dense_n: int = 400, segments: bool = True
) -> Body:
    if backend == "mock":
        return MockBody(device, dense_n=min(dense_n, 300), segments=segments)
    return AnnyBody(device, cache_dir, dense_n, segments)
