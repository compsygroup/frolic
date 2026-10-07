"""Biomechanical constraints for the Anny "default" rig (differentiable, torch).

Inputs are the per-frame local bone rotations of the posed bones (axis-angle deltas in each bone's rest frame), the world
bone heads and optional sole points; nothing here depends on a dataset. Terms (each with a flag and a weight):
  limits     per-DOF anatomical box limits on the clinical decomposition of each bone's local rotation; hinge joints get a
             hyperextension allowance and a knee axial range that grows with flexion; children get paediatric ranges
             blended by the Anny age slider.
  collision  self-penetration between non-adjacent segments as capsules whose radii are measured on the person's rest mesh.
  foot       foot heading relative to the knee plane, foot yaw velocity, sole floor penetration and no-skating contact.
  velocity   per-bone angular-velocity soft limit.

Anny axis conventions (rest frame: +x = person's left, -y = forward, +z = up; for every limb bone local y runs along the
bone, local x is the flexion axis, local z the ab/adduction axis):
  upperleg01 (hip)      +x extension          | +z L adduction / R abduction | +y L internal / R external rotation
  lowerleg01 (knee)     +x flexion            | +z varus/valgus              | +y L internal / R external (tibial)
  foot (ankle)          +x plantarflexion     | +z L adduction / R abduction | +y L eversion / R inversion
  upperarm01 (shoulder) +x flexion            | +z L abduction / R adduction | +y L internal / R external rotation
  lowerarm01 (elbow)    +x flexion            | +z carrying-angle plane      | +y pronation/supination (symmetric range)
  wrist                 x / z flexion-extension / deviation (palm side unresolved, symmetric ranges)
  spine05/03/01, neck01, head: +x flexion, +z lateral bend to the right, +y axial rotation
The rest pose is a standing A-pose (knee ~10 deg, elbow ~43 deg flexed, upper arm ~42 deg abducted), measured per person
by rest_offsets(); anatomical angle = rest offset + local swing component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import roma
import torch

D = math.pi / 180.0

# Anatomical ranges in degrees, (low, high) per DOF, sign convention: flex > 0, abd > 0, internal rotation > 0
# (ankle: dorsiflexion > 0, foot adduction > 0, inversion > 0). SOURCES documents every value.
ADULT: dict[str, dict[str, tuple[float, float]]] = {
    "hip": {"flex": (-30, 130), "abd": (-30, 60), "rot": (-60, 45)},
    "knee": {"flex": (-5, 160), "abd": (-10, 10), "rot": (-45, 25)},
    "ankle": {"flex": (-50, 40), "abd": (-25, 25), "rot": (-15, 35)},
    "shoulder": {"flex": (-60, 180), "abd": (-40, 180), "rot": (-90, 70)},
    "elbow": {"flex": (-5, 150), "abd": (-25, 25), "rot": (-85, 85)},
    "wrist": {"flex": (-80, 80), "abd": (-30, 30), "rot": (-85, 85)},
    "spine": {"flex": (-25, 80), "abd": (-45, 45), "rot": (-45, 45)},
    "neck": {"flex": (-45, 45), "abd": (-45, 45), "rot": (-60, 60)},
    "clavicle": {"flex": (-36, 36), "abd": (-36, 36), "rot": (-36, 36)},
}
CHILD: dict[str, dict[str, tuple[float, float]]] = {g: dict(d) for g, d in ADULT.items()}
CHILD["hip"].update(flex=(-30, 140), rot=(-60, 50))
CHILD["knee"].update(flex=(-10, 160))
CHILD["ankle"].update(flex=(-65, 40))
CHILD["elbow"].update(flex=(-10, 150), rot=(-90, 90))

# DOF -> (adult, child, source). "chosen" marks values that are not from a reference and were chosen for this fit.
SOURCES: dict[str, tuple[str, str, str]] = {
    "hip flex / ext": (
        "130 / 30",
        "140 / 30",
        "flex: CDC joint ROM tables (Soucie 2011) 20-44 y 130-134, 2-8 y 140.8; ext: AAOS 30",
    ),
    "hip abd / add": (
        "60 / 30",
        "same",
        "add: AAOS 30; abd 60: cross-legged sitting needs up to 57 (J Arthroplasty 2008, PubMed 18358387), AAOS active 45",
    ),
    "hip IR / ER": (
        "45 / 60",
        "50 / 60",
        "IR: AAOS 45, child Scherff 2024 prone 50; ER 60: cross-legged sitting up to 58 (PubMed 18358387)",
    ),
    "knee flex": (
        "160",
        "160",
        "loaded postures exceed active ROM: squatting heels-up 157 +- 6 (Hemmerich 2006); active CDC 20-44 y 141.9",
    ),
    "knee hyperext": (
        "5",
        "10",
        "CDC pooled mean 1.8 (SD 2.9) -> mean + 1 SD; child Scherff 4 (SD 4) -> mean + 1.5 SD, CDC 2-8 y F 5.4",
    ),
    "knee varus / valgus": ("10 / 10", "same", "chosen (not found)"),
    "knee IR / ER": (
        "25 / 45, scaled by flexion (20 % at 0 deg, full from 30 deg)",
        "same",
        "passive at 30-90 deg flexion (WikiMSK Knee Biomechanics; Mossberg & Smith 1983); flexion scaling is a chosen screw-home approximation",
    ),
    "ankle dorsi / plantar": (
        "40 / 50",
        "40 / 65",
        "dorsi 40: heels-down squat 38.5 +- 5.9 (Hemmerich 2006), AAOS active 20; plantar AAOS 50, CDC 2-8 y F 67.1",
    ),
    "ankle inversion / eversion": ("35 / 15", "same", "AAOS"),
    "foot ab / adduction": ("25 / 25", "same", "chosen between Rajagopal subtalar +-20 and SKEL +-45"),
    "shoulder flex / ext": ("180 / 60", "same", "AAOS"),
    "shoulder abd / add": ("180 / 40", "same", "AAOS"),
    "shoulder IR / ER": ("70 / 90", "same", "AAOS"),
    "elbow flex": ("150", "150", "AAOS 150; CDC 2-8 y 152.9"),
    "elbow off-axis": ("+-25", "same", "chosen (carrying-angle plane vs Anny hinge plane)"),
    "elbow hyperext": ("5", "10", "CDC 20-44 y F 4.7; child CDC 2-8 y F 6.8"),
    "forearm pron / sup": (
        "+-85 about rest",
        "+-90",
        "AAOS 80 / 80; CDC 2-8 y pron 84.6 / sup 93.7; neutral unresolved -> symmetric",
    ),
    "wrist flex / ext": ("+-80", "same", "AAOS 80 / 70; palm side unresolved -> symmetric, larger value"),
    "wrist deviation": ("+-30", "same", "AAOS radial 20 / ulnar 30"),
    "wrist twist": ("+-85", "same", "chosen: Anny puts part of the forearm pronation on the wrist bone"),
    "thoracolumbar flex / ext": ("80 / 25", "same", "AAOS"),
    "thoracolumbar lateral / rotation": ("45 / 45", "same", "AAOS (reproductions disagree 35 vs 45 -> larger)"),
    "spine per bone": ("<= 0.6 x chain range", "same", "chosen (Anny poses 3 spine bones)"),
    "cervical flex / ext / lat / rot": ("45 / 45 / 45 / 60", "same", "AAOS"),
    "clavicle all DOFs": ("+-36", "same", "SC-joint ROM not found -> SKEL scapula abduction limit +-36 (Keller 2023)"),
    "angular velocity": (
        "400 deg/s lower limb + trunk, 700 deg/s upper limb",
        "same",
        "knee peak in gait 399 +- 72 deg/s (Pan 2025); upper limb chosen (~0.3 x pitching elbow extension 2450 deg/s)",
    ),
    "foot yaw velocity": ("360 deg/s", "same", "chosen"),
    "foot heading vs knee plane": ("[-45, 35]", "same", "knee rotation range + foot ab/adduction allowance (chosen)"),
    "contact / no-skating": (
        "height 4 cm, speed sigmoid v_t 0.3 m/s, alpha 5",
        "same",
        "HuMoR (Rempe 2021) contact heights; WHAM (Shin 2024) velocity sigmoid",
    ),
    "capsule radius": ("0.85 x median rest-mesh distance", "same", "SMPLify 2016 capsule approximation; scale chosen"),
}

# bone -> (group, sign for z (abd), sign for y (internal rotation), sign for x (flexion))
JOINTS: dict[str, tuple[str, int, int, int]] = {
    "upperleg01.L": ("hip", -1, -1, +1), "upperleg01.R": ("hip", -1, +1, -1),
    "lowerleg01.L": ("knee", +1, +1, +1), "lowerleg01.R": ("knee", +1, -1, -1),
    "foot.L": ("ankle", -1, +1, -1), "foot.R": ("ankle", -1, -1, +1),
    "upperarm01.L": ("shoulder", +1, +1, +1), "upperarm01.R": ("shoulder", +1, -1, -1),
    "lowerarm01.L": ("elbow", +1, +1, +1), "lowerarm01.R": ("elbow", +1, -1, -1),
    "wrist.L": ("wrist", +1, +1, +1), "wrist.R": ("wrist", +1, -1, -1),
    "clavicle.L": ("clavicle", +1, +1, +1), "clavicle.R": ("clavicle", +1, -1, -1),
}  # fmt: skip
CHAINS: dict[str, list[str]] = {"spine": ["spine05", "spine03", "spine01"], "neck": ["neck01", "head"]}
CHAIN_FRAC = 0.6
OMEGA_MAX_DPS: dict[str, float] = {
    "hip": 400,
    "knee": 400,
    "ankle": 400,
    "shoulder": 700,
    "elbow": 700,
    "wrist": 700,
    "clavicle": 400,
    "spine": 400,
    "neck": 400,
}


@dataclass
class BiomechSettings:
    limits: bool = True
    collision: bool = True
    foot: bool = True
    velocity: bool = True
    arm_collision: bool = False
    foot_collision: bool = True
    w_limits: float = 0.5
    w_collision: float = 0.5
    w_foot_heading: float = 0.3
    w_foot_yaw: float = 0.1
    w_floor: float = 0.3
    w_skate: float = 0.5
    w_velocity: float = 1.0
    collision_radius_scale: float = 0.85
    child_age_lo: float = 1 / 3
    child_age_hi: float = 0.45
    limit_margin_deg: float = 0.0
    foot_yaw_max_dps: float = 360.0
    contact_h: float = 0.04

    @classmethod
    def from_config(cls, c) -> BiomechSettings:
        return cls(
            limits=c.limits, collision=c.collision, foot=c.foot, velocity=c.velocity, arm_collision=c.arm_collision,
            foot_collision=c.foot_collision, w_limits=c.w_limits, w_collision=c.w_collision, w_foot_heading=c.w_foot_heading,
            w_foot_yaw=c.w_foot_yaw, w_floor=c.w_floor, w_skate=c.w_skate, w_velocity=c.w_velocity,
            collision_radius_scale=c.collision_radius_scale, child_age_hi=c.child_age_hi,
        )  # fmt: skip


def hub(x: torch.Tensor, d: float) -> torch.Tensor:
    """Huber on a non-negative excess: quadratic below d, linear above, so impossible evidence costs a bounded force."""
    return torch.where(x < d, 0.5 * x * x / d, x - 0.5 * d)


def swing_twist(rv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local axis-angle (..., 3) -> (swing_x, swing_z, twist_y) in rad; twist about local y (the bone axis)."""
    q = roma.rotvec_to_unitquat(rv)
    q = torch.where(q[..., 3:4] < 0, -q, q)
    tw = 2 * torch.atan2(q[..., 1], q[..., 3])
    ht = 0.5 * tw
    qt = torch.stack([torch.zeros_like(ht), torch.sin(ht), torch.zeros_like(ht), torch.cos(ht)], -1)
    sv = roma.unitquat_to_rotvec(roma.quat_product(q, roma.quat_conjugation(qt)))
    return sv[..., 0], sv[..., 2], torch.atan2(torch.sin(tw), torch.cos(tw))


def _ang(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1), (u * v).sum(-1))


def rest_offsets(rest_heads: torch.Tensor, idx: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Anatomical (flexion, abduction) angles of the rest pose per limb bone from the rest skeleton (z up, -y forward)."""
    H = rest_heads
    o: dict[str, tuple[float, float]] = {}
    for s in "LR":
        if f"foot.{s}" not in idx or f"wrist.{s}" not in idx:
            continue
        thigh = H[idx["lowerleg01." + s]] - H[idx["upperleg01." + s]]
        shank = H[idx["foot." + s]] - H[idx["lowerleg01." + s]]
        ua = H[idx["lowerarm01." + s]] - H[idx["upperarm01." + s]]
        fa = H[idx["wrist." + s]] - H[idx["lowerarm01." + s]]
        lat = 1.0 if s == "L" else -1.0
        o["upperleg01." + s] = (float(torch.atan2(-thigh[1], -thigh[2])), float(torch.atan2(lat * thigh[0], -thigh[2])))
        o["lowerleg01." + s] = (float(_ang(thigh, shank)), 0.0)
        o["upperarm01." + s] = (float(torch.atan2(-ua[1], -ua[2])), float(torch.atan2(lat * ua[0], -ua[2])))
        o["lowerarm01." + s] = (float(_ang(ua, fa)), 0.0)
    return o


def blended_ranges(age01: float, lo: float, hi: float) -> tuple[dict[str, dict[str, tuple[float, float]]], float]:
    """Adult/child ranges blended by the Anny age slider: child weight 1 at <= lo, 0 at >= hi."""
    wc = min(1.0, max(0.0, (hi - float(age01)) / (hi - lo)))
    R = {
        g: {d: tuple((1 - wc) * ADULT[g][d][i] + wc * CHILD[g][d][i] for i in range(2)) for d in ADULT[g]}
        for g in ADULT
    }
    return R, wc


class BiomechConstraints(torch.nn.Module):
    """One instance per person (rest skeleton + rest mesh of that body shape); call inside the fitting loop.

    bone_labels: all rig bones; posed_bones: bones whose local rotations rv[:, k] are optimised; rest_heads (B, 3);
    rest_verts (V, 3) and vertex_dom_bone (V,) build the collision capsules; age01: Anny age slider.
    """

    def __init__(
        self,
        bone_labels: list[str],
        posed_bones: list[str],
        rest_heads: torch.Tensor,
        rest_verts: torch.Tensor | None = None,
        vertex_dom_bone: torch.Tensor | None = None,
        age01: float = 0.5,
        fps: float = 15.0,
        settings: BiomechSettings | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        s = settings or BiomechSettings()
        self.s = s
        self.dev = device or rest_heads.device
        self.L = list(bone_labels)
        self.idx = {b: i for i, b in enumerate(self.L)}
        self.posed = list(posed_bones)
        self.pidx = {b: k for k, b in enumerate(self.posed)}
        self.fps = fps
        self.ranges, self.child_w = blended_ranges(age01, s.child_age_lo, s.child_age_hi)
        off = rest_offsets(rest_heads.detach().float(), self.idx)
        self.rest_off_deg = {b: (v[0] / D, v[1] / D) for b, v in off.items()}
        self._build_joint_table(off)
        self._build_velocity_table()
        self.foot_yaw_lim = s.foot_yaw_max_dps * D / fps
        self.caps = False
        if s.collision and rest_verts is not None and vertex_dom_bone is not None:
            self._build_capsules(
                rest_heads.detach().float(), rest_verts.detach().float(), vertex_dom_bone, s.collision_radius_scale
            )

    def _build_joint_table(self, off: dict[str, tuple[float, float]]) -> None:
        rows = []
        m = self.s.limit_margin_deg
        for b, (g, sx, sz, ty) in JOINTS.items():
            if b not in self.pidx:
                continue
            fo, ao = off.get(b, (0.0, 0.0))
            rg = self.ranges[g]
            rows.append(
                (
                    self.pidx[b],
                    sx,
                    sz,
                    ty,
                    fo,
                    ao,
                    *[(rg[d][0] - m) * D for d in ("flex", "abd", "rot")],
                    *[(rg[d][1] + m) * D for d in ("flex", "abd", "rot")],
                    g == "knee",
                )
            )
        self.jnames = [b for b in JOINTS if b in self.pidx]
        self.jgroup = [JOINTS[b][0] for b in self.jnames]
        dev = self.dev

        def t(k: int) -> torch.Tensor:
            return torch.tensor([r[k] for r in rows], dtype=torch.float32, device=dev)

        self.J_pi = torch.tensor([r[0] for r in rows], device=dev, dtype=torch.long)
        self.J_sign = torch.stack([t(1), t(2), t(3)], -1)
        self.J_off = torch.stack([t(4), t(5), torch.zeros(len(rows), device=dev)], -1)
        self.J_lo = torch.stack([t(6), t(7), t(8)], -1)
        self.J_hi = torch.stack([t(9), t(10), t(11)], -1)
        self.J_knee = torch.tensor([r[12] for r in rows], device=dev)
        self.J_euler = torch.tensor([JOINTS[bn][0] != "shoulder" for bn in self.jnames], device=dev)
        self.chains = {g: [self.pidx[b] for b in bs if b in self.pidx] for g, bs in CHAINS.items()}

    def _build_velocity_table(self) -> None:
        vel_idx, vlim = [], []
        for b in self.posed:
            g = (
                JOINTS[b][0]
                if b in JOINTS
                else ("spine" if b.startswith("spine") else "neck" if b in ("neck01", "head") else None)
            )
            if g is None:
                continue
            vel_idx.append(self.pidx[b])
            vlim.append(OMEGA_MAX_DPS[g] * D / self.fps)
        self.vel_idx = torch.tensor(vel_idx, device=self.dev, dtype=torch.long)
        self.vel_lim = torch.tensor(vlim, device=self.dev)

    # ---------------------------------------------------------------- angles and limits
    def anat_angles(self, rv: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """rv (N, P, 3) -> (N, J, 3) anatomical (flex, abd, rot) in rad for the JOINTS bones plus chain sums.

        Hip, knee, ankle, elbow, wrist, clavicle: intrinsic Euler X-Z-Y (the clinical joint-coordinate-system order,
        Grood & Suntay 1983). Shoulder: swing-twist (Euler X-Z-Y has gimbal lock at 90 deg abduction).
        """
        q = rv[:, self.J_pi]
        sx, sz, ty = swing_twist(q)
        e = roma.rotmat_to_euler("XZY", roma.rotvec_to_rotmat(q))
        eu = self.J_euler[None, :]
        A = (
            torch.stack(
                [torch.where(eu, e[..., 0], sx), torch.where(eu, e[..., 1], sz), torch.where(eu, e[..., 2], ty)], -1
            )
            * self.J_sign
            + self.J_off
        )
        ch = {}
        for g, pis in self.chains.items():
            if pis:
                a, b, c = swing_twist(rv[:, pis])
                ch[g] = torch.stack([a, b, c], -1)
        return A, ch

    def _excess(
        self, A: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, knee: torch.Tensor | None = None
    ) -> torch.Tensor:
        e_hi = torch.relu(A - hi)
        e_lo = torch.relu(lo - A)
        if knee is not None and bool(knee.any()):
            f = (A[..., 0] / (30 * D)).clamp(0.2, 1.0)
            rl, rh = lo[..., 2] * f, hi[..., 2] * f
            e_rot = torch.relu(A[..., 2] - rh) + torch.relu(rl - A[..., 2])
            e_hi = e_hi.clone()
            e_lo = e_lo.clone()
            e_hi[..., 2] = torch.where(knee, e_rot, e_hi[..., 2])
            e_lo[..., 2] = torch.where(knee, torch.zeros_like(e_rot), e_lo[..., 2])
        return e_hi + e_lo

    def limit_excess(self, rv: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(N, J, 3) excess beyond range (rad) per joint and {chain: (N, 3)} (chain sum + per-bone share)."""
        A, ch = self.anat_angles(rv)
        E = self._excess(A, self.J_lo, self.J_hi, self.J_knee)
        EC = {}
        for g, C in ch.items():
            rg = self.ranges[g]
            lo = torch.tensor([rg[d][0] * D for d in ("flex", "abd", "rot")], device=rv.device)
            hi = torch.tensor([rg[d][1] * D for d in ("flex", "abd", "rot")], device=rv.device)
            EC[g] = self._excess(C.sum(1), lo, hi) + self._excess(C, CHAIN_FRAC * lo, CHAIN_FRAC * hi).sum(1)
        return E, EC

    def limit_loss(self, rv: torch.Tensor) -> torch.Tensor:
        E, EC = self.limit_excess(rv)
        d = 5 * D
        tot = hub(E, d).sum((1, 2)) + sum(hub(e, d).sum(-1) for e in EC.values())
        return self.s.w_limits * tot.mean()

    # ---------------------------------------------------------------- capsules
    def _build_capsules(self, H: torch.Tensor, RV: torch.Tensor, dom: torch.Tensor, scale: float) -> None:
        idx = self.idx
        domname = [self.L[int(i)] for i in dom.tolist()]

        def side(s: str) -> dict:
            return {
                "thigh": ("upperleg01." + s, "lowerleg01." + s, ("upperleg", "pelvis"), 0.3),
                "shin": ("lowerleg01." + s, "foot." + s, ("lowerleg",), 0.0),
                "foot": ("foot." + s, ("toe1-1." + s, "toe5-1." + s), ("foot", "toe"), 0.0),
                "uarm": ("upperarm01." + s, "lowerarm01." + s, ("upperarm",), 0.0),
                "farm": ("lowerarm01." + s, "wrist." + s, ("lowerarm",), 0.0),
            }

        def ends(spec) -> list[str]:
            return [spec] if isinstance(spec, str) else list(spec)

        segs = {}
        for s in "LR":
            for k, v in side(s).items():
                if all(e in idx for e in ends(v[0]) + ends(v[1])):
                    segs[f"{k}_{s}"] = (v[0], v[1], v[2], v[3], s)
        self.cap_names = list(segs) + ["torso_L", "torso_R"]
        rad = []
        self._cap_spec: list = []
        for _n, (b0, b1, pref, t0, s) in segs.items():
            self._cap_spec.append((ends(b0), ends(b1), t0))
            p0 = H[[idx[e] for e in ends(b0)]].mean(0)
            p1 = H[[idx[e] for e in ends(b1)]].mean(0)
            p0 = p0 + t0 * (p1 - p0)
            sel = torch.tensor(
                [any(d.startswith(p) for p in pref) and d.endswith("." + s) for d in domname], device=H.device
            )
            rad.append(self._seg_radius(RV[sel], p0, p1, 50) * scale)
        hipm = 0.5 * (H[idx["upperleg01.L"]] + H[idx["upperleg01.R"]])
        neck = H[idx["neck01"]]
        tsel = torch.tensor([d.startswith(("spine", "root", "pelvis", "breast")) for d in domname], device=H.device)
        for s in "LR":
            self._cap_spec.append(("torso", s))
            p0 = hipm + 0.5 * (H[idx["upperleg01." + s]] - hipm)
            p1 = neck + 0.5 * (H[idx["upperarm01." + s]] - neck)
            P = RV[tsel]
            P = P[(P[:, 0] > hipm[0]) if s == "L" else (P[:, 0] <= hipm[0])]
            rad.append(self._seg_radius(P, p0, p1, 30) * scale)
        self.cap_r = torch.stack(rad).to(self.dev)
        nm = {n: i for i, n in enumerate(self.cap_names)}
        pairs = []
        FP = (
            [("foot", "foot"), ("foot", "shin"), ("shin", "foot"), ("thigh", "foot"), ("foot", "thigh")]
            if self.s.foot_collision
            else []
        )
        for a_, b_ in (
            [("thigh", "thigh"), ("thigh", "shin"), ("shin", "thigh"), ("shin", "shin")]
            + FP
            + ([("farm", "farm")] if self.s.arm_collision else [])
        ):
            if f"{a_}_L" in nm and f"{b_}_R" in nm:
                pairs.append((nm[f"{a_}_L"], nm[f"{b_}_R"]))
        for s in "LR":
            for t_ in "LR":
                if f"shin_{s}" in nm:
                    pairs.append((nm[f"shin_{s}"], nm[f"torso_{t_}"]))
                if self.s.foot_collision and f"foot_{s}" in nm:
                    pairs.append((nm[f"foot_{s}"], nm[f"torso_{t_}"]))
                if self.s.arm_collision and f"farm_{s}" in nm:
                    pairs.append((nm[f"farm_{s}"], nm[f"torso_{t_}"]))
        self.pairs = torch.tensor(pairs, device=self.dev, dtype=torch.long).reshape(-1, 2)
        self.pair_names = [f"{self.cap_names[i]}-{self.cap_names[j]}" for i, j in pairs]
        self.caps = len(pairs) > 0

    @staticmethod
    def _seg_radius(P: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, pct: float) -> torch.Tensor:
        if len(P) < 10:
            return torch.tensor(0.03, device=p0.device)
        d = p1 - p0
        t = ((P - p0) @ d / d.dot(d)).clamp(0, 1)
        keep = (t > 0.15) & (t < 0.85)
        dist = torch.linalg.norm(P - (p0 + t[:, None] * d), dim=-1)[keep]
        return torch.quantile(dist, pct / 100.0) if len(dist) > 10 else torch.tensor(0.03, device=p0.device)

    def capsule_ends(self, heads: torch.Tensor) -> torch.Tensor:
        """heads (N, B, 3) -> (N, C, 2, 3) capsule end points."""
        idx = self.idx
        hipm = 0.5 * (heads[:, idx["upperleg01.L"]] + heads[:, idx["upperleg01.R"]])
        neck = heads[:, idx["neck01"]]
        out = []
        for spec in self._cap_spec:
            if spec[0] == "torso":
                s = spec[1]
                p0 = hipm + 0.5 * (heads[:, idx["upperleg01." + s]] - hipm)
                p1 = neck + 0.5 * (heads[:, idx["upperarm01." + s]] - neck)
            else:
                e0, e1, t0 = spec
                p0 = heads[:, [idx[e] for e in e0]].mean(1)
                p1 = heads[:, [idx[e] for e in e1]].mean(1)
                p0 = p0 + t0 * (p1 - p0)
            out.append(torch.stack([p0, p1], 1))
        return torch.stack(out, 1)

    @staticmethod
    def seg_seg_dist(p1, q1, p2, q2, eps: float = 1e-8) -> torch.Tensor:
        """Closest distance between segments [p1, q1] and [p2, q2] (Ericson, Real-Time Collision Detection 5.1.9)."""
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = (d1 * d1).sum(-1).clamp_min(eps)
        e = (d2 * d2).sum(-1).clamp_min(eps)
        f = (d2 * r).sum(-1)
        c = (d1 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        den = (a * e - b * b).clamp_min(eps)
        s = ((b * f - c * e) / den).clamp(0, 1)
        t = (b * s + f) / e
        s = torch.where(t < 0, (-c / a).clamp(0, 1), torch.where(t > 1, ((b - c) / a).clamp(0, 1), s))
        t = t.clamp(0, 1)
        v = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
        return torch.sqrt((v * v).sum(-1) + 1e-10)

    def penetration(self, heads: torch.Tensor) -> torch.Tensor:
        """(N, pairs) capsule overlap depth in m (> 0 = interpenetrating)."""
        C = self.capsule_ends(heads)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = self.seg_seg_dist(C[:, i, 0], C[:, i, 1], C[:, j, 0], C[:, j, 1])
        return self.cap_r[i] + self.cap_r[j] - d

    def collision_loss(self, heads: torch.Tensor) -> torch.Tensor:
        return self.s.w_collision * hub(torch.relu(self.penetration(heads)), 0.01).sum(-1).mean()

    # ---------------------------------------------------------------- feet
    def foot_vectors(self, heads: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = self.idx
        return {
            s: 0.5 * (heads[:, idx["toe1-1." + s]] + heads[:, idx["toe5-1." + s]]) - heads[:, idx["foot." + s]]
            for s in "LR"
            if "toe1-1." + s in idx
        }

    def foot_loss(
        self,
        heads: torch.Tensor,
        rv: torch.Tensor,
        sole: torch.Tensor | None = None,
        sole_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """heads (N, B, 3); rv (N, P, 3); sole: optional (N, K, 3) sole points in a z-up frame with the floor at z = 0."""
        N = heads.shape[0]
        loss = heads.sum() * 0
        A, _ = self.anat_angles(rv)
        for s in "LR":
            kb, ab = "lowerleg01." + s, "foot." + s
            if kb in self.jnames and ab in self.jnames:
                k, f = self.jnames.index(kb), self.jnames.index(ab)
                head = A[:, k, 2] + A[:, f, 1]
                loss = (
                    loss
                    + self.s.w_foot_heading * hub(torch.relu(head - 35 * D) + torch.relu(-45 * D - head), 5 * D).mean()
                )
        if N > 1:
            for v in self.foot_vectors(heads).values():
                yaw = torch.atan2(v[:, 1], v[:, 0])
                dy = yaw[1:] - yaw[:-1]
                dy = torch.atan2(torch.sin(dy), torch.cos(dy))
                horiz = torch.linalg.norm(v[:, :2], dim=-1) / torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
                wv = (horiz[1:] * horiz[:-1]).detach()
                loss = loss + self.s.w_foot_yaw * (wv * hub(torch.relu(dy.abs() - self.foot_yaw_lim), 5 * D)).mean()
        if sole is not None:
            z = sole[..., 2]
            sw = sole_w if sole_w is not None else torch.ones_like(z)
            loss = loss + self.s.w_floor * (sw * hub(torch.relu(-z - 0.01), 0.01)).sum(-1).mean()
            if N > 1:
                vh = (sole[1:, :, :2] - sole[:-1, :, :2]).pow(2).sum(-1) * self.fps**2
                c = torch.sigmoid((self.s.contact_h - z) / 0.01)
                c = (c[1:] * c[:-1] * sw[1:] * sw[:-1] * torch.sigmoid(-5 * (vh.sqrt() - 0.3) / 0.3)).detach()
                loss = loss + self.s.w_skate * (c * vh).sum(-1).mean() / self.fps
        return loss

    def velocity_loss(self, rv: torch.Tensor) -> torch.Tensor:
        if rv.shape[0] < 2 or len(self.vel_idx) == 0:
            return rv.sum() * 0
        R = roma.rotvec_to_rotmat(rv[:, self.vel_idx])
        rel = R[1:].transpose(-1, -2) @ R[:-1]
        tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        ang = torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
        return self.s.w_velocity * hub(torch.relu(ang - self.vel_lim), 5 * D).sum(-1).mean()

    def forward(
        self,
        rv: torch.Tensor,
        heads: torch.Tensor,
        sole: torch.Tensor | None = None,
        sole_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = rv.sum() * 0
        if self.s.limits:
            loss = loss + self.limit_loss(rv)
        if self.s.collision and self.caps:
            loss = loss + self.collision_loss(heads)
        if self.s.foot:
            loss = loss + self.foot_loss(heads, rv, sole, sole_w)
        if self.s.velocity:
            loss = loss + self.velocity_loss(rv)
        return loss

    @torch.no_grad()
    def report(self, rv: torch.Tensor, heads: torch.Tensor, tol_deg: float = 5.0) -> dict[str, int]:
        """Frames per joint group with any DOF > tol beyond range; capsule frames with overlap > 1 / 3 cm."""
        E, EC = self.limit_excess(rv)
        out: dict[str, int] = {}
        for g in sorted(set(self.jgroup)):
            k = [i for i, x in enumerate(self.jgroup) if x == g]
            out[f"lim_{g}"] = int((E[:, k].amax((1, 2)) > tol_deg * D).sum())
        for g, e in EC.items():
            out[f"lim_{g}"] = int((e.amax(-1) > tol_deg * D).sum())
        if self.caps:
            p = self.penetration(heads).amax(-1)
            out["cap_pen_gt1cm"] = int((p > 0.01).sum())
            out["cap_pen_gt3cm"] = int((p > 0.03).sum())
        return out


@torch.no_grad()
def leg_ik_init(
    W_hip0,
    W_knee0,
    hip0,
    knee0,
    hip,
    knee,
    ankle,
    valid,
    rest_knee_rad: float,
    lat_world,
    min_flex_deg: float = 20.0,
    smooth: int = 4,
):
    """Hinge-consistent analytic leg initialisation: the knee bends about its flexion axis only, the hip absorbs swing + twist.

    W_hip0 / W_knee0 (N, 3, 3): world orientation of the thigh and shank bones with zero local rotations; hip0 / knee0 (N, 3):
    their heads in that pose; hip / knee / ankle (N, 3): targets; valid (N,); lat_world (N, 3): the person's left axis.
    Returns local rotation vectors (N, 3) for the hip and the knee (nearest valid frame where invalid).
    """
    N = hip.shape[0]
    dev = hip.device
    t_loc = torch.einsum("nji,nj->ni", W_hip0, knee0 - hip0)
    t_loc = t_loc / t_loc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a_loc = torch.einsum("nji,nj->ni", W_hip0, W_knee0[..., 0])
    d1 = knee - hip
    d2 = ankle - knee
    d1 = d1 / d1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d2n = d2 / d2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    th = torch.atan2(torch.cross(d1, d2n, dim=-1).norm(dim=-1), (d1 * d2n).sum(-1))
    n = torch.cross(d1, d2n, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    use = (th > min_flex_deg * D) & ((n * lat_world).sum(-1) > -0.5) & valid
    wgt = torch.where(use, torch.sin(th).clamp_min(0), torch.zeros_like(th))
    k = 2 * smooth + 1
    ker = torch.ones(1, 1, k, device=dev)
    num = torch.nn.functional.conv1d((n * wgt[:, None]).T[:, None], ker, padding=smooth)[:, 0].T
    den = torch.nn.functional.conv1d(wgt[None, None], ker, padding=smooth)[0, 0]
    lw = 0.05 * k
    n = (num + lw * lat_world) / (den + lw)[:, None]
    n = n - (n * d1).sum(-1, keepdim=True) * d1
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a_o = a_loc - (a_loc * t_loc).sum(-1, keepdim=True) * t_loc
    a_o = a_o / a_o.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    Fl = torch.stack([t_loc, a_o, torch.cross(t_loc, a_o, dim=-1)], -1)
    Fw = torch.stack([d1, n, torch.cross(d1, n, dim=-1)], -1)
    dh = W_hip0.transpose(-1, -2) @ (Fw @ Fl.transpose(-1, -2))
    flex = torch.where(use, th, torch.zeros_like(th)) - rest_knee_rad
    dk = torch.zeros(N, 3, device=dev)
    dk[:, 0] = flex
    rh = roma.rotmat_to_rotvec(dh)
    ok = valid & torch.isfinite(rh).all(-1) & torch.isfinite(dk).all(-1)
    if not bool(ok.any()):
        return torch.zeros_like(rh), torch.zeros_like(dk)
    t = torch.arange(N, device=dev)
    g = t[ok]
    near = g[(t[:, None] - g[None, :]).abs().argmin(1)]
    return rh[near], dk[near]


def ranges_table() -> list[dict[str, str]]:
    """The SOURCES table as rows (for documentation)."""
    return [{"dof": k, "adult": a, "child": c, "source": s} for k, (a, c, s) in SOURCES.items()]


__all__ = [
    "ADULT",
    "CHILD",
    "SOURCES",
    "BiomechConstraints",
    "BiomechSettings",
    "blended_ranges",
    "hub",
    "leg_ik_init",
    "ranges_table",
    "rest_offsets",
    "swing_twist",
]
