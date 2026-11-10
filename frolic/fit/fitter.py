"""Per-person body fit: one body shape (phenotype + segment lengths) and per-frame pose.

Stage 1 fits shape and pose to the triangulated tracks (an unconstrained shape pass, then a constrained pose pass that
starts from a hinge-consistent leg initialisation). Stage 2 keeps the shape and refits the pose to the multi-view
SAM 3D Body estimate (joints, face/foot landmarks, fingers, dense surface correspondences) gated against the tracks, with
the biomechanical constraints, root-rotation continuity and reduced weight on single-view frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import roma
import torch

from frolic.config import FitConfig, PersonConfig
from frolic.fit.anny_model import Body, joint_weights
from frolic.fit.biomech import BiomechConstraints, BiomechSettings, leg_ik_init, rest_offsets
from frolic.fit.filters import bridge_single_view, fill_interp, jumpy, smooth_series, soft_fade, speed_weights
from frolic.fit.root import RootParam, geodesic_acceleration
from frolic.fit.shape import (
    GROUP_SEGMENT_LABELS,
    FixedPhenotype,
    LengthTargets,
    height_slider_for_stature,
    initial_age_from_leg,
    logit,
    segment_length_targets,
    standing_ankle_height,
    weight_muscle_for_mass,
)
from frolic.geometry import kabsch
from frolic.log import get_logger
from frolic.skeleton import (
    JOINT_NAMES,
    MHR70_BODY,
    MHR70_HIPS,
    MHR70_SOLE,
    MHR70_SURFACE,
    MHR70_WRIST_L,
    MHR70_WRIST_R,
    SEGMENTS,
    J,
)

log = get_logger(__name__)

KEY_JOINTS = [J["hip_l"], J["hip_r"], J["shoulder_l"], J["shoulder_r"], J["neck"]]
SAM_BODY_JOINTS = sorted(MHR70_BODY)
SAM_BODY_KP = [MHR70_BODY[j] for j in SAM_BODY_JOINTS]
SAM_SURF_KP = [k for k, _ in MHR70_SURFACE]


@dataclass
class FitInputs:
    X: np.ndarray  # (N, 17, 3) track joints, NaN where missing
    single_view: np.ndarray | None = None  # (N,)
    observed: np.ndarray | None = None  # (N,)
    sam_kp: np.ndarray | None = None  # (N, 70, 3) world (levelled frame)
    sam_verts: np.ndarray | None = None  # (N, V, 3) posed SAM 3D Body vertices
    cuts: list[int] = field(default_factory=list)
    person: PersonConfig | None = None


@dataclass
class FitResult:
    phenotype: np.ndarray
    phenotype_labels: list[str]
    rootrot: np.ndarray
    roottr: np.ndarray
    rotvecs: np.ndarray
    posed_bones: list[str]
    joints: np.ndarray
    target: np.ndarray
    have: np.ndarray
    single_view: np.ndarray
    bridged: np.ndarray
    observed: np.ndarray
    fps: float
    stature: float
    seg_labels: list[str]
    seg_values: np.ndarray
    biomech_report: dict
    diagnostics: dict = field(default_factory=dict)
    frame0: int = 0

    def save(self, path: Path, verts: np.ndarray | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, phenotype=self.phenotype, phenotype_labels=np.array(self.phenotype_labels), rootrot=self.rootrot, roottr=self.roottr,
            rotvecs=self.rotvecs, posed_bones=np.array(self.posed_bones), joints=self.joints, joint_names=np.array(JOINT_NAMES),
            target=self.target, have=self.have, single_view=self.single_view, bridged=self.bridged, observed=self.observed,
            frame0=self.frame0, fps=self.fps, stature=self.stature, seg_labels=np.array(self.seg_labels), seg_values=self.seg_values,
            biomech_report=json.dumps(self.biomech_report), diagnostics=json.dumps(self.diagnostics),
        )  # fmt: skip
        if verts is not None:
            np.save(path.with_name(path.stem + "_verts.npy"), verts.astype(np.float16))


def hub(r: torch.Tensor) -> torch.Tensor:
    return torch.where(r < 0.03, 0.5 * r**2 / 0.03, r - 0.015)


class PersonFitter:
    def __init__(self, body: Body, cfg: FitConfig, fps: float) -> None:
        self.body = body
        self.cfg = cfg
        self.fps = fps
        self.dev = body.device
        self.W = joint_weights(self.dev)
        self.nfade = round(cfg.fade_s * fps)
        self.bm_settings = BiomechSettings.from_config(cfg.biomech)
        self.labels = list(body.phenotype_labels)
        self.seg_labels = list(body.segment_labels) if cfg.segments else []
        self.sb_idx = torch.tensor(SAM_BODY_JOINTS, device=self.dev)

    # ------------------------------------------------------------------------------------------------ helpers
    def _skel(self, pl, root: RootParam, roottr, rv, segv):
        return self.body.skeleton(torch.sigmoid(pl), root.matrices()[: len(rv)], roottr, rv, self._seg(segv))

    def _seg(self, segv: torch.Tensor) -> torch.Tensor:
        return segv if self.seg_labels else torch.zeros(0, device=self.dev)

    def _acc(self, q: torch.Tensor, ws: torch.Tensor, cuts: list[int]) -> torch.Tensor:
        d = (q[2:] - 2 * q[1:-1] + q[:-2]).pow(2)
        if cuts:
            m = torch.ones(len(d), device=self.dev)
            for c in cuts:
                m[max(0, c - 2) : c] = 0
            ws = ws * m
        return (ws.view(-1, *([1] * (d.dim() - 1))) * d).mean()

    def _make_bm(self, pl: torch.Tensor, segv: torch.Tensor) -> BiomechConstraints | None:
        if not self.cfg.biomech.enabled:
            return None
        with torch.no_grad():
            H0, _, RV0 = self.body.rest(torch.sigmoid(pl).detach(), self._seg(segv.detach()))
        age = float(torch.sigmoid(pl)[self.labels.index("age")])
        return BiomechConstraints(
            self.body.bone_labels,
            self.body.posed_bones,
            H0,
            RV0,
            self.body.dominant_bone,
            age01=age,
            fps=self.fps,
            settings=self.bm_settings,
            device=self.dev,
        )

    def _stature(self, pheno: torch.Tensor, segv: torch.Tensor) -> float:
        with torch.no_grad():
            V = self.body.rest(pheno, self._seg(segv))[2]
        return float(V[:, 2].max() - V[:, 2].min())

    def _mesh_mass(self, pheno: torch.Tensor, segv: torch.Tensor) -> float:
        with torch.no_grad():
            V = self.body.rest(pheno, self._seg(segv))[2]
        F = torch.as_tensor(np.asarray(self.body.faces), dtype=torch.long, device=self.dev)
        vol = float((V[F[:, 0]] * torch.cross(V[F[:, 1]], V[F[:, 2]], dim=1)).sum() / 6)
        return abs(vol) * self.cfg.density_kg_m3

    def _rest_slot(self, rest_heads: torch.Tensor, rest_tails: torch.Tensor, j: int) -> torch.Tensor:
        from frolic.skeleton import ANNY_POINTS

        idx = self.body.idx
        if j == 0:
            return 0.5 * (rest_heads[idx["upperleg01.L"]] + rest_heads[idx["upperleg01.R"]])
        b, e, _ = ANNY_POINTS[j]
        return rest_heads[idx[b]] if e == "h" else rest_tails[idx[b]]

    def _rest_lengths(self, rest_heads: torch.Tensor, rest_tails: torch.Tensor, targets: LengthTargets) -> torch.Tensor:
        from frolic.fit.shape import LEN_GROUPS

        out = []
        for g in targets.groups:
            ls = [
                (self._rest_slot(rest_heads, rest_tails, a) - self._rest_slot(rest_heads, rest_tails, b)).norm()
                for a, b in (SEGMENTS[m] for m in LEN_GROUPS[g])
            ]
            out.append(torch.stack(ls).mean())
        return torch.stack(out)

    # ------------------------------------------------------------------------------------------------ initialisation
    def _root_init(self, X: np.ndarray, pl: torch.Tensor, segv: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        N = len(X)
        with torch.no_grad():
            o = self.body.skeleton(
                torch.sigmoid(pl),
                torch.eye(3, device=self.dev)[None],
                torch.zeros(1, 3, device=self.dev),
                torch.zeros(1, len(self.body.posed_bones), 3, device=self.dev),
                self._seg(segv),
            )
        P0 = o.points[0].cpu().numpy()
        rr = np.full((N, 3), np.nan, np.float32)
        tr = np.full((N, 3), np.nan, np.float32)
        for i in range(N):
            ok = [k for k in KEY_JOINTS if np.isfinite(X[i, k, 0])]
            if len(ok) >= 3 and np.isfinite(X[i, 0, 0]):
                R = kabsch(P0[ok], X[i, ok])
                rr[i] = roma.rotmat_to_rotvec(torch.tensor(R, dtype=torch.float32)).numpy()
                tr[i] = X[i, 0] - R @ P0[0]
        return rr, fill_interp(tr)

    def _ik_init(self, rv: torch.Tensor, pl, root: RootParam, roottr, segv, Xt: torch.Tensor, Mk: torch.Tensor) -> int:
        with torch.no_grad():
            o = self._skel(pl, root, roottr, rv, segv)
            idx = self.body.idx
            lat = root.matrices()[:, :, 0]
            ro = rest_offsets(o.rest_heads, idx)
            n = 0
            for sd, (sh, sk, sa) in (
                ("L", (J["hip_l"], J["knee_l"], J["ankle_l"])),
                ("R", (J["hip_r"], J["knee_r"], J["ankle_r"])),
            ):
                hb, kb = idx["upperleg01." + sd], idx["lowerleg01." + sd]
                val = (Mk[:, sh] > 0) & (Mk[:, sk] > 0) & (Mk[:, sa] > 0)
                rh, rk = leg_ik_init(
                    o.bone_rot[:, hb],
                    o.bone_rot[:, kb],
                    o.heads[:, hb],
                    o.heads[:, kb],
                    Xt[:, sh],
                    Xt[:, sk],
                    Xt[:, sa],
                    val,
                    ro["lowerleg01." + sd][0],
                    lat,
                )
                rv[:, self.body.posed_bones.index("upperleg01." + sd)] = rh
                rv[:, self.body.posed_bones.index("lowerleg01." + sd)] = rk
                n += int(val.sum())
        return n

    # ------------------------------------------------------------------------------------------------ main
    def fit(self, inp: FitInputs, pid: str = "") -> FitResult | None:
        cfg, dev, fps = self.cfg, self.dev, self.fps
        X_in = np.asarray(inp.X, np.float32)
        N = len(X_in)
        SVF = np.zeros(N, bool) if inp.single_view is None else np.asarray(inp.single_view, bool).copy()
        OBS = np.zeros(N, bool) if inp.observed is None else np.asarray(inp.observed, bool).copy()
        X_stats = X_in.copy()
        X_stats[SVF] = np.nan
        ank_h = standing_ankle_height(X_stats, fps) if cfg.ankle_bias else np.nan
        targets = segment_length_targets(X_stats)
        X, BRG = bridge_single_view(X_in, SVF, round(cfg.sv_bridge_s * fps))
        SVF = SVF | BRG
        have = np.isfinite(X[:, 0, 0])
        if have.sum() < 30:
            log.warning("%s: %d frames with 3D, too few; skipped", pid, int(have.sum()))
            return None
        if BRG.any():
            log.info("%s: %d frames bridged inside single-view runs", pid, int(BRG.sum()))
        Xt = torch.tensor(np.nan_to_num(X), device=dev)
        svw = np.where(SVF, cfg.sv_weight, 1.0)[:, None].astype(np.float32)
        Mk = torch.tensor(np.isfinite(X[..., 0]) * svw, device=dev).float() * self.W
        ws_t = torch.tensor(speed_weights(X, fps, cfg.smooth_v0, cfg.smooth_floor), device=dev)
        cuts = [int(c) for c in inp.cuts if 0 < c < N]

        fixed = FixedPhenotype.from_person(inp.person, self.body.age_from_years)
        age0 = fixed.values.get("age", initial_age_from_leg(X))
        pl = torch.zeros(len(self.labels), device=dev)
        pl[self.labels.index("age")] = logit(age0)
        fixed.apply(pl, self.labels)
        free_mask = fixed.mask(self.labels, dev)
        build_idx = [self.labels.index(k) for k in ("weight", "muscle") if k in self.labels and k not in fixed.values]
        segv = torch.zeros(len(self.seg_labels), device=dev)
        rr, tr = self._root_init(X, pl, segv)

        len_t = torch.tensor(targets.target, dtype=torch.float32, device=dev)
        len_w = torch.tensor(targets.weight, dtype=torch.float32, device=dev)
        len_ok = torch.tensor(targets.ok, device=dev)
        grp_active = torch.zeros(len(targets.groups), dtype=torch.bool, device=dev)
        seg_active = torch.zeros(len(self.seg_labels), device=dev)

        def len_loss(o):
            if not bool(grp_active.any()):
                return torch.zeros((), device=dev)
            la = self._rest_lengths(o.rest_heads, o.rest_tails, targets)
            return (len_w[len_ok] * (la[len_ok] / len_t[len_ok] - 1) ** 2).sum()

        passes = (
            [(False, True), (True, False)]
            if (cfg.biomech.enabled and cfg.biomech.shape_pass)
            else [(cfg.biomech.enabled, True)]
        )
        root = roottr = rv = None
        for use_bm, fit_shape in passes:
            root = RootParam(rr, dev, continuous=cfg.root_continuity)
            roottr = torch.tensor(tr, device=dev, requires_grad=True)
            rv = torch.zeros(N, len(self.body.posed_bones), 3, device=dev)
            if use_bm and cfg.biomech.ik_init:
                n_ik = self._ik_init(rv, pl, root, roottr, segv, Xt, Mk)
                log.info("%s: IK leg init on %d leg-frames", pid, n_ik)
            rv.requires_grad_(True)
            pl.requires_grad_(fit_shape)
            hooks = []
            if fit_shape:
                hooks.append(pl.register_hook(lambda g: g * free_mask))
                with torch.no_grad():
                    segv.zero_()
                segv.requires_grad_(bool(self.seg_labels))
                if self.seg_labels:
                    hooks.append(segv.register_hook(lambda g: g * seg_active))
                groups = [{"params": [root.delta, roottr, rv], "lr": 2e-2}, {"params": [pl], "lr": 2e-2}]
                if self.seg_labels:
                    groups.append({"params": [segv], "lr": 2e-2})
                opt = torch.optim.Adam(groups)
            else:
                segv.requires_grad_(False)
                opt = torch.optim.Adam([root.delta, roottr, rv], lr=2e-2)
            bm = self._make_bm(pl, segv) if use_bm else None
            for it in range(cfg.iters):
                if bm is not None and fit_shape and it in (cfg.iters // 4, cfg.iters // 2):
                    bm = self._make_bm(pl, segv)
                if it == cfg.iters // 2:
                    for g in opt.param_groups:
                        g["lr"] *= 0.25
                    if self.seg_labels and fit_shape:
                        self._gate_segments(
                            pl,
                            root,
                            roottr,
                            rv,
                            segv,
                            Xt,
                            Mk,
                            ank_h,
                            len_t,
                            len_ok,
                            len_w,
                            grp_active,
                            seg_active,
                            targets,
                            pid,
                        )
                opt.zero_grad()
                o = self._skel(pl, root, roottr, rv, segv)
                r = torch.linalg.norm(o.points - Xt, dim=-1)
                data = (Mk * hub(r)).sum() / Mk.sum()
                smooth = (
                    self._acc(rv, ws_t, cuts)
                    + geodesic_acceleration(root.matrices(), ws_t)
                    + 10 * self._acc(roottr, ws_t, cuts)
                )
                loss = data + cfg.w_smooth * smooth + 1e-3 * rv.pow(2).mean() + 1e-3 * pl.pow(2).mean()
                if cfg.w_build_prior > 0 and build_idx:
                    loss = loss + cfg.w_build_prior * pl[build_idx].pow(2).sum()
                if self.seg_labels:
                    loss = loss + cfg.w_segment * torch.tanh(segv).pow(2).sum() + cfg.w_length * len_loss(o)
                if bm is not None:
                    loss = loss + bm(rv, o.heads, o.surf[:, MHR70_SOLE])
                if not torch.isfinite(loss):
                    opt.zero_grad()
                    continue
                loss.backward()
                self._clean_grads(opt)
                opt.step()
            for h in hooks:
                h.remove()
            segv.requires_grad_(False)
            pl.requires_grad_(False)
        self._apply_fixed_size(pl, segv, fixed, pid)

        diag: dict = {}
        sam_used = False
        if inp.sam_kp is not None:
            self._stage2(inp, pid, pl, segv, root, roottr, rv, X, Xt, SVF, ws_t, cuts, fixed, diag)
            sam_used = True
        return self._finish(pid, pl, segv, root, roottr, rv, X, Xt, have, SVF, BRG, OBS, diag, sam_used)

    # ------------------------------------------------------------------------------------------------ stage-1 pieces
    @staticmethod
    def _clean_grads(opt: torch.optim.Optimizer) -> None:
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0, 0.0, 0.0)

    def _gate_segments(
        self,
        pl,
        root,
        roottr,
        rv,
        segv,
        Xt,
        Mk,
        ank_h,
        len_t,
        len_ok,
        len_w,
        grp_active,
        seg_active,
        targets: LengthTargets,
        pid: str,
    ) -> None:
        """Free only the segments the global body measurably misses; move ankle targets down the shin by the keypoint bias."""
        cfg = self.cfg
        with torch.no_grad():
            o = self._skel(pl, root, roottr[:1], rv[:1], segv)
            if cfg.ankle_bias and np.isfinite(ank_h):
                H_, _, RV_ = self.body.rest(torch.sigmoid(pl), self._seg(segv))
                idx = self.body.idx
                anny_ank = float(0.5 * (H_[idx["foot.L"], 2] + H_[idx["foot.R"], 2]) - RV_[:, 2].min())
                bias = min(max(0.0, ank_h - anny_ank), cfg.ankle_bias_max)
                if bias > 0:
                    for ka, kk in ((J["ankle_l"], J["knee_l"]), (J["ankle_r"], J["knee_r"])):
                        d_ = Xt[:, ka] - Xt[:, kk]
                        u_ = d_ / d_.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        okk = (Mk[:, ka] > 0) & (Mk[:, kk] > 0)
                        Xt[okk, ka] = Xt[okk, ka] + bias * u_[okk]
                    gi = targets.groups.index("shin")
                    len_t[gi] = len_t[gi] + bias
                log.info(
                    "%s: ankle keypoint %.1f cm vs model ankle %.1f cm above the sole -> shin target +%.1f cm",
                    pid,
                    ank_h * 100,
                    anny_ank * 100,
                    bias * 100,
                )
            mis = (self._rest_lengths(o.rest_heads, o.rest_tails, targets) / len_t - 1).abs()
            grp_active[:] = len_ok & (mis > cfg.segment_gate) & (len_w > 0.5)
            for gi, g in enumerate(targets.groups):
                if bool(grp_active[gi]):
                    for lab in GROUP_SEGMENT_LABELS[g]:
                        if lab in self.seg_labels:
                            seg_active[self.seg_labels.index(lab)] = 1.0
        log.info(
            "%s: segment mismatch (%%) %s -> corrected %s",
            pid,
            np.round((mis * 100).cpu().numpy(), 0).tolist(),
            [g for g, a in zip(targets.groups, grp_active.tolist()) if a],
        )

    def _apply_fixed_size(self, pl: torch.Tensor, segv: torch.Tensor, fixed: FixedPhenotype, pid: str) -> None:
        with torch.no_grad():
            if fixed.height_m is not None:
                ph0 = torch.sigmoid(pl).detach().clone()
                hv = height_slider_for_stature(lambda p: self._stature(p, segv), ph0, self.labels, fixed.height_m)
                pl[self.labels.index("height")] = logit(hv)
                fixed.values["height"] = hv
                log.info(
                    "%s: height %.2f m -> height slider %.2f (stature %.2f m)",
                    pid,
                    fixed.height_m,
                    hv,
                    self._stature(torch.sigmoid(pl), segv),
                )
            if fixed.weight_kg is not None:
                ph0 = torch.sigmoid(pl).detach().clone()
                wv, mv, ms = weight_muscle_for_mass(
                    lambda p: self._mesh_mass(p, segv), ph0, self.labels, fixed.weight_kg
                )
                pl[self.labels.index("weight")] = logit(wv)
                pl[self.labels.index("muscle")] = logit(mv)
                fixed.values["weight"], fixed.values["muscle"] = wv, mv
                log.info("%s: mass %.1f kg -> weight %.2f, muscle %.2f (%.1f kg)", pid, fixed.weight_kg, wv, mv, ms)

    # ------------------------------------------------------------------------------------------------ stage 2
    def _stage2(
        self,
        inp: FitInputs,
        pid: str,
        pl,
        segv,
        root: RootParam,
        roottr,
        rv,
        X,
        Xt,
        SVF,
        ws_t,
        cuts,
        fixed: FixedPhenotype,
        diag: dict,
    ) -> None:
        cfg, dev, fps, N = self.cfg, self.dev, self.fps, len(X)
        kp = smooth_series(np.asarray(inp.sam_kp, np.float32)[:N], fps, cfg.sam_euro_min_hz, cfg.sam_euro_beta, cuts)
        kb, ks = kp[:, SAM_BODY_KP], kp[:, SAM_SURF_KP]
        ours = X[:, SAM_BODY_JOINTS]
        with np.errstate(all="ignore"):
            dis = np.linalg.norm(kb - ours, axis=-1)
        dis[~np.isfinite(dis)] = 0
        jb, js = jumpy(kb, cfg.sam_jump_m, cuts=cuts), jumpy(ks, cfg.sam_jump_m, cuts=cuts)
        wb = (~jb & (dis <= cfg.sam_gate_m) & np.isfinite(kb[..., 0])).astype(np.float32)
        ws = (~js & np.isfinite(ks[..., 0])).astype(np.float32)
        with np.errstate(all="ignore"):
            dh = np.linalg.norm(kp[:, list(MHR70_HIPS)].mean(1) - X[:, 0], axis=-1)
        badf = ~np.isfinite(kp[:, 0, 0]) | (np.isfinite(dh) & (dh > cfg.sam_body_gate_m))
        wb[badf] = 0
        ws[badf] = 0
        log.info(
            "%s: SAM frames dropped whole %d (%.1f%%); body keypoints used %.0f%%, face/feet %.0f%%",
            pid,
            int(badf.sum()),
            100 * badf.mean(),
            100 * wb.mean(),
            100 * ws.mean(),
        )
        nf = self.nfade
        Wb = torch.tensor(soft_fade(wb, nf), device=dev)
        Ws = torch.tensor(soft_fade(ws, nf), device=dev)
        Sb = torch.tensor(np.nan_to_num(kb), device=dev)
        Ss = torch.tensor(np.nan_to_num(ks), device=dev)
        svw = np.where(SVF, cfg.sv_weight, 1.0)[:, None].astype(np.float32)
        Mk2 = torch.tensor(soft_fade(np.isfinite(X[..., 0]).astype(np.float32), nf) * svw, device=dev) * self.W
        Sh, Wh = self._hand_targets(kp, X, badf, cuts)
        Sd, Wd, _corr, keep = self._dense_targets(inp, pid, pl, segv, root, roottr, rv, badf, cuts)
        build = Sd is not None and cfg.dense_build and "weight" not in fixed.values
        if build:
            gm = torch.zeros(len(self.labels), device=dev)
            for k in ("weight", "muscle"):
                gm[self.labels.index(k)] = 1.0
            pl.requires_grad_(True)
            hook = pl.register_hook(lambda g: g * gm)
            opt = torch.optim.Adam([{"params": [root.delta, roottr, rv], "lr": 1e-2}, {"params": [pl], "lr": 5e-3}])
        else:
            hook = None
            pl.requires_grad_(False)
            opt = torch.optim.Adam([root.delta, roottr, rv], lr=1e-2)
        pl0 = pl.detach().clone()
        bm = self._make_bm(pl, segv)
        if bm is not None:
            with torch.no_grad():
                o = self._skel(pl, root, roottr, rv, segv)
            diag["biomech_stage1"] = bm.report(rv, o.heads)
        fingers = torch.tensor(
            [i for i, b in enumerate(self.body.posed_bones) if b.startswith("finger")], device=dev, dtype=torch.long
        )
        for it in range(cfg.iters):
            if it == cfg.iters // 2:
                for g in opt.param_groups:
                    g["lr"] *= 0.25
            opt.zero_grad()
            o = self._skel(pl, root, roottr, rv, segv)
            l_ours = (Mk2 * hub(torch.linalg.norm(o.points - Xt, dim=-1))).sum() / Mk2.sum().clamp_min(1)
            l_samb = (Wb * hub(torch.linalg.norm(o.points[:, self.sb_idx] - Sb, dim=-1))).sum() / Wb.sum().clamp_min(1)
            l_sams = (Ws * hub(torch.linalg.norm(o.surf - Ss, dim=-1))).sum() / Ws.sum().clamp_min(1)
            loss = cfg.w_ours * l_ours + cfg.w_sam * l_samb + cfg.w_surface * l_sams
            if Sh is not None and o.hand.shape[1] > 0:
                loss = loss + cfg.w_hand * (
                    Wh * hub(torch.linalg.norm((o.hand - o.hand_wrist) - Sh, dim=-1))
                ).sum() / Wh.sum().clamp_min(1)
            loss = loss + cfg.w_smooth * (
                self._acc(rv, ws_t, cuts)
                + geodesic_acceleration(root.matrices(), ws_t)
                + 10 * self._acc(roottr, ws_t, cuts)
            )
            loss = loss + 1e-3 * rv.pow(2).mean()
            if len(fingers):
                loss = loss + 1e-2 * rv[:, fingers].pow(2).mean()
            if bm is not None:
                loss = loss + bm(rv, o.heads, o.surf[:, MHR70_SOLE])
            if Sd is not None:
                loss = (
                    loss
                    + cfg.w_dense * (Wd * hub(torch.linalg.norm(o.dense - Sd, dim=-1))).sum() / Wd.sum().clamp_min(1)
                    + 1e-2 * (pl - pl0).pow(2).sum()
                )
            if build and cfg.w_build_prior > 0:
                bi = [self.labels.index(k) for k in ("weight", "muscle")]
                loss = loss + cfg.w_build_prior * pl[bi].pow(2).sum()
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            loss.backward()
            self._clean_grads(opt)
            opt.step()
        if hook is not None:
            hook.remove()
        pl.requires_grad_(False)
        with torch.no_grad():
            o = self._skel(pl, root, roottr, rv, segv)
            diag["sam_joint_err_m"] = (
                float(torch.linalg.norm(o.points[:, self.sb_idx] - Sb, dim=-1)[Wb > 0.5].median())
                if bool((Wb > 0.5).any())
                else None
            )
            diag["sam_surface_err_m"] = (
                float(torch.linalg.norm(o.surf - Ss, dim=-1)[Ws > 0.5].median()) if bool((Ws > 0.5).any()) else None
            )
            diag["sam_frames_dropped"] = int(badf.sum())
            if Sd is not None:
                dfr = torch.linalg.norm(o.dense - Sd, dim=-1)
                diag["dense_kept"] = int(keep.sum())
                diag["dense_residual_m"] = float(dfr[Wd > 0.5].median()) if bool((Wd > 0.5).any()) else None
        log.info("%s: stage 2 done; phenotype %s", pid, np.round(torch.sigmoid(pl).cpu().numpy(), 2).tolist())

    def _hand_targets(self, kp: np.ndarray, X: np.ndarray, badf: np.ndarray, cuts: list[int]):
        hp = self.body.hand_points
        if not hp or not self.cfg.fit_fingers:
            return None, None
        kh = kp[:, [k for k, _, _ in hp]]
        wh = (~jumpy(kh, self.cfg.sam_jump_m, cuts=cuts) & np.isfinite(kh[..., 0])).astype(np.float32)
        wh[badf] = 0
        for sd, wk, slot in (("R", MHR70_WRIST_R, J["wrist_r"]), ("L", MHR70_WRIST_L, J["wrist_l"])):
            with np.errstate(all="ignore"):
                dw = np.linalg.norm(kp[:, wk] - X[:, slot], axis=-1)
            bad = np.isfinite(dw) & (dw > self.cfg.sam_gate_m)
            cols = [i for i, (_, b, _) in enumerate(hp) if b.endswith("." + sd)]
            wh[np.ix_(bad, cols)] = 0
        kw = kp[:, [MHR70_WRIST_R if b.endswith(".R") else MHR70_WRIST_L for _, b, _ in hp]]
        return torch.tensor(np.nan_to_num(kh - kw), device=self.dev), torch.tensor(
            soft_fade(wh, self.nfade), device=self.dev
        )

    def _dense_targets(self, inp: FitInputs, pid: str, pl, segv, root, roottr, rv, badf: np.ndarray, cuts: list[int]):
        cfg, dev, N = self.cfg, self.dev, len(badf)
        if not cfg.dense or inp.sam_verts is None:
            return None, None, None, None
        SV = inp.sam_verts
        with torch.no_grad():
            AD = self._skel(pl, root, roottr, rv, segv).dense.cpu().numpy()
        okf = [
            i
            for i in range(0, N, 3)
            if not badf[i] and np.isfinite(np.asarray(SV[i][0], np.float32)).all() and np.isfinite(AD[i]).all()
        ]
        if len(okf) < 3:
            return None, None, None, None
        nn, dd = [], []
        for i in okf:
            dm = torch.cdist(torch.tensor(AD[i], device=dev), torch.tensor(np.asarray(SV[i], np.float32), device=dev))
            d_, j_ = dm.min(1)
            nn.append(j_.cpu().numpy())
            dd.append(d_.cpu().numpy())
        nn, dd = np.array(nn), np.array(dd)
        best = np.argsort(np.median(dd, 1))[: max(min(10, len(okf)), int(0.4 * len(okf)))]
        corr = np.array([np.bincount(nn[best, k]).argmax() for k in range(nn.shape[1])])
        cd = np.median(dd[best], 0)
        keep = cd < cfg.dense_max_d
        Sd_np = np.stack([np.asarray(SV[i], np.float32)[corr] for i in range(N)]).astype(np.float32)
        Sd_np = smooth_series(Sd_np, self.fps, cfg.sam_euro_min_hz, cfg.sam_euro_beta, cuts)
        wdf = soft_fade((~badf & np.isfinite(Sd_np[:, :, 0]).all(1)).astype(np.float32), self.nfade)
        Wd = torch.tensor(wdf, device=dev)[:, None] * torch.tensor(keep, dtype=torch.float32, device=dev)[None]
        log.info(
            "%s: dense %d/%d vertices (match median %.1f cm, from %d frames)",
            pid,
            int(keep.sum()),
            len(keep),
            np.median(cd) * 100,
            len(best),
        )
        return torch.tensor(np.nan_to_num(Sd_np), device=dev), Wd, corr, keep

    # ------------------------------------------------------------------------------------------------ output
    def _finish(
        self, pid, pl, segv, root: RootParam, roottr, rv, X, Xt, have, SVF, BRG, OBS, diag, sam_used
    ) -> FitResult:
        with torch.no_grad():
            o = self._skel(pl, root, roottr, rv, segv)
            err = torch.linalg.norm(o.points - Xt, dim=-1).cpu().numpy()
            err[~np.isfinite(X[..., 0])] = np.nan
            ph = torch.sigmoid(pl).cpu().numpy()
            stature = self._stature(torch.sigmoid(pl), segv)
            report = {}
            bm = self._make_bm(pl, segv)
            if bm is not None:
                report = bm.report(rv, o.heads)
        with np.errstate(all="ignore"):
            diag["joint_err_median_m"] = float(np.nanmedian(err)) if np.isfinite(err).any() else None
            diag["joint_err_p90_m"] = float(np.nanpercentile(err, 90)) if np.isfinite(err).any() else None
        diag["sam_used"] = sam_used
        log.info(
            "%s: %d/%d frames | joint error median %.1f cm | %s | stature %.2f m",
            pid,
            int(have.sum()),
            len(X),
            100 * (diag["joint_err_median_m"] or 0),
            " ".join(f"{k} {v:.2f}" for k, v in zip(self.labels, ph)),
            stature,
        )
        return FitResult(
            ph, self.labels, root.rotvecs().detach().cpu().numpy(), roottr.detach().cpu().numpy(), rv.detach().cpu().numpy(), list(self.body.posed_bones),
            o.points.cpu().numpy(), X, have, SVF, BRG, OBS | (have & ~SVF), self.fps, stature, self.seg_labels, torch.tanh(segv).cpu().numpy(), report, diag,
        )  # fmt: skip

    def vertices(self, res: FitResult, chunk: int = 150) -> np.ndarray:
        """Skinned mesh per frame, float16 (N, V, 3), computed in chunks."""
        N = len(res.rootrot)
        pheno = torch.tensor(res.phenotype, device=self.dev)
        segv = torch.tensor(np.arctanh(np.clip(res.seg_values, -0.999, 0.999)), dtype=torch.float32, device=self.dev)
        out = None
        with torch.no_grad():
            for c0 in range(0, N, chunk):
                c1 = min(N, c0 + chunk)
                R = roma.rotvec_to_rotmat(torch.tensor(res.rootrot[c0:c1], device=self.dev))
                V = self.body.skin(
                    pheno,
                    R,
                    torch.tensor(res.roottr[c0:c1], device=self.dev),
                    torch.tensor(res.rotvecs[c0:c1], device=self.dev),
                    self._seg(segv),
                )
                if out is None:
                    out = np.zeros((N, V.shape[1], 3), np.float16)
                out[c0:c1] = V.cpu().numpy().astype(np.float16)
        return out
