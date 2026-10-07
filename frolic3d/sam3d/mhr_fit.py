"""Multi-view MHR fit: fuse the per-view SAM 3D Body estimates of one person, then fit one body to every camera.

Fusion (initialisation): per observation weight = box size x not truncated x not overlapped x agreement with the 2D
pose; shape and scale = weighted median; per-frame body pose = weighted circular mean over views; world root rotation =
weighted chordal mean of per-view Kabsch alignments (camera -> world); translation = mid-hip on the triangulated track.
Fit: per-frame root rotation (6D), translation, body pose; per-person shape, scale and a global size factor. Loss =
robust (Geman-McClure) reprojection to the Halpe-26 keypoints in every camera + prior to the fused pose + weak anchor to
the triangulated track + joint acceleration + feet above the floor (z = 0 in the levelled frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import roma
import torch

from frolic3d.config import Sam3dConfig
from frolic3d.io.calib import CameraSet
from frolic3d.log import get_logger
from frolic3d.sam3d.adapter import MhrModel
from frolic3d.skeleton import HALPE26_TO_MHR70, MHR70_BODY, MHR70_FEET, MHR70_HIPS, J

log = get_logger(__name__)

HI = np.array(list(HALPE26_TO_MHR70))
MI = np.array(list(HALPE26_TO_MHR70.values()))
BONES = [
    (J["hip_l"], J["knee_l"]),
    (J["hip_r"], J["knee_r"]),
    (J["knee_l"], J["ankle_l"]),
    (J["knee_r"], J["ankle_r"]),
    (J["shoulder_l"], J["elbow_l"]),
    (J["shoulder_r"], J["elbow_r"]),
    (J["elbow_l"], J["wrist_l"]),
    (J["elbow_r"], J["wrist_r"]),
]
S2M = dict(sorted(MHR70_BODY.items()))


@dataclass
class PersonObservations:
    """Inputs of one person: prompts arrays (V = cameras in `cams` order) and per-view model outputs {cam_index: {frame: dict}}."""

    box: np.ndarray  # (V, N, 4)
    kps: np.ndarray  # (V, N, 26, 2)
    ks: np.ndarray  # (V, N, 26)
    P3: np.ndarray  # (N, 17, 3)
    views: dict[int, dict[int, dict[str, np.ndarray]]]
    sizes: np.ndarray  # (V, 2)


@dataclass
class MhrFit:
    rot: np.ndarray
    trans: np.ndarray
    body: np.ndarray
    hand: np.ndarray
    shape: np.ndarray
    scale: np.ndarray
    gscale: float
    gscale_init: float
    stature: float
    kp: np.ndarray
    kp_init: np.ndarray
    conf: np.ndarray
    have: np.ndarray
    n_obs: int
    n_rot_rejected: int
    holdout: str

    def save(self, path: Path | str, verts: np.ndarray | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items()})
        if verts is not None:
            np.save(path.with_name(path.stem + "_verts.npy"), verts.astype(np.float16))


def box_overlap(b1: np.ndarray, b2: np.ndarray) -> float:
    ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    i = ix * iy
    return float(i / min((b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (b2[3] - b2[1]))) if i > 0 else 0.0


def wmedian(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    o = np.argsort(x, axis=0)
    cw = np.cumsum(w[o], axis=0)
    idx = (cw >= cw[-1] / 2).argmax(0)
    return x[o[idx, np.arange(x.shape[1])], np.arange(x.shape[1])]


def chordal_mean(Rs: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = np.einsum("n,nij->ij", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


class MhrFitter:
    def __init__(self, model: MhrModel, cams: CameraSet, cfg: Sam3dConfig, fps: float = 15.0) -> None:
        self.model = model
        self.cams = cams
        self.cfg = cfg
        self.fps = fps
        self.dev = model.device
        self.Kt = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32, device=self.dev)
        self.Rt = torch.tensor(np.stack([c.R for c in cams]), dtype=torch.float32, device=self.dev)
        self.Tt = torch.tensor(np.stack([c.t for c in cams]), dtype=torch.float32, device=self.dev)

    def body_kp(self, rot6, trans, body, hand, scale, shape, gs=None):
        """World keypoints (B, 70, 3) and vertices: X = R_w (g X_model) + T, g = exp(gs)."""
        B = body.shape[0]
        verts, kp = self.model.forward(body, hand, scale.expand(B, -1), shape.expand(B, -1))
        Rw = roma.special_gramschmidt(rot6.view(-1, 3, 2)) if rot6.shape[-1] == 6 else rot6
        g = torch.exp(gs) if gs is not None else 1.0
        return (g * kp) @ Rw.transpose(1, 2) + trans[:, None], (g * verts) @ Rw.transpose(1, 2) + trans[:, None]

    def project(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Xc = torch.einsum("vij,bkj->vbki", self.Rt, X) + self.Tt[:, None, None]
        uv = torch.einsum("vij,vbkj->vbki", self.Kt, Xc)
        return uv[..., :2] / uv[..., 2:3].clamp(min=1e-3), Xc[..., 2]

    # ------------------------------------------------------------------------------------------------ observations
    def _collect(self, obs_in: PersonObservations, holdout: str) -> list[tuple[int, int, float, dict]]:
        obs = []
        for c, res in obs_in.views.items():
            if self.cams.names[c] == holdout:
                continue
            W_, H_ = obs_in.sizes[c]
            for i, o in res.items():
                b = obs_in.box[c, i]
                if not np.isfinite(b[0]):
                    continue
                h = b[3] - b[1]
                trunc = 0.5 if (b[0] < 5 or b[1] < 5 or b[2] > W_ - 5 or b[3] > H_ - 5) else 1.0
                ov = max([box_overlap(b, ob) for ob in obs_in.__dict__.get("others", {}).get((c, i), [])] + [0.0])
                kp, ks = obs_in.kps[c, i], obs_in.ks[c, i]
                m = ks[HI] > 0.5
                agree = (
                    np.median(np.linalg.norm(o["pred_keypoints_2d"][MI[m]] - kp[HI[m]], axis=1)) / h
                    if m.sum() >= 5
                    else 0.1
                )
                w = min(h / 400, 1.0) * trunc * (1 - 0.7 * ov) * np.exp(-((agree / 0.06) ** 2))
                obs.append((int(i), int(c), float(w), o))
        return obs

    def _world_rotations(self, obs: list) -> np.ndarray:
        Rw = np.zeros((len(obs), 3, 3))
        with torch.no_grad():
            for s in range(0, len(obs), 512):
                ch = obs[s : s + 512]
                T = lambda k, ch=ch: torch.tensor(np.stack([x[3][k] for x in ch]), dtype=torch.float32, device=self.dev)  # noqa: E731
                _, kc = self.model.forward(
                    T("body_pose_params"), T("hand_pose_params"), T("scale_params"), T("shape_params")
                )
                A = kc[:, :70]
                cidx = torch.tensor([x[1] for x in ch], device=self.dev)
                Bw = T("pred_keypoints_3d") @ self.Rt[cidx]
                A = A - A.mean(1, keepdim=True)
                Bw = Bw - Bw.mean(1, keepdim=True)
                U, _, Vh = torch.linalg.svd(A.transpose(1, 2) @ Bw)
                d = torch.sign(torch.linalg.det(U @ Vh))
                Dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1))
                Rw[s : s + len(ch)] = (U @ Dm @ Vh).transpose(1, 2).cpu().numpy()
        return Rw

    # ------------------------------------------------------------------------------------------------ fit
    def fit(
        self, obs_in: PersonObservations, pid: str = "", cuts: list[int] | None = None, holdout: str = ""
    ) -> MhrFit | None:
        cfg, dev, N = self.cfg, self.dev, obs_in.P3.shape[0]
        obs = self._collect(obs_in, holdout)
        if len(obs) < 20:
            log.warning("%s: only %d SAM 3D Body observations, skipped", pid, len(obs))
            return None
        W = np.array([x[2] for x in obs])
        shape0 = wmedian(np.stack([x[3]["shape_params"] for x in obs]), W)
        scale0 = wmedian(np.stack([x[3]["scale_params"] for x in obs]), W)
        Rw_obs = self._world_rotations(obs)
        nb, nh = obs[0][3]["body_pose_params"].shape[0], obs[0][3]["hand_pose_params"].shape[0]
        body0 = np.zeros((N, nb), np.float32)
        hand0 = np.zeros((N, nh), np.float32)
        Rw0 = np.tile(np.eye(3), (N, 1, 1))
        conf = np.zeros(N)
        byf: dict[int, list[int]] = {}
        for k, (i, _c, _w, _o) in enumerate(obs):
            byf.setdefault(i, []).append(k)
        have = np.zeros(N, bool)
        nrej = 0
        for i, ks_ in byf.items():
            w = np.array([obs[k][2] for k in ks_]) + 1e-6
            Rs = Rw_obs[ks_]
            Rm = chordal_mean(Rs, w)
            ang = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rs, Rm) - 1) / 2, -1, 1)))
            keep = ang < 35 if (ang < 35).any() else np.ones(len(ks_), bool)
            nrej += int((~keep).sum())
            w = w * keep
            Rw0[i] = chordal_mean(Rs, w)
            bp = np.stack([obs[k][3]["body_pose_params"] for k in ks_])
            body0[i] = np.arctan2((w[:, None] * np.sin(bp)).sum(0), (w[:, None] * np.cos(bp)).sum(0))
            hand0[i] = (w[:, None] * np.stack([obs[k][3]["hand_pose_params"] for k in ks_])).sum(0) / w.sum()
            conf[i] = min(w.sum(), 1.5)
            have[i] = True
        idx = np.flatnonzero(have)
        near = idx[np.abs(np.arange(N)[:, None] - idx[None]).argmin(1)]
        for arr in (body0, hand0, Rw0):
            arr[~have] = arr[near[~have]]
        P3 = obs_in.P3
        hipw = np.where(
            np.isfinite(P3[:, [J["hip_l"], J["hip_r"]], 0]).all(1)[:, None],
            P3[:, [J["hip_l"], J["hip_r"]]].mean(1),
            P3[:, J["pelvis"]],
        )
        tgt_np = np.where(np.isfinite(hipw), hipw, np.nan)
        for d in range(3):  # frames without a track take the nearest tracked position
            g = np.isfinite(tgt_np[:, d])
            tgt_np[:, d] = np.interp(np.arange(N), np.flatnonzero(g), tgt_np[g, d]) if g.any() else 0.0
        tgt = torch.tensor(tgt_np, dtype=torch.float32, device=dev)
        shape = torch.tensor(shape0, device=dev)[None].requires_grad_(True)
        scale = torch.tensor(scale0, device=dev)[None].requires_grad_(True)
        nbody = min(nb, 130)
        body = torch.tensor(body0[:, :nbody], device=dev).requires_grad_(True)
        bodyP = torch.tensor(body0[:, :nbody], device=dev)
        body_tail = torch.tensor(body0[:, nbody:], device=dev)
        hand = torch.tensor(hand0, device=dev)
        rot6 = torch.tensor(Rw0[:, :, :2].reshape(N, 6), dtype=torch.float32, device=dev).requires_grad_(True)
        rotP = torch.tensor(Rw0, dtype=torch.float32, device=dev)
        g0 = self._size_init(rot6, body, body_tail, hand, scale, shape, P3, pid)
        gs = torch.tensor(np.log(g0), dtype=torch.float32, device=dev).requires_grad_(True)
        with torch.no_grad():
            kp, _ = self.body_kp(
                rot6, torch.zeros(N, 3, device=dev), torch.cat([body, body_tail], 1), hand, scale, shape, gs
            )
            trans = (tgt - kp[:, list(MHR70_HIPS)].mean(1)).clone().requires_grad_(True)
        confT = torch.tensor(conf, dtype=torch.float32, device=dev)
        V = len(self.cams)
        kps = obs_in.kps[:, :, HI].copy()
        ks = obs_in.ks[:, :, HI].copy()
        ks[ks < 0.3] = 0
        kps[ks == 0] = 0
        kps = np.nan_to_num(kps)
        if holdout and holdout in self.cams.names:
            ks[self.cams.names.index(holdout)] = 0
        boxh = np.nan_to_num(obs_in.box[..., 3] - obs_in.box[..., 1], nan=1.0)
        nw = self._noise_weights(obs_in.kps, V)
        wt = torch.tensor(ks * nw[:, None, :], dtype=torch.float32, device=dev)
        tg = torch.tensor(kps, dtype=torch.float32, device=dev)
        sig = torch.tensor(cfg.sigma * boxh, dtype=torch.float32, device=dev)[..., None]
        P3m = np.stack([P3[:, s] for s in S2M], 1)
        a3 = torch.tensor(np.nan_to_num(P3m), dtype=torch.float32, device=dev)
        a3w = torch.tensor(np.isfinite(P3m[..., 0]).astype(np.float32), device=dev)
        a3i = list(S2M.values())
        nv2 = (ks.sum(-1) > 0).sum(0)
        few = torch.tensor(np.where(nv2 >= 2, 0.2, 1.0), dtype=torch.float32, device=dev)
        acc_mask = torch.ones(max(N - 2, 0), device=dev)
        for c in cuts or []:
            if 0 < c < N:
                acc_mask[max(0, c - 2) : c] = 0
        feet = [k for k in MHR70_FEET]

        def losses():
            kp, _ = self.body_kp(rot6, trans, torch.cat([body, body_tail], 1), hand, scale, shape, gs)
            uv, dep = self.project(kp[:, MI])
            r2 = ((uv - tg) ** 2).sum(-1) / sig**2
            L2d = (wt * r2 / (r2 + 1) * (dep > 0.1)).sum() / wt.sum().clamp(min=1)
            Lpr = (confT[:, None] * (1 - torch.cos(body - bodyP))).mean() * 2 + 0.1 * (
                confT[:, None, None] * (roma.special_gramschmidt(rot6.view(-1, 3, 2)) - rotP) ** 2
            ).mean()
            d3 = (kp[:, a3i] - a3).norm(dim=-1)
            Lan = (
                few[:, None]
                * a3w
                * torch.nn.functional.huber_loss(d3, torch.zeros_like(d3), delta=0.05, reduction="none")
            ).mean()
            Lacc = (
                (acc_mask[:, None, None] * (kp[2:] - 2 * kp[1:-1] + kp[:-2]) ** 2).sum(-1).mean()
                if N > 2
                else kp.sum() * 0
            )
            Lfl = torch.relu(-kp[:, feet, 2]).pow(2).mean()
            Lsh = (
                1e-3 * (shape**2).mean()
                + 1e-2 * ((scale - torch.tensor(scale0, device=dev)) ** 2).mean()
                + 10.0 * (gs - float(np.log(g0))) ** 2
            )
            return L2d, Lpr, Lan, Lacc, Lfl, Lsh, kp

        for stage, params, iters, lr in [
            (1, [rot6, trans, gs], cfg.iters // 4, 0.01),
            (2, [rot6, trans, body, shape, scale, gs], cfg.iters, 0.005),
        ]:
            opt = torch.optim.Adam(params, lr=lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(iters, 1))
            for it in range(iters):
                opt.zero_grad()
                L2d, Lpr, Lan, Lacc, Lfl, Lsh, _ = losses()
                L = L2d + cfg.w_prior * Lpr + cfg.w_anchor * Lan + cfg.w_acc * Lacc + cfg.w_floor * Lfl + Lsh
                if not torch.isfinite(L):
                    continue
                L.backward()
                opt.step()
                sch.step()
                if it % 150 == 0 or it == iters - 1:
                    log.debug(
                        "%s s%d it%d: 2d %.4f prior %.4f anchor %.4f acc %.2e floor %.2e",
                        pid,
                        stage,
                        it,
                        L2d.item(),
                        Lpr.item(),
                        Lan.item(),
                        Lacc.item(),
                        Lfl.item(),
                    )
        with torch.no_grad():
            vr, _ = self.model.forward(torch.zeros(1, nb, device=dev), torch.zeros(1, nh, device=dev), scale, shape)
            stature = float((vr[0, :, 1].max() - vr[0, :, 1].min()) * torch.exp(gs))
            _, _, _, _, _, _, kp = losses()
            kp0, _ = self.body_kp(
                torch.tensor(Rw0[:, :, :2].reshape(N, 6), dtype=torch.float32, device=dev),
                trans,
                torch.tensor(body0, device=dev),
                hand,
                torch.tensor(scale0, device=dev)[None],
                torch.tensor(shape0, device=dev)[None],
                torch.tensor(np.log(g0), device=dev),
            )
        log.info(
            "%s: %d obs, %d/%d frames with SAM 3D Body, %d view-rotations rejected, size factor %.2f, stature %.2f m",
            pid,
            len(obs),
            int(have.sum()),
            N,
            nrej,
            float(torch.exp(gs)),
            stature,
        )
        return MhrFit(
            roma.special_gramschmidt(rot6.detach().view(-1, 3, 2)).cpu().numpy(), trans.detach().cpu().numpy(), torch.cat([body, body_tail], 1).detach().cpu().numpy(), hand.cpu().numpy(),
            shape.detach().cpu().numpy()[0], scale.detach().cpu().numpy()[0], float(torch.exp(gs).item()), g0, stature, kp.cpu().numpy(), kp0.cpu().numpy(), conf, have, len(obs), nrej, holdout,
        )  # fmt: skip

    def _size_init(self, rot6, body, body_tail, hand, scale, shape, P3: np.ndarray, pid: str) -> float:
        """Global size factor from triangulated vs model limb lengths (legs only when symmetric and plausible)."""
        N = body.shape[0]
        with torch.no_grad():
            kp, _ = self.body_kp(
                rot6, torch.zeros(N, 3, device=self.dev), torch.cat([body, body_tail], 1), hand, scale, shape
            )
        kpn = kp.cpu().numpy()

        def L(X, i, j):
            return np.linalg.norm(X[:, i] - X[:, j], axis=-1)

        with np.errstate(all="ignore"):
            med = {b: np.nanmedian(L(P3, *b)) for b in BONES}
        arms = med[BONES[4]] + med[BONES[6]] + med[BONES[5]] + med[BONES[7]]
        legs = med[BONES[0]] + med[BONES[2]] + med[BONES[1]] + med[BONES[3]]
        sym = (
            all(
                abs(med[x] - med[y]) / (med[x] + med[y]) * 2 < 0.10
                for x, y in [(BONES[0], BONES[1]), (BONES[2], BONES[3])]
            )
            if np.isfinite(legs)
            else False
        )
        legs_ok = sym and np.isfinite(arms) and arms > 0 and 1.1 < legs / arms < 1.6
        use = BONES if legs_ok else BONES[4:]
        rat = []
        for s1, s2 in use:
            lt = L(P3, s1, s2)
            lm = L(kpn, S2M[s1], S2M[s2])
            ok = np.isfinite(lt) & (lm > 0.02)
            if ok.sum() > 10:
                rat.append(np.median(lt[ok] / lm[ok]))
        g0 = float(np.clip(np.median(rat), 0.4, 2.0)) if rat else 1.0
        log.info("%s: size init %.2f from %d bones (legs %s)", pid, g0, len(rat), "used" if legs_ok else "not used")
        return g0

    @staticmethod
    def _noise_weights(kps: np.ndarray, V: int) -> np.ndarray:
        """Per camera and joint: 2D jitter relative to the steadiest camera -> weight (best / this)^2 in [0.05, 1]."""
        with np.errstate(all="ignore"):
            tor = np.linalg.norm((kps[:, :, 11] + kps[:, :, 12]) / 2 - (kps[:, :, 5] + kps[:, :, 6]) / 2, axis=-1)
            acc = np.linalg.norm(kps[:, 2:, HI] - 2 * kps[:, 1:-1, HI] + kps[:, :-2, HI], axis=-1) / tor[:, 1:-1, None]
            jit = np.array(
                [
                    [
                        np.nanmedian(acc[c, :, j]) if np.isfinite(acc[c, :, j]).sum() >= 30 else np.nan
                        for j in range(len(HI))
                    ]
                    for c in range(V)
                ]
            )
            best = np.nanmin(jit, 0) if np.isfinite(jit).any() else np.ones(len(HI))
            nw = np.clip((best / jit) ** 2, 0.05, 1.0)
        nw[~np.isfinite(nw)] = 1.0
        return nw

    def vertices(self, fit: MhrFit, chunk: int = 150) -> np.ndarray:
        """Posed world vertices (N, V, 3) float16."""
        N = len(fit.rot)
        out = None
        with torch.no_grad():
            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                T = lambda k, s=s, e=e: torch.tensor(getattr(fit, k)[s:e], dtype=torch.float32, device=self.dev)  # noqa: E731
                _, v = self.body_kp(
                    T("rot"),
                    T("trans"),
                    T("body"),
                    T("hand"),
                    torch.tensor(fit.scale, device=self.dev)[None],
                    torch.tensor(fit.shape, device=self.dev)[None],
                    torch.tensor(np.log(fit.gscale), device=self.dev),
                )
                if out is None:
                    out = np.zeros((N, v.shape[1], 3), np.float16)
                out[s:e] = v.cpu().numpy().astype(np.float16)
        return out
