"""Per person / camera / track frame: the detection that built the 3D track, undistorted into the camera's pinhole K."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frolic.geometry import Camera
from frolic.io.calib import CameraSet
from frolic.io.detections import N_KP, Detections
from frolic.io.tracks import Tracks
from frolic.log import get_logger
from frolic.skeleton import HALPE26_HIP, N_JOINTS, J

log = get_logger(__name__)


@dataclass
class Prompts:
    ids: list[str]
    cameras: list[str]
    box: np.ndarray  # (P, V, N, 4) undistorted pixels, NaN when absent
    kps: np.ndarray  # (P, V, N, 26, 2)
    ks: np.ndarray  # (P, V, N, 26)
    P3: np.ndarray  # (P, N, 17, 3)
    frame0: int
    stride: int
    fps: float
    sizes: np.ndarray  # (V, 2) width, height

    @property
    def n_frames(self) -> int:
        return self.box.shape[2]

    def present(self, cam_index: int, frame: int) -> list[int]:
        return [p for p in range(len(self.ids)) if np.isfinite(self.box[p, cam_index, frame, 0])]

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ids=np.array(self.ids),
            cameras=np.array(self.cameras),
            box=self.box,
            kps=self.kps,
            ks=self.ks,
            P3=self.P3,
            frame0=self.frame0,
            stride=self.stride,
            fps=self.fps,
            sizes=self.sizes,
        )

    @classmethod
    def load(cls, path: Path | str) -> Prompts:
        z = np.load(path)
        return cls(
            list(z["ids"]),
            list(z["cameras"]),
            z["box"],
            z["kps"],
            z["ks"],
            z["P3"],
            int(z["frame0"]),
            int(z["stride"]),
            float(z["fps"]),
            z["sizes"],
        )


def undistorted_box(cam: Camera, b: np.ndarray) -> np.ndarray:
    pts = np.array(
        [
            [b[0], b[1]],
            [b[2], b[1]],
            [b[0], b[3]],
            [b[2], b[3]],
            [(b[0] + b[2]) / 2, b[1]],
            [(b[0] + b[2]) / 2, b[3]],
            [b[0], (b[1] + b[3]) / 2],
            [b[2], (b[1] + b[3]) / 2],
        ]
    )
    u = cam.undistort_points(pts)
    return np.array([u[:, 0].min(), u[:, 1].min(), u[:, 0].max(), u[:, 1].max()], np.float32)


def gather_prompts(tracks: Tracks, dets: dict[str, Detections], cams: CameraSet, gate_px: float = 80.0) -> Prompts:
    P, N, V = tracks.n_people, tracks.n_frames, len(cams)
    box = np.full((P, V, N, 4), np.nan, np.float32)
    kps = np.full((P, V, N, N_KP, 2), np.nan, np.float32)
    ks = np.zeros((P, V, N, N_KP), np.float32)
    gated = 0
    for c, cam in enumerate(cams):
        ci = tracks.cameras.index(cam.name)
        d = dets[cam.name]
        for p in range(P):
            rows = tracks.support[p, :, ci]
            for i in np.flatnonzero(rows >= 0):
                r = int(rows[i])
                if r >= len(d):
                    continue
                kp = cam.undistort_points(d.kps[r])
                pel = tracks.xyz[p, i, J["pelvis"]]
                if np.isfinite(pel).all():
                    pr, z = cam.project(pel)
                    hip = 0.5 * (kp[11] + kp[12]) if d.kps_score[r][HALPE26_HIP] < 0.3 else kp[HALPE26_HIP]
                    if z <= 0 or np.linalg.norm(pr - hip) > gate_px:
                        gated += 1
                        continue
                box[p, c, i] = undistorted_box(cam, d.box[r])
                kps[p, c, i] = kp
                ks[p, c, i] = d.kps_score[r]
    nv = np.isfinite(box[..., 0]).sum(1)
    for p, pid in enumerate(tracks.ids):
        log.info(
            "%s: boxes on %d frames, views/frame %.2f",
            pid,
            int((nv[p] > 0).sum()),
            float(nv[p][nv[p] > 0].mean()) if (nv[p] > 0).any() else 0.0,
        )
    if gated:
        log.info("pelvis gate dropped %d detections", gated)
    sizes = np.array([[c.width, c.height] for c in cams], np.int32)
    P3 = tracks.xyz.astype(np.float32).reshape(P, N, N_JOINTS, 3)
    return Prompts(
        list(tracks.ids), list(cams.names), box, kps, ks, P3, tracks.frame0, tracks.stride, tracks.fps, sizes
    )
