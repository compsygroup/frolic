"""SAM 3D Body on every prompted box of one camera, frame by frame along the track grid (frames stay in memory)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from frolic.geometry import Camera
from frolic.io.video import iter_frames
from frolic.log import get_logger
from frolic.sam3d.adapter import KEEP, Sam3dBody
from frolic.sam3d.prompts import Prompts

log = get_logger(__name__)


class ViewResults:
    """Per-view outputs stacked along M observations: `pid_index` (M,), `frame` (M,) and every model array."""

    def __init__(self, pid_index: np.ndarray, frame: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
        self.pid_index = np.asarray(pid_index, np.int32)
        self.frame = np.asarray(frame, np.int32)
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.frame)

    def by_person(self, p: int) -> dict[int, dict[str, np.ndarray]]:
        rows = np.flatnonzero(self.pid_index == p)
        return {int(self.frame[r]): {k: v[r] for k, v in self.arrays.items()} for r in rows}

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pid_index=self.pid_index, frame=self.frame, **self.arrays)

    @classmethod
    def load(cls, path: Path | str) -> ViewResults:
        z = np.load(path)
        return cls(z["pid_index"], z["frame"], {k: z[k] for k in z.files if k not in ("pid_index", "frame")})

    @classmethod
    def empty(cls) -> ViewResults:
        return cls(np.zeros(0, np.int32), np.zeros(0, np.int32), {})


def run_view(
    cam: Camera,
    video: Path | str,
    frame_offset: int,
    prompts: Prompts,
    estimator: Sam3dBody,
    out: Path | str,
    max_frames: int = 0,
    frames: list[int] | None = None,
) -> ViewResults:
    """Decode the aligned frames of the track grid, undistort to the camera's K and estimate every present person."""
    c = prompts.cameras.index(cam.name)
    N = prompts.n_frames if not max_frames else min(prompts.n_frames, max_frames)
    wanted = sorted(set(range(N)) if frames is None else {f for f in frames if 0 <= f < N})
    m1, m2 = cv2.initUndistortRectifyMap(cam.K, cam.dist, None, cam.K, (cam.width, cam.height), cv2.CV_32FC1)
    pid_index, frame_idx, arrays = [], [], {k: [] for k in KEEP}
    t0 = time.time()
    n_crops = 0
    todo = [i for i in wanted if prompts.present(c, i)]
    if todo:
        start = prompts.frame0 + todo[0] * prompts.stride + frame_offset
        stop = prompts.frame0 + todo[-1] * prompts.stride + frame_offset + 1
        want = {prompts.frame0 + i * prompts.stride + frame_offset: i for i in todo}
        for f, img in iter_frames(video, start=start, stop=stop, step=prompts.stride):
            i = want.get(f)
            if i is None:
                continue
            who = prompts.present(c, i)
            rgb = cv2.cvtColor(cv2.remap(img, m1, m2, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            outs = estimator.estimate(rgb, np.stack([prompts.box[p, c, i] for p in who]), cam.K)
            for p, o in zip(who, outs):
                pid_index.append(p)
                frame_idx.append(i)
                for k in KEEP:
                    arrays[k].append(o.get(k))
                n_crops += 1
            if n_crops and n_crops % 200 == 0:
                log.info(
                    "%s: frame %d/%d, %d crops, %.2f s/frame",
                    cam.name,
                    i,
                    N,
                    n_crops,
                    (time.time() - t0) / max(1, len([x for x in todo if x <= i])),
                )
    stacked = {k: np.stack(v).astype(np.float32) for k, v in arrays.items() if v and all(x is not None for x in v)}
    res = ViewResults(np.array(pid_index), np.array(frame_idx), stacked)
    res.save(out)
    log.info("%s: %d person-crops in %.1f min -> %s", cam.name, n_crops, (time.time() - t0) / 60, out)
    return res
