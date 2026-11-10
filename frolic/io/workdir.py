"""Work directory layout, done markers and stage logs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

STAGES: list[str] = ["sync", "detect", "calibrate", "track", "select", "singleview", "sam3d", "fit", "export", "render"]
STAGE_DIRS: dict[str, str] = {s: s for s in STAGES}
STAGE_DIRS["calibrate"] = "calib"


class WorkDir:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def __truediv__(self, other: str) -> Path:
        return self.root / other

    def stage_dir(self, stage: str) -> Path:
        d = self.root / STAGE_DIRS[stage]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def done_dir(self) -> Path:
        d = self.root / "done"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def log_dir(self) -> Path:
        d = self.root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def marker(self, stage: str) -> Path:
        return self.done_dir / f"{stage}.done"

    def is_done(self, stage: str) -> bool:
        return self.marker(stage).exists()

    def mark_done(self, stage: str, info: dict[str, Any] | None = None) -> None:
        with open(self.marker(stage), "w") as fh:
            json.dump({"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, fh, indent=1)

    def clear(self, stage: str) -> None:
        m = self.marker(stage)
        if m.exists():
            m.unlink()

    def clear_from(self, stage: str) -> None:
        for s in STAGES[STAGES.index(stage) :]:
            self.clear(s)

    def status(self) -> dict[str, dict[str, Any]]:
        out = {}
        for s in STAGES:
            m = self.marker(s)
            if m.exists():
                with open(m) as fh:
                    out[s] = {"done": True, **json.load(fh)}
            else:
                out[s] = {"done": False}
        return out

    # canonical file locations shared by the stages
    @property
    def videos_json(self) -> Path:
        return self.root / "videos.json"

    @property
    def cameras(self) -> Path:
        return self.stage_dir("calibrate") / "cameras.json"

    @property
    def cameras_level(self) -> Path:
        return self.stage_dir("track") / "cameras.json"

    @property
    def tracks_raw(self) -> Path:
        return self.stage_dir("track") / "tracks_raw.npz"

    @property
    def tracks(self) -> Path:
        return self.stage_dir("track") / "tracks.npz"

    @property
    def tracks_selected(self) -> Path:
        return self.stage_dir("select") / "tracks.npz"

    @property
    def tracks_singleview(self) -> Path:
        return self.stage_dir("singleview") / "tracks.npz"

    @property
    def fit_input_tracks(self) -> Path:
        sv = self.tracks_singleview
        return sv if sv.exists() else self.tracks_selected

    def read_videos(self) -> dict[str, Any]:
        with open(self.videos_json) as fh:
            return json.load(fh)

    def write_videos(self, d: dict[str, Any]) -> None:
        with open(self.videos_json, "w") as fh:
            json.dump(d, fh, indent=1)
