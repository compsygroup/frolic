"""Stage registry and the idempotent runner (done markers, --force / --from, per-stage interpreters)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from frolic.config import FrolicConfig
from frolic.io.workdir import STAGES, WorkDir
from frolic.log import get_logger

log = get_logger("pipeline")

STAGE_MODULES: dict[str, str] = {
    "sync": "frolic.sync.stage",
    "calibrate": "frolic.calib.stage",
    "detect": "frolic.detect.stage",
    "track": "frolic.track.stage",
    "select": "frolic.select.stage",
    "singleview": "frolic.singleview.stage",
    "sam3d": "frolic.sam3d.stage",
    "fit": "frolic.fit.stage",
    "export": "frolic.export.stage",
    "render": "frolic.render.stage",
}
GPU_STAGES: set[str] = {"detect", "track", "sam3d", "fit", "render"}
StageFn = Callable[[FrolicConfig, WorkDir], dict[str, Any]]


def stage_function(stage: str) -> StageFn:
    mod = importlib.import_module(STAGE_MODULES[stage])
    return mod.run


def stages_from(start: str | None, stop: str | None = None) -> list[str]:
    i0 = 0 if start is None else STAGES.index(start)
    i1 = len(STAGES) if stop is None else STAGES.index(stop) + 1
    return STAGES[i0:i1]


def _other_interpreter(cfg: FrolicConfig, stage: str) -> Path | None:
    py = cfg.compute.stage_python.get(stage)
    if py is None:
        return None
    py = Path(py).resolve()
    return None if py == Path(sys.executable).resolve() else py


def run_stage(
    cfg: FrolicConfig, work: WorkDir, stage: str, config_path: Path | None = None, force: bool = False
) -> bool:
    """Run one stage unless its done marker exists. Returns True when work was done."""
    if work.is_done(stage) and not force:
        log.info("%s: already done (use --force to redo)", stage)
        return False
    work.clear_from(stage)
    py = _other_interpreter(cfg, stage)
    t0 = time.time()
    if py is not None and config_path is not None:
        log.info("%s: running in %s", stage, py)
        env = dict(os.environ)
        env.setdefault("FROLIC_CHILD", "1")
        cmd = [str(py), "-m", "frolic.cli", stage, "-c", str(config_path), "--force"]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            raise RuntimeError(f"stage {stage} failed in {py} (rc={rc})")
        return True
    log.info("%s: start", stage)
    info = stage_function(stage)(cfg, work) or {}
    info["seconds"] = round(time.time() - t0, 1)
    work.mark_done(stage, info)
    log.info("%s: done in %.0f s", stage, info["seconds"])
    return True


def run_pipeline(
    cfg: FrolicConfig,
    work: WorkDir,
    config_path: Path | None = None,
    start: str | None = None,
    stop: str | None = None,
    force: bool = False,
) -> list[str]:
    ran = []
    if start is not None and force:
        work.clear_from(start)
    for stage in stages_from(start, stop):
        if run_stage(cfg, work, stage, config_path, force=force and (start is None or stage == start)):
            ran.append(stage)
    return ran
