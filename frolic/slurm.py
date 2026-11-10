"""Slurm submission: one sbatch job per pending stage, chained with afterok dependencies."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from frolic3d.config import FrolicConfig
from frolic3d.io.workdir import WorkDir
from frolic3d.log import get_logger
from frolic3d.pipeline import GPU_STAGES, stages_from

log = get_logger(__name__)


def sbatch_script(cfg: FrolicConfig, work: WorkDir, stage: str, config_path: Path, force: bool) -> str:
    s = cfg.compute.slurm
    gpu = stage in GPU_STAGES
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=frolic_{stage}",
        f"#SBATCH --partition={s.partition_gpu if gpu else s.partition_cpu}",
        f"#SBATCH --cpus-per-task={s.cpus}",
        f"#SBATCH --mem={s.mem}",
        f"#SBATCH --time={s.time_gpu if gpu else s.time_cpu}",
        f"#SBATCH --output={work.log_dir}/slurm_{stage}_%j.out",
    ]
    if gpu:
        lines.append(f"#SBATCH --gres={s.gres_gpu}")
    if s.account:
        lines.append(f"#SBATCH --account={s.account}")
    lines += [f"#SBATCH {x}" for x in s.extra_sbatch]
    lines += ["set -euo pipefail", f"export OMP_NUM_THREADS={cfg.compute.threads}"]
    lines += list(s.setup)
    cmd = ["frolic", stage, "-c", str(config_path), "--threads", str(cfg.compute.threads)]
    if force:
        cmd.append("--force")
    lines.append(" ".join(shlex.quote(c) for c in cmd))
    return "\n".join(lines) + "\n"


def write_scripts(
    cfg: FrolicConfig, work: WorkDir, config_path: Path, stages: list[str], force: bool
) -> dict[str, Path]:
    out_dir = work.root / "slurm"
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts = {}
    for k, stage in enumerate(stages):
        p = out_dir / f"{stage}.sbatch"
        p.write_text(sbatch_script(cfg, work, stage, config_path, force and k == 0))
        scripts[stage] = p
    return scripts


def submit(script: Path, depend: str | None) -> str:
    cmd = ["sbatch", "--parsable"]
    if depend:
        cmd += [f"--dependency=afterok:{depend}", "--kill-on-invalid-dep=yes"]
    cmd.append(str(script))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script.name}: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


def submit_pipeline(
    cfg: FrolicConfig,
    work: WorkDir,
    config_path: Path,
    start: str | None = None,
    stop: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, str]:
    """Returns stage -> job id (or script path when dry_run). Finished stages are skipped unless force."""
    if start is not None and force:
        work.clear_from(start)
    stages = [s for s in stages_from(start, stop) if force or not work.is_done(s)]
    if not stages:
        log.info("nothing to submit (all stages done)")
        return {}
    scripts = write_scripts(cfg, work, config_path, stages, force)
    if dry_run:
        return {s: str(p) for s, p in scripts.items()}
    jobs: dict[str, str] = {}
    depend = None
    for stage in stages:
        depend = submit(scripts[stage], depend)
        jobs[stage] = depend
        log.info("%s -> job %s", stage, depend)
    with open(work.root / "slurm" / "jobs.tsv", "w") as fh:
        fh.writelines(f"{s}\t{j}\n" for s, j in jobs.items())
    return jobs
