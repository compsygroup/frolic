"""`frolic` command line: one sub-command per stage plus init / run / status."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from frolic3d import __version__
from frolic3d.config import FrolicConfig, load_config, template
from frolic3d.io.workdir import STAGES, WorkDir
from frolic3d.log import configure, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="FROLIC 3D: multi-view, multi-person, child-aware 3D human mesh pipeline.",
)
log = get_logger("cli")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="frolic.yaml")]
ForceOpt = Annotated[bool, typer.Option("--force", help="redo this stage (and clear every later marker)")]
DeviceOpt = Annotated[str | None, typer.Option("--device", help="cuda, cuda:0 or cpu")]
ThreadsOpt = Annotated[int | None, typer.Option("--threads", help="CPU threads for torch / OpenMP")]
LogOpt = Annotated[str, typer.Option("--log-level", help="DEBUG, INFO, WARNING")]


def _setup(
    config: Path, device: str | None, threads: int | None, log_level: str, stage: str | None = None
) -> tuple[FrolicConfig, WorkDir]:
    cfg = load_config(config)
    if device:
        cfg.compute.device = device
        cfg.fit.device = device
    if threads:
        cfg.compute.threads = threads
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.compute.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cfg.compute.threads))
    work = WorkDir(cfg.work_dir)
    work.root.mkdir(parents=True, exist_ok=True)
    configure(log_level, work.log_dir / f"{stage or 'frolic'}.log")
    return cfg, work


def _stage_command(stage: str):
    def cmd(
        config: ConfigOpt = Path("frolic.yaml"),
        force: ForceOpt = False,
        device: DeviceOpt = None,
        threads: ThreadsOpt = None,
        log_level: LogOpt = "INFO",
    ) -> None:
        from frolic3d.pipeline import run_stage

        cfg, work = _setup(config, device, threads, log_level, stage)
        run_stage(cfg, work, stage, config_path=config.resolve(), force=force)

    cmd.__name__ = stage
    cmd.__doc__ = _STAGE_HELP[stage]
    return cmd


_STAGE_HELP = {
    "sync": "Align the camera videos in time (audio cross-correlation) and write work/videos.json.",
    "calibrate": "Camera calibration: ChArUco board videos, person-based bootstrap, or a given file; floor levelling.",
    "detect": "Per-camera 2D person detection and pose (YOLOX + RTMPose halpe26).",
    "track": "Cross-view association, identity tracking (appearance re-ID), track refinement, floor levelling.",
    "select": "Choose which people to reconstruct (auto filters, interactive sheet, or config ids) and label them.",
    "singleview": "Lift frames seen by only one camera (floor / stature priors) so people do not vanish.",
    "sam3d": "SAM 3D Body per camera on each person's box and the multi-view MHR fit.",
    "fit": "Anny body fit: shape from the tracks, pose from SAM 3D Body, biomechanical constraints.",
    "export": "Per-person npz meshes, metrics.json and summary.csv.",
    "render": "Overlay / grid / room / trajectory videos.",
}
for _s in STAGES:
    app.command(_s)(_stage_command(_s))


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="where to write the template")] = Path("frolic.yaml"),
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write a commented configuration template."""
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} exists (use --force)")
    path.write_text(template())
    typer.echo(f"wrote {path}")


@app.command()
def run(
    config: ConfigOpt = Path("frolic.yaml"),
    from_stage: Annotated[
        str | None, typer.Option("--from", help="start at this stage (clears later markers with --force)")
    ] = None,
    to_stage: Annotated[str | None, typer.Option("--to", help="stop after this stage")] = None,
    force: ForceOpt = False,
    slurm: Annotated[bool, typer.Option("--slurm", help="submit one sbatch job per stage with dependencies")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="with --slurm: write the scripts, do not submit")] = False,
    device: DeviceOpt = None,
    threads: ThreadsOpt = None,
    log_level: LogOpt = "INFO",
) -> None:
    """Run every stage in order (idempotent: finished stages are skipped)."""
    cfg, work = _setup(config, device, threads, log_level, "run")
    for s in (from_stage, to_stage):
        if s is not None and s not in STAGES:
            raise typer.BadParameter(f"unknown stage {s!r}; stages: {', '.join(STAGES)}")
    if slurm:
        from frolic3d.slurm import submit_pipeline

        jobs = submit_pipeline(
            cfg, work, config.resolve(), start=from_stage, stop=to_stage, force=force, dry_run=dry_run
        )
        for stage, job in jobs.items():
            typer.echo(f"{stage}: {job}")
        return
    from frolic3d.pipeline import run_pipeline

    ran = run_pipeline(cfg, work, config.resolve(), start=from_stage, stop=to_stage, force=force)
    typer.echo("ran: " + (", ".join(ran) if ran else "nothing (all done)"))


@app.command()
def status(config: ConfigOpt = Path("frolic.yaml"), as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show which stages are done."""
    cfg = load_config(config)
    work = WorkDir(cfg.work_dir)
    st = work.status()
    if as_json:
        typer.echo(json.dumps(st, indent=1))
        return
    for s in STAGES:
        d = st[s]
        mark = "done" if d["done"] else "-"
        extra = f"  {d.get('time', '')}  {d.get('seconds', '')}s" if d["done"] else ""
        typer.echo(f"{s:11s} {mark}{extra}")


@app.command()
def version() -> None:
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
