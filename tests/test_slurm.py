from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from frolic.config import FrolicConfig
from frolic.io.workdir import STAGES, WorkDir
from frolic.slurm import sbatch_script, submit_pipeline


@pytest.fixture
def cfg(tmp_path):
    c = FrolicConfig(calib={"mode": "given", "given": str(tmp_path / "c.json")})
    c.input.work_dir = tmp_path / "work"
    c.compute.slurm.account = "proj"
    c.compute.slurm.setup = ["module load cuda", "conda activate frolic"]
    c.compute.slurm.partition_gpu = "gpuq"
    return c


def test_sbatch_script_headers(cfg, tmp_path):
    work = WorkDir(cfg.work_dir)
    s = sbatch_script(cfg, work, "fit", tmp_path / "frolic.yaml", force=True)
    assert s.startswith("#!/bin/bash\n#SBATCH --job-name=frolic_fit")
    assert "--partition=gpuq" in s and "--gres=gpu:1" in s and "--account=proj" in s
    assert "module load cuda\nconda activate frolic\n" in s
    assert s.rstrip().endswith("frolic fit -c " + str(tmp_path / "frolic.yaml") + " --threads 8 --force")
    cpu = sbatch_script(cfg, work, "export", tmp_path / "frolic.yaml", force=False)
    assert "--partition=cpu" in cpu and "--gres" not in cpu and "--force" not in cpu


def test_dry_run_writes_scripts(cfg, tmp_path):
    work = WorkDir(cfg.work_dir)
    work.mark_done("sync")
    work.mark_done("detect")
    jobs = submit_pipeline(cfg, work, tmp_path / "frolic.yaml", dry_run=True)
    assert list(jobs) == STAGES[2:]
    assert all(Path(p).exists() for p in jobs.values())
    assert submit_pipeline(cfg, work, tmp_path / "frolic.yaml", start="fit", stop="export", dry_run=True).keys() == {
        "fit",
        "export",
    }


def test_submit_chains_dependencies(cfg, tmp_path, monkeypatch):
    work = WorkDir(cfg.work_dir)
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = f"{1000 + len(calls)};cluster\n"
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    jobs = submit_pipeline(cfg, work, tmp_path / "frolic.yaml", start="fit", stop="render")
    assert jobs == {"fit": "1001", "export": "1002", "render": "1003"}
    assert "--dependency" not in " ".join(calls[0])
    assert "--dependency=afterok:1001" in calls[1] and "--kill-on-invalid-dep=yes" in calls[1]
    assert "--dependency=afterok:1002" in calls[2]
    assert (work.root / "slurm" / "jobs.tsv").read_text().splitlines()[0] == "fit\t1001"


def test_force_from_clears_markers(cfg, tmp_path):
    work = WorkDir(cfg.work_dir)
    for s in STAGES:
        work.mark_done(s)
    assert submit_pipeline(cfg, work, tmp_path / "f.yaml", dry_run=True) == {}
    jobs = submit_pipeline(cfg, work, tmp_path / "f.yaml", start="export", force=True, dry_run=True)
    assert list(jobs) == ["export", "render"] and not work.is_done("render") and work.is_done("fit")
