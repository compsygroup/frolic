from __future__ import annotations

import sys
import types

import yaml
from typer.testing import CliRunner

from frolic import pipeline
from frolic.cli import app
from frolic.config import FrolicConfig, load_config
from frolic.io.workdir import STAGES, WorkDir

runner = CliRunner()


def test_init_writes_template(tmp_path):
    p = tmp_path / "frolic.yaml"
    r = runner.invoke(app, ["init", str(p)])
    assert r.exit_code == 0 and p.exists()
    assert yaml.safe_load(p.read_text())["track"]["fps"] == 15.0
    r2 = runner.invoke(app, ["init", str(p)])
    assert r2.exit_code != 0
    assert runner.invoke(app, ["init", str(p), "--force"]).exit_code == 0


def _config(tmp_path):
    p = tmp_path / "frolic.yaml"
    p.write_text(yaml.safe_dump({"input": {"work_dir": "work"}, "calib": {"mode": "given", "given": "c.json"}}))
    return p


def test_status(tmp_path):
    p = _config(tmp_path)
    r = runner.invoke(app, ["status", "-c", str(p)])
    assert r.exit_code == 0 and "sync" in r.output and "render" in r.output
    r = runner.invoke(app, ["status", "-c", str(p), "--json"])
    assert r.exit_code == 0 and '"done": false' in r.output


def _fake_stages(monkeypatch, calls):
    for s in STAGES:
        mod = types.ModuleType(f"fake_{s}")

        def run(cfg, work, _s=s):
            calls.append(_s)
            return {"touched": _s}

        mod.run = run
        sys.modules[f"fake_{s}"] = mod
    monkeypatch.setattr(pipeline, "STAGE_MODULES", {s: f"fake_{s}" for s in STAGES})


def test_stage_idempotency_force_and_from(tmp_path, monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    cfg = FrolicConfig(calib={"mode": "given", "given": "c.json"})
    cfg.input.work_dir = tmp_path / "work"
    work = WorkDir(cfg.work_dir)
    assert pipeline.run_pipeline(cfg, work) == STAGES
    assert all(work.is_done(s) for s in STAGES)
    assert pipeline.run_pipeline(cfg, work) == []  # second run skips everything
    assert pipeline.run_stage(cfg, work, "fit") is False
    calls.clear()
    assert pipeline.run_stage(cfg, work, "fit", force=True) is True
    assert calls == ["fit"] and work.is_done("fit") and not work.is_done("export") and not work.is_done("render")
    assert work.is_done("sam3d")
    assert pipeline.run_pipeline(cfg, work) == ["export", "render"]
    calls.clear()
    assert pipeline.run_pipeline(cfg, work, start="sam3d", force=True) == ["sam3d", "fit", "export", "render"]
    assert calls == ["sam3d", "fit", "export", "render"]
    assert pipeline.run_pipeline(cfg, work, start="export", stop="export") == []
    st = work.status()
    assert st["fit"]["touched"] == "fit" and "seconds" in st["fit"]


def test_stage_python_same_interpreter_runs_in_process(tmp_path, monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    cfg = FrolicConfig(calib={"mode": "given", "given": "c.json"})
    cfg.input.work_dir = tmp_path / "work"
    cfg.compute.stage_python = {"fit": sys.executable}
    work = WorkDir(cfg.work_dir)
    assert pipeline.run_stage(cfg, work, "fit", config_path=tmp_path / "frolic.yaml") is True
    assert calls == ["fit"]


def test_stage_python_other_interpreter_subprocess(tmp_path, monkeypatch):
    cfg = FrolicConfig(calib={"mode": "given", "given": "c.json"})
    cfg.input.work_dir = tmp_path / "work"
    cfg.compute.stage_python = {"fit": tmp_path / "other_python"}
    work = WorkDir(cfg.work_dir)
    seen = {}

    def fake_run(cmd, env):
        seen["cmd"] = cmd
        work.mark_done("fit")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    assert pipeline.run_stage(cfg, work, "fit", config_path=tmp_path / "frolic.yaml") is True
    assert (
        seen["cmd"][:3] == [str((tmp_path / "other_python").resolve()), "-m", "frolic.cli"] and "fit" in seen["cmd"]
    )


def test_cli_stage_command_uses_markers(tmp_path, monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    p = _config(tmp_path)
    r = runner.invoke(app, ["export", "-c", str(p)])
    assert r.exit_code == 0, r.output
    assert calls == ["export"]
    work = WorkDir(load_config(p).work_dir)
    assert work.is_done("export")
    r = runner.invoke(app, ["run", "-c", str(p), "--to", "track"])
    assert r.exit_code == 0 and "ran: sync, detect, calibrate, track" in r.output
    r = runner.invoke(app, ["run", "-c", str(p), "--from", "nowhere"])
    assert r.exit_code != 0
