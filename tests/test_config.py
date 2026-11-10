from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from frolic.config import FrolicConfig, load_config, save_config, template


def test_defaults_validate():
    cfg = FrolicConfig(calib={"mode": "given", "given": "cams.json"})
    assert cfg.track.fps == 15.0 and cfg.fit_fps == 15.0
    cfg.fit.fps = 10.0
    assert cfg.fit_fps == 10.0


def test_template_parses_and_validates():
    t = template()
    d = yaml.safe_load(t)
    cfg = FrolicConfig.model_validate(d)
    assert cfg.calib.mode == "given" and str(cfg.calib.given).endswith("cameras.json")
    assert "# folder with one video per camera" in t
    assert set(d) >= {"input", "sync", "calib", "detect", "track", "select", "fit", "export", "render", "compute"}


@pytest.mark.parametrize(
    "patch",
    [
        {"calib": {"mode": "given"}},
        {"calib": {"mode": "bootstrap"}},
        {"calib": {"mode": "given", "given": "x"}, "select": {"mode": "config"}},
        {"calib": {"mode": "given", "given": "x"}, "track": {"min_views": 1}},
        {"calib": {"mode": "given", "given": "x"}, "people": {"p0": {"height_m": 3.0}}},
        {"calib": {"mode": "given", "given": "x"}, "people": {"p0": {"age_years": -1}}},
        {"calib": {"mode": "given", "given": "x"}, "render": {"what": ["movie"]}},
        {"calib": {"mode": "given", "given": "x"}, "unknown_key": 1},
    ],
)
def test_invalid_configs_raise(patch):
    with pytest.raises((ValidationError, ValueError)):
        FrolicConfig.model_validate(patch)


def test_load_config_resolves_relative_paths(tmp_path):
    (tmp_path / "frolic.yaml").write_text(
        yaml.safe_dump(
            {
                "input": {"video_dir": "vids", "work_dir": "out"},
                "calib": {"mode": "given", "given": "cal/cameras.json"},
                "detect": {"detector": "mock", "mock_dir": "gt"},
            }
        )
    )
    cfg = load_config(tmp_path / "frolic.yaml")
    assert cfg.input.video_dir == (tmp_path / "vids").resolve()
    assert cfg.work_dir == (tmp_path / "out").resolve()
    assert cfg.calib.given == (tmp_path / "cal" / "cameras.json").resolve()
    assert cfg.detect.mock_dir == (tmp_path / "gt").resolve()


def test_save_roundtrip(tmp_path):
    cfg = FrolicConfig(
        calib={"mode": "given", "given": "cams.json"}, people={"p0": {"label": "A", "age_years": 4, "sex": "female"}}
    )
    save_config(cfg, tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert back.people["p0"].label == "A" and back.people["p0"].sex == "female"
