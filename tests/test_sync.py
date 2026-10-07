import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import wavfile

from frolic3d.config import FrolicConfig, load_config
from frolic3d.io.workdir import WorkDir
from frolic3d.sync import audio as A
from frolic3d.sync.stage import common_window, run
from frolic3d.synthetic import make_scene

SR = 8000


def _bursty_audio(rng, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    x = rng.normal(0, 0.02, n)
    for _ in range(int(seconds * 3)):
        s = rng.integers(0, n - SR // 4)
        burst = rng.normal(0, 1.0, SR // 8) * np.hanning(SR // 8)
        x[s : s + SR // 8] += burst * rng.uniform(0.3, 1.0)
    return x.astype(np.float32)


def _shift(x: np.ndarray, lag_s: float, rng, noise=0.05) -> np.ndarray:
    """Recording that starts lag_s after x (an event at t in x is at t - lag_s here)."""
    k = round(lag_s * SR)
    y = x[k:] if k >= 0 else np.r_[np.zeros(-k, np.float32), x]
    return (y + rng.normal(0, noise, len(y))).astype(np.float32)


def test_pair_lag_recovers_shift():
    rng = np.random.default_rng(0)
    x = _bursty_audio(rng, 40)
    for lag in (1.234, -2.5, 0.0171):
        y = _shift(x, lag, rng)
        got, coarse, ncc, _peak, overlap = A.pair_lag(x, y, SR, 10.0, 0.08)
        assert abs(got + lag) < 1.5e-3, (lag, got, coarse)
        assert abs(coarse + lag) < 0.02
        assert ncc > 0.5 and overlap > 30


def test_closure_and_offsets():
    rng = np.random.default_rng(3)
    x = _bursty_audio(rng, 40)
    lags = {"a": 0.0, "b": 0.8, "c": -1.3}
    audio = {k: _shift(x, v, rng) for k, v in lags.items()}
    pairs = A.pairwise_lags(audio, SR, 10.0, 0.08)
    assert len(pairs) == 3
    cl = A.closure_errors(pairs)
    assert list(cl) == ["a-b-c"] and abs(cl["a-b-c"]) < 3e-3
    off = A.offsets_from_pairs(pairs, list(audio), "a")
    assert abs(off["a"]) < 1e-6
    assert abs(off["b"] + 0.8) < 2e-3 and abs(off["c"] - 1.3) < 2e-3


def test_ncc_min_overlap_masks_lags():
    a = np.random.default_rng(1).normal(size=200)
    lags, r = A.ncc_lags(a, a, 150)
    assert np.isnan(r[np.abs(lags) > 50]).all()
    assert abs(r[lags == 0][0] - 1) < 1e-9


def test_gcc_phat_subsample():
    rng = np.random.default_rng(2)
    x = _bursty_audio(rng, 5)
    y = np.r_[np.zeros(37, np.float32), x]
    d, peak = A.gcc_phat(x[: len(y)], y, SR, 0.02)
    assert abs(d - 37 / SR) < 0.3 / SR
    assert peak > 0.3


def test_common_window():
    from frolic3d.io.video import VideoInfo

    infos = {k: VideoInfo(Path(k), 1, 1, 30.0, 300, 10.0, False) for k in "ab"}
    t0, t1 = common_window(infos, {"a": 0.0, "b": 2.0}, 0.0, 0.0)
    assert (t0, t1) == (0.0, 8.0)
    t0, t1 = common_window(infos, {"a": 0.0, "b": -2.0}, 0.5, 3.0)
    assert (t0, t1) == (2.5, 5.5)
    with pytest.raises(RuntimeError):
        common_window(infos, {"a": 0.0, "b": 20.0}, 0.0, 0.0)


def _write_cfg(path: Path, scene: Path, work: Path, **over) -> Path:
    cfg = {
        "input": {"video_dir": str(scene / "videos"), "work_dir": str(work), **over.pop("input", {})},
        "calib": {"given": str(scene / "calib" / "cameras.json")},
        **over,
    }
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    make_scene(d, n_cams=3, n_frames=24, fps=12.0, videos=True)
    return d


def test_stage_synced(scene, tmp_path):
    cfg = load_config(_write_cfg(tmp_path / "f.yaml", scene, tmp_path / "work"))
    work = WorkDir(cfg.work_dir)
    info = run(cfg, work)
    doc = work.read_videos()
    assert doc["cameras"] == ["cam1", "cam2", "cam3"] and info["method"] == "assumed_synced"
    assert doc["n_frames"] == 24 and doc["fps"] == pytest.approx(12.0) and not doc["trimmed"]
    assert all(doc["frame_offset"][c] == 0 for c in doc["cameras"])
    assert (work.stage_dir("sync") / "offsets.json").exists()


def test_stage_manual_offsets_trim(scene, tmp_path):
    cfg = load_config(
        _write_cfg(
            tmp_path / "f.yaml",
            scene,
            tmp_path / "work",
            sync={"manual_offsets_s": {"cam1": 0.0, "cam2": 0.5, "cam3": -0.25}, "trim": True},
        )
    )
    work = WorkDir(cfg.work_dir)
    run(cfg, work)
    doc = work.read_videos()
    assert doc["trimmed"] and doc["window_s"] == pytest.approx([0.25, 1.5])
    assert doc["n_frames"] == 15
    assert all(Path(doc["videos"][c]).exists() for c in doc["cameras"])


def test_stage_manual_offsets_no_trim(scene, tmp_path):
    cfg = load_config(
        _write_cfg(
            tmp_path / "f.yaml",
            scene,
            tmp_path / "work",
            sync={"manual_offsets_s": {"cam1": 0.0, "cam2": 0.5, "cam3": -0.25}, "trim": False},
        )
    )
    work = WorkDir(cfg.work_dir)
    run(cfg, work)
    doc = work.read_videos()
    assert doc["frame_offset"] == {"cam1": 3, "cam2": 9, "cam3": 0}
    assert doc["n_frames"] == 15 and Path(doc["videos"]["cam1"]) == scene / "videos" / "cam1.mp4"


def test_stage_audio(scene, tmp_path):
    rng = np.random.default_rng(5)
    x = _bursty_audio(rng, 30)
    lags = {"cam1": 0.0, "cam2": 0.9, "cam3": -0.6}
    vdir = tmp_path / "videos"
    vdir.mkdir()
    for c, lag in lags.items():
        wav = tmp_path / f"{c}.wav"
        wavfile.write(wav, SR, _shift(x, lag, rng))
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-stream_loop",
                "20",
                "-i",
                str(scene / "videos" / f"{c}.mp4"),
                "-i",
                str(wav),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(vdir / f"{c}.mp4"),
            ],
            capture_output=True,
        )
        assert r.returncode == 0, r.stderr
    cfgp = tmp_path / "f.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "input": {"video_dir": str(vdir), "work_dir": str(tmp_path / "work"), "synced": False},
                "sync": {"trim": False, "max_audio_s": 0, "min_overlap_s": 10},
                "calib": {"given": "x.json"},
            }
        )
    )
    cfg = load_config(cfgp)
    work = WorkDir(cfg.work_dir)
    info = run(cfg, work)
    doc = work.read_videos()
    assert info["method"] == "audio" and doc["sync"]["ok"]
    for c, lag in lags.items():
        assert abs(doc["offsets_s"][c] + lag) < 5e-3, (c, doc["offsets_s"])
    assert json.loads(Path(work.stage_dir("sync") / "offsets.json").read_text())["reference"] == "cam1"


def test_config_template_roundtrip():
    from frolic3d.config import template

    FrolicConfig.model_validate(yaml.safe_load(template()))
