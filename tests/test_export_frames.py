"""Export frame bookkeeping: the fit's frame0 (a source frame) is added once to the stride grid."""
import numpy as np


def _grid(src0: int, n: int, stride: int, src_fps: float):
    source_frames = (src0 + np.arange(n) * stride).astype(np.int32)
    return source_frames, source_frames / src_fps


def test_offset_added_once():
    sf, t = _grid(600, 5, 4, 60.0)
    assert sf[0] == 600 and sf[-1] == 616
    assert abs(t[0] - 10.0) < 1e-9


def test_zero_start_unchanged():
    sf, t = _grid(0, 4, 4, 60.0)
    assert list(sf) == [0, 4, 8, 12]
    assert np.allclose(t, np.arange(4) / 15.0)
