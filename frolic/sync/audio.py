"""Audio-based time offsets between cameras: onset-envelope cross-correlation (coarse) + GCC-PHAT (fine)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import signal

from frolic.log import get_logger

log = get_logger(__name__)


@dataclass
class PairLag:
    a: str
    b: str
    lag_s: float  # an event at time t in `a` occurs at t + lag_s in `b`
    coarse_lag_s: float
    ncc: float
    phat_peak: float
    overlap_s: float


def onset_envelope(
    x: np.ndarray,
    sr: int,
    band_hz: tuple[float, float] = (300.0, 3000.0),
    frame_ms: float = 10.0,
    median_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Log band energy per frame minus its running median -> (envelope, frame rate in Hz)."""
    lo, hi = band_hz
    hi = min(hi, 0.45 * sr)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64)) if len(x) > 50 else x.astype(np.float64)
    hop = max(1, round(sr * frame_ms / 1000.0))
    n = len(y) // hop
    if n < 2:
        return np.zeros(max(n, 0)), sr / hop
    e = np.log(np.square(y[: n * hop]).reshape(n, hop).mean(1) + 1e-10)
    k = round(median_s * sr / hop) | 1
    base = signal.medfilt(e, min(k, (len(e) - 1) | 1)) if len(e) > 3 else e
    return (e - base).astype(np.float64), sr / hop


def ncc_lags(a: np.ndarray, b: np.ndarray, min_overlap: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation r[k] over the overlap for every shift k with b[t + k] ~ a[t]; lags with < min_overlap samples are NaN."""
    na, nb = len(a), len(b)
    m = 1
    while m < na + nb:
        m *= 2
    fa, fb = np.fft.rfft(a, m), np.fft.rfft(b, m)
    raw = np.fft.irfft(fb * np.conj(fa), m)
    lags = np.arange(-(nb - 1), na)
    sab = raw[lags % m]
    ca, ca2 = np.r_[0, np.cumsum(a)], np.r_[0, np.cumsum(a * a)]
    cb, cb2 = np.r_[0, np.cumsum(b)], np.r_[0, np.cumsum(b * b)]
    t0 = np.maximum(0, -lags)
    t1 = np.minimum(na, nb - lags)
    n = (t1 - t0).astype(np.float64)
    ok = n >= max(min_overlap, 2)
    t0c, t1c = np.where(ok, t0, 0), np.where(ok, t1, 0)
    sa, sa2 = ca[t1c] - ca[t0c], ca2[t1c] - ca2[t0c]
    sb, sb2 = cb[t1c + lags * ok] - cb[t0c + lags * ok], cb2[t1c + lags * ok] - cb2[t0c + lags * ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab - sa * sb / n
        var = (sa2 - sa * sa / n) * (sb2 - sb * sb / n)
        r = np.where(ok & (var > 0), cov / np.sqrt(np.maximum(var, 1e-20)), np.nan)
    return lags, r


def coarse_lag(
    a: np.ndarray, b: np.ndarray, sr: int, min_overlap_s: float, band_hz=(300.0, 3000.0), frame_ms: float = 10.0
) -> tuple[float, float, float]:
    """Envelope NCC -> (lag_s with b(t + lag) ~ a(t), peak ncc, overlap_s)."""
    ea, rate = onset_envelope(a, sr, band_hz, frame_ms)
    eb, _ = onset_envelope(b, sr, band_hz, frame_ms)
    lags, r = ncc_lags(ea, eb, int(min_overlap_s * rate))
    if not np.isfinite(r).any():
        raise ValueError("no lag with enough overlap")
    k = int(np.nanargmax(r))
    lag_s = lags[k] / rate
    overlap = (min(len(ea), len(eb) - lags[k]) - max(0, -lags[k])) / rate
    return float(lag_s), float(r[k]), float(overlap)


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_shift_s: float) -> tuple[float, float]:
    """Sub-sample delay of b relative to a (b(t + d) ~ a(t)) within +-max_shift_s, with the PHAT peak height."""
    n = len(a) + len(b)
    m = 1
    while m < n:
        m *= 2
    fa, fb = np.fft.rfft(a, m), np.fft.rfft(b, m)
    cross = fb * np.conj(fa)
    cc = np.fft.irfft(cross / (np.abs(cross) + 1e-12), m)
    w = round(max_shift_s * sr)
    idx = np.r_[np.arange(0, w + 1), np.arange(m - w, m)] if w > 0 else np.array([0])
    vals = cc[idx]
    j = int(np.argmax(vals))
    k = idx[j]
    shift = k if k <= m // 2 else k - m
    if 0 < j < len(vals) - 1 and w > 0:
        y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            shift = shift + 0.5 * (y0 - y2) / den
    return float(shift / sr), float(vals[j] / (np.abs(cc).max() + 1e-12))


def pair_lag(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    min_overlap_s: float,
    fine_window_s: float,
    band_hz=(300.0, 3000.0),
    frame_ms: float = 10.0,
) -> tuple[float, float, float, float, float]:
    """(lag_s, coarse_lag_s, ncc, phat_peak, overlap_s) with b(t + lag) ~ a(t)."""
    coarse, ncc, overlap = coarse_lag(a, b, sr, min_overlap_s, band_hz, frame_ms)
    shift = round(coarse * sr)
    if shift >= 0:
        sa, sb = a[: len(b) - shift] if len(b) - shift < len(a) else a, b[shift:]
    else:
        sa, sb = a[-shift:], b[: len(a) + shift] if len(a) + shift < len(b) else b
    n = min(len(sa), len(sb))
    if n < sr:
        return coarse, coarse, ncc, 0.0, overlap
    fine, peak = gcc_phat(sa[:n], sb[:n], sr, fine_window_s)
    return coarse + fine, coarse, ncc, peak, overlap


def pairwise_lags(
    audio: dict[str, np.ndarray],
    sr: int,
    min_overlap_s: float,
    fine_window_s: float,
    band_hz=(300.0, 3000.0),
    frame_ms: float = 10.0,
) -> list[PairLag]:
    out = []
    for a, b in itertools.combinations(audio, 2):
        try:
            lag, coarse, ncc, peak, ov = pair_lag(
                audio[a], audio[b], sr, min_overlap_s, fine_window_s, band_hz, frame_ms
            )
        except ValueError as e:
            log.warning("%s-%s: %s", a, b, e)
            continue
        out.append(PairLag(a, b, lag, coarse, ncc, peak, ov))
    return out


def closure_errors(pairs: list[PairLag]) -> dict[str, float]:
    """Triangle closure lag(A,B) + lag(B,C) - lag(A,C) in seconds for every camera triple."""
    lag = {}
    for p in pairs:
        lag[(p.a, p.b)] = p.lag_s
        lag[(p.b, p.a)] = -p.lag_s
    names = sorted({n for p in pairs for n in (p.a, p.b)})
    out = {}
    for a, b, c in itertools.combinations(names, 3):
        if all(k in lag for k in ((a, b), (b, c), (a, c))):
            out[f"{a}-{b}-{c}"] = float(lag[(a, b)] + lag[(b, c)] - lag[(a, c)])
    return out


def offsets_from_pairs(pairs: list[PairLag], names: list[str], reference: str) -> dict[str, float]:
    """Least-squares offsets (event at t in the reference occurs at t + offset in each camera), weighted by NCC."""
    idx = {n: i for i, n in enumerate(names)}
    rows, rhs, w = [], [], []
    for p in pairs:
        r = np.zeros(len(names))
        r[idx[p.b]] = 1.0
        r[idx[p.a]] = -1.0
        rows.append(r)
        rhs.append(p.lag_s)
        w.append(max(p.ncc, 1e-3))
    r = np.zeros(len(names))
    r[idx[reference]] = 1.0
    rows.append(r)
    rhs.append(0.0)
    w.append(1e3)
    A = np.array(rows) * np.sqrt(np.array(w))[:, None]
    x, *_ = np.linalg.lstsq(A, np.array(rhs) * np.sqrt(np.array(w)), rcond=None)
    return {n: float(x[idx[n]]) for n in names}
