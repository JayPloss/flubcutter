import os

import numpy as np
import pytest

from flubcutter import edit, enhance

R = enhance.RATE


def speechlike(seconds=6.0, noise=0.003, seed=0):
    """Bursts of a harmonic tone (syllables) over steady noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * R)) / R
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((180, 360, 540, 1200, 3000), 1))
    gate = (np.sin(2 * np.pi * 2.5 * t) > 0.2) & (t % 2 < 1.5)
    return (0.2 * voice * gate + noise * rng.standard_normal(len(t))).astype(np.float32)


def lag(a, b, max_lag=2000):
    """Samples by which b is late relative to a."""
    c = np.correlate(b[max_lag:-max_lag], a, "valid")
    return max_lag - int(np.argmax(np.abs(c)))


def test_loudness_hits_target_and_peak_ceiling():
    x = speechlike() * 3  # peaky and loud
    out = enhance.loudness(x)
    m = enhance.measure(out)
    assert len(out) == len(x)
    assert abs(m["lufs"] - enhance.TARGET_LUFS) < 0.7
    assert m["true_peak"] <= enhance.TRUE_PEAK + 0.1


def test_loudness_is_one_gain_when_peaks_allow():
    t = np.arange(4 * R) / R
    x = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # quiet, crest factor 3 dB
    out = enhance.loudness(x)
    ratio = out[R:-R] / np.where(np.abs(x[R:-R]) > 1e-3, x[R:-R], np.nan)
    assert np.nanstd(ratio) / np.nanmean(ratio) < 1e-3


def test_polish_keeps_timing_and_removes_rumble():
    t = np.arange(6 * R) / R
    x = speechlike() + (0.05 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    out = enhance.polish(x)
    assert len(out) == len(x)
    # Timing: bursts of (aperiodic) noise come out where they went in.
    rng = np.random.default_rng(1)
    bursts = (rng.standard_normal(len(t)) * 0.1 * (t % 1 < 0.3)).astype(np.float32)
    assert abs(lag(bursts, enhance.polish(bursts))) < 0.003 * R

    def band(y, f):
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / R)
        return spec[np.abs(freqs - f) < 2].max()

    # 30 Hz rumble falls well below the 180 Hz voice fundamental.
    assert band(out, 30) / band(out, 180) < 0.5 * band(x, 30) / band(x, 180)


def test_reduction_follows_the_noise():
    assert enhance.reduction(speechlike(noise=0.0003)) == enhance.MIN_ATTEN_DB
    assert enhance.reduction(speechlike(noise=0.03)) == enhance.MAX_ATTEN_DB


def test_master_is_cached_per_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(enhance, "enhance", lambda a, rate, progress=None: calls.append(1) or a * 0.5)
    src = tmp_path / "rec.wav"
    edit.write_wav(speechlike(1.0), src)
    cache_dir = tmp_path / "cache"
    first = enhance.master(src, cache_dir)
    second = enhance.master(src, cache_dir)
    assert len(calls) == 1 and np.array_equal(first, second)
    # A new recording under the same name replaces the old master.
    edit.write_wav(speechlike(1.0, seed=1), src)
    os.utime(src, ns=(1, 1))
    enhance.master(src, cache_dir)
    assert len(calls) == 2
    assert len(list(cache_dir.glob("*.master.wav"))) == 1


def model_ready():
    try:
        from dpdfnet.models import available_model_entries
        return any(r["name"] == enhance.MODEL and r["cached"] for r in available_model_entries())
    except Exception:
        return False


@pytest.mark.skipif(not model_ready(), reason="DPDFNet model not downloaded")
def test_denoise_lowers_floor_keeps_timing_and_chunks_join(monkeypatch):
    x = speechlike(8.0, noise=0.01)
    whole = enhance.denoise(x, 20.0)
    assert len(whole) == len(x)
    assert lag(x, whole) == 0
    floor_in, floor_out = edit.Acoustics(x).floor, edit.Acoustics(whole).floor
    assert floor_out < floor_in - 6
    # Chunked processing matches one pass over the whole thing.
    monkeypatch.setattr(enhance, "CHUNK", 2.5)
    chunked = enhance.denoise(x, 20.0)
    err = np.sum((chunked - whole) ** 2) / np.sum(whole ** 2)
    assert 10 * np.log10(err) < -20
