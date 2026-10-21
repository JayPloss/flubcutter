"""Make a home recording sound like a studio one, without changing a word.

Two passes, at either end of the pipeline:

1. **Master** (`enhance`): each original recording is cleaned once and cached;
   cutting and rendering then work on this master. (Transcription still uses
   the original: the speech model was trained on real-world audio and gains
   nothing from cleaning.)
   - Denoise with DPDFNet (`dpdfnet2_48khz_hr`, ONNX, CPU), a *masking*
     denoiser: it turns time-frequency cells of the recording down, it never
     generates audio, so it can't invent or swap words. The reduction is
     limited by blending the dry signal back in: only a few dB come off
     (FLOOR_GAP), so the natural room tone survives: total silence between words sounds dead and
     makes every cut audible, and edit.py takes its pause fill from that room
     tone.
   - Level the speech as it goes (`level_curve`): a slow gain that follows
     the reading minute to minute, so a passage read close to the iPad and
     one read leaning back come out alike, and one shout can't set the level
     for the rest of the chapter.
   - Then ffmpeg: a highpass for rumble and handling noise, a light presence
     lift for clarity, a de-esser (the lift brings up sibilance too), and a
     gentle 2:1 compressor so soft and loud passages sit closer together.
   The master keeps the original's timing to the sample, so word timestamps
   from the original transcript still line up.
2. **Loudness** (`loudness`), last, on the rendered chapter: a single gain to
   the target loudness, with a true-peak limiter only for the few peaks
   that would go over the ceiling (ACX: -23 to -18 dB RMS, peaks below
   -3 dB, noise floor below -60 dB). Mono stays mono.

The model (~10 MB) downloads on first use.
"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from flubcutter import edit

RATE = edit.RATE          # 48 kHz, the denoiser's native rate
MODEL = "dpdfnet2_48khz_hr"
# Noise reduction is light on purpose: listeners get used to a little room
# noise at once, but heavy denoising makes the voice thin and watery, which
# they never do. It is only as strong as needed to leave the noise floor
# FLOOR_GAP dB below the speech's loudness, within these limits; the rest of
# the noise is the dry signal blended back in.
MIN_ATTEN_DB = 4.0
MAX_ATTEN_DB = 10.0
FLOOR_GAP = 40.0

# Denoising runs in chunks in parallel. The model is recurrent and normalises
# its input with running averages, so each chunk starts PRE_ROLL early (that
# output is thrown away) and neighbours overlap by XFADE.
CHUNK = 30.0
PRE_ROLL = 3.0
XFADE = 0.05

SPEECH_DB = -20.0         # speech level going into the compressor, dBFS
LEVEL_BLOCK = 3.0         # the speech level is measured in blocks this long, s
LEVEL_WINDOW = 30.0       # and follows the median over this long, s: slow
                          # enough not to pump, quick enough for a new seat
LEVEL_RANGE = 24.0        # most the leveling turns a passage up or down, dB
RIDE_FRAME = 0.1          # then a fast rider holds each second of speech there,
RIDE_WINDOW = 1.0         # s: its level is the power of the voiced frames
RIDE_LOOK = 0.4           # within RIDE_WINDOW; it starts turning down this
                          # early, so a loud passage never comes through first
RIDE_DOWN = 15.0          # most it turns a second down, dB
RIDE_UP = 6.0             # and up (little, so word tails and breaths aren't raised)
MASTER_LUFS = -20.0       # master level: near the final one, so it plays (and
                          # transcribes) like the finished chapter
MASTER_PEAK = 0.9         # a limiter keeps the master's few loud peaks under this
EQ = ",".join([
    "highpass=f=80:poles=2",
    "equalizer=f=4000:t=q:w=0.9:g=1.5",             # presence
    "deesser=i=0.4:m=0.5:f=0.5:s=o",
    "acompressor=threshold=-24dB:ratio=2:attack=15:release=200:knee=4",
])

TARGET_LUFS = -19.0
TRUE_PEAK = -3.0          # dBTP ceiling
LIMIT_MARGIN = 0.5        # limiter set this far below the ceiling, dB

# Part of the cache key: change it whenever the master's processing changes.
VERSION = (f"4:{MODEL}:{MIN_ATTEN_DB}-{MAX_ATTEN_DB}/{FLOOR_GAP}:{SPEECH_DB}:"
           f"{LEVEL_BLOCK}/{LEVEL_WINDOW}/{LEVEL_RANGE}:"
           f"{RIDE_FRAME}/{RIDE_WINDOW}/{RIDE_LOOK}/{RIDE_DOWN}/{RIDE_UP}:{EQ}:{MASTER_LUFS}/{MASTER_PEAK}")


# ---------- denoise ----------

@cache
def _runtime():
    from dpdfnet.models import resolve_model
    from dpdfnet.onnx_backend import build_runtime_model

    path = resolve_model(model=MODEL, onnx_path=None, auto_download=True, verbose=False).onnx_path
    return build_runtime_model(path)


def _denoise_chunk(x: np.ndarray, atten_db: float) -> np.ndarray:
    """DPDFNet on one stretch of audio, same length and timing out as in."""
    from dpdfnet.audio import (apply_attn_limit, fit_length, make_stft_config,
                               postprocess_spec, preprocess_waveform)
    from dpdfnet.onnx_backend import infer_win_len

    rt = _runtime()
    cfg = make_stft_config(infer_win_len(rt.session, RATE))
    spec = preprocess_waveform(np.pad(x, (0, cfg.win_len)), cfg)
    state, frames = rt.init_state.copy(), []
    for t in range(spec.shape[1]):
        out, state = rt.session.run(
            [rt.out_spec_name, rt.out_state_name],
            {rt.in_spec_name: np.ascontiguousarray(spec[:, t:t + 1]), rt.in_state_name: state})
        frames.append(out)
    # The attenuation limit blends the dry spectrum back in (aligned for the
    # model's latency, which postprocess_spec then removes).
    spec_e = apply_attn_limit(spec, np.concatenate(frames, axis=1), atten_db)
    return fit_length(postprocess_spec(spec_e, cfg), len(x))


def reduction(audio: np.ndarray) -> float:
    """How much noise reduction this recording needs, in dB (see FLOOR_GAP)."""
    gap = measure(audio)["lufs"] - edit.Acoustics(audio).floor
    return float(np.clip(FLOOR_GAP - gap, MIN_ATTEN_DB, MAX_ATTEN_DB)) if np.isfinite(gap) else MIN_ATTEN_DB


def denoise(audio: np.ndarray, atten_db: float = MAX_ATTEN_DB, workers: int | None = None,
            progress=None) -> np.ndarray:
    """DPDFNet over a whole recording, in parallel chunks joined by linear
    crossfades (both sides are estimates of the same signal, so they add up
    coherently)."""
    import os

    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    step, pre, xf = int(CHUNK * RATE), int(PRE_ROLL * RATE), int(XFADE * RATE)
    if n == 0:
        return audio.copy()
    starts = list(range(0, n, step))
    if len(starts) > 1 and n - starts[-1] < RATE:  # no tiny last chunk
        starts.pop()
    bounds = list(zip(starts, starts[1:] + [n]))

    def run(b):
        s, e = b
        a, z = max(0, s - pre - xf), min(n, e + xf)
        return a, _denoise_chunk(audio[a:z], atten_db)

    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    workers = workers or max(1, min(len(bounds), (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(workers) as pool:  # onnxruntime releases the GIL
        for i, ((s, e), (a, y)) in enumerate(zip(bounds, pool.map(run, bounds))):
            # Weight: 1 over the chunk itself, ramping over xf on each side.
            lo, hi = max(0, s - xf), min(n, e + xf)
            w = np.ones(hi - lo, dtype=np.float32)
            if s > 0:
                w[:2 * xf] = np.linspace(0, 1, 2 * xf, dtype=np.float32)
            if e < n:
                w[-2 * xf:] = np.minimum(w[-2 * xf:], np.linspace(1, 0, 2 * xf, dtype=np.float32))
            out[lo:hi] += w * y[lo - a:hi - a]
            weight[lo:hi] += w
            if progress:
                progress((i + 1) / len(bounds))
    return out / np.maximum(weight, 1e-6)


# ---------- EQ and dynamics ----------

def _ffmpeg(audio: np.ndarray, filters: str, capture_log: bool = False):
    """Run mono float audio through an ffmpeg filter chain at RATE."""
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "info" if capture_log else "error",
         "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-i", "-",
         "-af", filters, "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
        input=np.asarray(audio, dtype="<f4").tobytes(), check=True, capture_output=True,
    )
    out = np.frombuffer(p.stdout, dtype=np.float32)
    return (out, p.stderr.decode(errors="replace")) if capture_log else out


def speech_level(audio: np.ndarray) -> float:
    """Average level of the speech (frames well above the noise floor), dBFS."""
    ac = edit.Acoustics(audio)
    loud = ac.db[ac.db > ac.floor + 15]
    if not len(loud):
        return float(np.max(ac.db))
    return float(10 * np.log10(np.mean(10 ** (loud / 10))))


def level_curve(audio: np.ndarray) -> np.ndarray:
    """Per-sample gain that brings the speech to SPEECH_DB, moving slowly.

    The speech level is the median of the voiced frames in each LEVEL_BLOCK
    (a median, so a shout or a bang barely moves it), then the median over
    LEVEL_WINDOW, smoothed; blocks with no speech take their neighbours'
    level, so pauses and room tone aren't pulled up."""
    ac = edit.Acoustics(audio)
    per = max(1, int(LEVEL_BLOCK / edit.HOP))
    n = len(ac.db) // per
    if n == 0:
        return np.full(len(audio), 10 ** ((SPEECH_DB - speech_level(audio)) / 20), np.float32)
    blocks = ac.db[:n * per].reshape(n, per)
    level = np.full(n, np.nan)
    for i, b in enumerate(blocks):
        voiced = b[b > ac.floor + 15]
        if len(voiced) > per // 10:
            level[i] = np.median(voiced)
    if np.isnan(level).all():
        return np.full(len(audio), 10 ** ((SPEECH_DB - speech_level(audio)) / 20), np.float32)
    w = max(1, int(LEVEL_WINDOW / LEVEL_BLOCK) // 2)
    med = np.array([np.nanmedian(level[max(0, i - w):i + w + 1]) if not np.isnan(level[max(0, i - w):i + w + 1]).all()
                    else np.nan for i in range(n)])
    have = ~np.isnan(med)
    med = np.interp(np.arange(n), np.flatnonzero(have), med[have])
    k = np.hanning(2 * w + 3)[1:-1]
    smooth = np.convolve(np.pad(med, w, mode="edge"), k / k.sum(), mode="valid")[:n]
    gain_db = np.clip(SPEECH_DB - smooth, -LEVEL_RANGE, LEVEL_RANGE)
    centres = (np.arange(n) + 0.5) * per * edit.HOP * RATE
    return (10 ** (np.interp(np.arange(len(audio)), centres, gain_db) / 20)).astype(np.float32)


def ride(audio: np.ndarray) -> np.ndarray:
    """Per-sample gain that holds each second of speech at SPEECH_DB.

    `level_curve` follows the reading over half a minute, so a loud stretch
    of a few seconds (a voice, a laugh, leaning in) came through as a surge.
    This follows the voiced power over RIDE_WINDOW instead, mostly turning
    down; the gain is the lowest within RIDE_LOOK either side, so it is
    already down when the loud part starts, and it holds through pauses."""
    ac = edit.Acoustics(audio)
    per = max(1, int(round(RIDE_FRAME / edit.HOP)))
    n = len(ac.db) // per
    if n == 0:
        return np.ones(len(audio), np.float32)
    power = (10 ** (ac.db[:n * per] / 10)).reshape(n, per).mean(1)
    voiced = 10 * np.log10(power + 1e-12) > ac.floor + 15
    w = int(RIDE_WINDOW / RIDE_FRAME) // 2
    sums = np.concatenate([[0], np.cumsum(np.where(voiced, power, 0))])
    counts = np.concatenate([[0], np.cumsum(voiced)])
    lo, hi = np.clip(np.arange(n) - w, 0, n), np.clip(np.arange(n) + w + 1, 0, n)
    count = counts[hi] - counts[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.where(count >= 3, 10 * np.log10((sums[hi] - sums[lo]) / count + 1e-12), np.nan)
    gain_db = np.clip(SPEECH_DB - level, -RIDE_DOWN, RIDE_UP)
    have = ~np.isnan(gain_db)
    if not have.any():
        return np.ones(len(audio), np.float32)
    gain_db = np.interp(np.arange(n), np.flatnonzero(have), gain_db[have])
    look = int(RIDE_LOOK / RIDE_FRAME)
    gain_db = np.lib.stride_tricks.sliding_window_view(np.pad(gain_db, look, mode="edge"), 2 * look + 1).min(1)
    k = np.hanning(2 * look + 3)[1:-1]
    gain_db = np.convolve(np.pad(gain_db, look, mode="edge"), k / k.sum(), mode="valid")[:n]
    centres = (np.arange(n) + 0.5) * per * edit.HOP * RATE
    return (10 ** (np.interp(np.arange(len(audio)), centres, gain_db) / 20)).astype(np.float32)


def polish(audio: np.ndarray) -> np.ndarray:
    """Level, highpass, presence, de-ess and compress. The compressor's
    threshold is fixed, so the speech is first brought to a set level
    (SPEECH_DB), followed minute to minute by `level_curve` and second to
    second by `ride`."""
    audio = audio * level_curve(audio)
    gain = ride(audio)
    # The de-esser goes unstable on samples past full scale (a decoded shout
    # can reach 3x), so those are clipped first, as a recorder would have.
    out = _fit(_ffmpeg(np.clip(audio * gain, -1.0, 1.0), EQ), len(audio))
    # Then to the master level. One shout or bang can peak 30 dB over the
    # speech, so scaling until that peak fits would bury the reading; a
    # limiter takes those few peaks down instead (the master is saved as 16-bit).
    lufs, peak = measure(out)["lufs"], float(np.max(np.abs(out), initial=0.0))
    if not np.isfinite(lufs) or peak == 0:
        return out
    gain = np.float32(10 ** ((MASTER_LUFS - lufs) / 20))
    out = out * gain
    if peak * gain > MASTER_PEAK:
        out = _fit(_ffmpeg(out, f"aresample=192000,alimiter=limit={MASTER_PEAK}:attack=1:"
                                f"release=60:level=disabled,aresample={RATE}"), len(audio))
    return np.clip(out, -1.0, 1.0)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """x cut or zero-padded to n samples."""
    return x[:n] if len(x) >= n else np.concatenate([x, np.zeros(n - len(x), np.float32)])


def enhance(audio: np.ndarray, rate: int = RATE, progress=None) -> np.ndarray:
    """The working master: denoised, EQ'd and compressed, same length and
    timing as the input (mono float32 at RATE)."""
    if rate != RATE:
        raise ValueError(f"enhance works at {RATE} Hz, got {rate}")
    clean = denoise(audio, reduction(audio), progress=(lambda f: progress(0.95 * f)) if progress else None)
    out = polish(clean)
    if progress:
        progress(1.0)
    return out


# ---------- loudness ----------

def measure(audio: np.ndarray) -> dict:
    """Integrated loudness (LUFS) and true peak (dBTP), per EBU R128."""
    _, log = _ffmpeg(audio, "loudnorm=print_format=json", capture_log=True)
    m = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    return {"lufs": float(m["input_i"]), "true_peak": float(m["input_tp"])}


def loudness(audio: np.ndarray, target: float = TARGET_LUFS, peak: float = TRUE_PEAK) -> np.ndarray:
    """Bring a finished chapter to `target` LUFS with true peaks under `peak`.

    One gain for the whole file (so the noise floor moves with the speech and
    nothing pumps); a limiter, run oversampled so it catches peaks between
    samples, touches only the few peaks that gain would push over."""
    m = measure(audio)
    if not np.isfinite(m["lufs"]):  # silence
        return np.asarray(audio, dtype=np.float32)
    gain_db = target - m["lufs"]
    out = np.asarray(audio, dtype=np.float32) * np.float32(10 ** (gain_db / 20))
    if m["true_peak"] + gain_db > peak - 0.1:
        limit = 10 ** ((peak - LIMIT_MARGIN) / 20)
        out = _fit(_ffmpeg(out, f"aresample=192000,alimiter=limit={limit:.4f}:attack=1:"
                                       f"release=60:level=disabled,aresample={RATE}"), len(audio))
    return out


# ---------- cache ----------

def master_path(src: Path, cache_dir: Path) -> Path:
    """Where the master of recording `src` is cached: keyed on the file
    (path, size, modification time) and the processing settings."""
    st = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{VERSION}".encode()).hexdigest()[:12]
    return cache_dir / f"{src.stem}.{key}.master.wav"


def master(src: Path, cache_dir: Path, progress=None) -> np.ndarray:
    """The enhanced master of recording `src` (48 kHz mono), made once and
    cached in `cache_dir`; stale masters of the same recording are removed."""
    out = master_path(src, cache_dir)
    if not out.exists():
        audio = enhance(edit.load(src), RATE, progress)
        for old in cache_dir.glob(f"{src.stem}.{'?' * 12}.master.wav"):
            old.unlink(missing_ok=True)
        tmp = out.with_suffix(".tmp.wav")
        edit.write_wav(audio, tmp)
        tmp.replace(out)
    return edit.load(out)
