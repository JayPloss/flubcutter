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
MASTER_LUFS = -20.0       # master level: near the final one, so it plays (and
                          # transcribes) like the finished chapter
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
VERSION = f"2:{MODEL}:{MIN_ATTEN_DB}-{MAX_ATTEN_DB}/{FLOOR_GAP}:{SPEECH_DB}:{EQ}:{MASTER_LUFS}"


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


def polish(audio: np.ndarray) -> np.ndarray:
    """Highpass, presence, de-ess and compress. The compressor's threshold
    is fixed, so the speech is first brought to a set level (SPEECH_DB)."""
    gain = 10 ** ((SPEECH_DB - speech_level(audio)) / 20)
    out = _fit(_ffmpeg(audio * gain, EQ), len(audio))
    # Then to the master level, short of clipping (the master is saved as 16-bit).
    lufs, peak = measure(out)["lufs"], float(np.max(np.abs(out), initial=0.0))
    if not np.isfinite(lufs) or peak == 0:
        return out
    return out * np.float32(min(10 ** ((MASTER_LUFS - lufs) / 20), 0.9 / peak))


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
