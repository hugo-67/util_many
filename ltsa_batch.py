#!/usr/bin/env python3
"""High-throughput Long-Term Spectral Average (LTSA) batch processor.

This program is the batch companion to ``audio_annotatorv2.html``.  It reads
WAV/RF64/BW64/FLAC recordings in bounded blocks, optionally anti-aliases and
downsamples them *before* the FFT, computes a numeric LTSA matrix, and writes
the portable ``.ltsa`` format understood by the browser viewer.

The implementation is designed for very large passive-acoustic-monitoring
archives:

* automatic anti-aliased low-rate analysis for low-frequency targets;
* adaptive RAM budgeting with headroom for the OS and other applications;
* parallelism across files plus threaded batched FFTs within each CPU worker;
* optional CuPy/CUDA resampling and FFTs with a hard GPU-memory limit;
* optional Omura's-whale candidate surfacing from the LTSA (screening only);
* atomic outputs, resumable runs, manifests, CSV candidates, and Audacity labels.

CPU quick start (nothing permanently installed):

    uv run --with numpy --with scipy --with soundfile --with psutil \\
      python ltsa_batch.py recordings/ --recursive \\
      --config sample/ltsa_omuras_whale_fast_scan.json --output-dir ltsa_out

CUDA quick start (choose the CuPy wheel matching the installed CUDA runtime):

    uv run --with numpy --with scipy --with soundfile --with psutil \\
      --with cupy-cuda12x python ltsa_batch.py recordings/ --recursive \\
      --config sample/ltsa_omuras_whale_fast_scan.json \\
      --backend auto --jobs 1 --output-dir ltsa_out

``.ltsa`` version 1 layout:

* 8 bytes: ``b"LTSA1\\0\\0\\0"``
* 4 bytes: little-endian unsigned JSON-header length
* N bytes: UTF-8 JSON header
* remaining bytes: little-endian float32 matrix, time-major
  ``[time_bin, frequency_bin]``

Important scientific limitation
--------------------------------
Candidate rows are *potential* call-like events, not species identifications.
Ship noise, flow noise, seismic pulses, electrical hum, and other baleen whales
can overlap 10--100 Hz.  Verify candidates in the native-rate spectrogram and
look for the species' stereotyped 8--16 s phrase and roughly 2--3 min rhythm.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Sequence

MAGIC = b"LTSA1\0\0\0"
FORMAT_NAME = "util_many_ltsa"
CONFIG_FORMAT = "util_many_ltsa_config"
VERSION = 1
AUDIO_SUFFIXES = {".wav", ".wave", ".flac"}
MIB = 1024 * 1024
GIB = 1024 * MIB

# Browser-compatible values remain accepted.  The batch program additionally
# permits larger powers of two when a user explicitly needs them.
SUPPORTED_FFT = tuple(2**power for power in range(5, 21))

DEFAULT_CONFIG: dict[str, Any] = {
    "format": CONFIG_FORMAT,
    "version": VERSION,
    "analysis": {
        "fft_size": 8192,
        "window": "hann",
        "overlap": 0.0,
        "average_seconds": 60.0,
        "channel": 0,
        "frequency_min_hz": 0.0,
        "frequency_max_hz": None,
    },
    "processing": {
        # None preserves the source rate.  For Omura's-whale work, 256 or
        # 512 Hz is normally sufficient and is vastly faster than 48--384 kHz.
        "analysis_sample_rate_hz": None,
        "backend": "auto",
        "memory_fraction": 0.75,
        "gpu_memory_fraction": 0.75,
        "chunk_seconds": None,
        "fft_batch_windows": None,
    },
    "display": {
        "db_min": -120.0,
        "db_max": 0.0,
        "colormap": "ember",
    },
    "detection": {
        "enabled": False,
        "profile": None,
    },
}

OMURAS_DETECTION_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "profile": "omuras_whale",
    "primary_band_hz": [13.0, 72.0],
    "peak_band_hz": [20.0, 45.0],
    "lower_sideband_hz": [5.0, 12.0],
    "upper_sideband_hz": [80.0, 115.0],
    "min_band_snr_db": 2.5,
    "min_above_baseline_db": 3.0,
    "baseline_window_seconds": 300.0,
    "min_duration_seconds": 5.0,
    "max_duration_seconds": 24.0,
    "bridge_gap_seconds": 2.5,
    "ici_min_seconds": 110.0,
    "ici_max_seconds": 240.0,
    "minimum_sequence_calls": 2,
}


@dataclasses.dataclass(frozen=True)
class Plan:
    """Validated numerical plan for one recording.

    ``sample_rate`` and ``frames`` intentionally retain their historic names
    for compatibility with older callers; they refer to the *analysis-rate*
    stream after optional downsampling.  Original metadata is stored separately.
    """

    fft_size: int
    hop: int
    average_seconds: float
    channel: int
    frequency_min_hz: float
    frequency_max_hz: float
    first_frequency_bin: int
    frequency_bins: int
    time_bins: int
    sample_rate: int
    frames: int
    channels: int
    source_sample_rate: int
    source_frames: int
    duration_seconds: float
    resample_up: int
    resample_down: int
    fft_windows: int


@dataclasses.dataclass(frozen=True)
class ResourcePlan:
    """Host-level execution plan shared by all jobs in one invocation."""

    backend: str
    jobs: int
    threads_per_worker: int
    total_cpu_threads: int
    available_memory_bytes: int
    memory_budget_bytes: int
    memory_per_worker_bytes: int
    memory_fraction: float
    gpu_memory_fraction: float


@dataclasses.dataclass
class Candidate:
    """One call-like interval surfaced by the optional screening heuristic."""

    start_seconds: float
    end_seconds: float
    peak_seconds: float
    duration_seconds: float
    score: float
    band_snr_db: float
    above_baseline_db: float
    peak_band_db: float
    peak_frequency_hz: float
    rhythmic_match: bool = False
    previous_ici_seconds: float | None = None
    next_ici_seconds: float | None = None
    mains_line_flag: bool = False


@dataclasses.dataclass
class ComputeResult:
    """Internal result bundle; public ``compute_ltsa`` still returns a tuple."""

    plan: Plan
    matrix: Any
    backend: str
    elapsed_seconds: float
    chunk_source_frames: int
    fft_batch_windows: int
    candidates: list[Candidate]


def _deps():
    """Import mandatory numerical packages lazily.

    Lazy imports let ``--help`` and configuration inspection work even before
    dependencies are installed and prevent parent processes from initializing
    large numerical runtimes before worker thread limits are applied.
    """

    try:
        import numpy as np
        import soundfile as sf
        import scipy.fft as sp_fft
        import scipy.ndimage as ndimage
        import scipy.signal as sp_signal
    except ImportError as exc:  # pragma: no cover - exercised by user setups
        raise SystemExit(
            "ltsa_batch.py needs numpy, scipy, and soundfile. Run it with:\n"
            "  uv run --with numpy --with scipy --with soundfile --with psutil "
            "python ltsa_batch.py ..."
        ) from exc
    return np, sf, sp_fft, sp_signal, ndimage


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a validated-shape configuration with defaults recursively merged."""

    base = json.loads(json.dumps(DEFAULT_CONFIG))
    if config:
        if config.get("format") not in (None, CONFIG_FORMAT):
            raise ValueError(f"Unsupported config format: {config.get('format')!r}")
        if int(config.get("version", VERSION)) != VERSION:
            raise ValueError(f"Unsupported LTSA config version: {config.get('version')!r}")
        _deep_merge(base, config)
    base["format"] = CONFIG_FORMAT
    base["version"] = VERSION
    if base.get("detection", {}).get("profile") == "omuras_whale":
        detection = json.loads(json.dumps(OMURAS_DETECTION_DEFAULTS))
        _deep_merge(detection, base.get("detection", {}))
        base["detection"] = detection
    return base


def load_config(path: Path | None) -> dict[str, Any]:
    """Load JSON configuration or return defaults when no path is supplied."""

    if path is None:
        return merged_config(None)
    return merged_config(json.loads(path.read_text(encoding="utf-8")))


def _positive_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def make_plan(config: dict[str, Any], sample_rate: int, frames: int, channels: int) -> Plan:
    """Validate one file and derive its downsampling, FFT, and output dimensions."""

    if sample_rate <= 0 or frames <= 0 or channels <= 0:
        raise ValueError("Audio metadata must contain positive rate, frames, and channels")
    cfg = merged_config(config)
    analysis = cfg["analysis"]
    processing = cfg.get("processing", {})

    fft_size = int(analysis["fft_size"])
    if fft_size not in SUPPORTED_FFT:
        raise ValueError(f"fft_size must be a power of two in {SUPPORTED_FFT[0]}..{SUPPORTED_FFT[-1]}")
    if str(analysis.get("window", "hann")).lower() != "hann":
        raise ValueError("Only the Hann window is supported so CPU, CUDA, and browser values stay comparable")
    overlap = float(analysis["overlap"])
    if not 0.0 <= overlap < 0.95:
        raise ValueError("overlap must be >= 0 and < 0.95")
    average_seconds = _positive_finite(analysis["average_seconds"], "average_seconds")
    channel = int(analysis["channel"])
    if channel < -1 or channel >= channels:
        raise ValueError(f"channel must be -1 (mix) or 0..{channels - 1}")

    requested_rate = processing.get("analysis_sample_rate_hz")
    if requested_rate in (None, "", 0):
        analysis_rate = int(sample_rate)
    else:
        analysis_rate = int(round(_positive_finite(requested_rate, "analysis_sample_rate_hz")))
        if analysis_rate > sample_rate:
            # Upsampling cannot add information and wastes compute/RAM.
            analysis_rate = int(sample_rate)

    ratio = Fraction(analysis_rate, sample_rate)
    up, down = ratio.numerator, ratio.denominator
    analysis_frames = int(math.ceil(frames * up / down))
    duration = frames / sample_rate
    nyquist = analysis_rate / 2.0
    fmin = max(0.0, float(analysis.get("frequency_min_hz", 0.0) or 0.0))
    raw_max = analysis.get("frequency_max_hz")
    fmax = nyquist if raw_max is None else min(nyquist, float(raw_max))
    if not math.isfinite(fmin) or not math.isfinite(fmax) or fmax <= fmin:
        raise ValueError(
            f"Invalid stored frequency band {fmin:g}..{fmax:g} Hz for analysis Nyquist {nyquist:g} Hz"
        )
    # Leave at least a small anti-alias transition band when downsampling.
    if down > up and raw_max is not None and float(raw_max) > nyquist * 0.94:
        raise ValueError(
            f"frequency_max_hz={float(raw_max):g} is too close to the {nyquist:g} Hz analysis Nyquist; "
            "raise analysis_sample_rate_hz or lower the stored maximum"
        )

    hz_per_bin = analysis_rate / fft_size
    first = max(0, int(math.ceil(fmin / hz_per_bin - 1e-12)))
    last = min(fft_size // 2, int(math.floor(fmax / hz_per_bin + 1e-12)))
    frequency_bins = last - first + 1
    if frequency_bins <= 0:
        raise ValueError("The requested frequency range contains no FFT bins")
    time_bins = max(1, int(math.ceil(duration / average_seconds)))
    hop = max(1, int(round(fft_size * (1.0 - overlap))))
    # FFT frames are retained while their centre lies inside the recording.
    fft_windows = max(0, int(math.ceil(max(0.0, analysis_frames - fft_size / 2.0) / hop)))

    return Plan(
        fft_size=fft_size,
        hop=hop,
        average_seconds=average_seconds,
        channel=channel,
        frequency_min_hz=first * hz_per_bin,
        frequency_max_hz=last * hz_per_bin,
        first_frequency_bin=first,
        frequency_bins=frequency_bins,
        time_bins=time_bins,
        sample_rate=analysis_rate,
        frames=analysis_frames,
        channels=channels,
        source_sample_rate=sample_rate,
        source_frames=frames,
        duration_seconds=duration,
        resample_up=up,
        resample_down=down,
        fft_windows=fft_windows,
    )


def estimate_bytes(plan: Plan) -> int:
    """Bytes occupied by the final float32 LTSA matrix only."""

    return plan.time_bins * plan.frequency_bins * 4


def _affinity_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def available_memory_bytes() -> int:
    """Best-effort currently available RAM, not total physical RAM."""

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    # Conservative fallback when the platform exposes no availability API.
    return 4 * GIB


def _set_thread_limits(threads: int) -> None:
    """Prevent nested BLAS/OpenMP pools from oversubscribing CPU workers."""

    value = str(max(1, int(threads)))
    # SciPy FFT receives an explicit worker count.  Keep unrelated BLAS pools at
    # one thread because FFT batches and file-level processes already parallelize.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ["SCIPY_FFT_WORKERS"] = value


def resolve_resources(
    file_count: int,
    *,
    requested_backend: str,
    requested_jobs: int,
    requested_threads: int,
    memory_fraction: float,
    gpu_memory_fraction: float,
) -> ResourcePlan:
    """Choose file workers, FFT threads, and per-worker RAM without oversubscription."""

    if requested_backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")
    if not 0.1 <= memory_fraction <= 0.95:
        raise ValueError("memory_fraction must be between 0.1 and 0.95")
    if not 0.1 <= gpu_memory_fraction <= 0.95:
        raise ValueError("gpu_memory_fraction must be between 0.1 and 0.95")

    cpus = _affinity_cpu_count()
    available = available_memory_bytes()
    # Keep explicit headroom for the OS, filesystem cache, desktop, and transient
    # codec allocations.  This is deliberately based on *available* RAM.
    reserve = max(1 * GIB, int(available * 0.15))
    budget = max(512 * MIB, min(int(available * memory_fraction), max(512 * MIB, available - reserve)))

    # Enforce the RAM limit even when the user requests an aggressive worker
    # count.  Explicit overrides are preferences, never permission to overcommit.
    minimum_worker_memory = 512 * MIB
    jobs_by_ram = max(1, budget // minimum_worker_memory)
    if requested_backend == "cuda":
        jobs = 1
    elif requested_jobs > 0:
        jobs = max(1, min(file_count, requested_jobs, cpus, jobs_by_ram))
    else:
        jobs = max(1, min(file_count, cpus, jobs_by_ram))
    if requested_threads > 0:
        threads = min(cpus, requested_threads)
    else:
        threads = max(1, cpus // jobs)
    # Never permit total FFT workers to exceed the process affinity.
    threads = max(1, min(threads, max(1, cpus // jobs)))
    return ResourcePlan(
        backend=requested_backend,
        jobs=jobs,
        threads_per_worker=threads,
        total_cpu_threads=cpus,
        available_memory_bytes=available,
        memory_budget_bytes=budget,
        memory_per_worker_bytes=max(1, budget // jobs),
        memory_fraction=memory_fraction,
        gpu_memory_fraction=gpu_memory_fraction,
    )


class CPUBackend:
    """NumPy/SciPy execution backend with threaded batched real FFTs."""

    name = "cpu"

    def __init__(self, threads: int):
        np, _, sp_fft, sp_signal, _ = _deps()
        self.np = np
        self.sp_fft = sp_fft
        self.sp_signal = sp_signal
        self.threads = max(1, int(threads))

    def asarray(self, value):
        return self.np.asarray(value, dtype=self.np.float32)

    def zeros(self, length: int):
        return self.np.zeros(length, dtype=self.np.float32)

    def concatenate(self, arrays: Sequence[Any]):
        return self.np.concatenate(arrays)

    def resample(self, signal, up: int, down: int, fir):
        if up == down == 1:
            return self.asarray(signal)
        return self.sp_signal.resample_poly(
            self.asarray(signal), up, down, window=fir, padtype="constant"
        ).astype(self.np.float32, copy=False)

    def aggregate_fft(self, windows, window, factors, f0: int, f1: int, bins):
        np = self.np
        weighted = np.asarray(windows, dtype=np.float32) * window
        spectrum = self.sp_fft.rfft(
            weighted, axis=1, workers=self.threads, overwrite_x=True
        )
        amplitude = np.abs(spectrum) * factors
        powers = (amplitude[:, f0:f1] ** 2).astype(np.float32, copy=False)
        return _aggregate_contiguous_numpy(powers, bins)

    def to_numpy(self, value):
        return self.np.asarray(value)

    def synchronize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


class CUDABackend:
    """CuPy backend for GPU polyphase resampling and batched real FFTs."""

    name = "cuda"

    def __init__(self, gpu_memory_fraction: float):
        try:
            import cupy as cp
            import cupyx.scipy.signal as cp_signal
        except ImportError as exc:  # pragma: no cover - no CUDA in CI
            raise RuntimeError(
                "CUDA backend requested but CuPy is unavailable. Install the wheel "
                "matching your CUDA runtime, for example cupy-cuda12x."
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA backend requested but no CUDA device is visible")
        self.cp = cp
        self.cp_signal = cp_signal
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        # CUDA context/libraries allocate outside the pool, so cap the pool to a
        # fraction of currently free memory rather than total device memory.
        limit = int(free_bytes * gpu_memory_fraction)
        if limit < 256 * MIB:
            raise RuntimeError(
                f"Only {free_bytes / MIB:.1f} MiB of GPU memory is free; "
                "at least 256 MiB of bounded CuPy pool space is required"
            )
        cp.get_default_memory_pool().set_limit(size=limit)
        self.memory_limit = limit
        self._fir_cache: dict[int, Any] = {}
        # Warm the CUDA context and cuFFT once before timing/processing.
        cp.fft.rfft(cp.zeros(1024, dtype=cp.float32))
        cp.cuda.Stream.null.synchronize()

    def asarray(self, value):
        return self.cp.asarray(value, dtype=self.cp.float32)

    def zeros(self, length: int):
        return self.cp.zeros(length, dtype=self.cp.float32)

    def concatenate(self, arrays: Sequence[Any]):
        return self.cp.concatenate(arrays)

    def resample(self, signal, up: int, down: int, fir):
        if up == down == 1:
            return self.asarray(signal)
        key = id(fir)
        gpu_fir = self._fir_cache.get(key)
        if gpu_fir is None:
            gpu_fir = self.asarray(fir)
            self._fir_cache[key] = gpu_fir
        return self.cp_signal.resample_poly(
            self.asarray(signal), up, down, window=gpu_fir, padtype="constant"
        ).astype(self.cp.float32, copy=False)

    def aggregate_fft(self, windows, window, factors, f0: int, f1: int, bins):
        cp = self.cp
        weighted = windows.astype(cp.float32, copy=False) * window
        spectrum = cp.fft.rfft(weighted, axis=1)
        amplitude = cp.abs(spectrum) * factors
        powers = (amplitude[:, f0:f1] ** 2).astype(cp.float32, copy=False)
        # Group contiguous equal LTSA-bin indices on device, then transfer only
        # one float64 vector per LTSA bin rather than the full spectrogram.
        boundaries = [0]
        for i in range(1, len(bins)):
            if bins[i] != bins[i - 1]:
                boundaries.append(i)
        boundaries.append(len(bins))
        keys: list[int] = []
        vectors = []
        counts: list[int] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            keys.append(int(bins[left]))
            counts.append(right - left)
            vectors.append(cp.sum(powers[left:right], axis=0, dtype=cp.float64))
        stacked = cp.stack(vectors) if vectors else cp.empty((0, f1 - f0), dtype=cp.float64)
        host = cp.asnumpy(stacked)
        return [(key, host[i], counts[i]) for i, key in enumerate(keys)]

    def to_numpy(self, value):
        return self.cp.asnumpy(value)

    def synchronize(self) -> None:
        self.cp.cuda.Stream.null.synchronize()

    def cleanup(self) -> None:
        self._fir_cache.clear()
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _aggregate_contiguous_numpy(powers, bins) -> list[tuple[int, Any, int]]:
    """Sum chronological rows sharing the same LTSA time bin."""

    if len(bins) == 0:
        return []
    np = powers.__array_namespace__() if hasattr(powers, "__array_namespace__") else None
    # NumPy is guaranteed here; use its dtype keyword for stable accumulation.
    import numpy as real_np

    boundaries = real_np.flatnonzero(real_np.r_[True, bins[1:] != bins[:-1], True])
    out: list[tuple[int, Any, int]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        out.append(
            (
                int(bins[left]),
                real_np.sum(powers[left:right], axis=0, dtype=real_np.float64),
                int(right - left),
            )
        )
    return out


def _design_resample_filter(plan: Plan):
    """Create an explicit high-quality FIR shared by SciPy and CuPy.

    Explicit coefficients keep CPU/CUDA behavior comparable and let the block
    reader know exactly how much guard audio to discard at every chunk edge.
    """

    np, _, _, sp_signal, _ = _deps()
    if plan.resample_up == plan.resample_down == 1:
        return np.array([1.0], dtype=np.float32), 0
    max_rate = max(plan.resample_up, plan.resample_down)
    half_len = 12 * max_rate
    fir = sp_signal.firwin(
        2 * half_len + 1,
        1.0 / max_rate,
        window=("kaiser", 8.6),
    ).astype(np.float32)
    # Group delay is expressed on the upsampled grid.  Convert it back to input
    # samples and align the guard to the rational-resampling denominator so that
    # every core chunk maps to an exact integer output index.
    guard = int(math.ceil(half_len / plan.resample_up)) + plan.resample_down
    guard = int(math.ceil(guard / plan.resample_down) * plan.resample_down)
    return fir, guard


def _select_channel(block, channel: int):
    """Reduce a decoded 2-D block to one float32 analysis stream on the CPU."""

    np, _, _, _, _ = _deps()
    if channel == -1:
        return block.mean(axis=1, dtype=np.float32)
    return block[:, channel].astype(np.float32, copy=False)


def _read_padded(audio, start: int, stop: int, channel: int):
    """Read ``[start, stop)`` source frames, zero-padding beyond either EOF."""

    np, _, _, _, _ = _deps()
    length = stop - start
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    out = np.zeros(length, dtype=np.float32)
    actual_start = max(0, start)
    actual_stop = min(len(audio), stop)
    if actual_stop <= actual_start:
        return out
    audio.seek(actual_start)
    raw = audio.read(actual_stop - actual_start, dtype="float32", always_2d=True)
    selected = _select_channel(raw, channel)
    out[actual_start - start : actual_stop - start] = selected
    return out


def _choose_chunk_source_frames(
    plan: Plan,
    *,
    allocation_bytes: int,
    chunk_seconds: float | None,
) -> int:
    """Choose an aligned source block within its explicit RAM allocation.

    SoundFile first decodes all channels, then the selected/mixed float32 stream
    and its resampled counterpart coexist briefly.  The estimate therefore
    includes all three representations.  A requested ``chunk_seconds`` value is
    capped rather than trusted blindly, so convenience options cannot trigger an
    out-of-memory crash.
    """

    output_per_source_frame = 4.0 * plan.resample_up / plan.resample_down
    bytes_per_source_frame = max(1, plan.channels) * 4 + 4 + output_per_source_frame
    safe_frames = max(plan.resample_down, int(allocation_bytes / max(1.0, bytes_per_source_frame)))
    if chunk_seconds is not None:
        requested = max(plan.resample_down, int(float(chunk_seconds) * plan.source_sample_rate))
    else:
        requested = safe_frames
    # Several minutes amortise filesystem and codec overhead; larger monolithic
    # allocations add little throughput and make cancellation/recovery worse.
    requested = min(requested, safe_frames, int(plan.source_sample_rate * 600))
    requested = max(plan.resample_down, requested)
    return max(plan.resample_down, requested // plan.resample_down * plan.resample_down)

def _choose_fft_batch_windows(
    plan: Plan,
    *,
    allocation_bytes: int,
    requested: int | None,
) -> int:
    """Choose/cap the FFT batch within an explicit temporary-memory allocation."""

    # Approximate simultaneous bytes per row: weighted real input, complex rFFT,
    # amplitude, selected-band powers, plus workspace/copy slack.  The 1.35
    # multiplier deliberately covers allocator rounding and cuFFT/SciPy scratch.
    per_window = int(1.35 * (
        plan.fft_size * 4
        + (plan.fft_size // 2 + 1) * 8
        + (plan.fft_size // 2 + 1) * 4
        + plan.frequency_bins * 4
    ))
    safe = max(1, min(131072, allocation_bytes // max(1, per_window)))
    if requested is not None and int(requested) > 0:
        return max(1, min(int(requested), safe))
    return safe

def _iter_resampled_chunks(audio, plan: Plan, backend, fir, guard: int, core_source_frames: int):
    """Yield contiguous analysis-rate chunks with chunk-edge transients removed."""

    padded_source_frames = int(math.ceil(plan.source_frames / plan.resample_down) * plan.resample_down)
    trim_target = guard * plan.resample_up // plan.resample_down
    for core_start in range(0, padded_source_frames, core_source_frames):
        core_stop = min(padded_source_frames, core_start + core_source_frames)
        read_start = core_start - guard
        read_stop = core_stop + guard
        source = _read_padded(audio, read_start, read_stop, plan.channel)
        resampled = backend.resample(source, plan.resample_up, plan.resample_down, fir)
        core_target_length = (core_stop - core_start) * plan.resample_up // plan.resample_down
        core = resampled[trim_target : trim_target + core_target_length]
        target_start = core_start * plan.resample_up // plan.resample_down
        if target_start >= plan.frames:
            break
        keep = min(core_target_length, plan.frames - target_start)
        yield target_start, core[:keep]


class _LTSAAccumulator:
    """Chronological linear-power accumulator converted to dBFS on flush."""

    def __init__(self, plan: Plan):
        np, _, _, _, _ = _deps()
        self.plan = plan
        self.matrix = np.full((plan.time_bins, plan.frequency_bins), np.nan, dtype="<f4")
        self.current_bin = -1
        self.current_sum = np.zeros(plan.frequency_bins, dtype=np.float64)
        self.current_count = 0

    def add(self, bin_index: int, vector, count: int) -> None:
        if bin_index < 0 or bin_index >= self.plan.time_bins or count <= 0:
            return
        if self.current_bin != bin_index:
            self.flush()
            self.current_bin = bin_index
        self.current_sum += vector
        self.current_count += count

    def flush(self) -> None:
        np, _, _, _, _ = _deps()
        if self.current_bin >= 0 and self.current_count:
            mean_power = self.current_sum / self.current_count
            self.matrix[self.current_bin] = 10.0 * np.log10(np.maximum(mean_power, 1e-18))
        self.current_sum.fill(0.0)
        self.current_count = 0

    def finish(self):
        self.flush()
        return self.matrix


def _process_stream(
    audio,
    plan: Plan,
    backend,
    *,
    fir,
    guard: int,
    core_source_frames: int,
    fft_batch_windows: int,
):
    """Stream source blocks through resampling and batched STFT accumulation."""

    np, _, _, _, _ = _deps()
    accumulator = _LTSAAccumulator(plan)
    window_np = np.hanning(plan.fft_size).astype(np.float32)
    coherent = float(window_np.sum())
    factors_np = np.full(plan.fft_size // 2 + 1, 2.0 / coherent, dtype=np.float32)
    factors_np[0] = 1.0 / coherent
    if plan.fft_size % 2 == 0:
        factors_np[-1] = 1.0 / coherent
    window = backend.asarray(window_np)
    factors = backend.asarray(factors_np)
    f0 = plan.first_frequency_bin
    f1 = f0 + plan.frequency_bins

    buffer = backend.zeros(0)
    buffer_start = 0
    buffer_end = 0
    next_window_start = 0
    processed_windows = 0

    def consume_available() -> None:
        nonlocal buffer, buffer_start, next_window_start, processed_windows
        available_windows = max(0, (buffer_end - next_window_start - plan.fft_size) // plan.hop + 1)
        while available_windows > 0 and processed_windows < plan.fft_windows:
            count = min(available_windows, fft_batch_windows, plan.fft_windows - processed_windows)
            offset0 = next_window_start - buffer_start
            # A strided view avoids duplicating the source buffer.  Multiplication
            # by the Hann window creates the contiguous work array used by FFT.
            xp = backend.cp if backend.name == "cuda" else np
            view = xp.lib.stride_tricks.sliding_window_view(buffer, plan.fft_size)
            offsets = offset0 + xp.arange(count) * plan.hop
            windows = view[offsets]
            starts = next_window_start + np.arange(count, dtype=np.int64) * plan.hop
            centers = (starts + plan.fft_size / 2.0) / plan.sample_rate
            bins = np.floor(centers / plan.average_seconds).astype(np.int64)
            groups = backend.aggregate_fft(windows, window, factors, f0, f1, bins)
            for bin_index, vector, rows in groups:
                accumulator.add(bin_index, vector, rows)
            next_window_start += count * plan.hop
            processed_windows += count
            available_windows -= count
        discard = max(0, next_window_start - buffer_start)
        if discard:
            buffer = buffer[discard:]
            buffer_start += discard

    for target_start, chunk in _iter_resampled_chunks(
        audio, plan, backend, fir, guard, core_source_frames
    ):
        if len(buffer) == 0:
            buffer = chunk
            buffer_start = target_start
            buffer_end = target_start + len(chunk)
        else:
            if target_start != buffer_end:
                raise RuntimeError(
                    f"Internal resampling discontinuity: expected {buffer_end}, got {target_start}"
                )
            buffer = backend.concatenate((buffer, chunk))
            buffer_end += len(chunk)
        consume_available()

    # Zero-pad only enough to process final windows whose centres remain inside
    # the recording; this matches the historical browser/Python convention.
    remaining = plan.fft_windows - processed_windows
    if remaining > 0:
        needed_end = next_window_start + (remaining - 1) * plan.hop + plan.fft_size
        if needed_end > buffer_end:
            buffer = backend.concatenate((buffer, backend.zeros(needed_end - buffer_end)))
            buffer_end = needed_end
        consume_available()
    if processed_windows != plan.fft_windows:
        raise RuntimeError(f"Processed {processed_windows} FFT windows; expected {plan.fft_windows}")
    backend.synchronize()
    return accumulator.finish()


def _make_backend(name: str, threads: int, gpu_memory_fraction: float):
    if name == "cpu":
        return CPUBackend(threads)
    if name == "cuda":
        return CUDABackend(gpu_memory_fraction)
    raise ValueError(f"Unknown backend {name!r}")


def _benchmark_backend_choice(
    audio,
    plan: Plan,
    *,
    threads: int,
    gpu_memory_fraction: float,
    fir,
    cpu_parallel_jobs: int = 1,
) -> str:
    """Benchmark one representative block and choose CPU or CUDA.

    The benchmark is cached only for this file/process.  It excludes file I/O,
    because decoding is identical for both paths, and includes synchronization.
    CUDA must beat CPU by at least 10% to justify transfers and complexity.
    """

    if not _cuda_available():
        return "cpu"
    np, _, _, _, _ = _deps()
    probe_frames = min(plan.source_frames, max(plan.source_sample_rate * 10, plan.resample_down * 64))
    probe_frames = min(probe_frames, 4_000_000)
    audio.seek(0)
    raw = audio.read(probe_frames, dtype="float32", always_2d=True)
    probe = _select_channel(raw, plan.channel)
    if len(probe) < plan.resample_down * 2:
        return "cpu"

    cpu = CPUBackend(threads)
    start = time.perf_counter()
    y_cpu = cpu.resample(probe, plan.resample_up, plan.resample_down, fir)
    if len(y_cpu) >= plan.fft_size:
        rows = min(256, max(1, (len(y_cpu) - plan.fft_size) // plan.hop + 1))
        view = np.lib.stride_tricks.sliding_window_view(y_cpu, plan.fft_size)
        cpu.aggregate_fft(
            view[np.arange(rows) * plan.hop],
            np.hanning(plan.fft_size).astype(np.float32),
            np.ones(plan.fft_size // 2 + 1, dtype=np.float32),
            plan.first_frequency_bin,
            plan.first_frequency_bin + plan.frequency_bins,
            np.zeros(rows, dtype=np.int64),
        )
    cpu_seconds = time.perf_counter() - start

    try:
        cuda = CUDABackend(gpu_memory_fraction)
        start = time.perf_counter()
        y_gpu = cuda.resample(probe, plan.resample_up, plan.resample_down, fir)
        if len(y_gpu) >= plan.fft_size:
            rows = min(256, max(1, (len(y_gpu) - plan.fft_size) // plan.hop + 1))
            view = cuda.cp.lib.stride_tricks.sliding_window_view(y_gpu, plan.fft_size)
            cuda.aggregate_fft(
                view[cuda.cp.arange(rows) * plan.hop],
                cuda.asarray(np.hanning(plan.fft_size).astype(np.float32)),
                cuda.asarray(np.ones(plan.fft_size // 2 + 1, dtype=np.float32)),
                plan.first_frequency_bin,
                plan.first_frequency_bin + plan.frequency_bins,
                np.zeros(rows, dtype=np.int64),
            )
        cuda.synchronize()
        gpu_seconds = time.perf_counter() - start
        cuda.cleanup()
    except Exception:
        return "cpu"
    # For an archive, compare one GPU stream with the approximate aggregate
    # throughput of all planned CPU file workers, not merely one CPU worker.
    cpu_archive_equivalent = cpu_seconds / max(1, int(cpu_parallel_jobs))
    return "cuda" if gpu_seconds < cpu_archive_equivalent * 0.90 else "cpu"


def _band_db(matrix, frequencies, low: float, high: float):
    """Mean linear power within a band, returned in dB for every time bin.

    Empty/all-NaN rows remain NaN without emitting NumPy ``Mean of empty slice``
    warnings, which is common when an LTSA bin contains no centred FFT frame.
    """

    np, _, _, _, _ = _deps()
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        return np.full(matrix.shape[0], np.nan, dtype=np.float64)
    values = matrix[:, mask].astype(np.float64)
    finite = np.isfinite(values)
    power = np.zeros_like(values, dtype=np.float64)
    power[finite] = np.power(10.0, np.clip(values[finite], -300.0, 100.0) / 10.0)
    counts = finite.sum(axis=1)
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = 10.0 * np.log10(np.maximum(power[valid].sum(axis=1) / counts[valid], 1e-18))
    return out


def _finite_max(values, fallback: float = -math.inf) -> float:
    """Return a finite-aware maximum without all-NaN runtime warnings."""

    np, _, _, _, _ = _deps()
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return float(finite.max()) if finite.size else fallback

def surface_candidates(matrix, plan: Plan, config: dict[str, Any]) -> list[Candidate]:
    """Surface Omura-call-like intervals using robust LTSA-domain screening.

    This is intentionally a high-recall triage heuristic, not a classifier.  It
    combines target-band versus side-band contrast, a rolling local baseline,
    expected 5--24 s event duration, and optional 110--240 s rhythmic spacing.
    """

    cfg = merged_config(config).get("detection", {})
    if not cfg.get("enabled"):
        return []
    if cfg.get("profile") not in (None, "omuras_whale"):
        raise ValueError(f"Unsupported detection profile: {cfg.get('profile')!r}")
    np, _, _, _, ndimage = _deps()
    frequencies = plan.frequency_min_hz + np.arange(plan.frequency_bins) * (plan.sample_rate / plan.fft_size)

    primary = _band_db(matrix, frequencies, *map(float, cfg["primary_band_hz"]))
    peak = _band_db(matrix, frequencies, *map(float, cfg["peak_band_hz"]))
    lower = _band_db(matrix, frequencies, *map(float, cfg["lower_sideband_hz"]))
    upper = _band_db(matrix, frequencies, *map(float, cfg["upper_sideband_hz"]))
    # np.fmax selects the finite side when only one is stored and leaves NaN
    # only when both are absent, without an all-NaN reduction warning.
    side = np.fmax(lower, upper)
    # If only one side band exists in a narrowly stored LTSA, use it.  If neither
    # exists, disable sideband gating rather than silently rejecting everything.
    side_missing = ~np.isfinite(side)
    side[side_missing] = primary[side_missing] - float(cfg["min_band_snr_db"])
    band_snr = primary - side

    width = max(3, int(round(float(cfg["baseline_window_seconds"]) / plan.average_seconds)))
    if width % 2 == 0:
        width += 1
    available_primary = primary[np.isfinite(primary)]
    if not available_primary.size:
        return []
    finite_primary = np.where(np.isfinite(primary), primary, float(np.median(available_primary)))
    baseline = ndimage.median_filter(finite_primary, size=width, mode="nearest")
    excess = primary - baseline
    mask = (
        np.isfinite(primary)
        & (band_snr >= float(cfg["min_band_snr_db"]))
        & (excess >= float(cfg["min_above_baseline_db"]))
    )
    bridge_bins = max(1, int(round(float(cfg["bridge_gap_seconds"]) / plan.average_seconds)))
    mask = ndimage.binary_closing(mask, structure=np.ones(bridge_bins + 1, dtype=bool))
    labels, count = ndimage.label(mask)

    min_duration = float(cfg["min_duration_seconds"])
    max_duration = float(cfg["max_duration_seconds"])
    candidates: list[Candidate] = []
    for label_id in range(1, count + 1):
        indices = np.flatnonzero(labels == label_id)
        if indices.size == 0:
            continue
        start = indices[0] * plan.average_seconds
        end = min(plan.duration_seconds, (indices[-1] + 1) * plan.average_seconds)
        duration = end - start
        if duration < min_duration or duration > max_duration:
            continue
        local_values = primary[indices]
        if not np.isfinite(local_values).any():
            continue
        local = indices[int(np.nanargmax(local_values))]
        peak_time = (local + 0.5) * plan.average_seconds
        snr_value = _finite_max(band_snr[indices])
        excess_value = _finite_max(excess[indices])
        peak_value = _finite_max(peak[indices])

        peak_low, peak_high = map(float, cfg["peak_band_hz"])
        peak_frequency_mask = (frequencies >= peak_low) & (frequencies <= peak_high)
        peak_frequency = math.nan
        if np.any(peak_frequency_mask):
            segment = matrix[np.ix_(indices, np.flatnonzero(peak_frequency_mask))].astype(np.float64)
            finite_segment = np.isfinite(segment)
            counts = finite_segment.sum(axis=0)
            spectral_profile = np.full(segment.shape[1], np.nan, dtype=np.float64)
            valid_columns = counts > 0
            spectral_profile[valid_columns] = np.where(finite_segment, segment, 0.0)[:, valid_columns].sum(axis=0) / counts[valid_columns]
            if np.isfinite(spectral_profile).any():
                peak_frequency = float(frequencies[peak_frequency_mask][int(np.nanargmax(spectral_profile))])

        # Spectral-line flag catches common 50/60 Hz electrical contamination.
        line50 = _band_db(matrix[indices], frequencies, 49.0, 51.0)
        line60 = _band_db(matrix[indices], frequencies, 59.0, 61.0)
        line_level = _finite_max(np.r_[line50, line60])
        primary_level = _finite_max(primary[indices])
        candidates.append(
            Candidate(
                start_seconds=start,
                end_seconds=end,
                peak_seconds=peak_time,
                duration_seconds=duration,
                score=snr_value + excess_value + 0.25 * max(0.0, peak_value - primary[local]),
                band_snr_db=snr_value,
                above_baseline_db=excess_value,
                peak_band_db=peak_value,
                peak_frequency_hz=peak_frequency,
                mains_line_flag=line_level > primary_level + 3.0,
            )
        )

    ici_min = float(cfg["ici_min_seconds"])
    ici_max = float(cfg["ici_max_seconds"])
    minimum_calls = int(cfg.get("minimum_sequence_calls", 2))
    starts = [c.start_seconds for c in candidates]
    for i, candidate in enumerate(candidates):
        previous = starts[i] - starts[i - 1] if i else None
        following = starts[i + 1] - starts[i] if i + 1 < len(starts) else None
        candidate.previous_ici_seconds = previous
        candidate.next_ici_seconds = following
        matches = int(previous is not None and ici_min <= previous <= ici_max) + int(
            following is not None and ici_min <= following <= ici_max
        )
        candidate.rhythmic_match = matches >= max(1, minimum_calls - 1)
    return candidates


def compute_ltsa_detailed(
    path: Path,
    config: dict[str, Any],
    *,
    chunk_windows: int | None = None,
    max_output_mib: float = 4096.0,
    backend: str | None = None,
    threads: int = 1,
    memory_budget_bytes: int | None = None,
    gpu_memory_fraction: float | None = None,
    chunk_seconds: float | None = None,
) -> ComputeResult:
    """Compute one LTSA with adaptive downsampling, RAM use, and CPU/CUDA backend."""

    np, sf, _, _, _ = _deps()
    cfg = merged_config(config)
    processing = cfg.get("processing", {})
    requested_backend = backend or str(processing.get("backend", "auto"))
    gpu_fraction = float(
        gpu_memory_fraction if gpu_memory_fraction is not None else processing.get("gpu_memory_fraction", 0.75)
    )
    memory_budget = int(memory_budget_bytes or max(512 * MIB, available_memory_bytes() * 0.5))

    with sf.SoundFile(path) as audio:
        plan = make_plan(cfg, int(audio.samplerate), int(len(audio)), int(audio.channels))
        output_bytes = estimate_bytes(plan)
        if output_bytes > max_output_mib * MIB:
            needed = plan.duration_seconds * plan.frequency_bins * 4 / (max_output_mib * MIB)
            raise ValueError(
                f"Estimated LTSA matrix is {output_bytes / MIB:.1f} MiB, above the {max_output_mib:g} MiB limit. "
                f"Increase average_seconds to about {max(plan.average_seconds, needed):.3g}s or narrow the stored band."
            )
        if output_bytes > memory_budget * 0.45:
            raise MemoryError(
                f"LTSA output alone needs {output_bytes / MIB:.1f} MiB, exceeding 45% of this worker's "
                f"{memory_budget / MIB:.1f} MiB RAM budget. Increase average_seconds, narrow the band, "
                "reduce --jobs, or raise --memory-fraction."
            )

        fir, guard = _design_resample_filter(plan)
        chosen_backend = requested_backend
        if requested_backend == "auto":
            # GPU auto-selection only makes sense in a single-file worker; caller
            # controls that by setting jobs=1.  The benchmark uses real audio.
            chosen_backend = _benchmark_backend_choice(
                audio,
                plan,
                threads=threads,
                gpu_memory_fraction=gpu_fraction,
                fir=fir,
            )
        if chosen_backend == "cuda" and not _cuda_available():
            if requested_backend == "cuda":
                raise RuntimeError("CUDA requested but no compatible CuPy device is available")
            chosen_backend = "cpu"
        engine = _make_backend(chosen_backend, threads, gpu_fraction)

        dynamic_available = available_memory_bytes()
        safe_budget = min(memory_budget, int(dynamic_available * 0.80))
        fixed_overhead = max(96 * MIB, int(safe_budget * 0.08))
        working_budget = safe_budget - output_bytes - fixed_overhead
        if working_budget < 96 * MIB:
            raise MemoryError(
                f"Only {working_budget / MIB:.1f} MiB remains for decoding/resampling/FFTs after reserving "
                "the output and runtime overhead. Reduce --jobs, increase --memory-fraction, or use a coarser LTSA."
            )
        chunk_allocation = int(working_budget * 0.38)
        fft_allocation = working_budget - chunk_allocation
        if chosen_backend == "cuda":
            device_limit = int(getattr(engine, "memory_limit", 0))
            device_working = max(0, device_limit - 128 * MIB)
            if device_working < 128 * MIB:
                raise MemoryError("Insufficient bounded GPU memory remains after CUDA runtime headroom")
            chunk_allocation = min(chunk_allocation, int(device_working * 0.35))
            fft_allocation = min(fft_allocation, int(device_working * 0.55))

        chosen_chunk_seconds = chunk_seconds
        if chosen_chunk_seconds is None and processing.get("chunk_seconds") not in (None, ""):
            chosen_chunk_seconds = float(processing["chunk_seconds"])
        core_frames = _choose_chunk_source_frames(
            plan, allocation_bytes=chunk_allocation, chunk_seconds=chosen_chunk_seconds
        )
        requested_batch = chunk_windows
        if requested_batch is None and processing.get("fft_batch_windows") not in (None, ""):
            requested_batch = int(processing["fft_batch_windows"])
        batch_windows = _choose_fft_batch_windows(
            plan, allocation_bytes=fft_allocation, requested=requested_batch
        )

        audio.seek(0)
        started = time.perf_counter()
        try:
            matrix = _process_stream(
                audio,
                plan,
                engine,
                fir=fir,
                guard=guard,
                core_source_frames=core_frames,
                fft_batch_windows=batch_windows,
            )
            candidates = surface_candidates(matrix, plan, cfg)
        finally:
            engine.cleanup()
        elapsed = time.perf_counter() - started
        return ComputeResult(
            plan=plan,
            matrix=np.asarray(matrix, dtype="<f4", order="C"),
            backend=chosen_backend,
            elapsed_seconds=elapsed,
            chunk_source_frames=core_frames,
            fft_batch_windows=batch_windows,
            candidates=candidates,
        )


def compute_ltsa(
    path: Path,
    config: dict[str, Any],
    *,
    chunk_windows: int = 256,
    max_output_mib: float = 4096.0,
):
    """Backward-compatible API returning ``(Plan, matrix)``.

    Programmatic callers needing backend/resource/candidate metadata should use
    :func:`compute_ltsa_detailed`.
    """

    result = compute_ltsa_detailed(
        path,
        config,
        chunk_windows=chunk_windows,
        max_output_mib=max_output_mib,
        backend="cpu",
        threads=1,
    )
    return result.plan, result.matrix


def make_header(
    path: Path,
    config: dict[str, Any],
    plan: Plan,
    *,
    backend: str = "cpu",
    elapsed_seconds: float | None = None,
    chunk_source_frames: int | None = None,
    fft_batch_windows: int | None = None,
) -> dict[str, Any]:
    """Build a self-describing header for the portable LTSA matrix."""

    stat = path.stat()
    header: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": VERSION,
        "source": {
            "name": path.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "config": merged_config(config),
        # ``sample_rate`` remains the analysis rate for v1 compatibility.
        "sample_rate": plan.sample_rate,
        "analysis_sample_rate": plan.sample_rate,
        "source_sample_rate": plan.source_sample_rate,
        "channels": plan.channels,
        "source_frames": plan.source_frames,
        "analysis_frames": plan.frames,
        "duration_seconds": plan.duration_seconds,
        "time_bins": plan.time_bins,
        "frequency_bins": plan.frequency_bins,
        "time_bin_seconds": plan.average_seconds,
        "frequency_bin_hz": plan.sample_rate / plan.fft_size,
        "frequency_start_hz": plan.frequency_min_hz,
        "frequency_end_hz": plan.frequency_max_hz,
        "first_frequency_bin": plan.first_frequency_bin,
        "fft_size": plan.fft_size,
        "fft_hop_samples": plan.hop,
        "fft_windows": plan.fft_windows,
        "resample_up": plan.resample_up,
        "resample_down": plan.resample_down,
        "processing_backend": backend,
        "data_dtype": "float32_le",
        "data_order": "time_frequency",
        "quantity": "power_dbfs",
    }
    if elapsed_seconds is not None:
        header["processing_elapsed_seconds"] = elapsed_seconds
    if chunk_source_frames is not None:
        header["chunk_source_frames"] = chunk_source_frames
    if fft_batch_windows is not None:
        header["fft_batch_windows"] = fft_batch_windows
    return header


def write_ltsa(path: Path, header: dict[str, Any], data) -> None:
    """Atomically write a v1 LTSA file; incomplete jobs never replace output."""

    np, _, _, _, _ = _deps()
    raw_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    matrix = np.asarray(data, dtype="<f4", order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(MAGIC)
            out.write(struct.pack("<I", len(raw_header)))
            out.write(raw_header)
            matrix.tofile(out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def read_ltsa(path: Path):
    """Read and strictly validate one v1 LTSA file."""

    np, _, _, _, _ = _deps()
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("Not a util_many LTSA file")
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError("Truncated LTSA header")
        header_length = struct.unpack("<I", raw_length)[0]
        if header_length > 16 * MIB:
            raise ValueError("LTSA header is unreasonably large")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError("Truncated LTSA JSON header")
        header = json.loads(raw_header.decode("utf-8"))
        if header.get("format") != FORMAT_NAME or int(header.get("version", 0)) != VERSION:
            raise ValueError("Unsupported LTSA format/version")
        count = int(header["time_bins"]) * int(header["frequency_bins"])
        data = np.fromfile(stream, dtype="<f4", count=count)
        if data.size != count or stream.read(1):
            raise ValueError("LTSA payload size does not match header")
        return header, data.reshape((int(header["time_bins"]), int(header["frequency_bins"])))


def discover(inputs: list[Path], recursive: bool) -> list[Path]:
    """Discover supported audio files while retaining deterministic ordering."""

    found: list[Path] = []
    for item in inputs:
        item = item.expanduser()
        if item.is_file():
            if item.suffix.lower() in AUDIO_SUFFIXES:
                found.append(item.resolve())
            continue
        if item.is_dir():
            iterator = item.rglob("*") if recursive else item.glob("*")
            found.extend(
                candidate.resolve()
                for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_SUFFIXES
            )
            continue
        raise FileNotFoundError(item)
    return sorted(dict.fromkeys(found))


def _output_name(source: Path, seen: set[str]) -> str:
    """Avoid silent collisions when different directories contain equal stems."""

    base = source.stem + ".ltsa"
    if base not in seen:
        seen.add(base)
        return base
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    unique = f"{source.stem}.{digest}.ltsa"
    seen.add(unique)
    return unique


def _write_candidates(base: Path, source: Path, candidates: Sequence[Candidate]) -> tuple[Path, Path]:
    csv_path = base.with_suffix(".omuras_candidates.csv")
    txt_path = base.with_suffix(".omuras_candidates.txt")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "source",
                "start_seconds",
                "end_seconds",
                "peak_seconds",
                "duration_seconds",
                "score",
                "band_snr_db",
                "above_baseline_db",
                "peak_band_dbfs",
                "peak_frequency_hz",
                "rhythmic_match",
                "previous_ici_seconds",
                "next_ici_seconds",
                "mains_line_flag",
            ]
        )
        for row in candidates:
            writer.writerow(
                [
                    str(source),
                    f"{row.start_seconds:.6f}",
                    f"{row.end_seconds:.6f}",
                    f"{row.peak_seconds:.6f}",
                    f"{row.duration_seconds:.6f}",
                    f"{row.score:.4f}",
                    f"{row.band_snr_db:.4f}",
                    f"{row.above_baseline_db:.4f}",
                    f"{row.peak_band_db:.4f}",
                    "" if not math.isfinite(row.peak_frequency_hz) else f"{row.peak_frequency_hz:.4f}",
                    int(row.rhythmic_match),
                    "" if row.previous_ici_seconds is None else f"{row.previous_ici_seconds:.6f}",
                    "" if row.next_ici_seconds is None else f"{row.next_ici_seconds:.6f}",
                    int(row.mains_line_flag),
                ]
            )
    with txt_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in candidates:
            flags = []
            if row.rhythmic_match:
                flags.append("rhythmic")
            if row.mains_line_flag:
                flags.append("50/60Hz-line")
            suffix = ",".join(flags) if flags else "single"
            stream.write(
                f"{row.start_seconds:.6f}\t{row.end_seconds:.6f}\t"
                f"Omura_candidate score={row.score:.2f} snr={row.band_snr_db:.2f}dB "
                f"fpeak={row.peak_frequency_hz:.2f}Hz {suffix}\n"
            )
    return csv_path, txt_path


def _inspect_audio(path: Path):
    _, sf, _, _, _ = _deps()
    info = sf.info(path)
    return int(info.samplerate), int(info.frames), int(info.channels), str(info.format), str(info.subtype)


def _one(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point; all values are pickle-safe for spawn multiprocessing."""

    _set_thread_limits(int(job["threads"]))
    source = Path(job["input"])
    output = Path(job["output"])
    try:
        result = compute_ltsa_detailed(
            source,
            job["config"],
            chunk_windows=job.get("chunk_windows"),
            max_output_mib=float(job["max_output_mib"]),
            backend=str(job["backend"]),
            threads=int(job["threads"]),
            memory_budget_bytes=int(job["memory_budget_bytes"]),
            gpu_memory_fraction=float(job["gpu_memory_fraction"]),
            chunk_seconds=job.get("chunk_seconds"),
        )
        header = make_header(
            source,
            job["config"],
            result.plan,
            backend=result.backend,
            elapsed_seconds=result.elapsed_seconds,
            chunk_source_frames=result.chunk_source_frames,
            fft_batch_windows=result.fft_batch_windows,
        )
        write_ltsa(output, header, result.matrix)
        candidate_csv = candidate_txt = None
        if job.get("write_candidates") and job["config"].get("detection", {}).get("enabled"):
            candidate_csv, candidate_txt = _write_candidates(output, source, result.candidates)
        return {
            "input": str(source),
            "output": str(output),
            "status": "ok",
            "backend": result.backend,
            "elapsed_seconds": result.elapsed_seconds,
            "source_sample_rate": result.plan.source_sample_rate,
            "analysis_sample_rate": result.plan.sample_rate,
            "resample_ratio": f"{result.plan.resample_up}/{result.plan.resample_down}",
            "time_bins": result.plan.time_bins,
            "frequency_bins": result.plan.frequency_bins,
            "fft_windows": result.plan.fft_windows,
            "chunk_source_frames": result.chunk_source_frames,
            "fft_batch_windows": result.fft_batch_windows,
            "size_mib": output.stat().st_size / MIB,
            "candidate_count": len(result.candidates),
            "rhythmic_candidate_count": sum(candidate.rhythmic_match for candidate in result.candidates),
            "candidate_csv": str(candidate_csv) if candidate_csv else None,
            "candidate_audacity_txt": str(candidate_txt) if candidate_txt else None,
        }
    except Exception as exc:
        return {
            "input": str(source),
            "output": str(output),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def choose_archive_backend(
    first_file: Path,
    config: dict[str, Any],
    resources: ResourcePlan,
    gpu_memory_fraction: float,
) -> str:
    """Benchmark real audio and choose GPU only when it beats aggregate CPU throughput."""

    if not _cuda_available():
        return "cpu"
    _, sf, _, _, _ = _deps()
    with sf.SoundFile(first_file) as audio:
        plan = make_plan(config, int(audio.samplerate), int(len(audio)), int(audio.channels))
        fir, _ = _design_resample_filter(plan)
        return _benchmark_backend_choice(
            audio,
            plan,
            threads=resources.threads_per_worker,
            gpu_memory_fraction=gpu_memory_fraction,
            fir=fir,
            cpu_parallel_jobs=resources.jobs,
        )


def _format_bytes(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    return f"{value / MIB:.1f} MiB"


def _print_resource_plan(plan: ResourcePlan) -> None:
    print(
        "Resource plan: "
        f"backend={plan.backend}, file_workers={plan.jobs}, "
        f"FFT_threads/worker={plan.threads_per_worker}, CPUs={plan.total_cpu_threads}, "
        f"available_RAM={_format_bytes(plan.available_memory_bytes)}, "
        f"usable_RAM={_format_bytes(plan.memory_budget_bytes)}, "
        f"RAM/worker={_format_bytes(plan.memory_per_worker_bytes)}"
    )


def _dry_run(files: Sequence[Path], config: dict[str, Any], resources: ResourcePlan) -> int:
    print("Dry run; no output will be written.")
    _print_resource_plan(resources)
    total_input = 0
    total_output = 0
    for source in files:
        rate, frames, channels, fmt, subtype = _inspect_audio(source)
        plan = make_plan(config, rate, frames, channels)
        size = source.stat().st_size
        estimated = estimate_bytes(plan)
        total_input += size
        total_output += estimated
        print(
            f"{source}: {fmt}/{subtype}, {channels}ch, {rate} Hz, {plan.duration_seconds / 3600:.2f} h, "
            f"analysis {plan.sample_rate} Hz, FFT {plan.fft_size}, Δf={plan.sample_rate / plan.fft_size:.4f} Hz, "
            f"{plan.fft_windows:,} windows, LTSA {plan.time_bins:,}×{plan.frequency_bins:,} "
            f"({_format_bytes(estimated)})"
        )
    print(f"Totals: input={_format_bytes(total_input)}, estimated LTSA matrices={_format_bytes(total_output)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="Audio file(s) or directories")
    parser.add_argument("--config", type=Path, help="LTSA JSON configuration")
    parser.add_argument("--output-dir", type=Path, default=Path("ltsa_out"))
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), help="Override config processing.backend")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel files; 0 chooses automatically")
    parser.add_argument("--threads-per-worker", type=int, default=0, help="Threaded FFT workers per file; 0=auto")
    parser.add_argument("--memory-fraction", type=float, help="Fraction of currently available RAM usable by all workers")
    parser.add_argument("--gpu-memory-fraction", type=float, help="Fraction of currently free GPU RAM available to CuPy")
    parser.add_argument("--chunk-seconds", type=float, help="Override adaptive source-read chunk duration")
    parser.add_argument(
        "--chunk-windows",
        type=int,
        help="Maximum FFT windows per batch (legacy name; auto when omitted)",
    )
    parser.add_argument("--max-output-mib", type=float, default=4096.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-candidates", action="store_true", help="Do not write candidate CSV/Audacity files")
    parser.add_argument("--dry-run", action="store_true", help="Inspect files and print compute/RAM plan only")
    parser.add_argument("--write-default-config", type=Path, help="Write the default configuration and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write_default_config:
        args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_default_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
        return 0
    if not args.inputs:
        parser.error("provide at least one audio file/directory, or use --write-default-config")

    config = load_config(args.config)
    processing = config.setdefault("processing", {})
    backend = args.backend or str(processing.get("backend", "auto"))
    memory_fraction = float(
        args.memory_fraction if args.memory_fraction is not None else processing.get("memory_fraction", 0.75)
    )
    gpu_memory_fraction = float(
        args.gpu_memory_fraction
        if args.gpu_memory_fraction is not None
        else processing.get("gpu_memory_fraction", 0.75)
    )
    files = discover(args.inputs, args.recursive)
    if not files:
        parser.error("no WAV/WAVE/FLAC files found")

    resources = resolve_resources(
        len(files),
        requested_backend=backend,
        requested_jobs=args.jobs,
        requested_threads=args.threads_per_worker,
        memory_fraction=memory_fraction,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    worker_backend = backend
    if backend == "auto":
        worker_backend = choose_archive_backend(files[0], config, resources, gpu_memory_fraction)
        print(
            f"Auto backend benchmark selected {worker_backend}; GPU is chosen only when one GPU stream "
            f"beats the estimated aggregate throughput of {resources.jobs} CPU file worker(s)."
        )
    if worker_backend == "cuda":
        resources = dataclasses.replace(
            resources,
            backend=worker_backend,
            jobs=1,
            threads_per_worker=min(resources.total_cpu_threads, max(1, args.threads_per_worker or 4)),
            memory_per_worker_bytes=resources.memory_budget_bytes,
        )
    else:
        resources = dataclasses.replace(resources, backend=worker_backend)

    if args.dry_run:
        return _dry_run(files, config, resources)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _print_resource_plan(resources)
    print(
        f"Processing {len(files)} recording(s); backend dispatched as {worker_backend}. "
        "Use --dry-run to inspect dimensions before large runs."
    )

    seen: set[str] = set()
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source in files:
        destination = args.output_dir / _output_name(source, seen)
        if destination.exists() and not args.overwrite:
            skipped.append(
                {
                    "input": str(source),
                    "output": str(destination),
                    "status": "skipped",
                    "reason": "exists",
                }
            )
            continue
        jobs.append(
            {
                "input": str(source),
                "output": str(destination),
                "config": config,
                "backend": worker_backend,
                "threads": resources.threads_per_worker,
                "memory_budget_bytes": resources.memory_per_worker_bytes,
                "gpu_memory_fraction": gpu_memory_fraction,
                "chunk_seconds": args.chunk_seconds,
                "chunk_windows": args.chunk_windows,
                "max_output_mib": args.max_output_mib,
                "write_candidates": not args.no_candidates,
            }
        )

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    if resources.jobs <= 1:
        for index, job in enumerate(jobs, 1):
            result = _one(job)
            results.append(result)
            detail = f" backend={result.get('backend')} {result.get('elapsed_seconds', 0):.1f}s" if result["status"] == "ok" else ""
            print(f"[{index}/{len(jobs)}] {result['status']:>7} {result['input']}{detail}")
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=resources.jobs, mp_context=context) as pool:
            futures = {pool.submit(_one, job): job for job in jobs}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                detail = f" backend={result.get('backend')} {result.get('elapsed_seconds', 0):.1f}s" if result["status"] == "ok" else ""
                print(f"[{index}/{len(jobs)}] {result['status']:>7} {result['input']}{detail}")

    elapsed = time.perf_counter() - started
    manifest = {
        "format": "util_many_ltsa_manifest",
        "version": VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "config": config,
        "resource_plan": dataclasses.asdict(resources),
        "dispatched_backend": worker_backend,
        "results": sorted(results + skipped, key=lambda item: item["input"]),
    }
    manifest_path = args.output_dir / "ltsa_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in results if item["status"] == "failed"]
    print(f"Manifest: {manifest_path}")
    print(f"Elapsed: {elapsed:.1f}s; successful={len(results) - len(failed)}, failed={len(failed)}, skipped={len(skipped)}")
    if failed:
        for item in failed:
            print(f"FAILED: {item['input']}: {item['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
