#!/usr/bin/env python3
"""Batch Long-Term Spectral Average (LTSA) generator for Audio Annotator v2.

The browser annotator and this CLI share the same JSON configuration and .ltsa
binary format. Audio is read in bounded blocks; the whole recording is never
loaded into RAM.

Quick start:
  uv run --with numpy --with soundfile python ltsa_batch.py recording.wav
  uv run --with numpy --with soundfile python ltsa_batch.py recordings/ \
      --config sample/ltsa_config.json --output-dir ltsa_out --recursive

The .ltsa payload is:
  8 bytes  magic: b"LTSA1\\0\\0\\0"
  4 bytes  little-endian uint32 JSON header length
  N bytes  UTF-8 JSON header
  ...      little-endian float32 matrix, time-major [time_bin, frequency_bin]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

MAGIC = b"LTSA1\\0\\0\\0"
FORMAT_NAME = "util_many_ltsa"
CONFIG_FORMAT = "util_many_ltsa_config"
VERSION = 1
SUPPORTED_FFT = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
AUDIO_SUFFIXES = {".wav", ".wave", ".flac"}

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
    "display": {
        "db_min": -120.0,
        "db_max": 0.0,
        "colormap": "ember",
    },
}


@dataclasses.dataclass(frozen=True)
class Plan:
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


def _deps():
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "ltsa_batch.py needs numpy and soundfile. Run it with:\\n"
            "  uv run --with numpy --with soundfile python ltsa_batch.py ..."
        ) from exc
    return np, sf


def merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    base = json.loads(json.dumps(DEFAULT_CONFIG))
    if not config:
        return base
    if config.get("format") not in (None, CONFIG_FORMAT):
        raise ValueError(f"Unsupported config format: {config.get('format')!r}")
    if int(config.get("version", VERSION)) != VERSION:
        raise ValueError(f"Unsupported LTSA config version: {config.get('version')!r}")
    for section in ("analysis", "display"):
        if isinstance(config.get(section), dict):
            base[section].update(config[section])
    base["format"] = CONFIG_FORMAT
    base["version"] = VERSION
    return base


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return merged_config(None)
    return merged_config(json.loads(path.read_text(encoding="utf-8")))


def make_plan(config: dict[str, Any], sample_rate: int, frames: int, channels: int) -> Plan:
    a = config["analysis"]
    fft_size = int(a["fft_size"])
    if fft_size not in SUPPORTED_FFT:
        raise ValueError(f"fft_size must be one of {SUPPORTED_FFT}")
    if str(a.get("window", "hann")).lower() != "hann":
        raise ValueError("Only the Hann window is supported so browser and Python results stay identical.")
    overlap = float(a["overlap"])
    if not 0.0 <= overlap < 0.95:
        raise ValueError("overlap must be >= 0 and < 0.95")
    hop = max(1, int(round(fft_size * (1.0 - overlap))))
    average_seconds = float(a["average_seconds"])
    if not math.isfinite(average_seconds) or average_seconds <= 0:
        raise ValueError("average_seconds must be a positive finite value")
    channel = int(a["channel"])
    if channel < -1 or channel >= channels:
        raise ValueError(f"channel must be -1 (mix) or 0..{channels - 1}")
    nyquist = sample_rate / 2.0
    fmin = max(0.0, float(a.get("frequency_min_hz", 0.0) or 0.0))
    raw_max = a.get("frequency_max_hz")
    fmax = nyquist if raw_max is None else min(nyquist, float(raw_max))
    if not math.isfinite(fmin) or not math.isfinite(fmax) or fmax <= fmin:
        raise ValueError(f"Invalid frequency band {fmin:g}..{fmax:g} Hz for Nyquist {nyquist:g} Hz")
    hz_per_bin = sample_rate / fft_size
    first = max(0, int(math.ceil(fmin / hz_per_bin - 1e-12)))
    last = min(fft_size // 2, int(math.floor(fmax / hz_per_bin + 1e-12)))
    frequency_bins = last - first + 1
    duration = frames / sample_rate
    time_bins = max(1, int(math.ceil(duration / average_seconds)))
    return Plan(
        fft_size=fft_size, hop=hop, average_seconds=average_seconds, channel=channel,
        frequency_min_hz=first * hz_per_bin, frequency_max_hz=last * hz_per_bin,
        first_frequency_bin=first, frequency_bins=frequency_bins, time_bins=time_bins,
        sample_rate=sample_rate, frames=frames, channels=channels,
    )


def estimate_bytes(plan: Plan) -> int:
    return plan.time_bins * plan.frequency_bins * 4


def compute_ltsa(path: Path, config: dict[str, Any], *, chunk_windows: int = 256,
                 max_output_mib: float = 4096.0):
    np, sf = _deps()
    with sf.SoundFile(path) as audio:
        sample_rate = int(audio.samplerate)
        frames = int(len(audio))
        channels = int(audio.channels)
        plan = make_plan(config, sample_rate, frames, channels)
        output_bytes = estimate_bytes(plan)
        if output_bytes > max_output_mib * 1048576:
            needed = frames / sample_rate * plan.frequency_bins * 4 / (max_output_mib * 1048576)
            raise ValueError(
                f"Estimated LTSA matrix is {output_bytes / 1048576:.1f} MiB, above the "
                f"{max_output_mib:g} MiB limit. Increase average_seconds to about "
                f"{max(plan.average_seconds, needed):.3g}s or narrow the stored frequency band."
            )

        result = np.full((plan.time_bins, plan.frequency_bins), np.nan, dtype="<f4")
        window = np.hanning(plan.fft_size).astype(np.float64)
        coherent = float(window.sum())
        factors = np.ones(plan.fft_size // 2 + 1, dtype=np.float64) * 2.0
        factors[0] = 1.0
        if plan.fft_size % 2 == 0:
            factors[-1] = 1.0
        factors /= coherent
        f0 = plan.first_frequency_bin
        f1 = f0 + plan.frequency_bins

        current_bin = -1
        current_sum = np.zeros(plan.frequency_bins, dtype=np.float64)
        current_count = 0

        def flush(bin_index: int):
            nonlocal current_sum, current_count
            if 0 <= bin_index < plan.time_bins and current_count:
                result[bin_index] = (10.0 * np.log10(np.maximum(current_sum / current_count, 1e-20))).astype(np.float32)
            current_sum.fill(0.0)
            current_count = 0

        frame_starts = range(0, frames, plan.hop)
        batch: list[int] = []
        for frame_start in frame_starts:
            batch.append(frame_start)
            if len(batch) < max(1, chunk_windows) and frame_start + plan.hop < frames:
                continue

            read_start = batch[0]
            read_stop = min(frames, batch[-1] + plan.fft_size)
            audio.seek(read_start)
            block = audio.read(read_stop - read_start, dtype="float32", always_2d=True)
            if plan.channel == -1:
                signal = block.mean(axis=1, dtype=np.float64)
            else:
                signal = block[:, plan.channel].astype(np.float64, copy=False)

            need = batch[-1] - read_start + plan.fft_size
            if len(signal) < need:
                padded = np.zeros(need, dtype=np.float64)
                padded[:len(signal)] = signal
                signal = padded

            base = np.lib.stride_tricks.sliding_window_view(signal, plan.fft_size)
            offsets = np.asarray(batch, dtype=np.int64) - read_start
            windows = base[offsets] * window
            spectrum = np.fft.rfft(windows, axis=1)
            amplitude = np.abs(spectrum) * factors
            powers = (amplitude[:, f0:f1] ** 2).astype(np.float64, copy=False)
            centers = (np.asarray(batch, dtype=np.float64) + plan.fft_size / 2.0) / sample_rate
            bins = np.floor(centers / plan.average_seconds).astype(np.int64)
            for b in np.unique(bins):
                if b >= plan.time_bins:
                    continue
                if current_bin != int(b):
                    if current_bin >= 0:
                        flush(current_bin)
                    current_bin = int(b)
                mask = bins == b
                current_sum += powers[mask].sum(axis=0)
                current_count += int(mask.sum())
            batch.clear()

        if current_bin >= 0:
            flush(current_bin)

    return plan, result


def make_header(path: Path, config: dict[str, Any], plan: Plan) -> dict[str, Any]:
    stat = path.stat()
    return {
        "format": FORMAT_NAME,
        "version": VERSION,
        "source": {
            "name": path.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "config": config,
        "sample_rate": plan.sample_rate,
        "channels": plan.channels,
        "duration_seconds": plan.frames / plan.sample_rate,
        "time_bins": plan.time_bins,
        "frequency_bins": plan.frequency_bins,
        "time_bin_seconds": plan.average_seconds,
        "frequency_bin_hz": plan.sample_rate / plan.fft_size,
        "frequency_start_hz": plan.frequency_min_hz,
        "frequency_end_hz": plan.frequency_max_hz,
        "first_frequency_bin": plan.first_frequency_bin,
        "data_dtype": "float32_le",
        "data_order": "time_frequency",
        "quantity": "power_dbfs",
    }


def write_ltsa(path: Path, header: dict[str, Any], data) -> None:
    np, _ = _deps()
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
    np, _ = _deps()
    with path.open("rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("Not a util_many LTSA file")
        header_len_raw = f.read(4)
        if len(header_len_raw) != 4:
            raise ValueError("Truncated LTSA header")
        header_len = struct.unpack("<I", header_len_raw)[0]
        if header_len > 16 * 1048576:
            raise ValueError("LTSA header is unreasonably large")
        header = json.loads(f.read(header_len).decode("utf-8"))
        if header.get("format") != FORMAT_NAME or int(header.get("version", 0)) != VERSION:
            raise ValueError("Unsupported LTSA format/version")
        count = int(header["time_bins"]) * int(header["frequency_bins"])
        data = np.fromfile(f, dtype="<f4", count=count)
        if data.size != count or f.read(1):
            raise ValueError("LTSA payload size does not match header")
        return header, data.reshape((int(header["time_bins"]), int(header["frequency_bins"])))


def discover(inputs: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        item = item.expanduser()
        if item.is_file():
            if item.suffix.lower() in AUDIO_SUFFIXES:
                found.append(item.resolve())
            continue
        if item.is_dir():
            it = item.rglob("*") if recursive else item.glob("*")
            found.extend(p.resolve() for p in it if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
            continue
        raise FileNotFoundError(item)
    return sorted(dict.fromkeys(found))


def _one(job: tuple[str, str, dict[str, Any], int, float]) -> dict[str, Any]:
    input_name, output_name, config, chunk_windows, max_output_mib = job
    src, dst = Path(input_name), Path(output_name)
    try:
        plan, data = compute_ltsa(src, config, chunk_windows=chunk_windows, max_output_mib=max_output_mib)
        header = make_header(src, config, plan)
        write_ltsa(dst, header, data)
        return {
            "input": str(src), "output": str(dst), "status": "ok",
            "time_bins": plan.time_bins, "frequency_bins": plan.frequency_bins,
            "size_mib": dst.stat().st_size / 1048576,
        }
    except Exception as exc:
        return {"input": str(src), "output": str(dst), "status": "failed", "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio file(s) or directories")
    parser.add_argument("--config", type=Path, help="Shared LTSA JSON config exported by audio_annotatorv2.html")
    parser.add_argument("--output-dir", type=Path, default=Path("ltsa_out"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel files; 0 = use available CPU cores")
    parser.add_argument("--chunk-windows", type=int, default=256, help="FFT windows vectorized per bounded read")
    parser.add_argument("--max-output-mib", type=float, default=4096.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-default-config", type=Path, help="Write the default shared config and exit")
    args = parser.parse_args(argv)

    if args.write_default_config:
        args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_default_config.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\\n", encoding="utf-8")
        return 0

    config = load_config(args.config)
    files = discover(args.inputs, args.recursive)
    if not files:
        parser.error("No WAV/WAVE/FLAC files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = []
    for src in files:
        dst = args.output_dir / (src.stem + ".ltsa")
        if dst.exists() and not args.overwrite:
            skipped.append({"input": str(src), "output": str(dst), "status": "skipped", "reason": "exists"})
        else:
            jobs.append((str(src), str(dst), config, max(1, args.chunk_windows), args.max_output_mib))

    workers = args.jobs if args.jobs > 0 else min(len(jobs), max(1, os.cpu_count() or 1))
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = _one(job)
            results.append(r)
            print(f"[{i}/{len(jobs)}] {r['status']:>6}  {r['input']}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, job): job for job in jobs}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                r = future.result()
                results.append(r)
                print(f"[{i}/{len(jobs)}] {r['status']:>6}  {r['input']}")

    manifest = {
        "format": "util_many_ltsa_manifest",
        "version": VERSION,
        "config": config,
        "results": sorted(results + skipped, key=lambda x: x["input"]),
    }
    manifest_path = args.output_dir / "ltsa_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
    failed = [r for r in results if r["status"] == "failed"]
    print(f"Manifest: {manifest_path}")
    if failed:
        for r in failed:
            print(f"FAILED: {r['input']}: {r['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
