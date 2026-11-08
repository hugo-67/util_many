# LTSA in Audio Annotator v2

Audio Annotator v2 can compute, view, import, and export **Long-Term Spectral Average (LTSA)** data without uploading audio. The browser and `ltsa_batch.py` share a versioned JSON configuration and the same portable numeric `.ltsa` format.

For the recorder-specific Omura’s whale workflow, literature review, tuned configurations, candidate interpretation, and high-throughput commands, see **[OMURAS_WHALE_LTSA.md](OMURAS_WHALE_LTSA.md)**.

## Recommended interactive workflow

1. Open `audio_annotatorv2.html` and load a WAV/RF64/BW64/FLAC recording.
2. Expand **Long-Term Spectral Average (LTSA)**.
3. Tune FFT size, overlap, averaging interval, channel, and stored frequency band.
4. Use **Preview current view** while tuning. Preview is limited to a short bounded interval.
5. Click the LTSA plot to move the annotator cursor/view to that time.
6. Use **Build full recording** when the estimated browser result is within the browser memory guard.
7. Export **LTSA config JSON** and use the same config for a folder-scale Python run.
8. Export/import `.ltsa` files when collaborators should inspect a precomputed result without reprocessing the raw recording.

The browser LTSA worker is separate from the annotator analysis/playback workers, so a long LTSA job does not monopolize waveform/spectrogram rendering.

## What is averaged

Each frame uses a Hann window and a real FFT. The one-sided amplitude spectrum is normalized by Hann coherent gain. LTSA bins average **linear power** across FFT frames and convert the mean to dBFS with `10*log10(power)`.

FFT frame starts are aligned to absolute sample zero. This makes equivalent browser and Python analyses comparable when no Python-only downsampling is configured.

## Shared configuration

The generic example is `sample/ltsa_config.json`. Targeted examples can add `processing` and `detection` sections.

- `analysis.fft_size`: power-of-two FFT size.
- `analysis.overlap`: fraction from 0 to `<0.95`.
- `analysis.average_seconds`: LTSA time-bin width.
- `analysis.channel`: zero-based channel index; `-1` averages all channels.
- `analysis.frequency_min_hz` / `frequency_max_hz`: band actually stored in the LTSA matrix. `null` means Nyquist.
- `processing.analysis_sample_rate_hz`: optional Python-only anti-aliased analysis rate. This is the largest speedup for low-frequency targets in high-rate recordings.
- `processing.backend`: `auto`, `cpu`, or `cuda`.
- `processing.memory_fraction` / `gpu_memory_fraction`: bounded resource use.
- `display`: viewer-only dB range and colormap; changing these does not recompute data.
- `detection`: optional target-specific candidate screening.

The browser always analyses the source sample rate. The Python batch processor may downsample first when `analysis_sample_rate_hz` is set, and records both source and analysis rates in the `.ltsa` header.

## Browser memory guard

The browser estimates the matrix before a full build:

```text
time_bins × frequency_bins × 4 bytes
```

It refuses results above its safety cap. Increase `average_seconds`, narrow the stored band, or use Python when a full LTSA would be too large.

The browser never loads the whole recording into one AudioBuffer. It reuses the annotator’s bounded native WAV/FLAC reader and computes LTSA in a dedicated worker.

## Accelerated Python batch processing

Install nothing permanently with `uv`:

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  python ltsa_batch.py recordings/ \
  --config sample/ltsa_config.json \
  --output-dir ltsa_out \
  --recursive
```

The processor provides:

- anti-aliased polyphase downsampling before FFT;
- adaptive multiprocessing across files;
- threaded SciPy FFT batches inside each CPU worker;
- optional CuPy/CUDA resampling and FFTs;
- a real-audio CPU-versus-GPU auto benchmark;
- adaptive RAM and GPU-memory budgets with OS/runtime headroom;
- capped user overrides that cannot intentionally overcommit the calculated budget;
- atomic output and resumable directory runs;
- exact processing metadata and timings in `ltsa_manifest.json`.

Useful options:

- `--dry-run`: print source/analysis rates, dimensions, FFT work, and resource plan without writing output.
- `--backend auto|cpu|cuda`: choose execution backend.
- `--jobs 0`: choose file-process count from CPU affinity and available RAM.
- `--threads-per-worker 0`: divide FFT threads automatically among file workers.
- `--memory-fraction 0.75`: fraction of currently available RAM eligible for use.
- `--gpu-memory-fraction 0.75`: hard CuPy pool fraction of currently free GPU RAM.
- `--chunk-seconds N`: requested source-block duration; capped to the safe RAM allocation.
- `--chunk-windows N`: requested FFT batch; capped to the safe temporary-memory allocation.
- `--max-output-mib N`: refuse unexpectedly huge output matrices.
- `--no-candidates`: suppress target-profile candidate CSV/TXT files.
- `--overwrite`: replace completed outputs.

A batch run writes one `.ltsa` per audio file plus `ltsa_manifest.json`. Configurations with candidate screening enabled also write a CSV and Audacity-compatible label file next to each LTSA.

## `.ltsa` binary format

Version 1 is deliberately simple and portable:

1. 8-byte magic: `LTSA1\0\0\0`
2. little-endian uint32 JSON-header length
3. UTF-8 JSON header
4. little-endian Float32 matrix in **time-major** order: `[time_bin, frequency_bin]`

The header includes source and analysis sample rates, duration, dimensions, bin widths, source metadata, FFT/resampling parameters, processing backend, quantity (`power_dbfs`), elapsed time, and the full shared configuration.

## Scientific and operational notes

- LTSA is a review/navigation summary, not a replacement for the native waveform or detailed spectrogram.
- Short or narrow events can be diluted by long averaging bins.
- The stored frequency band is irreversible without recomputation.
- Display floor/ceiling and colormap are rendering choices and can change later without changing data.
- Candidate detectors surface intervals for review; they do not establish species identity.
- For large archives, use HTML for tuning/review and Python for repeatable batch generation.
