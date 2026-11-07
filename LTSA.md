# LTSA in Audio Annotator v2

Audio Annotator v2 can compute, view, import and export **Long-Term Spectral Average (LTSA)** data without uploading audio. The browser and `ltsa_batch.py` use the same JSON configuration and the same binary `.ltsa` format.

## Recommended workflow

1. Open `audio_annotatorv2.html` and load a WAV/RF64/BW64/FLAC recording.
2. Expand **Long-Term Spectral Average (LTSA)**.
3. Tune FFT size, overlap, averaging interval, channel and stored frequency band.
4. Use **Preview current view** while tuning. Preview is limited to a short bounded interval.
5. Click the LTSA plot to move the annotator cursor/view to that time.
6. Use **Build full recording** when the estimated browser result is within the browser memory guard.
7. Export **LTSA config JSON** and use the same config for a folder-scale Python run.
8. Export/import `.ltsa` files when you want collaborators to inspect a precomputed result without reprocessing the raw recording.

The LTSA worker is separate from the annotator analysis/playback workers, so long LTSA jobs do not monopolize waveform/spectrogram rendering.

## What is averaged

Each frame uses a Hann window and a native-rate real FFT. The one-sided amplitude spectrum is normalized by Hann coherent gain. LTSA bins average **linear power** across FFT frames and convert the mean to dBFS with `10*log10(power)`.

FFT frame starts are aligned to absolute sample zero. This makes a browser preview and a full-file Python run comparable for the same configuration.

## Shared configuration

See `sample/ltsa_config.json`.

- `fft_size`: one of 32..16384 powers of two supported by the browser.
- `overlap`: 0 to <0.95. For very long ultrasonic recordings, 0 overlap is a much faster starting point.
- `average_seconds`: LTSA time-bin width.
- `channel`: zero-based channel index; `-1` means average/mix all channels.
- `frequency_min_hz` / `frequency_max_hz`: the band actually stored in the LTSA matrix. `null` max means Nyquist.
- `display`: viewer-only dB range and colormap; changing these does not recompute an imported/built LTSA.

For a 384 kHz recording with FFT 8192, one FFT spans about 21.3 ms. A 60 s LTSA bin can therefore contain many FFT frames. Start with 0 overlap for long deployments; add overlap only when it materially improves what you need to see.

## Browser memory guard

The browser estimates the matrix before a full build:

`time_bins × frequency_bins × 4 bytes`

and refuses results above its safety cap. Increase `average_seconds`, narrow the stored band, or use Python when a full LTSA would be too large.

The browser never loads the whole recording into one AudioBuffer. It reuses the annotator's bounded native WAV/FLAC reader and computes LTSA in a dedicated worker.

## Python batch processing

Install nothing permanently with `uv`:

```bash
uv run --with numpy --with soundfile python ltsa_batch.py recordings/ \
  --config sample/ltsa_config.json \
  --output-dir ltsa_out \
  --recursive
```

Useful options:

- `--jobs 0`: use available CPU cores across files (default).
- `--jobs 1`: conservative single-file processing.
- `--chunk-windows N`: number of FFT windows vectorized per bounded block.
- `--max-output-mib N`: refuse unexpectedly huge output matrices.
- `--overwrite`: replace existing outputs.

A batch run writes one `.ltsa` per audio file plus `ltsa_manifest.json` recording successes/failures and the exact configuration.

## .ltsa binary format

Version 1 is deliberately simple and portable:

1. 8-byte magic: `LTSA1\0\0\0`
2. little-endian uint32 JSON-header length
3. UTF-8 JSON header
4. little-endian Float32 matrix in **time-major** order: `[time_bin, frequency_bin]`

The header includes sample rate, duration, dimensions, time/frequency bin widths, source metadata, quantity (`power_dbfs`) and the full shared configuration.

## Scientific/operational notes

- LTSA is a review/navigation summary, not a replacement for the native waveform or detailed spectrogram.
- Narrow/short clicks can be visually diluted by long averaging bins. Use LTSA to find periods of interest, then inspect the detailed native-rate spectrogram.
- The stored band is irreversible without recomputation. For maximum flexibility, store 0..Nyquist when the matrix size is reasonable.
- Display floor/ceiling and colormap are rendering choices and can be changed later without changing the data.
- For large archives, use HTML for tuning/review and Python for repeatable batch generation.
