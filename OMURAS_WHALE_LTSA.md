# Omura’s whale LTSA and candidate-screening workflow

Research review and implementation guide, updated **2026-11-08**.

## 1. Species-name correction

`Balaenoptera omurai` is **Omura’s whale**, not Bryde’s whale. The name “Bryde’s whale” normally refers to members of the *Balaenoptera edeni/brydei* complex. Older Australian reports sometimes attributed the low-frequency song described below to Bryde’s whales before Omura’s whale song was formally described and compared across regions. This workflow follows the scientific name supplied and therefore targets **Omura’s whale**.

Do not treat an acoustic candidate as a confirmed species record. Omura’s whale overlaps other baleen whales, ships, seismic pulses, flow noise, earthquakes, and 50/60 Hz electrical contamination. Confirmation requires manual review and, for a new region, preferably independent visual, genetic, or multi-source evidence.

## 2. Best available reference sounds

### Official downloadable WAV

The Curtin University Centre for Marine Science and Technology (CMST) hosts an Omura’s whale spectrogram and downloadable WAV:

- Information page: <https://cmst.curtin.edu.au/research/marine-mammal-bioacoustics/omuras-whale/>
- Direct WAV: <https://cmst.curtin.edu.au/wp-content/uploads/sites/4/2017/06/t2809_171_49E7E99A_LowFreqSplodge.wav>

The sound is mostly below ordinary human hearing. Inspect the native spectrogram; listening may require temporary speed-up, but do not use the speed-altered audio for quantitative measurements.

### Confirmed Madagascar reference

The Omura’s Whale Project provides spectrograms and an explanation of the visually/genetically associated Madagascar song:

- <https://www.omuraswhale.org/second-page>

This is the strongest reference for attribution because recordings were made in the presence of confirmed Omura’s whales. It shows the stereotyped pulsative unit, a rhythmic series, and chorusing by several apparent singers.

### Detector reference implementation

CMST publishes information and licensed MATLAB code for its Omura’s whale detector:

- <https://cmst.curtin.edu.au/products/signal-detection-software/>

The published detector work compared spectrogram cross-correlation, Shannon-entropy, and spectral-blob methods. Cross-correlation performed best quantitatively in that test corpus, while all three were useful in some conditions. The LTSA candidate routine in this repository is deliberately a **high-recall triage layer**, not a replacement for those validated detectors or manual review.

## 3. Known acoustic specifications

Only one broad vocalisation type is currently securely attributed to Omura’s whale: low-frequency amplitude-modulated song, occurring as a one-unit **singlet** or two-unit **doublet** depending on region. The phrase may be pulsative/noisy and, in the doublet, followed by a more tonal unit.

| Region / form | Frequency | Peak frequency | Duration | Repetition / notes |
|---|---:|---:|---:|---|
| Northwest Madagascar, confirmed | about 15–50 Hz; measured mean low 14.9 ± 5.4 Hz and high 52.9 ± 2.2 Hz | 36.1 ± 6.2 Hz | 9.2 ± 0.92 s | approximately 134–177 s; commonly described as 2–3 min; long bouts and choruses |
| Northwest Australia, doublet | 17–50 Hz (earlier descriptions approximately 20–60 or 15–62 Hz) | 25.5 Hz | approximately 15 s; commonly 12–15 s | approximately 160–210 s; extended song bouts |
| Northwest Australia, singlet | 13–72 Hz | 26.6 Hz | approximately 8 s | repeated as song; less common and geographically restricted in the analysed data |
| Chagos / Diego Garcia candidate population | 17–50 Hz | not consistently reported | 13.1 ± 0.6 s | 186 ± 18 s |
| Equatorial Atlantic | 20–47 Hz | not consistently reported | 10.42 ± 1.16 s | 172.18 ± 32.5 s |
| Great Barrier Reef candidate population | first unit approximately 15–60 Hz | 24.92 ± 0.57 Hz | 13.87 ± 0.37 s | Omura-like doublet; attribution remains less secure without simultaneous visual confirmation |
| Cross-region practical envelope | **13–72 Hz** | commonly **20–45 Hz**, often near 25–36 Hz | **8–16 s** typical; allow **5–24 s** during screening | usually **110–240 s** for a tolerant search interval |

### Practical detector envelope used here

The supplied screening profile deliberately surrounds the published values:

- primary signal band: **13–72 Hz**;
- expected peak band: **20–45 Hz**;
- lower comparison band: **5–12 Hz**;
- upper comparison band: **80–115 Hz**;
- candidate duration: **5–24 s**;
- song interval: **110–240 s**;
- local-baseline window: **300 s**;
- in-band/side-band threshold: normally **2–3 dB**.

The wide envelope is intentional. Song differs geographically, weak calls blur into noise, recorder response is imperfect below 30 Hz, and a long LTSA average broadens event boundaries. Candidate output therefore favors recall over precision.

## 4. Recorder-specific considerations

### SoundTrap

SoundTrap STD models are commonly specified as approximately flat from 20 Hz upward, but independent measurements of ST500/ST600 units found meaningful sensitivity roll-off below roughly 30 Hz. Omura’s song often peaks around 25–36 Hz, so:

1. retain the 13–30 Hz region even when it appears weak;
2. do not compare absolute received levels among recorders without calibration/correction;
3. inspect a known background spectrum and, where possible, apply the unit/model frequency-response correction outside this LTSA tool;
4. avoid a display floor that hides the low-frequency first unit.

### HydroMoth

HydroMoth is inexpensive and capable of high sample rates, but its integrated transducer, underwater case, gain, orientation, and unit-to-unit response differ from a calibrated SoundTrap. Published comparisons show lower overall performance than SoundTrap in some conditions, while newer low-frequency calibration work indicates useful and reasonably consistent measurements after calibration in the tested 100–1000 Hz range. The Omura target extends below that tested range, so treat HydroMoth amplitudes as relative screening evidence unless the exact device/housing has been calibrated down to at least 10–100 Hz.

For both systems, deployment motion and current can create strong flow noise below about 10–20 Hz, while ships and electrical interference frequently occupy 20–100 Hz. Rhythm and phrase shape are therefore as important as frequency.

## 5. Why downsample before LTSA

A 384 kHz recording contains 1,500 times more samples than a 256 Hz analysis stream. Omura’s known song is below approximately 72 Hz, so carrying ultrasonic bandwidth into every FFT wastes nearly all compute and RAM.

The accelerated batch script:

1. reads source audio in bounded chunks;
2. selects one channel or averages channels;
3. applies an explicit Kaiser-windowed polyphase anti-alias filter;
4. down-samples to 256 or 512 Hz;
5. runs batched Hann-windowed real FFTs;
6. averages **linear power**, not dB values, inside each LTSA time bin;
7. converts the power mean to dBFS;
8. writes the numeric `.ltsa` matrix atomically;
9. optionally writes candidate CSV and Audacity labels.

Chunk boundaries use FIR guard samples and exact rational-rate alignment. Tests compare different chunk sizes and require numerically identical output.

## 6. Supplied configurations

### A. Fast archive scan

`sample/ltsa_omuras_whale_fast_scan.json`

| Setting | Value | Rationale |
|---|---:|---|
| Analysis rate | 256 Hz | Keeps a guarded 5–110 Hz band while discarding unnecessary high-frequency data |
| FFT | 1024 samples | 4 s window, 0.25 Hz frequency spacing |
| Overlap | 75% | 1 s hop, enough temporal continuity for 8–16 s calls |
| LTSA bin | 2 s | Good event visibility at moderate storage/compute cost |
| Stored band | 5–110 Hz | Target, low-frequency context, and upper noise reference |
| Candidate threshold | 2.5 dB side-band, 3 dB local baseline | High-recall first pass |

Use this first over the complete archive.

### B. Sensitive review

`sample/ltsa_omuras_whale_sensitive_review.json`

| Setting | Value | Rationale |
|---|---:|---|
| Analysis rate | 512 Hz | Preserves context and possible harmonics to 180 Hz |
| FFT | 4096 samples | 8 s window, 0.125 Hz frequency spacing |
| Overlap | 87.5% | 1 s hop |
| LTSA bin | 1 s | Better phrase boundaries and weak-call review |
| Stored band | 5–180 Hz | More diagnostic context around interference |
| Candidate threshold | 2.0 dB side-band, 2.5 dB baseline | More sensitive, with more false positives |

Run this on candidate days, sites, or files—not necessarily the entire archive.

### C. Long song/rhythm context

`sample/ltsa_omuras_whale_song_context.json`

| Setting | Value |
|---|---:|
| Analysis rate | 256 Hz |
| FFT | 1024 samples |
| Overlap | 50% |
| LTSA bin | 10 s |
| Stored band | 5–110 Hz |
| Automatic candidates | off |

This produces compact long-duration views where repeated units, 2–3 min rhythm, chorusing, ship passages, and deployment-wide patterns are easy to see. It is not the best matrix for exact event timing.

## 7. High-throughput execution design

### Adaptive CPU execution

The default `--backend auto` mode examines current CPU affinity and **currently available**, not total, RAM. It then:

- chooses as many file processes as CPU and RAM safely permit;
- divides CPU threads among those processes;
- uses `scipy.fft.rfft(..., workers=N)` for threaded batched FFTs;
- prevents nested BLAS/OpenMP pools from oversubscribing the machine;
- dynamically caps source chunks and FFT batches;
- reserves RAM for the output matrix, codec, Python runtime, OS, and filesystem cache;
- caps aggressive user overrides instead of allowing them to overcommit memory.

Parallelism is mostly across files because PAM archives usually contain many recordings. This also avoids forcing one huge recording through a single serial file queue.

### CUDA execution

If CuPy and a compatible NVIDIA GPU are available, auto mode benchmarks a representative real-audio block. It chooses one CUDA stream only when that stream is estimated to beat the **aggregate throughput of all planned CPU file workers**, not merely one CPU core/process.

CUDA mode:

- uses `cupyx.scipy.signal.resample_poly` for anti-aliased resampling;
- performs batched `cupy.fft.rfft` transforms;
- keeps per-window spectra on the device;
- transfers only one accumulated vector per LTSA bin;
- applies a hard CuPy memory-pool limit based on currently free GPU RAM;
- forces one file worker so processes do not fight for one GPU.

For very low-rate Omura analysis, multicore CPU can outperform GPU because downsampling leaves little FFT work and file decoding remains on CPU. The benchmark makes the choice instead of assuming GPU is faster.

### RAM safety

`--memory-fraction` applies to currently available RAM. The planner also reserves explicit OS headroom. Within each worker it separately budgets:

- final LTSA matrix;
- runtime/codec overhead;
- decoded + selected + resampled source chunks;
- FFT windows, complex spectra, powers, and workspace.

An output matrix that cannot fit safely is rejected before processing, with instructions to increase LTSA averaging, narrow the stored band, reduce jobs, or increase the permitted memory fraction.

## 8. Installation

CPU-only, without creating a permanent environment:

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  python ltsa_batch.py --help
```

For an existing environment:

```bash
uv add numpy scipy soundfile psutil
```

Optional NVIDIA CUDA support requires the CuPy package matching the installed CUDA runtime. For a CUDA 12 system, for example:

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  --with cupy-cuda12x python ltsa_batch.py --help
```

Do not install multiple incompatible CuPy binary packages in the same environment.

## 9. Recommended workflow

### Step 1: inspect the plan without writing data

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  python ltsa_batch.py /data/PAM \
  --recursive \
  --config sample/ltsa_omuras_whale_fast_scan.json \
  --output-dir /data/LTSA/omura_fast \
  --dry-run
```

Check:

- every file’s source and analysis sample rate;
- FFT frequency spacing;
- number of FFT windows;
- estimated LTSA dimensions;
- file-worker count, FFT threads, and RAM per worker.

### Step 2: process the archive with automatic CPU/GPU choice

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  --with cupy-cuda12x \
  python ltsa_batch.py /data/PAM \
  --recursive \
  --config sample/ltsa_omuras_whale_fast_scan.json \
  --output-dir /data/LTSA/omura_fast
```

Omit the CuPy dependency on a CPU-only machine. Existing `.ltsa` outputs are skipped, so rerunning is resumable. Add `--overwrite` only when replacing completed outputs intentionally.

### Step 3: CPU-only maximum-throughput run

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  python ltsa_batch.py /data/PAM \
  --recursive \
  --config sample/ltsa_omuras_whale_fast_scan.json \
  --backend cpu \
  --jobs 0 \
  --threads-per-worker 0 \
  --memory-fraction 0.80 \
  --output-dir /data/LTSA/omura_fast
```

`0` means adaptive. A higher memory fraction is reasonable on a dedicated workstation; lower it on a desktop in active use.

### Step 4: force the RTX-class GPU

```bash
uv run --with numpy --with scipy --with soundfile --with psutil \
  --with cupy-cuda12x \
  python ltsa_batch.py /data/PAM \
  --recursive \
  --config sample/ltsa_omuras_whale_sensitive_review.json \
  --backend cuda \
  --jobs 1 \
  --gpu-memory-fraction 0.80 \
  --output-dir /data/LTSA/omura_sensitive
```

Use forced CUDA only after confirming the automatic benchmark or your own timed run shows an advantage.

### Step 5: review outputs

For each recording the batch processor can write:

- `recording.ltsa`: numeric LTSA matrix for `audio_annotatorv2.html`;
- `recording.omuras_candidates.csv`: candidate metrics;
- `recording.omuras_candidates.txt`: Audacity-compatible time labels;
- `ltsa_manifest.json`: exact configuration, hardware/resource plan, backend, timings, successes, failures, and output paths.

Open the original WAV/FLAC and corresponding `.ltsa` in `audio_annotatorv2.html`. Click the LTSA at a candidate time to jump into the native waveform/spectrogram.

## 10. Candidate columns and interpretation

| Field | Meaning |
|---|---|
| `start_seconds`, `end_seconds` | LTSA-derived event interval; boundaries are approximate |
| `peak_seconds` | strongest target-band time bin |
| `peak_frequency_hz` | strongest mean frequency inside 20–45 Hz over the candidate |
| `band_snr_db` | 13–72 Hz energy relative to the stronger side band |
| `above_baseline_db` | target-band energy above a rolling local median |
| `score` | heuristic ranking score, not a probability |
| `rhythmic_match` | adjacent candidate spacing falls within 110–240 s |
| `previous_ici_seconds`, `next_ici_seconds` | candidate inter-call intervals |
| `mains_line_flag` | 50/60 Hz narrow-line contamination may dominate |

Prioritize candidates that:

1. last roughly 8–16 s;
2. show a pulsative AM unit or a compatible doublet;
3. peak around 20–45 Hz but occupy a broader 13–72 Hz envelope;
4. repeat near 2–3 min for multiple cycles;
5. appear in long bouts or overlapping choruses;
6. do not consist solely of a stationary 50/60 Hz line;
7. remain visible across adjacent channels/recorders when available.

Reject or down-rank:

- continuous ship tonals;
- broadband flow-noise surges;
- electrical hum with perfectly stationary frequency;
- seismic pulses lacking the phrase envelope/rhythm;
- isolated low-SNR blobs with no repeated structure.

## 11. Scientific publications most relevant to acoustic identification

This is a best-effort comprehensive list of the directly acoustic and distribution-via-acoustics literature located through 2026-11-08. It is not a list of every biological paper ever written about Omura’s whale.

### Primary descriptions, regional occurrence, and song structure

1. Cerchio, S., et al. (2015). **Omura’s whales (*Balaenoptera omurai*) off northwest Madagascar: ecology, behaviour and conservation needs.** *Royal Society Open Science*, 2, 150301. <https://doi.org/10.1098/rsos.150301>
2. Sousa, A. G., & Harris, D. (2015). **Description and seasonal detection of two potential whale calls recorded in the Indian Ocean.** *Journal of the Acoustical Society of America*, 138, 1379–1388. <https://doi.org/10.1121/1.4928719>
3. Cerchio, S., Dorning, S., Andrianantenaina, B., & Cholewiak, D. (2017). **A first description of rhythmic song in Omura’s whale (*Balaenoptera omurai*).** *Journal of the Acoustical Society of America*, 141, 3544 (meeting abstract). <https://doi.org/10.1121/1.4987495>
4. Erbe, C., et al. (2017). **Review of Underwater and In-Air Sounds Emitted by Australian and Antarctic Marine Mammals.** *Acoustics Australia*. <https://doi.org/10.1007/s40857-017-0101-z>
5. Cerchio, S., Yamada, T. K., & Brownell, R. L. Jr. (2019). **Global Distribution of Omura’s Whales (*Balaenoptera omurai*) and Assessment of Range-Wide Threats.** *Frontiers in Marine Science*, 6, 67. <https://doi.org/10.3389/fmars.2019.00067>
6. Moreira, S. C., et al. (2020). **Occurrence of Omura’s whale, *Balaenoptera omurai*, in the Equatorial Atlantic Ocean based on Passive Acoustic Monitoring.** *Journal of Mammalogy*, 101, 1727–1735. <https://doi.org/10.1093/jmammal/gyaa130>
7. Leroy, E. C., et al. (2021). **Multiple pygmy blue whale acoustic populations in the Indian Ocean: whale song identifies a possible new population.** *Scientific Reports*, 11. Includes comparative treatment of the Diego Garcia/Omura song. <https://doi.org/10.1038/s41598-021-88062-5>
8. Cerchio, S. (2022). **The Omura’s Whale: Exploring the Enigma.** In *Natural History of the Cetaceans*. Review/chapter synthesizing distribution and song.
9. Browne, C. E., Erbe, C., & McCauley, R. D. (2024). **Distribution and Seasonality of the Omura’s Whale in Australia Based on Passive Acoustic Recordings.** *Animals*, 14, 2944. <https://doi.org/10.3390/ani14202944>
10. Browne, C. E., Erbe, C., & McCauley, R. D. (2025). **Omura’s whale song features and structure in northern Australian waters.** *JASA Express Letters*, 5, 071201. <https://doi.org/10.1121/10.0037187>
11. Li, C., Yang, H., Huang, H., Cheng, J., & Cui, H. (2026). **First Acoustic Evidence of Omura’s Whale in the South China Sea.** *Marine Mammal Science*, 42, e70154. <https://doi.org/10.1111/mms.70154>
12. Browne, C. E., McCauley, R. D., & Erbe, C. (2026). **Temporal Changes in Omura’s Whale Song Off Northwest Australia: Investigating Diel, Lunar, and Seasonal Patterns.** *Acoustics Australia*. <https://doi.org/10.1007/s40857-026-00390-6>
13. Browne, C. E., et al. (2026). **Warm, Productive Waters Shape the Habitat of Vocalizing Omura’s Whales in Northwest Australia.** *Marine Mammal Science*. <https://doi.org/10.1111/mms.70279>

### Detection and source-level methodology

14. Madhusudhana, S., Murray, A., & Erbe, C. (2020). **Automatic detectors for low-frequency vocalizations of Omura’s whales: A performance comparison.** *Journal of the Acoustical Society of America*, 147, 3078–3090. <https://doi.org/10.1121/10.0001108>
15. Gavrilov, A., Sidenko, E., McCauley, R. D., Browne, C. E., & Erbe, C. (2025). **Source levels of Western Australian Omura’s whale sound.** *Journal of the Acoustical Society of America*, 157, 4196–4202. <https://doi.org/10.1121/10.0036847>

### Useful reports and proceedings

- McCauley, R. D. (2011 and related CMST reports): early northwest-Australian descriptions, originally attributed before firm Omura comparison.
- McPherson et al. (2016): Barossa-area passive-acoustic monitoring report.
- Cerchio et al. (2018): International Whaling Commission report on distribution/field observations.
- Browne et al. (2025): **Omura’s whale song features, patterns, and geographic variation**, Australian Acoustical Society proceedings.

## 12. Limits and next improvements

LTSA is excellent for rapid review of massive archives, but it intentionally discards fine temporal structure. The strongest next-stage detector should operate on candidate-centered native-rate or low-rate spectrogram clips and combine:

- normalized spectrogram cross-correlation against confirmed regional templates;
- the published Shannon-entropy/blob features;
- explicit doublet-envelope matching;
- sequence-level rhythm scoring over several calls;
- hard-negative training with ships, seismic pulses, flow noise, fin/blue/Bryde’s calls, and electrical hum;
- recorder-specific calibration/response correction;
- manual validation with provenance retained.

The current candidate CSV/Audacity output is designed to make that second stage straightforward without slowing the initial archive-wide scan.
