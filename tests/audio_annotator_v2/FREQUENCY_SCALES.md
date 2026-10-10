# Frequency-axis scales in Audio Annotator v2

The spectrogram's **Y-axis scale** selector offers six mappings, corresponding to
the Spectrogram Scale choices in Audacity. The default remains **Linear**.
Sonic Visualiser also offers Linear, Logarithmic and Mel displays.

These are **visual remappings of native-rate Hann FFT bins**, not changes to
the underlying samples, pitch, frequency resolution, STFT or exported marker
coordinates. Mel/Bark/ERB displays are **not** filterbank energies. All marker
frequency values remain **Hz**, including when importing and exporting files.

## Options

| Choice | Forward transform of frequency f (Hz) | When it helps |
| --- | --- | --- |
| Linear | `f` | Ultrasonic clicks, harmonics, calibrated native frequency spacing |
| Logarithmic | `ln(f)` (f ≥ 1 Hz) | Equal vertical distance for octaves; tonal/pitch inspection |
| Mel | `1127 ln(1 + f/700)` | Approximate perceptual pitch spacing |
| Bark | Traunmüller's corrected critical-band formula | Psychoacoustic comparisons |
| ERB | `11.17268 ln(1 + 46.06538 f/(f + 14678.49))` | Auditory filter bandwidth-oriented display |
| Period | `−1/f` (with a positive display floor) | A reciprocal-frequency viewpoint; useful mainly for Audacity's Pitch/EAC mode |

For **Bark**, set `z = 26.81 f/(1960 + f) − 0.53`. Use
`z + 0.15(2 − z)` for `z < 2`, `z + 0.22(z − 20.1)` for
`z > 20.1`, otherwise `z`. These are Audacity's corrected
Traunmüller formulas, not the distinct Zwicker approximation.

Mel here follows **Audacity's natural-logarithm formula**; other software may
use HTK Mel (`2595 log10(1 + f/700)`) or Slaney's piecewise-linear/log Mel.
These definitions are similar but should not be assumed numerically identical.

### Coordinates and zoom

For a chosen scale transform `T`, minimum `a`, maximum `b`:

```text
fraction_from_bottom(f) = (T(f) - T(a)) / (T(b) - T(a))
y_in_pixels(f)        = top + (1 - fraction_from_bottom(f)) * height
frequency_at_y(y)     = inverse_T(T(a) + (1 - (y-top)/height) * (T(b)-T(a)))
```

The main thread constructs **one native-Hz frequency edge per image row** using
this inverse. The audio worker uses the edges to select the corresponding FFT
bins. Spectral labels, tooltips, pointer hit-testing, annotation rectangles,
vertical ticks and Alt+wheel zoom use that same transform.

For Alt+wheel, the span is expanded or contracted in transformed space,
keeping the frequency under the pointer as the anchor unless the chosen
window reaches the allowed min/max boundary.

The **Logarithmic** and **Period** transforms are singular at zero.
As in Audacity, an entered minimum of **0 Hz is treated as 1 Hz** for
these views. The input retains the entered `0`, allowing the full 0 Hz band
to return when switching to Linear/Mel/Bark/ERB. The effective band and its
limit appear in diagnostics. No frequency can exceed the file's native
Nyquist limit.

For Period, only the Y-axis changes: unlike Audacity's **Pitch (EAC)**
algorithm, this tool continues to show the ordinary FFT spectrogram.
At typical full-band ultrasonic rates, most Period pixels are concentrated
near the very lowest frequencies. Prefer Linear or ERB for many click analyses.

At narrow zooms, multiple vertical pixels may refer to the same FFT bin:
a display scale **cannot increase actual frequency resolution**, which is
approximately sample_rate / FFT_size. The existing time sampling and
wide-recording performance safeguards still apply.

## Testing

The repository's existing browser test runner invokes the additional suite:

```bash
python tests/audio_annotator_v2/run.py --suite frequency_scale
# or run everything, with optional large synthetic recordings:
python tests/audio_annotator_v2/run.py --large
```

The frequency suite builds a two-tone 96 kHz WAV in memory and verifies
all six scales, effective zero handling, transform inverses, monotonic
frequencies, renderer completion, time-frequency annotation coordinates,
Alt+wheel zoom anchoring, unchanged native PCM and JSON view metadata.

The mathematical transforms have also been checked independently at
sample rates of 44.1, 96, 384 and 1536 kHz (24 combinations) with
maximal tested round-trip error below **6e−9 Hz**. That numerical
check does **not** replace execution of the Chromium regression suite.

## References

- [Audacity Spectrogram Settings (six scale choices)](https://manual.audacityteam.org/man/spectrogram_settings.html)
- [Audacity NumberScale implementation (formulas)](https://fossies.org/dox/audacity-sources-4.0.0/numberscale_8h_source.html)
- [Audacity Spectrogram View (linear, logarithmic, perceptual, period)](https://manual.audacityteam.org/man/spectrogram_view.html)
- [Sonic Visualiser 5.2 reference (linear, logarithmic, Mel)](https://www.sonicvisualiser.org/doc/reference/5.2.1/en/)
- [librosa Mel convention differences](https://librosa.org/doc/main/api/generated/librosa.mel_frequencies.html)
