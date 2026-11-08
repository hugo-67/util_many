"""LTSA numeric, acceleration, detector, format, and browser regression checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ltsa_batch as ltsa  # noqa: E402


class LTSATest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = self.dir / "tone.wav"
        sr = 48000
        seconds = 2.0
        t = np.arange(int(sr * seconds), dtype=np.float64) / sr
        # A strong 6 kHz tone plus a much weaker 2 kHz tone.
        x = 0.65 * np.sin(2 * np.pi * 6000 * t) + 0.08 * np.sin(2 * np.pi * 2000 * t)
        pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
        with wave.open(str(self.wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(pcm.tobytes())

    def tearDown(self):
        self.tmp.cleanup()

    def config(self):
        c = ltsa.merged_config(None)
        c["analysis"].update({
            "fft_size": 1024,
            "overlap": 0.5,
            "average_seconds": 0.5,
            "channel": 0,
            "frequency_min_hz": 0.0,
            "frequency_max_hz": None,
        })
        return c

    def test_detects_tone_and_dimensions(self):
        plan, matrix = ltsa.compute_ltsa(self.wav, self.config(), chunk_windows=64, max_output_mib=64)
        self.assertEqual(matrix.shape, (4, 513))
        self.assertEqual(plan.time_bins, 4)
        self.assertTrue(np.isfinite(matrix).any())
        mean_db = np.nanmean(matrix, axis=0)
        peak_hz = np.nanargmax(mean_db) * 48000 / 1024
        self.assertLess(abs(peak_hz - 6000), 48000 / 1024 * 1.1)
        weak_bin = round(2000 / (48000 / 1024))
        strong_bin = round(6000 / (48000 / 1024))
        self.assertGreater(float(mean_db[strong_bin] - mean_db[weak_bin]), 10.0)

    def test_binary_round_trip(self):
        config = self.config()
        plan, matrix = ltsa.compute_ltsa(self.wav, config, chunk_windows=64, max_output_mib=64)
        out = self.dir / "tone.ltsa"
        header = ltsa.make_header(self.wav, config, plan)
        ltsa.write_ltsa(out, header, matrix)
        got_header, got = ltsa.read_ltsa(out)
        self.assertEqual(got_header["format"], "util_many_ltsa")
        self.assertEqual(got_header["quantity"], "power_dbfs")
        self.assertEqual(got_header["source_sample_rate"], 48000)
        self.assertEqual(got.shape, matrix.shape)
        np.testing.assert_allclose(got, matrix, rtol=0, atol=0, equal_nan=True)

    def test_size_guard_gives_actionable_error(self):
        config = self.config()
        config["analysis"]["average_seconds"] = 0.0001
        with self.assertRaisesRegex(ValueError, "Increase average_seconds"):
            ltsa.compute_ltsa(self.wav, config, chunk_windows=64, max_output_mib=0.01)

    def test_config_schema(self):
        config = ltsa.merged_config({"analysis": {"fft_size": 4096}, "display": {"db_min": -90}})
        self.assertEqual(config["format"], "util_many_ltsa_config")
        self.assertEqual(config["analysis"]["fft_size"], 4096)
        self.assertEqual(config["analysis"]["window"], "hann")
        self.assertEqual(config["display"]["db_min"], -90)
        self.assertIn("processing", config)

    def test_downsampled_chunks_are_boundary_invariant_and_surface_song(self):
        # Use a modest source rate for CI while exercising the same rational
        # polyphase path used by 48--384 kHz SoundTrap/HydroMoth recordings.
        sr = 4800
        duration = 190
        rng = np.random.default_rng(7)
        signal = rng.normal(0, 0.002, sr * duration).astype(np.float32)
        for start in (10, 160):
            t = np.arange(sr * 12, dtype=np.float64) / sr
            envelope = np.sin(np.pi * np.arange(t.size) / t.size) ** 2
            call = 0.16 * np.sin(2 * np.pi * 25 * t) * (1 + 0.35 * np.sin(2 * np.pi * 3 * t)) * envelope
            signal[start * sr:(start + 12) * sr] += call.astype(np.float32)
        source = self.dir / "synthetic_omura.wav"
        sf.write(source, signal, sr, subtype="PCM_16")
        config = ltsa.merged_config({
            "analysis": {
                "fft_size": 1024,
                "overlap": 0.75,
                "average_seconds": 2.0,
                "channel": 0,
                "frequency_min_hz": 5.0,
                "frequency_max_hz": 110.0,
            },
            "processing": {"analysis_sample_rate_hz": 256, "backend": "cpu"},
            "detection": {"enabled": True, "profile": "omuras_whale"},
        })
        short = ltsa.compute_ltsa_detailed(
            source, config, backend="cpu", threads=2,
            memory_budget_bytes=512 * ltsa.MIB, chunk_seconds=3,
        )
        long = ltsa.compute_ltsa_detailed(
            source, config, backend="cpu", threads=1,
            memory_budget_bytes=512 * ltsa.MIB, chunk_seconds=20,
        )
        self.assertEqual(short.plan.source_sample_rate, sr)
        self.assertEqual(short.plan.sample_rate, 256)
        self.assertAlmostEqual(short.plan.sample_rate / short.plan.fft_size, 0.25)
        np.testing.assert_allclose(short.matrix, long.matrix, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(len(short.candidates), 2)
        self.assertTrue(all(candidate.rhythmic_match for candidate in short.candidates))
        self.assertTrue(all(abs(candidate.peak_frequency_hz - 25.0) <= 0.5 for candidate in short.candidates))
        self.assertAlmostEqual(short.candidates[1].start_seconds - short.candidates[0].start_seconds, 150, delta=4)

    def test_resource_planner_caps_aggressive_overrides(self):
        with mock.patch.object(ltsa, "available_memory_bytes", return_value=4 * ltsa.GIB), \
             mock.patch.object(ltsa, "_affinity_cpu_count", return_value=16):
            plan = ltsa.resolve_resources(
                100,
                requested_backend="cpu",
                requested_jobs=999,
                requested_threads=999,
                memory_fraction=0.5,
                gpu_memory_fraction=0.75,
            )
        self.assertLessEqual(plan.jobs, 4)  # 2 GiB budget / 512 MiB minimum
        self.assertLessEqual(plan.jobs * plan.memory_per_worker_bytes, plan.memory_budget_bytes)
        self.assertLessEqual(plan.jobs * plan.threads_per_worker, plan.total_cpu_threads)

    def test_omura_configs_are_valid_and_have_expected_resolution(self):
        expected = {
            "ltsa_omuras_whale_fast_scan.json": (256, 1024, 0.25),
            "ltsa_omuras_whale_sensitive_review.json": (512, 4096, 0.125),
            "ltsa_omuras_whale_song_context.json": (256, 1024, 0.25),
        }
        for name, (rate, fft, resolution) in expected.items():
            config = ltsa.load_config(ROOT / "sample" / name)
            plan = ltsa.make_plan(config, 48000, 48000 * 60, 1)
            self.assertEqual(plan.sample_rate, rate)
            self.assertEqual(plan.fft_size, fft)
            self.assertAlmostEqual(plan.sample_rate / plan.fft_size, resolution)
            self.assertGreaterEqual(plan.frequency_min_hz, 0)
            self.assertLess(plan.frequency_max_hz, rate / 2)

    def test_html_contains_dedicated_ltsa_surface_and_worker_api(self):
        html = (ROOT / "audio_annotatorv2.html").read_text(encoding="utf-8")
        for token in (
            'id="ltsaPlot"', 'id="ltsaCanvas"', 'id="ltsaBuildFull"',
            'id="ltsaPreview"', 'id="ltsaImport"', 'id="ltsaExport"',
            "ltsaCompute", "ltsaCancel", "util_many_ltsa_config",
        ):
            self.assertIn(token, html)

    def test_browser_preview_and_click_navigation(self):
        try:
            from playwright.sync_api import sync_playwright
            from common import F, launch, mount
        except ImportError as exc:
            self.skipTest(f"Playwright browser test unavailable: {exc}")
        audio = F / "ultrasonic_PCM_16.wav"
        if not audio.is_file():
            self.skipTest("Run tests/audio_annotator_v2/run.py so synthetic browser fixtures are generated")
        with sync_playwright() as p:
            browser = launch(p)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                mount(page)
                page.locator("#fileInput").set_input_files(str(audio))
                page.wait_for_function("() => !!window.AudioAnnotatorV2?.inspect().meta")
                page.locator("#ltsaFft").select_option("1024")
                page.locator("#ltsaOverlap").fill("0")
                page.locator("#ltsaAverage").fill("0.5")
                page.locator("#ltsaPreview").click()
                page.wait_for_function(
                    "() => { const s=AudioAnnotatorV2.inspect().ltsa; return !!s.header && !s.running; }",
                    timeout=30000,
                )
                state = page.evaluate("() => AudioAnnotatorV2.inspect()")
                header = state["ltsa"]["header"]
                self.assertEqual(header["sample_rate"], 384000)
                self.assertEqual(header["time_bins"], 4)
                self.assertEqual(header["frequency_bins"], 513)
                self.assertEqual(state["ltsa"]["values"], 4 * 513)
                self.assertNotIn("failed", page.locator("#ltsaStatus").inner_text().lower())

                overlay = page.locator("#ltsaOverlay")
                box = overlay.bounding_box()
                self.assertIsNotNone(box)
                data_width = max(20, box["width"] - 80)
                overlay.click(position={"x": 64 + data_width * 0.5, "y": box["height"] * 0.5})
                page.wait_for_function(
                    "() => Math.abs(AudioAnnotatorV2.inspect().cursor - 1) < 0.1",
                    timeout=5000,
                )
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
