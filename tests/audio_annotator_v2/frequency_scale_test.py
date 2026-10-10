"""Browser regression for all six spectrogram frequency scales.

Run with: python tests/audio_annotator_v2/run.py --suite frequency_scale
Uses real Chromium, a generated tone WAV and the existing in-repository harness.
"""
import io
import json
import math
import struct
import wave

from playwright.sync_api import sync_playwright
from common import OUT, launch, mount

SCALES = ('linear', 'log', 'mel', 'bark', 'erb', 'period')
RATE = 96000
NYQUIST = RATE / 2


def fixture():
    # A short native 96 kHz two-tone PCM16 file, generated entirely in memory.
    with io.BytesIO() as stream:
        with wave.open(stream, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(struct.pack('<h', round(
                10500 * math.sin(2 * math.pi * 1200 * i / RATE)
                + 9500 * math.sin(2 * math.pi * 28000 * i / RATE)))
                for i in range(RATE // 2)))
        return stream.getvalue()


def assert_near(actual, expected, *, hz=0.001):
    assert math.isfinite(actual) and abs(actual - expected) <= hz, (actual, expected)


def wait(page, scale):
    page.wait_for_function("""name => {
      const s = AudioAnnotatorV2.inspect();
      return !!s.meta && !s.renderPending && !!s.render && s.render.scale === name;
    }""", arg=scale, timeout=60000)


def screen_xy(page, hz, x=.35):
    box = page.locator('#specOverlay').bounding_box()
    fraction = page.evaluate('(f)=>AudioAnnotatorV2.frequencyPosition(f)', hz)
    # Match the canvas' native plot margins, not the full DOM bounding box.
    return box['x'] + 64 + x * (box['width'] - 80), box['y'] + 23 + (1 - fraction) * (box['height'] - 51)


def drag(page, lower, upper):
    page.locator('#specOverlay').scroll_into_view_if_needed()
    x1, y1 = screen_xy(page, lower, .2)
    x2, y2 = screen_xy(page, upper, .65)
    page.mouse.move(x1, y1)
    page.mouse.down()
    page.mouse.move(x2, y2, steps=8)
    page.mouse.up()


def main():
    records = []
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={'width': 1500, 'height': 1200}, accept_downloads=True)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        mount(page)
        page.locator('#fileInput').set_input_files({
            'name': 'frequency_scales_96k.wav', 'mimeType': 'audio/wav', 'buffer': fixture()
        })
        wait(page, 'linear')
        native = page.evaluate('AudioAnnotatorV2.readNativeSamples(100, 16)')
        page.locator('#fftSize').select_option('4096')
        wait(page, 'linear')

        for scale in SCALES:
            page.locator('#fullBandBtn').click()
            page.locator('#frequencyScale').select_option(scale)
            wait(page, scale)
            info = page.evaluate('AudioAnnotatorV2.frequencyAxis()')
            assert info['scale'] == scale
            assert_near(info['max'], NYQUIST)
            assert_near(info['min'], 1 if scale in ('log', 'period') else 0)
            axis = page.evaluate("""() =>
                Array.from({length:101},(_,i)=>AudioAnnotatorV2.frequencyAtPosition(i/100))""")
            assert all(math.isfinite(v) for v in axis)
            assert all(b > a for a, b in zip(axis, axis[1:])), (scale, axis)
            assert_near(axis[0], info['min'])
            assert_near(axis[-1], info['max'])
            for hz in (info['min'], 2, 12, 100, 1200, 5000, 28000, NYQUIST):
                if hz < info['min']:
                    continue
                fraction = page.evaluate('(f)=>AudioAnnotatorV2.frequencyPosition(f)', hz)
                restored = page.evaluate('(f)=>AudioAnnotatorV2.frequencyAtPosition(f)', fraction)
                assert_near(restored, hz, hz=max(.002, hz * 1e-9))
            # Hit testing and annotation drawing use precisely the same mapping.
            start = 2 if scale == 'period' else 1200
            end = 8 if scale == 'period' else 28000
            page.locator('input[name=mode][value=frequency_range]').check()
            before = page.evaluate('AudioAnnotatorV2.inspect().markers.length')
            drag(page, start, end)
            page.wait_for_function('(n)=>AudioAnnotatorV2.inspect().markers.length===n', arg=before+1)
            markers = page.evaluate('AudioAnnotatorV2.inspect().markers')
            newest = markers[-1]
            assert_near(newest['freqMin'], start, hz=max(.1, start*.003))
            assert_near(newest['freqMax'], end, hz=max(.1, end*.003))
            assert newest['type'] == 'frequency_range'
            # Alt+wheel zoom is about the pointer in transformed coordinates.
            page.locator('#fullBandBtn').click()
            wait(page, scale)
            mid_hz = page.evaluate('AudioAnnotatorV2.frequencyAtPosition(.5)')
            px, py = screen_xy(page, mid_hz)
            page.mouse.move(px, py)
            page.keyboard.down('Alt')
            page.mouse.wheel(0, -100)
            page.keyboard.up('Alt')
            page.wait_for_function('!AudioAnnotatorV2.inspect().renderPending', timeout=60000)
            assert page.evaluate('AudioAnnotatorV2.inspect().render.scale') == scale
            mid_after = page.evaluate('(f)=>AudioAnnotatorV2.frequencyPosition(f)', mid_hz)
            assert_near(mid_after, .5, hz=.005)
            # No sampling/resampling is caused by switching axes.
            assert page.evaluate('AudioAnnotatorV2.readNativeSamples(100, 16)') == native
            records.append({'scale': scale, 'status': 'PASS', 'axis': info,
                            'frequency_range_hz': [newest['freqMin'], newest['freqMax']],
                            'zoom_anchor_fraction': mid_after})

        # A band-preserving change of Y scale must not alter existing annotation Hz.
        saved = page.evaluate('AudioAnnotatorV2.inspect().markers')
        for scale in SCALES:
            page.locator('#frequencyScale').select_option(scale)
            wait(page, scale)
            assert page.evaluate('AudioAnnotatorV2.inspect().markers') == saved
        # The selected scale is recorded as view metadata, not marker coordinates.
        with page.expect_download() as event:
            page.locator('#exportJson').click()
        location = OUT / 'frequency-axis-annotations.json'
        event.value.save_as(location)
        data = json.loads(location.read_text(encoding='utf8'))
        assert data['view']['frequencyScale'] == SCALES[-1]
        assert len(data['markers']) == len(SCALES)
        assert not errors, errors
        (OUT / 'frequency-scale-results.json').write_text(
            json.dumps({'browser': browser.version, 'tests': records, 'pageErrors': errors}, indent=2),
            encoding='utf8')
        browser.close()
    print(f'PASS: {len(records)} frequency scales with render, inverse Hz mapping, '
          'annotations, zoom anchor, native audio and JSON metadata.')


if __name__ == '__main__':
    main()
