// Run with: node tests/audio_annotator_v2/waveform_connected_test.cjs
// No npm dependencies. Tests the actual inline worker and canvas trace functions.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', '..', 'audio_annotatorv2.html'), 'utf8');
const worker = html.match(/<script id="audio-worker" type="text\/plain">([\s\S]*?)<\/script>/)?.[1];
const main = html.match(/<script>\s*([\s\S]*?)<\/script>/)?.[1];
assert.ok(worker && main, 'Inline scripts must be present');
function extract(s, begin, end) {
    const a = s.indexOf(begin);
    const b = s.indexOf(end, a + begin.length);
    assert.ok(a >= 0 && b > a, 'Expected source markers: ' + begin + ', ' + end);
    return s.slice(a, b);
}
const workerScope = {
    Float32Array, Number, Math,
    source: {
        sampleRate: 384000,
        frames: 1000000,
        async read(start, count) {
            assert.ok(Number.isInteger(start) && Number.isInteger(count) && count > 0 && count <= 65536);
            return Float32Array.from({ length: count }, (_, i) => .002 * Math.sin((start + i) * .37));
        }
    },
    guard() {},
    async sleep() {}
};
vm.createContext(workerScope);
vm.runInContext(extract(worker, 'async function envelope(', '\nfunction statistics(') + '\nthis.runEnvelope = envelope;', workerScope);
const traceScope = { Float32Array, Number, Math };
vm.createContext(traceScope);
vm.runInContext(extract(main, '    function drawWaveformTrace(', '    function drawBase(') + '\nthis.trace = drawWaveformTrace;', traceScope);
const g = { top: 23, h: 190 };
function draw(result, amplitude) {
    const events = [], canvas = {
        beginPath: () => events.push(['beginPath']),
        moveTo: (x, y) => events.push(['moveTo', x, y]),
        lineTo: (x, y) => events.push(['lineTo', x, y]),
        stroke: () => events.push(['stroke'])
    };
    traceScope.trace(canvas, g, { ...result, start: 0, span: result.span || 72 / 384000 },
        64, 960, amplitude, 384000);
    return events;
}
(async () => {
    const narrow = await workerScope.runEnvelope(0, 72 / 384000, 100, 0, 1);
    assert.equal(narrow.mode, 'samples', 'Close zoom must preserve native sample order');
    assert.equal(narrow.samples.length, 72);
    assert.equal(narrow.firstSample, 0);
    for (const amp of [1, 16, 128, 512]) {
        const events = draw({ ...narrow, span: 72 / 384000 }, amp);
        const move = events.filter(e => e[0] === 'moveTo');
        const lines = events.filter(e => e[0] === 'lineTo');
        assert.equal(move.length, 1, 'One continuous path, no isolated segments');
        assert.equal(lines.length, 71, 'Every adjacent native sample pair connected');
        assert.ok(lines.every((e, i) => i === 0 || e[1] > lines[i - 1][1]), 'Monotonic time');
        assert.ok(lines.every(e => Number.isFinite(e[2])), 'Finite drawing coordinates');
    }
    const wide = await workerScope.runEnvelope(0, 1200 / 384000, 100, 0, 1);
    assert.equal(wide.mode, 'exact', 'Zoomed-out view uses bounded envelopes');
    assert.equal(wide.wave.length, 200);
    for (const amp of [1, 16, 128, 512]) {
        const events = draw({ ...wide, span: 1200 / 384000 }, amp);
        assert.equal(events.filter(e => e[0] === 'moveTo').length, 1, 'Envelope columns connect');
        assert.equal(events.filter(e => e[0] === 'lineTo').length, 199, 'No detached min/max strokes');
    }
    const single = draw({ samples: new Float32Array([.001]), firstSample: 0, span: 1 / 384000 }, 512);
    assert.equal(single.filter(e => e[0] === 'lineTo').length, 1, 'Single sample is visible');
    assert.ok(worker.includes('result.samples ? [result.samples.buffer]'), 'Transfer native samples to the UI');
    console.log('PASS waveform samples/envelopes at 1×, 16×, 128× and 512×; single-sample window');
})().catch(error => { console.error(error); process.exitCode = 1; });
