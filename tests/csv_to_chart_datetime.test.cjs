// Run with: node tests/csv_to_chart_datetime.test.cjs
// No browser, package manager or third-party dependencies required.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'csv_to_chart.html'), 'utf8');
const match = html.match(/<script>\s*([\s\S]*?)<\/script>/);
assert.ok(match, 'inline application script exists');

const elements = new Map();
let tableRows = [];
let echartsOption, plotlyResult;
function element(id = '') {
  if (!elements.has(id)) {
    elements.set(id, {
      id, style: {}, value: id === 'xColumnSel' ? 'auto' : '', checked: false,
      children: [], options: [], addEventListener() {},
      appendChild(child) { this.children.push(child); },
      append(value) { this.children.push(value); },
      add(option) { this.options.push(option); },
      querySelectorAll() { return []; },
      set innerHTML(value) { this.html = value; this.options = []; this.children = []; },
      get innerHTML() { return this.html; },
    });
  }
  return elements.get(id);
}
const plotly = {
  react(id, traces, layout) { plotlyResult = { id, traces, layout }; },
  purge() {},
};
const context = vm.createContext({
  console,
  window: { addEventListener() {}, Plotly: plotly },
  Plotly: plotly,
  document: {
    getElementById: element,
    createElement() { return element(Symbol()); },
    createDocumentFragment() { return element(Symbol()); },
    querySelectorAll(selector) {
      return selector === '#table-container table tr' ? tableRows : [];
    },
  },
  Option: function(label, value) { this.label = label; this.value = value; },
  requestAnimationFrame() {},
  fetch: () => Promise.reject(new Error('sample.csv unavailable in tests')),
  Blob: class { constructor(parts) { this.size = parts.join('').length; } },
  echarts: { init() {
    return {
      setOption(option) { echartsOption = option; },
      dispose() {}, resize() {},
    };
  } },
});
vm.runInContext(match[1], context, { filename: 'csv_to_chart.html' });

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log('PASS', name);
}
const parse = context.parseDateCell;
test('ISO 8601 with timezone offsets, Z and fractional precision', () => {
  assert.equal(parse('2026-11-12T10:20:30.123+08:00'), Date.UTC(2026, 10, 12, 2, 20, 30, 123));
  assert.equal(parse('2026-11-12T02:20:30Z'), Date.UTC(2026, 10, 12, 2, 20, 30));
  assert.equal(parse('2026-11-12 02:20:30.123456Z'), Date.UTC(2026, 10, 12, 2, 20, 30, 123));
  assert.equal(parse('2026-11-12T10:20:30+99:00'), null);
});

test('strict calendar validation rejects numeric IDs and impossible dates', () => {
  for (const value of ['2026-02-30', '2026-13-01', '2026-11-12T25:00', '03/13/2026',
    '12345', '1234567890', '2026-04-31', '2026-11-12T10:60']) {
    assert.equal(parse(value), null, value);
  }
  assert.ok(parse('2024-02-29') !== null);
  assert.equal(parse('2025-02-29'), null);
});

test('non-ISO and English-month formats', () => {
  assert.ok(parse('31/12/2026', 'DMY') !== null);
  assert.ok(parse('12/31/2026', 'MDY') !== null);
  assert.ok(parse('31.12.2026', 'DMY') !== null);
  assert.ok(parse('2026/12/31') !== null);
  assert.ok(parse('31 Oct 2026') !== null);
  assert.ok(parse('Oct 31, 2026') !== null);
  assert.equal(parse('not a date'), null);
});

test('numeric epochs/compact dates require an appropriate header context', () => {
  assert.equal(parse('1735689600'), null);
  assert.equal(parse('20261231'), null);
  assert.equal(parse('1735689600', 'DMY', true), Date.UTC(2025, 0, 1));
  assert.equal(parse('1735689600.25', 'DMY', true), Date.UTC(2025, 0, 1, 0, 0, 0, 250));
  assert.equal(parse('1735689600000', 'DMY', true), Date.UTC(2025, 0, 1));
  assert.ok(parse('20261231', 'DMY', false, false, true) !== null);
});

test('date ranking uses header, completeness, precision and format', () => {
  const rows = [
    ['1', '15.3', '2026-11-12T10:20:30+08:00', '2026-11-12', '12/31/2026'],
    ['2', '16.3', '2026-11-13T10:20:30+08:00', '2026-11-13', '01/01/2027'],
    ['3', '17.3', '2026-11-14T10:20:30+08:00', '2026-11-14', '01/02/2027'],
  ];
  const candidates = context.analyzeDateColumns(
    ['id', 'value', 'event_timestamp', 'date', 'us_date'], rows);
  assert.equal(candidates[0], null);
  assert.equal(candidates[1], null);
  assert.ok(candidates[2].score > candidates[3].score);
  assert.equal(candidates[4].order, 'MDY');
  assert.equal(context.analyzeDateColumns(['id'], [['1'], ['2']])[0], null);
});

test('sampling remains bounded on a 120,000-row dataset', () => {
  assert.equal(context.sampleRowIndices(120000).length, 96);
  const rows = Array.from({ length: 120000 }, (_, i) =>
    [String(i), new Date(Date.UTC(2026, 0, 1) + i * 60000).toISOString(), '1']);
  const candidates = context.analyzeDateColumns(['id', 'event_time', 'value'], rows);
  assert.ok(candidates[1]);
  assert.equal(candidates[0], null);
  assert.ok(Array.isArray(context.computeNightRanges(rows.map(row => row[1]), 1, 103)));
  assert.equal(context.computeNightRanges([null, null], 1, 103).length, 0);
});

test('CSV load and both renderers use the detected non-first X column', () => {
  const csv = [
    'id,value,event_timestamp,date',
    '1,4,2026-11-12T10:00:00+08:00,2026-11-12',
    '2,5,2026-11-13T10:00:00+08:00,2026-11-13',
    '3,6,2026-11-14T10:00:00+08:00,2026-11-14',
  ].join('\n');
  context.loadCSV(csv, 800000);
  assert.match(element('status').textContent, /X: event_timestamp/);
  assert.match(element('xColumnSel').options[0].label, /Auto: event_timestamp/);
  assert.equal(element('col-toggles').children.length, 3);
  context.renderECharts();
  assert.equal(echartsOption.xAxis.type, 'time');
  assert.equal(echartsOption.xAxis.name, 'event_timestamp');
  assert.ok(echartsOption.series.every(series => series.name !== 'event_timestamp'));
  assert.equal(echartsOption.series[0].data[0][0], '2026-11-12T02:00:00.000Z');
  context.renderPlotly();
  assert.equal(plotlyResult.layout.xaxis.type, 'date');
  assert.equal(plotlyResult.layout.xaxis.title.text, 'event_timestamp');
  assert.ok(plotlyResult.traces.every(trace => trace.name !== 'event_timestamp'));
  assert.equal(plotlyResult.traces[0].x[0], '2026-11-12T02:00:00.000Z');
});

test('manual override and category fallback work in both renderers', () => {
  element('xColumnSel').value = '0';
  context.onXColumnChange();
  context.renderECharts();
  assert.equal(echartsOption.xAxis.type, 'category');
  assert.equal(echartsOption.xAxis.name, 'id');
  context.renderPlotly();
  assert.equal(plotlyResult.layout.xaxis.type, 'category');
  assert.equal(plotlyResult.layout.xaxis.title.text, 'id');
  context.loadCSV('id,value\nA,2\nB,3', 15);
  context.renderECharts();
  assert.equal(echartsOption.xAxis.type, 'category');
  assert.match(element('status').textContent, /X: id \(category\)/);
});

test('table edits recompute date detection and normalize changed values', () => {
  context.loadCSV('id,date,value\n1,2026-01-01,10\n2,2026-01-02,11\n3,2026-01-03,12', 80);
  tableRows = [
    { cells: [{ textContent: 'id' }, { textContent: 'date' }, { textContent: 'value' }] },
    { cells: [{ textContent: '1' }, { textContent: '2026-01-01' }, { textContent: '10' }] },
    { cells: [{ textContent: '2' }, { textContent: 'not a date' }, { textContent: '11' }] },
    { cells: [{ textContent: '3' }, { textContent: '2026-01-03' }, { textContent: '12' }] },
  ];
  context.onCellEdit();
  context.renderECharts();
  assert.equal(echartsOption.xAxis.type, 'category'); // 2/3 valid < 85%
  tableRows[2].cells[1].textContent = '2026-01-02';
  context.onCellEdit();
  context.renderECharts();
  assert.equal(echartsOption.xAxis.type, 'time');
  assert.equal(echartsOption.xAxis.name, 'date');
});
console.log('\n' + passed + ' datetime detection/integration test groups passed.');
