// The page: controls, the canvas and the timing panel. The logic is in demo-core.js.
import { emeansSql, highDimSql, makeBlobs, palette, readResult, toCanvas } from './demo-core.js';
import { startDuckDB } from './duckdb.js';

const $ = id => document.getElementById(id);
const PAD = 12;
const state = { points: [], labels: [], centroids: [], db: null };

const dark = () => matchMedia('(prefers-color-scheme: dark)').matches;
const token = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function setStatus(text) {
  $('status').textContent = text;
}

function controls() {
  const value = id => Number($(id).value);
  return { n: value('n'), centres: value('centres'), spread: value('spread'), k: value('k'),
           nepochs: value('nepochs'), seed: value('seed') };
}

function showControlValues() {
  for (const id of ['n', 'centres', 'spread', 'k', 'nepochs']) {
    $(`${id}-value`).textContent = $(id).value;
  }
}

//===--------------------------------------------------------------------===//
// Drawing
//===--------------------------------------------------------------------===//

function sizeCanvas(canvas) {
  const size = canvas.clientWidth * devicePixelRatio;
  canvas.width = size;
  canvas.height = size;
  return size;
}

// Colour 0 is for points not clustered yet; labels are 1-based.
function colours(k) {
  return [token('--muted'), ...palette(k, dark() ? 65 : 42)];
}

function drawPoints(ctx, size, fill) {
  const r = 2.2 * devicePixelRatio;
  state.points.forEach((p, i) => {
    const [x, y] = toCanvas(p, size, PAD * devicePixelRatio);
    ctx.fillStyle = fill[state.labels[i] ?? 0];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function drawCentroids(ctx, size, fill) {
  const r = 7 * devicePixelRatio;
  ctx.lineWidth = 2.5 * devicePixelRatio;
  ctx.strokeStyle = token('--fg');
  state.centroids.forEach((c, j) => {
    const [x, y] = toCanvas(c, size, PAD * devicePixelRatio);
    ctx.fillStyle = fill[j + 1];
    ctx.beginPath();
    ctx.rect(x - r, y - r, 2 * r, 2 * r);
    ctx.fill();
    ctx.stroke();
  });
}

function draw() {
  const canvas = $('plot');
  const size = sizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, size, size);
  const fill = colours(state.centroids.length);
  drawPoints(ctx, size, fill);
  drawCentroids(ctx, size, fill);
}

//===--------------------------------------------------------------------===//
// Playground
//===--------------------------------------------------------------------===//

function newPoints() {
  const c = controls();
  state.points = makeBlobs({ n: c.n, centres: c.centres, spread: c.spread, seed: c.seed });
  state.labels = [];
  state.centroids = [];
  $('result').textContent = 'Not clustered yet.';
  draw();
}

function showResult(r, ms, sql) {
  const obj = r.obj.map(o => o.toFixed(3)).join(', ');
  $('result').textContent =
    `${ms.toFixed(0)} ms in DuckDB-WASM. Batch size ${r.batchSize}, alpha0 ${r.alpha0.toFixed(4)}, ` +
    `${r.nSplits} dead clusters revived. Objective per pass: ${obj}.`;
  $('sql').textContent = sql.replace(/\[\[.*\]\]/, '[[x, y], ...]');
}

async function cluster() {
  const c = controls();
  const sql = emeansSql({ points: state.points, k: c.k, nepochs: c.nepochs, seed: c.seed });
  const start = performance.now();
  const rows = await state.db.query(sql);
  const r = readResult(rows[0].r);
  state.labels = r.labels;
  state.centroids = r.centroids;
  draw();
  showResult(r, performance.now() - start, sql);
}

//===--------------------------------------------------------------------===//
// High-dimensional timing
//===--------------------------------------------------------------------===//

async function timeHighDim() {
  const value = id => Number($(id).value);
  const q = highDimSql({ n: value('hd-n'), d: value('hd-d'), groups: 16, k: value('hd-k'), nepochs: 2 });
  $('hd-result').textContent = 'Running...';
  const t0 = performance.now();
  await state.db.query(q.create);
  const t1 = performance.now();
  const r = readResult((await state.db.query(q.cluster))[0].r);
  const t2 = performance.now();
  $('hd-result').textContent =
    `Generating the vectors took ${(t1 - t0).toFixed(0)} ms; clustering them ${(t2 - t1).toFixed(0)} ms ` +
    `(batch size ${r.batchSize}, ${r.obj.length} passes, final objective ${r.obj.at(-1).toFixed(1)}).`;
  $('hd-sql').textContent = `${q.create};\n${q.cluster};`;
}

//===--------------------------------------------------------------------===//
// Start
//===--------------------------------------------------------------------===//

// Runs an action from a button, reporting errors in the status line instead of failing silently.
function guarded(action) {
  return async () => {
    try {
      await action();
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
  };
}

function wire() {
  for (const id of ['n', 'centres', 'spread', 'k', 'nepochs']) {
    $(id).addEventListener('input', showControlValues);
  }
  $('new-points').addEventListener('click', guarded(newPoints));
  $('cluster').addEventListener('click', guarded(cluster));
  $('hd-run').addEventListener('click', guarded(timeHighDim));
  addEventListener('resize', draw);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
}

async function start() {
  wire();
  showControlValues();
  newPoints();
  state.db = await startDuckDB('emeans.duckdb_extension.wasm');
  const version = await state.db.query('SELECT version() AS v');
  setStatus(`DuckDB ${version[0].v} with emeans, running in this browser.`);
  for (const id of ['cluster', 'hd-run']) {
    $(id).disabled = false;
  }
  await cluster();
}

guarded(start)();
