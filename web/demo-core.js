// The demo's logic, without the DOM or DuckDB: data generation, the SQL it sends, and the conversion of
// DuckDB-WASM's Arrow results to plain values. Tested by demo-core.test.js (npm test).

// A small seeded generator (mulberry32), so the same seed draws the same points in every browser.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A standard normal draw (Box-Muller); 1 - u keeps the logarithm finite.
export function gaussian(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

// n points around `centres` centres placed uniformly in [0.15, 0.85]^2, each with N(0, spread^2) noise.
export function makeBlobs({ n, centres, spread, seed }) {
  const rng = mulberry32(seed);
  const middles = Array.from({ length: centres }, () => [0.15 + 0.7 * rng(), 0.15 + 0.7 * rng()]);
  return Array.from({ length: n }, (_, i) => {
    const [cx, cy] = middles[i % centres];
    return [cx + spread * gaussian(rng), cy + spread * gaussian(rng)];
  });
}

function requireInteger(name, value, min, max) {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be a whole number from ${min} to ${max}, got ${value}`);
  }
}

function requireNepochs(nepochs) {
  if (!(nepochs > 0 && nepochs <= 50)) {
    throw new Error(`nepochs must be above 0 and at most 50, got ${nepochs}`);
  }
}

// The points as a SQL list literal of FLOAT[2].
export function matrixLiteral(points) {
  return `[${points.map(([x, y]) => `[${x.toFixed(5)},${y.toFixed(5)}]`).join(',')}]::FLOAT[][]`;
}

// The query for one clustering of the points; every number is checked before it reaches the SQL.
export function emeansSql({ points, k, nepochs, seed }) {
  requireInteger('k', k, 1, Math.max(1, points.length));
  requireNepochs(nepochs);
  requireInteger('seed', seed, 0, 4294967295);
  return `SELECT emeans(${matrixLiteral(points)}, ${k}, ${nepochs}, ${seed}) AS r`;
}

// n vectors of dimension d around `groups` centres, built in SQL so nothing large crosses into JavaScript,
// and the clustering of them into k.
export function highDimSql({ n, d, groups, k, nepochs }) {
  requireInteger('n', n, 1, 200000);
  requireInteger('d', d, 1, 4096);
  requireInteger('groups', groups, 1, 1024);
  requireInteger('k', k, 1, n);
  requireNepochs(nepochs);
  return {
    create:
      `CREATE OR REPLACE TEMP TABLE hd AS SELECT i, list_transform(range(${d}), ` +
      `lambda j: ((i % ${groups}) * 7 + j) % 13 * 0.25 + random() - 0.5)::FLOAT[] AS v FROM range(${n}) t(i)`,
    cluster: `SELECT emeans(list(v ORDER BY i), ${k}, ${nepochs}, 1) AS r FROM hd`,
  };
}

// The points of one data set, x0 y0 x1 y1 ..., as CSV with a row number.
export function pointsCsv(xy) {
  const lines = ['i,x,y'];
  for (let i = 0; i < xy.length / 2; i++) {
    lines.push(`${i},${xy[2 * i]},${xy[2 * i + 1]}`);
  }
  return lines.join('\n');
}

// One recorded run over the tables train and test (each a single row v FLOAT[][]); see emeans_trace.
export function traceSql({ k, nepochs, seed, method, framesPerEpoch, batchSize, revival }) {
  requireInteger('k', k, 1, 1e7);
  requireNepochs(nepochs);
  requireInteger('seed', seed, 0, 4294967295);
  requireInteger('batchSize', batchSize, 0, 1e9);
  if (method !== 'emeans' && method !== 'lloyd') {
    throw new Error(`method must be 'emeans' or 'lloyd', got ${method}`);
  }
  return `SELECT emeans_trace((SELECT v FROM train), (SELECT v FROM test), ${k}, ${nepochs}, ${seed}, ` +
    `'${method}', ${Number(framesPerEpoch)}, ${batchSize}, ${revival === true}) AS t`;
}

// The emeans_trace struct as plain JavaScript, centroids flattened to x0 y0 x1 y1 ...
export function readTrace(t) {
  const s = plain(typeof t.toJSON === 'function' ? t.toJSON() : t);
  return {
    batchSize: s.batch_size,
    alpha0: s.alpha0,
    frames: s.frames.map(f => ({
      epoch: f.epoch, step: f.step, trainMse: f.train_mse, testMse: f.test_mse ?? NaN, centroids: f.centroids.flat(),
    })),
  };
}

// DuckDB-WASM hands back Arrow values: vectors with toArray(), struct rows with toJSON(), typed arrays, and
// BigInt for BIGINT.
export function plain(value) {
  if (typeof value === 'bigint') {
    return Number(value);
  }
  // Struct rows have toArray() too, but it drops the field names: toJSON() first.
  if (value && typeof value.toJSON === 'function') {
    return plain(value.toJSON());
  }
  if (value && typeof value.toArray === 'function') {
    return plain(value.toArray());
  }
  if (ArrayBuffer.isView(value) || Array.isArray(value)) {
    return Array.from(value, plain);
  }
  return isRecord(value) ? Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)])) : value;
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype;
}

// The emeans() struct as plain JavaScript.
export function readResult(r) {
  const s = typeof r.toJSON === 'function' ? r.toJSON() : r;
  return {
    centroids: plain(s.centroids),
    labels: plain(s.labels),
    obj: plain(s.obj),
    batchSize: plain(s.batch_size),
    alpha0: plain(s.alpha0),
    nSplits: plain(s.n_splits),
  };
}

// k distinct hues around the colour wheel, offset by the golden angle so neighbours differ. lightness is
// the HSL lightness: about 42 against a light background, 65 against a dark one.
export function palette(k, lightness = 45) {
  return Array.from({ length: k }, (_, j) => `hsl(${Math.round((j * 137.508) % 360)} 70% ${lightness}%)`);
}

// [0, 1]^2 (a little beyond, for the noise) to canvas pixels, y up.
export function toCanvas([x, y], size, pad) {
  const span = size - 2 * pad;
  return [pad + x * span, size - pad - y * span];
}
