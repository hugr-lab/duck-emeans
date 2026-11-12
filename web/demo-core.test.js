// node --test web/ (npm test). The pure parts of the demo; the page itself is checked by hand.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  emeansSql, gaussian, highDimSql, makeBlobs, matrixLiteral, mulberry32, palette, plain, readResult, toCanvas,
} from './demo-core.js';

test('mulberry32 repeats for a seed and stays in [0, 1)', () => {
  const a = mulberry32(7), b = mulberry32(7), c = mulberry32(8);
  const xs = Array.from({ length: 1000 }, a);
  assert.deepEqual(xs.slice(0, 5), Array.from({ length: 5 }, b));
  assert.notEqual(xs[0], c());
  assert.ok(xs.every(x => x >= 0 && x < 1));
});

test('gaussian has mean about 0 and variance about 1', () => {
  const rng = mulberry32(1);
  const xs = Array.from({ length: 20000 }, () => gaussian(rng));
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
  assert.ok(Math.abs(mean) < 0.03, `mean ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.05, `variance ${variance}`);
});

test('makeBlobs puts row i near centre i % centres', () => {
  const points = makeBlobs({ n: 300, centres: 3, spread: 0, seed: 5 });
  assert.equal(points.length, 300);
  assert.deepEqual(points[3], points[0]);
  assert.deepEqual(points[4], points[1]);
  assert.notDeepEqual(points[0], points[1]);
  assert.ok(points.every(([x, y]) => x >= 0.15 && x <= 0.85 && y >= 0.15 && y <= 0.85));
});

test('matrixLiteral and emeansSql', () => {
  assert.equal(matrixLiteral([[0, 1], [0.123456, 2]]), '[[0.00000,1.00000],[0.12346,2.00000]]::FLOAT[][]');
  assert.equal(emeansSql({ points: [[0, 0], [1, 1]], k: 2, nepochs: 2, seed: 9 }),
    'SELECT emeans([[0.00000,0.00000],[1.00000,1.00000]]::FLOAT[][], 2, 2, 9) AS r');
  assert.throws(() => emeansSql({ points: [[0, 0]], k: 2, nepochs: 2, seed: 1 }), /k must be a whole number from 1 to 1/);
  assert.throws(() => emeansSql({ points: [[0, 0]], k: 1, nepochs: 0, seed: 1 }), /nepochs/);
  assert.throws(() => emeansSql({ points: [[0, 0]], k: 1, nepochs: 1, seed: -1 }), /seed/);
  assert.throws(() => emeansSql({ points: [[0, 0]], k: 1.5, nepochs: 1, seed: 1 }), /k must/);
});

test('highDimSql', () => {
  const q = highDimSql({ n: 100, d: 768, groups: 8, k: 16, nepochs: 2 });
  assert.match(q.create, /range\(768\)/);
  assert.match(q.create, /range\(100\) t\(i\)/);
  assert.equal(q.cluster, 'SELECT emeans(list(v ORDER BY i), 16, 2, 1) AS r FROM hd');
  assert.throws(() => highDimSql({ n: 10, d: 768, groups: 8, k: 11, nepochs: 2 }), /k must/);
  assert.throws(() => highDimSql({ n: 10, d: 5000, groups: 8, k: 2, nepochs: 2 }), /d must/);
});

test('plain and readResult turn Arrow-like values into plain ones', () => {
  const vector = values => ({ toArray: () => values });
  assert.equal(plain(5n), 5);
  assert.deepEqual(plain(new Float32Array([1, 2])), [1, 2]);
  assert.deepEqual(plain(vector([vector(new Float32Array([0.5, 1.5])), vector([2, 3])])), [[0.5, 1.5], [2, 3]]);
  const struct = {
    centroids: vector([vector(new Float32Array([1, 2]))]), labels: vector(new Int32Array([1, 1])),
    obj: vector(new Float64Array([3, 2])), batch_size: 1n, alpha0: 0.25, n_splits: 0n,
  };
  assert.deepEqual(readResult({ toJSON: () => struct }),
    { centroids: [[1, 2]], labels: [1, 1], obj: [3, 2], batchSize: 1, alpha0: 0.25, nSplits: 0 });
  assert.deepEqual(readResult(struct).labels, [1, 1]);
});

test('palette and toCanvas', () => {
  const colours = palette(12);
  assert.equal(new Set(colours).size, 12);
  assert.equal(colours[0], 'hsl(0 70% 45%)');
  assert.equal(palette(2, 65)[1], 'hsl(138 70% 65%)');
  assert.deepEqual(toCanvas([0, 0], 100, 10), [10, 90]);
  assert.deepEqual(toCanvas([1, 1], 100, 10), [90, 10]);
});
