---
title: Demos
sidebar_position: 11
---

# Demos

Both run the extension compiled to WebAssembly inside [DuckDB-WASM](https://duckdb.org/docs/api/wasm/overview):
nothing is installed and no data leaves the page.

## [Playground](https://hugr-lab.github.io/duck-emeans/demo/playground.html)

Lloyd's k-means and e-means side by side on 2-D data, with their Voronoi cells and the train and held-out
test MSE over epochs, both run by [`emeans_trace`](./functions/recipe-and-trace.md#emeans_trace) and then
replayed. It follows the [playground](https://e-means.github.io/playground.html) of the e-means authors,
with the computation in SQL.

## [Demo](https://hugr-lab.github.io/duck-emeans/demo/)

Clusters points in the plane and times embedding-sized vectors (d = 768 or 1536). In the browser the
extension runs on one thread, without Accelerate: 10,000 vectors of dimension 768 into 64 clusters take
about 1 s in Chromium on an M5 Pro.

## Running them locally

They live in `web/` in the repository. `duckdb.js` starts DuckDB-WASM (the build of DuckDB v2.0.0) with
unsigned extensions allowed and loads the `wasm_eh` build of the extension from the distribution pipeline.
The playground is a Scala.js + Laminar app in `web/playground/`, built with Mill. To run both (needs Java
and Node):

```sh
gh run download <run-id> -n emeans-v2.0.0-extension-wasm_eh -D web/
make -C web/playground serve   # http://localhost:8099/playground.html
```
