# Changelog

## Unreleased

- `emeans_trace()`: a run of e-means or Lloyd recorded as frames of centroids with train and held-out test
  MSE. The core gains `constant_step` (with full batches and alpha0 = 1, Lloyd) and a frame callback.
- The playground (`web/playground/`, Scala.js + Laminar): Lloyd and e-means side by side with Voronoi
  cells and the loss over epochs, run in DuckDB-WASM and replayed.
- Browser demo (`web/`, GitHub Pages): DuckDB-WASM with the `wasm_eh` build of the extension, clustering
  points in the plane and timing embedding-sized vectors (d up to 1536).
- Threads: a large batch's assignment is split across up to `emeans_threads` threads (default: DuckDB's
  `threads` setting), with identical results on any number. POSIX threads, Win32 threads, and one thread
  in WASM.
- macOS: the kernel's dot products go to Accelerate's `cblas_sgemm`. A million 128-d vectors into 256
  clusters: 1.23 s on 18 threads, 2.44 s on one (was 6.30 s); FAISS takes 2.52 s.
- A blocked distance kernel (8 rows × 256 centroids, centroids transposed, vectorised by the compiler),
  with the chosen centroid's distance recomputed exactly: `emeans()` is 2.3× to 5.4× faster (6.3 s for a
  million 128-d vectors into 256 clusters, was 33.8 s).
- `emeans_spherical()` and `emeans_assign_spherical()`: the cosine variant of the reference, for unit
  vectors; within 0.3% of `NumpyEmeans(spherical=True)`.
- `emeans_assign` prepares the centroids once per distinct list in a chunk, so constant centroids (a
  literal, parameter, scalar subquery or one-row cross join) cost once per 2048 rows: 0.4 s for 100,000
  rows against 256 × 128 centroids, was 4.7 s.
- Validation against faiss-cpu 1.16.0 (`make compare_faiss`, also in CI): the recipe matches for 220
  shapes, the final objective is within 1% of `NumpyEmeans` (−0.43% at d = 16, +0.36% at d = 128), and
  `emeans_assign` agrees with `faiss.knn`. Benchmark (`make bench`) in docs/BENCHMARK.md.
- SQL: `emeans(vectors, k [, nepochs [, seed]])` returns centroids, 1-based labels, the objective per pass,
  the batch size, alpha0 and the number of splits; `emeans_assign(vector, centroids)` returns the nearest
  centroid and the squared distance. Vectors as FLOAT, DOUBLE or integer lists or fixed-size arrays.
  The setting `emeans_max_work` (default 1e11) bounds the work of one call.
- The e-means loop in the C core (`emeans_fit`): EMA centroid updates on mini-batches with the cosine
  step schedule, dead-cluster revival every 5 steps, and the closing pass. PCG32 random numbers, exact
  nearest-centroid assignment, a work limit, one allocation per call. Not yet callable from SQL.
- Repository scaffold: C11 extension on the stable C API, CMake and Makefile, CI (distribution
  pipeline on DuckDB v2.0.0, core tests under ASan/UBSan and TSan, CRAP gate).
- `emeans_recipe(n, k, d, nepochs [, batch_law])`: batch size, initial step, law, number of passes and
  total steps from the scaling laws in `faiss/contrib/e_means/base.py`.
