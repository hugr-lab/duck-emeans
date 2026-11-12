# Changelog

## Unreleased

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
