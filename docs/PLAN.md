# Plan

e-means has no C or C++ source of its own: the reference is Python, with FAISS doing the nearest-centroid
search on CPU and Triton/CuTe kernels on GPU. This extension is a reimplementation in C11 of the
algorithm in `base.py` and `e_means.py` (about 1,100 lines together), against the stable DuckDB C
extension API, for DuckDB 2.0.

## SQL interface

The data goes through a LIST, as in `cpt`: the C API has no table-in/table-out functions, and e-means
needs the whole dataset for its random permutations anyway.

```sql
SELECT emeans(list(embedding ORDER BY id), 1000) FROM docs;
-- STRUCT(centroids LIST<LIST<FLOAT>>, labels LIST<INTEGER>, obj LIST<DOUBLE>,
--        batch_size BIGINT, alpha0 DOUBLE, n_splits INTEGER)

SELECT id, emeans_assign(embedding, c.centroids) FROM docs, (SELECT emeans(...) AS c FROM docs);
-- STRUCT(label INTEGER, distance DOUBLE)

SELECT emeans_recipe(1000000, 1000, 128, 2.0);
-- STRUCT(batch_size, alpha0, batch_law, n_passes, total_steps)
```

Overloads: `emeans(x, k)`, `emeans(x, k, nepochs)`, `emeans(x, k, nepochs, seed)`, and
`emeans_spherical(...)`. Vectors are accepted as `LIST<DOUBLE>`, `LIST<FLOAT>` and the fixed-size
`FLOAT[d]` array that the `vss` extension uses. Centroids come back as `LIST<LIST<FLOAT>>` first;
returning `FLOAT[d]` needs the bind callback to see `d`. `emeans_assign` converts a constant centroid
argument once in bind and caches the norms, so the hot path is a plain n × k × d loop.

## Core (`src/core/`, no DuckDB dependency)

- `recipe.c` (done): batch laws B = c · k^0.134749 · n^0.876473 · T^0.813344 with c = 0.035134
  (throughput) or 0.00439175 (quality; `auto` picks it for d ≤ 16), rounded to the nearest power of
  two and capped; step law α0 = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057 clamped to
  [1e-4, 1]; pass schedule; cosine schedule α = α0 · (0.1 + 0.9 · ½(1 + cos(π · step/total))).
- `rng.c` (done): PCG32 and a partial Fisher–Yates shuffle. Numpy's permutation cannot be matched
  bit for bit, so tests compare objectives and assignments, not centroid bits.
- `assign.c` (done, plain loop): exact nearest centroid, squared differences in double. Phase 5 replaces
  it with ‖x‖² + ‖c‖² − 2 x·c in blocked loops (64 rows × 64 centroids × d) so the compiler can
  vectorise. No BLAS.
- `emeans.c` (done): state is the centroid c_j and mass N_j (initially 2^-16). Per batch: assign, accumulate
  n_j and s_j; N_j ← (1 − α) N_j for all j; for n_j > 0, N_j += α n_j and
  c_j += (α n_j / N_j)(s_j / n_j − c_j). α = 1 is one Lloyd step. Every 5 steps: revival, with
  threshold τ ΣN_j / k (τ = 0.1, none for d ≤ 2), donors sampled with Gumbel keys
  log N_j − log(−log u), split at ±1e-4 relative offset with the mass halved. After the last pass the
  closing mean c_j = pass_sums_j / pass_counts_j; labels are the last pass's assignments. Spherical:
  normalised centroids, argmax dot product, objective Σ(2 − 2·sim).
- Budget: a scalar function cannot be interrupted, so the assignment loop charges a budget and a
  setting bounds it. Threads inside one call come later: DuckDB gives one `emeans()` call one thread,
  and the assignment is where the time goes.

## Phases

1. Scaffold (done): repository, build, CI, test gates, `emeans_recipe`.
2. Core (done): rng, assign kernel, the loop; C tests, Lloyd equivalence (B = n, α0 = 1 reproduces two
   Lloyd iterations computed by hand on a 6-point set), revival on a set with two dead initial centroids,
   closing means, determinism, fault injection, 8 threads under TSan.
3. SQL (done): `emeans`, `emeans_assign`, LIST and ARRAY inputs (arrays through the implicit cast),
   NULL handling, errors, the `emeans_max_work` setting, sqllogictests, a threads test.
4. Validation (done): the recipe matches faiss-cpu 1.16.0 for 220 shapes; the final objective is within
   1% of `NumpyEmeans` on Gaussian mixtures (n = 100k, d = 16 and 128, k = 256); `emeans_assign` agrees
   with `faiss.knn`; benchmark in docs/BENCHMARK.md (2.5× slower than FAISS at d = 16, 13× at d = 128).
5. Done: the blocked kernel, spherical, `emeans_assign` reusing prepared centroids within a chunk, the
   budget setting (phase 3). Not done: `FLOAT[d]` output, because the stable C API cannot set a return
   type at bind (cast in SQL instead); folding constant centroids at bind, because asking for a bind
   argument fails for scalar subqueries.
6. Done: threads inside one call (`src/core/parallel.c`: POSIX threads, Win32 threads, one thread in
   WASM), used only for batches with at least 4 million units of work per thread, and Accelerate's sgemm
   for the kernel on macOS. `emeans()` now beats NumpyEmeans on an M5 Pro (docs/BENCHMARK.md).
7. GPU, as a separate build that users choose: the community extension stays CPU-only, and a second
   build adds a GPU backend for the kernel (MLX or Metal on Apple silicon, CUDA elsewhere). e-means makes
   many sequential steps with small batches, so per-step launch overhead matters; the reference needed
   custom Triton and CuTe kernels for it.
8. Done: the DuckDB-WASM demo (`web/`, GitHub Pages).
9. Done: the playground in the style of the e-means site's (Lloyd and e-means side by side with Voronoi
   cells, train and held-out test MSE over epochs, playback), in Scala.js and Laminar, on `emeans_trace`
   and the core's constant-step (Lloyd) mode.
10. Submission to `duckdb/community-extensions`.

Later, if wanted: an aggregate `emeans_agg(vector, k)` (buffers rows in the state, runs the algorithm
in finalize; the EMA is order-dependent, so combine concatenates), a DuckDB-WASM 2.0 build for a
browser demo, and an "IVF-Flat in SQL" example where `emeans_assign` gives the partition key.

## Risks

- Speed against FAISS: settled on macOS (docs/BENCHMARK.md). On Linux the portable kernel has not been
  measured against FAISS with OpenBLAS or MKL.
- Memory: the `list(embedding)` path holds the data once in DuckDB and once as float32 in the
  function; 10M × 128 is 10 GB + 5 GB.
- Reference drift: blog, abstract and code disagree on the speed-up figures and on the scaling-law
  coefficients. Pin the FAISS commit the tests compare against.
