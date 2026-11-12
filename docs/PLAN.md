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
10. Done: spatial clustering. `emeans_agg` (aggregate and window function over POINT geometries or vectors)
    and `emeans_assign_point`, after PostGIS's `ST_ClusterKMeans`; geographic CRSs on the sphere.
11. Move from the stable C API to DuckDB's C++ API (in progress), keeping the numerical core in C.
12. Submission to `duckdb/community-extensions`.

## Moving to the C++ API

The stable C API was chosen so one binary would load in every DuckDB release. emeans targets DuckDB 2.0
only, so that advantage is small, and the C API keeps blocking things the C++ API has:

- Memory that spills. The C API's only allocator is `duckdb_malloc`, a heap wrapper: emeans's buffers
  (a float copy of the input, the aggregate's points) are invisible to `memory_limit` and cannot go to
  disk. The C++ API's `BufferManager` gives spillable, accounted blocks.
- A real `ST_ClusterKMeans`. A C++ window function (`ExtensionLoader::RegisterFunction(WindowFunction)`
  in v2.0.0), or an aggregate's window callback, which receives the current row (`idx_t rid`), can
  return each row's cluster id in one call, where the C API needs `emeans_assign_point` as a second.
- Argument types at bind (a GEOMETRY's CRS without the ANY parameter), return types at bind (`FLOAT[d]`
  centroids), interruptible queries (the work limit becomes optional), DuckDB's task scheduler (in place
  of the thread shim), and bind-time constant folding without the scalar-subquery failure.

The cost: the extension builds against DuckDB's C++ headers (the C++ extension template, DuckDB as a
submodule, compiled in CI), each DuckDB release needs a rebuild (the community pipeline does this), and
internal APIs can change between releases.

Stages, each with every SQL test passing:

1. Scaffold (done on the `cpp-api` branch): the C++ template's build, DuckDB v2.0.0 as a submodule, and
   a C++ entry point that registers the existing C functions unchanged. Inside DuckDB a
   `duckdb_connection` is a `Connection *` (see `duckdb_connect` in `src/main/capi/v1/duckdb-c.cpp`), so
   the C++ `Load` hands the C registrations a `Connection` on the loader's database. The C sources
   include a shim `duckdb_extension.h`, so their calls bind to the C API inside DuckDB. The CRAP gate
   takes its SQL coverage from a DuckDB build with only the extension instrumented, so it scores the
   C++ as well as the C. The C API build stays available (`make -f Makefile.capi`) until stage 4.
2. Spilling memory (done): the input copy, the fit's workspace and the aggregate's points in
   `BufferManager` blocks (`src/include/emeans_mem.h`, `src/cpp/emeans_mem.cpp`). An aggregate state's
   block is pinned only inside a callback, so between callbacks it can go to `temp_directory`.
   `test/sql/emeans_memory.test` runs a `GROUP BY` whose states exceed `memory_limit` and checks that a
   block above the limit fails with DuckDB's error.
3. `emeans_cluster_id(geom, k) OVER (...)`, the per-row id in one call, compared again with PostGIS.
4. Native C++ for the remaining functions, removing the ANY parameter, the thread shim and the C API
   build; `FLOAT[d]` output; interrupts.

Later, if wanted: an "IVF-Flat in SQL" example where `emeans_assign` gives the partition key (see
docs/QDRANT.md for how such an index compares).

## Risks

- Speed against FAISS: settled on macOS (docs/BENCHMARK.md). On Linux the portable kernel has not been
  measured against FAISS with OpenBLAS or MKL.
- Memory: the `list(embedding)` path holds the data once in DuckDB and once as float32 in the
  function; 10M × 128 is 10 GB + 5 GB. Since stage 2 the float32 copy is counted against
  `memory_limit`, but it is pinned for the whole fit, so it cannot spill. The window operator also
  keeps a partition's rows pinned, so `emeans_agg(...) OVER (PARTITION BY ...)` does not spill as the
  `GROUP BY` form does.
- Reference drift: blog, abstract and code disagree on the speed-up figures and on the scaling-law
  coefficients. Pin the FAISS commit the tests compare against.
