# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

`emeans`: a DuckDB loadable extension for e-means clustering (Bessa et al., arXiv 2611.09501): mini-batch
k-means with EMA centroid updates, batch size and step from fitted scaling laws, dead-cluster revival
and a closing mean pass. For DuckDB 2.0.0. The numerics and SQL functions are C11; the extension is
moving from the stable C API to DuckDB's C++ API (docs/PLAN.md, "Moving to the C++ API"): stage 1
builds it as a C++ extension whose entry point registers the C functions through the C API.

## Hard rules

- `src/core/` and `src/functions/` are C11. C++ goes in `src/cpp/` and follows the stages in
  docs/PLAN.md; do not add C++ elsewhere until a stage says so.
- Until stage 4, `src/functions/` must also build as the C API extension (`make -f Makefile.capi`):
  only call C API functions present in `duckdb_capi/` (extension API v1.5.6, what DuckDB 2.0.0 ships).
  No `duckdb_extension_v2.h`, no `USE_UNSTABLE_C_API`. Tests run on DuckDB 2.0.0 only.
- `src/core/` has no DuckDB dependency.
- The reference is `faiss/contrib/e_means/base.py` and `e_means.py` (MIT), not the paper: the shipped
  scaling-law coefficients differ from the paper's Table 2. Do not copy the Python; reimplement.
- Random permutations cannot match numpy bit for bit, so tests compare objectives and assignments
  against the reference, and exact values only where the arithmetic is deterministic (recipe, Lloyd
  special case with B = n and alpha0 = 1).

## Commands

```sh
GEN=ninja make release              # DuckDB + build/release/extension/emeans/emeans.duckdb_extension
make test_release                   # SQL tests in DuckDB's unittest
make test_core                      # C core tests under ASan/UBSan
make test_threads                   # core on 8 threads under ThreadSanitizer
make crap                           # CRAP < 10 for every C and C++ function; CI fails otherwise
make compare_faiss                  # against faiss-cpu NumpyEmeans (uv, pinned Python 3.12 env); CI too
make bench                          # timings for docs/BENCHMARK.md; manual
```

Use `rtk` as a command prefix in this environment (see the user's global instructions).

## Where things are

- `src/core/recipe.c`: scaling laws, pass schedule, cosine step schedule. `rng.c`: PCG32 and the partial
  shuffle. `assign.c`: the blocked kernel (`emeans_kernel_*`, used everywhere) and the exact loop it is tested
  against. `emeans.c`: the loop, revival and closing (`emeans_fit`, Euclidean and spherical), one
  allocation per call; the kernel's transposed centroids are refreshed whenever a centroid moves.
  `parallel.c`: the fork-join used to split a large batch. On macOS the kernel uses Accelerate's
  `cblas_sgemm` (`EMEANS_USE_ACCELERATE`, set by CMake and the Makefile); `use_blas = 0` forces the
  portable loop, so the tests cover both.
- `src/core/geo.c`: WKB points, geographic CRSs, longitude/latitude to unit vectors, great-circle
  distance, the row sort. `src/functions/emeans_agg.c` (aggregate) and `emeans_geo.c`
  (`emeans_assign_point`) take the point as ANY: a GEOMETRY parameter would cast `GEOMETRY('OGC:CRS84')`
  to plain GEOMETRY and lose the CRS. Core DuckDB knows `OGC:CRS84` without the spatial extension, so the
  tests use it.
- `src/functions/*.c`: one file per SQL function (`emeans_cluster.c` is `emeans()`); `vec_io.c` reads
  lists of vectors into float matrices and writes nested lists. Labels are 1-based in SQL, 0-based in the core.
- `src/cpp/`: the C++ side. `emeans_extension.cpp` is the entry point; `emeans_mem.cpp` backs
  `emeans_mem.h` with the buffer manager (every row-sized buffer in `src/functions` comes from there);
  `emeans_window.cpp` is `emeans_cluster_id`, a DuckDB `WindowFunction` that fits each partition with
  `emeans_agg`'s C code through `emeans_points.h`. It must stay a `WindowFunction`: a window-only
  aggregate is rewritten into a GROUP BY by the window self-join optimizer and fails.
- `docs/PLAN.md`: the phased plan; `docs/ALGORITHM.md`: the formulas.
- `web/playground/`: the playground, Scala.js + Laminar built with Mill (`make -C web/playground test`,
  `make -C web/playground dist`); it reaches DuckDB through `web/playground-bridge.js`.
- `web/`: the browser demo. `demo-core.js` is pure and tested (`npm test`); `demo.js` is the page;
  `duckdb.js` pins DuckDB-WASM `1.33.1-dev114.0`, the build of DuckDB v2.0.0 (extensions only load in the
  release they were built for). `.github/workflows/pages.yml` deploys it with the `wasm_eh` build.

## Gotchas

- The C++ build needs the shim header first: DuckDB's own `src/include/duckdb_extension.h` turns every
  `duckdb_*` call into a function-table lookup that the C++ entry point never fills
  (`target_include_directories(... BEFORE ...)` in CMakeLists.txt).
- Without `OVERRIDE_GIT_DESCRIBE=v2.0.0` a DuckDB 2.0 build is stamped with the commit hash, and the
  binary refuses to load into the released DuckDB.
- DuckDB 2.0 rejects `x -> ...` lambdas by default. Use `lambda x: ...` in tests and docs.
- `generate_series` has no DOUBLE overload; build grids from `range()`.
- The sqllogictest runner stops at the first failing file.
- Every C function stays at CRAP < 10 (`make crap`): small, single-purpose functions; core allocation
  through `emeans_malloc`/`emeans_calloc`/`emeans_realloc` so fault injection reaches the out-of-memory
  paths.
- No mutable global or static state in src/: calls run concurrently on DuckDB's threads.
- A C API scalar function cannot be interrupted: `emeans_fit` refuses work above `max_work` (rows × k × d
  over all passes) up front; `emeans()` takes it from the setting `emeans_max_work` at bind.
- With no git commits, set `EXTENSION_VERSION` or the metadata footer is corrupt.
- Expected values in tests must be derived from the model, not copied from output. Carry the arithmetic
  in a comment.
