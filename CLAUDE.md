# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

`emeans`: a DuckDB loadable extension for e-means clustering (Bessa et al., arXiv 2611.09501): mini-batch
k-means with EMA centroid updates, batch size and step from fitted scaling laws, dead-cluster revival
and a closing mean pass. Written in C11 against the **stable** DuckDB C extension API, for DuckDB 2.0.

## Hard rules

- C11 only. No C++, no DuckDB internals, no `duckdb_extension_v2.h`, no `USE_UNSTABLE_C_API`.
- `duckdb_capi/` holds the extension API v1.5.6 headers. That is the stable C API DuckDB 2.0.0 ships
  (its header accepts no other major version and adds no functions), so `TARGET_DUCKDB_VERSION` stays
  `v1.5.6`. Only call API functions present there. Tests run on DuckDB 2.0 only.
- `src/core/` has no DuckDB dependency.
- The reference is `faiss/contrib/e_means/base.py` and `e_means.py` (MIT), not the paper: the shipped
  scaling-law coefficients differ from the paper's Table 2. Do not copy the Python; reimplement.
- Random permutations cannot match numpy bit for bit, so tests compare objectives and assignments
  against the reference, and exact values only where the arithmetic is deterministic (recipe, Lloyd
  special case with B = n and alpha0 = 1).

## Commands

```sh
make configure && make release      # build/release/emeans.duckdb_extension
make test_release                   # SQL tests, latest DuckDB
make test_core                      # C core tests under ASan/UBSan
make test_threads                   # core on 8 threads under ThreadSanitizer
make crap                           # CRAP < 10 for every C function; CI fails otherwise
```

Use `rtk` as a command prefix in this environment (see the user's global instructions).

## Where things are

- `src/core/recipe.c`: scaling laws, pass schedule, cosine step schedule. Coming: `rng.c`, `assign.c`
  (exact nearest centroid, float32, blocked), `emeans.c` (the loop, revival, closing).
- `src/functions/*.c`: one file per SQL function.
- `docs/PLAN.md`: the phased plan; `docs/ALGORITHM.md`: the formulas.

## Gotchas

- DuckDB 2.0 rejects `x -> ...` lambdas by default. Use `lambda x: ...` in tests and docs.
- `generate_series` has no DOUBLE overload; build grids from `range()`.
- The sqllogictest runner stops at the first failing file.
- Every C function stays at CRAP < 10 (`make crap`): small, single-purpose functions; core allocation
  through `emeans_malloc`/`emeans_calloc`/`emeans_realloc` so fault injection reaches the out-of-memory
  paths.
- No mutable global or static state in src/: calls run concurrently on DuckDB's threads.
- A C API scalar function cannot be interrupted: long loops must charge a budget (coming with the
  clustering loop) so a setting keeps bounding run time.
- With no git commits, set `EXTENSION_VERSION` or the metadata footer is corrupt.
- Expected values in tests must be derived from the model, not copied from output. Carry the arithmetic
  in a comment.
