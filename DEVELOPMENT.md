# Development

## Toolchain

- CMake ≥ 3.5, a C11 and C++17 compiler (clang or gcc; MSVC through the CI toolchain), Python 3, git,
  ninja (optional, `GEN=ninja`).
- `duckdb` (v2.0.0) and `extension-ci-tools` are git submodules (`git submodule update --init`).

## Build

The extension is moving from the stable C API to DuckDB's C++ API (docs/PLAN.md, "Moving to the C++
API"). Two builds exist until that is finished:

```sh
GEN=ninja make release   # DuckDB from the submodule + build/release/extension/emeans/emeans.duckdb_extension
GEN=ninja make debug     # same, debug
make -f Makefile.capi configure release   # the C API build: build/release/emeans.duckdb_extension
```

`make release` is the extension template's build: it compiles DuckDB with the extension linked into
the shell and `unittest`, and writes the loadable binary. The first build takes several minutes;
later ones only recompile what changed. The binary loads into DuckDB v2.0.0 only, including the
Python wheel. `OVERRIDE_GIT_DESCRIBE=v2.0.0` in the Makefile stamps that version: DuckDB 2.0 does not
read it from git, and without it the binary names the submodule's commit hash.

In the C++ build `src/cpp/emeans_extension.cpp` is the entry point, and the C sources in
`src/functions` include `src/include/cpp_shim/duckdb_extension.h` instead of `duckdb_capi/`'s, so
their `duckdb_*` calls go straight to the C API inside DuckDB.

`compile_commands.json` is written to `build/release/`.

## DuckDB version

The extension is built and tested for DuckDB 2.0.0, the `duckdb` submodule. A C++ extension is tied
to the exact DuckDB version it was built against; a new DuckDB release means moving the submodule and
`duckdb_version` in `.github/workflows/MainDistributionPipeline.yml`.

For the C API build (Makefile.capi): the stable C extension API that DuckDB 2.0.0 ships is v1.5.6.
Its `duckdb_extension.h` refuses any other major version and has the same function set as the 1.5.6
header, so `TARGET_DUCKDB_VERSION=v1.5.6` names the API version the binary declares, and
`duckdb_capi/*.h` are the headers from that tag.

## Tests

```sh
GEN=ninja make release test_release   # SQL logic tests in DuckDB's unittest
make test_core test_threads
make crap           # CRAP < 10 for every C and C++ function (lizard via uvx, llvm-cov, ninja)
uvx pytest tools/crap   # the CRAP tooling's own tests
```

Before a commit, run the first line. The CRAP gate runs in CI.

`make crap` builds the C tests and the extension instrumented, runs them, writes one LCOV report and
scores it with `tools/crap/crap_cpp.py`: lizard's cyclomatic complexity per function and the share of
the function's own lines that ran. That scorer is shared, byte for byte, with
`hugr-lab/mssql-extension` and `hugr-lab/duck-changepoint`; `python3 tools/crap/crap_cpp.py --selftest`
checks it without lizard or a report. To score the Python tooling itself with
[crapper](https://github.com/unclebob/crapper), give it an interpreter it can install into:
`uv venv tools/crap/.venv && uv pip install -p tools/crap/.venv/bin/python pytest coverage`.

## Code conventions

- C11, tabs, 120 columns (`.clang-format`). Functions and types are `snake_case` with an `emeans_`
  prefix. SQL callbacks are `CamelCaseFunction` as in the DuckDB template.
- `src/core` must not include DuckDB headers.
- Errors go through `char err[EMEANS_ERR_LEN]` and a non-zero return, and the SQL layer turns them into
  `duckdb_scalar_function_set_error`. Allocation failures are reported, not ignored.
- No per-iteration allocation in algorithm loops; workspaces are allocated once per call.
- Allocate in `src/core` with `emeans_malloc` / `emeans_calloc` / `emeans_realloc` (`emeans_alloc.h`),
  so that `test_core`'s fault injection covers the out-of-memory path.
- Keep every function at CRAP < 10 (`make crap`). In practice: cyclomatic complexity below 10 and tests
  for every branch.
