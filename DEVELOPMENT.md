# Development

## Toolchain

- CMake ≥ 3.5, a C11 compiler (clang or gcc; MSVC through the CI toolchain), Python 3, git.
- No external libraries: libc, libm and the DuckDB C API headers in `duckdb_capi/`.
- `extension-ci-tools` is a git submodule (`git submodule update --init`).

## Build

```sh
make configure      # configure/venv (DuckDB + sqllogictest runner), configure/platform.txt, version
make release        # cmake_build/release -> build/release/emeans.duckdb_extension (with metadata footer)
make debug          # same, debug
```

The version in the footer comes from `git describe` (`configure/extension_version.txt`). In a checkout
without commits, pass `EXTENSION_VERSION=v0.0.1-dev` to `make configure`. Otherwise the footer gets
git's error text and DuckDB refuses the file with "Unknown ABI type".

`compile_commands.json` is written to `cmake_build/release/`. `.clangd` points there.

## DuckDB version

The extension is built and tested for DuckDB 2.0. The stable C extension API that DuckDB 2.0.0 ships
is v1.5.6: its `duckdb_extension.h` refuses any other major version and has the same function set as
the 1.5.6 header. So `TARGET_DUCKDB_VERSION=v1.5.6` in the Makefile names the API version the binary
declares, and `duckdb_capi/*.h` are the headers from that tag. The CI pipeline builds with DuckDB
v2.0.0 and `make test_release` runs on the latest DuckDB from PyPI.

## Tests

```sh
make release test_release test_core test_threads
make crap           # CRAP < 10 for every C function (needs uv for lizard, llvm-cov)
uvx pytest tools/crap   # the CRAP tool's own pure functions
```

Before a commit, run the first line. The CRAP gate runs in CI. To score the Python tooling itself with
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
