---
title: Development
sidebar_position: 12
---

# Development

The extension is a C++ extension for DuckDB 2.0, built against DuckDB's C++ API from the `duckdb` git
submodule. The numerical core (`src/core/`) and most SQL functions (`src/functions/`) are C11, and the core
has no DuckDB dependency; what needs the C++ API, such as the window function, spilling memory and the
streaming functions, is in `src/cpp/`. A binary loads only into the DuckDB release it was built for.

## Build

Needs CMake ≥ 3.5, a C11 and C++17 compiler, Python 3, git, and ninja (optional).

```sh
git clone --recurse-submodules https://github.com/hugr-lab/duck-emeans
cd duck-emeans
GEN=ninja make release   # DuckDB v2.0.0 from the submodule, then build/release/extension/emeans/emeans.duckdb_extension
```

The first build takes several minutes; later ones recompile only what changed. Load the result into a DuckDB
2.0.0 started with `-unsigned`, or into the shell it built, `build/release/duckdb`.

## Tests

```sh
make test_release        # SQL logic tests in DuckDB's unittest
make test_core           # the C core under AddressSanitizer and UndefinedBehaviorSanitizer
make test_threads        # the core on 8 threads under ThreadSanitizer
make test_interrupt      # emeans_stream stops on con.interrupt() (Python via uv)
make crap                # CRAP < 10 for every C and C++ function
make compare_faiss       # against faiss-cpu's NumpyEmeans (Python via uv)
```

CI runs all of these on every pull request, with the distribution pipeline's builds for Linux, Windows and
WASM (on `main` also macOS, Linux arm64, MinGW and the other WASM variants). `main` is protected: changes go through a pull request that passes every check and has
one approving review.

## This site

The site is [Docusaurus](https://docusaurus.io/) in `website/`:

```sh
cd website
npm ci
npm start      # live preview at http://localhost:3000/duck-emeans/
npm run build  # what CI builds
```

`.github/workflows/pages.yml` publishes it with the demos after every successful build on `main`.

More in the repository: [DEVELOPMENT.md](https://github.com/hugr-lab/duck-emeans/blob/main/DEVELOPMENT.md)
(toolchain and conventions) and [docs/PLAN.md](https://github.com/hugr-lab/duck-emeans/blob/main/docs/PLAN.md)
(the phases and design decisions).
