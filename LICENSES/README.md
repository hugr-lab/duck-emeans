# Third-party licenses

| Component | Where | License |
|---|---|---|
| DuckDB v2.0.0 (built with the extension, linked into the loadable binary) | `duckdb/` (submodule) | MIT, [`duckdb-MIT.txt`](duckdb-MIT.txt) |

`extension-ci-tools` is a git submodule used only at build time and is not redistributed in the
extension binary.

The algorithm is a reimplementation in C of e-means (Bessa et al., arXiv 2611.09501). The reference
code in `faiss/contrib/e_means` is MIT-licensed and is not copied here; the scaling-law coefficients
and constants in `src/core/recipe.c` are taken from it.
