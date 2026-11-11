# duck-emeans

`emeans` is a DuckDB extension for e-means clustering: mini-batch k-means with exponential-moving-average
centroid updates, whose batch size and step come from fitted scaling laws (Bessa, Mazaré, Szilvasy,
Touvron, Picard, Douze and Jégou, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), reference code
in [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)).
The paper reports Lloyd's 25-epoch loss beaten in 1.4 to 2.6 epochs.

It is written in C11 on the stable DuckDB C extension API and built for DuckDB 2.0.

## Status

Phases 1 and 2 of [the plan](docs/PLAN.md): the build and test gates, the recipe, and the e-means loop in
the C core (`emeans_fit` in `src/core/emeans.c`). The SQL functions that call the loop come in phase 3.

| Function | State |
|---|---|
| `emeans_recipe(n, k, d, nepochs [, batch_law])` | done: batch size, alpha0, law, passes and steps |
| `emeans(vectors, k [, nepochs [, seed]])` | planned: centroids, labels, objective per pass |
| `emeans_assign(vector, centroids)` | planned: nearest centroid and distance, row by row |
| `emeans_spherical(...)` | planned: cosine variant |

```sql
LOAD emeans;
SELECT emeans_recipe(1000000, 1000, 128, 2.0);
-- {'batch_size': 32768, 'alpha0': 0.7339706421150827, 'batch_law': throughput, 'n_passes': 2, 'total_steps': 62}
```

## Building

```sh
git clone --recurse-submodules https://github.com/hugr-lab/duck-emeans
cd duck-emeans
make configure && make release
make test_release test_core test_threads
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## Licence

MIT. The DuckDB headers are MIT (see [LICENSES](LICENSES/README.md)). The algorithm is reimplemented
from the MIT-licensed reference; no code is copied from it.
