# duck-emeans

`emeans` is a DuckDB extension for e-means clustering: mini-batch k-means with exponential-moving-average
centroid updates, whose batch size and step come from fitted scaling laws (Bessa, Mazaré, Szilvasy,
Touvron, Picard, Douze and Jégou, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), reference code
in [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)).
The paper reports Lloyd's 25-epoch loss beaten in 1.4 to 2.6 epochs.

It is written in C11 on the stable DuckDB C extension API and built for DuckDB 2.0.

## Status

Phases 1 to 4 of [the plan](docs/PLAN.md) are done: the C core, the SQL functions below, and validation
against the reference implementation in faiss-cpu. Still to come: a faster distance kernel, the cosine
variant, and a browser demo.

| Function | Returns |
|---|---|
| `emeans(vectors, k [, nepochs [, seed]])` | `STRUCT(centroids FLOAT[][], labels INTEGER[], obj DOUBLE[], batch_size, alpha0, n_splits)` |
| `emeans_assign(vector, centroids)` | `STRUCT(label INTEGER, sq_distance DOUBLE)`, the nearest centroid |
| `emeans_recipe(n, k, d, nepochs [, batch_law])` | the batch size, step, law, passes and steps e-means would use |

`vectors` is a list of equal-length vectors, as `FLOAT[]`, `DOUBLE[]`, integer lists or fixed-size arrays
such as `FLOAT[384]`; build it with `list(embedding ORDER BY id)`. Labels are 1-based, so
`centroids[labels[i]]` is row i's centroid, and they are the assignments of the last pass. `nepochs`
defaults to 5 and `seed` to 1234; the same seed gives the same result.

```sql
LOAD emeans;

-- Cluster every document's chunks into 8 topics, then label each chunk with its topic
CREATE TABLE topics AS
SELECT doc_id, emeans(list(embedding ORDER BY chunk_id), 8, 2.0) AS r
FROM chunks GROUP BY doc_id;

SELECT c.doc_id, c.chunk_id, (emeans_assign(c.embedding, t.r.centroids)).label AS topic
FROM chunks c JOIN topics t USING (doc_id);
```

## How it compares

Against faiss-cpu's `NumpyEmeans` on Gaussian mixtures (n = 100,000, k = 256, 2 epochs), the final
objective is within 0.5% (−0.43% at d = 16, +0.36% at d = 128), both beat Lloyd at 25 iterations, and
the recipe matches for 220 shapes. Speed is the gap: the distances are a plain C loop, 2.5× slower than
FAISS at d = 16 and 13× at d = 128, where FAISS uses BLAS. See [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Limits

A running scalar function cannot be interrupted, so one `emeans()` call does at most
`emeans_max_work` units of work (rows × k × d summed over the passes; default 1e11, 0 = no limit):

```sql
SET emeans_max_work = 1e12;
```

## Building

```sh
git clone --recurse-submodules https://github.com/hugr-lab/duck-emeans
cd duck-emeans
make configure && make release
make test_release test_core test_threads
make compare_faiss bench   # against faiss-cpu; needs uv
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## Licence

MIT. The DuckDB headers are MIT (see [LICENSES](LICENSES/README.md)). The algorithm is reimplemented
from the MIT-licensed reference; no code is copied from it.
