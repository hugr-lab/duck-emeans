# duck-emeans

`emeans` is a DuckDB extension for e-means clustering: mini-batch k-means with exponential-moving-average
centroid updates, whose batch size and step come from fitted scaling laws (Bessa, Mazaré, Szilvasy,
Touvron, Picard, Douze and Jégou, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), reference code
in [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)).
The paper reports Lloyd's 25-epoch loss beaten in 1.4 to 2.6 epochs.

It is written in C11 on the stable DuckDB C extension API and built for DuckDB 2.0.

**[Try it in your browser](https://hugr-lab.github.io/duck-emeans/)**: the extension compiled to
WebAssembly, loaded into DuckDB-WASM, clustering points in the plane and timing embedding-sized vectors
(d = 768 or 1536). Nothing is installed and no data leaves the page.

## Status

Phases 1 to 6 of [the plan](docs/PLAN.md) are done: the C core with a vectorised distance kernel, threads
and Accelerate on macOS, the SQL functions below with a cosine variant, validation against the reference
implementation in faiss-cpu, and the browser demo. Still to come: a playground in the style of the
e-means site's (Lloyd and e-means side by side), a GPU build, and the community-extension submission.

| Function | Returns |
|---|---|
| `emeans(vectors, k [, nepochs [, seed]])` | `STRUCT(centroids FLOAT[][], labels INTEGER[], obj DOUBLE[], batch_size, alpha0, n_splits)` |
| `emeans_assign(vector, centroids)` | `STRUCT(label INTEGER, sq_distance DOUBLE)`, the nearest centroid |
| `emeans_spherical(vectors, k [, nepochs [, seed]])` | as `emeans`, for unit vectors: by direction, unit centroids |
| `emeans_assign_spherical(vector, centroids)` | `STRUCT(label INTEGER, similarity DOUBLE)`, the closest direction |
| `emeans_recipe(n, k, d, nepochs [, batch_law])` | the batch size, step, law, passes and steps e-means would use |

`vectors` is a list of equal-length vectors, as `FLOAT[]`, `DOUBLE[]`, integer lists or fixed-size arrays
such as `FLOAT[384]`; build it with `list(embedding ORDER BY id)`. Labels are 1-based, so
`centroids[labels[i]]` is row i's centroid, and they are the assignments of the last pass. `nepochs`
defaults to 5 and `seed` to 1234; the same seed gives the same result. The spherical functions assign by
the largest inner product with the centroid directions, for embeddings compared by cosine similarity;
like the reference they do not normalise the vectors, so pass unit vectors. Centroids come back as
`FLOAT[][]`; cast them to `FLOAT[384][]` if you need the array type.

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
objective is within 0.5% (−0.43% at d = 16, +0.38% at d = 128, −0.30% spherical), both beat Lloyd at 25
iterations, and the recipe matches for 220 shapes. On an Apple M5 Pro, `emeans()` clusters a million
128-d vectors into 256 clusters in 1.23 s on 18 threads (2.44 s on one), against 2.52 s for FAISS; on
macOS it uses Accelerate for the dot products. See [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Limits

A running scalar function cannot be interrupted, so one `emeans()` call does at most
`emeans_max_work` units of work (rows × k × d summed over the passes; default 1e11, 0 = no limit):

```sql
SET emeans_max_work = 1e12;
```

A call splits a large batch across up to `emeans_threads` threads (default 0, meaning DuckDB's `threads`
setting). Small calls, such as one per group in a `GROUP BY`, stay on one thread. The result does not
depend on the thread count.

## Building

```sh
git clone --recurse-submodules https://github.com/hugr-lab/duck-emeans
cd duck-emeans
make configure && make release
make test_release test_core test_threads
make compare_faiss bench   # against faiss-cpu; needs uv
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## The browser demo

`web/` is a static page: `duckdb.js` starts DuckDB-WASM (the build of DuckDB v2.0.0) with unsigned
extensions allowed and loads `emeans.duckdb_extension.wasm`, the `wasm_eh` build from the distribution
pipeline; `demo-core.js` holds the logic (`npm test`) and `demo.js` the page. The Pages workflow publishes
it after every successful build on `main`. To run it locally, put the `wasm_eh` build next to the files
and serve the directory:

```sh
gh run download <run-id> -n emeans-v2.0.0-extension-wasm_eh -D web/
python3 -m http.server -d web 8000   # http://localhost:8000
```

In the browser the extension runs on one thread, without Accelerate: 10,000 vectors of dimension 768
into 64 clusters take about 1 s in Chromium on an M5 Pro.

## Licence

MIT. The DuckDB headers are MIT (see [LICENSES](LICENSES/README.md)). The algorithm is reimplemented
from the MIT-licensed reference; no code is copied from it.
