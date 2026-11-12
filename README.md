# duck-emeans

`emeans` is a DuckDB extension for e-means clustering: mini-batch k-means with exponential-moving-average
centroid updates, whose batch size and step come from fitted scaling laws (Bessa, Mazaré, Szilvasy,
Touvron, Picard, Douze and Jégou, [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), reference code
in [faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)).
The paper reports Lloyd's 25-epoch loss beaten in 1.4 to 2.6 epochs.

It is written in C11 on the stable DuckDB C extension API and built for DuckDB 2.0.

**[The playground](https://hugr-lab.github.io/duck-emeans/playground.html)**: Lloyd's k-means and e-means
side by side on 2-D data, with their Voronoi cells and the train and held-out test MSE over epochs, both
run by the extension compiled to WebAssembly inside DuckDB-WASM and then replayed. It follows the
[playground](https://e-means.github.io/playground.html) of the e-means authors, with the computation in
SQL. A [simpler demo](https://hugr-lab.github.io/duck-emeans/) clusters points in the plane and times
embedding-sized vectors (d = 768 or 1536). Nothing is installed and no data leaves the page.

## Status

Done (see [the plan](docs/PLAN.md)): the C core with a vectorised distance kernel, threads and Accelerate
on macOS; the SQL functions below, with a cosine variant and spatial clustering of POINT geometries;
validation against the reference implementation in faiss-cpu; comparisons with PostGIS and Qdrant; and
two browser demos. Still to come: a GPU build and the community-extension submission.

| Function | Returns |
|---|---|
| `emeans(vectors, k [, nepochs [, seed]])` | `STRUCT(centroids FLOAT[][], labels INTEGER[], obj DOUBLE[], batch_size, alpha0, n_splits)` |
| `emeans_assign(vector, centroids)` | `STRUCT(label INTEGER, sq_distance DOUBLE)`, the nearest centroid |
| `emeans_spherical(vectors, k [, nepochs [, seed]])` | as `emeans`, for unit vectors: by direction, unit centroids |
| `emeans_assign_spherical(vector, centroids)` | `STRUCT(label INTEGER, similarity DOUBLE)`, the closest direction |
| `emeans_agg(points, k [, nepochs [, seed]])` | aggregate, also a window function: centroids of a group's POINT geometries or vectors |
| `emeans_cluster_id(points, k [, nepochs [, seed]]) OVER (...)` | window function: each row's 1-based cluster within its partition, like `ST_ClusterKMeans` |
| `emeans_assign_point(point, centroids)` | `STRUCT(label INTEGER, distance DOUBLE)`, a POINT's nearest centroid (metres for geographic CRSs) |
| `emeans_recipe(n, k, d, nepochs [, batch_law])` | the batch size, step, law, passes and steps e-means would use |
| `emeans_trace(train, test, k, nepochs, seed, method, frames_per_epoch, batch_size, revival)` | frames of a run (`'emeans'` or `'lloyd'`): centroids, train and test MSE; for playgrounds and teaching |

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

## Spatial clustering

`emeans_cluster_id` gives every row its cluster within its window partition, in one call, the way
PostGIS's [`ST_ClusterKMeans`](https://postgis.net/docs/ST_ClusterKMeans.html) does:

```sql
-- PostGIS: ST_ClusterKMeans(geom, 8) OVER (PARTITION BY city), ids from 0
SELECT id, city, emeans_cluster_id(geom, 8) OVER (PARTITION BY city) AS cluster
FROM places;
```

`emeans_agg` gives the centroids instead, per group or per window partition:

```sql
SELECT city, (emeans_agg(geom, 8)).centroids FROM places GROUP BY city;
```

The two agree: `emeans_cluster_id(geom, k) OVER w` is the label
`emeans_assign_point(geom, (emeans_agg(geom, k) OVER w).centroids)` gives, the row's nearest final
centroid. After the closing pass every centroid is the mean of the points assigned to it, so this is
the cluster a converged k-means gives the row. That two-call form was the only way before the
extension moved to DuckDB's C++ API: the stable C API has no window functions, and an aggregate never
learns which row it is computed for.

Labels start at 1, so `centroids[cluster]` is the row's centroid; NULL points and `POINT EMPTY` get
NULL. `emeans_cluster_id` clusters the whole partition, like `ntile`, whatever the `ORDER BY` or frame.
Other differences from PostGIS: there is no `max_radius`, and the result of a partition does not depend
on the order of its rows (the points are sorted before clustering). Both functions start from k-means++
seeds, unlike `emeans()`.

Against PostGIS 3.7 the DuckDB side is as fast on 10,000 points and up to 8.5× faster on more, its clusters are within 3% of PostGIS's in
within-cluster variance, and it handles the antimeridian, NULLs and row order differently. See
[docs/POSTGIS.md](docs/POSTGIS.md) for the full comparison and the figure.

The coordinate system decides the geometry. For a geographic CRS (`OGC:CRS84`, `EPSG:4326`, `EPSG:4258`,
`EPSG:4269` and a few others; x is longitude) the points are clustered on the sphere, so clusters across
the antimeridian or around a pole are right, centroids come back as `[longitude, latitude]`, and
`emeans_assign_point` measures metres along the great circle. Any other CRS is planar: centroids and
distances are in its units, and projected coordinates keep their precision. Both also take vectors
(`FLOAT[]`, `DOUBLE[]`, integer lists, fixed-size arrays), labelled as `emeans_assign` labels them. Both
read `emeans_max_work` and `emeans_threads` as `emeans()` does.

## How it compares

Against faiss-cpu's `NumpyEmeans` on Gaussian mixtures (n = 100,000, k = 256, 2 epochs), the final
objective is within 0.5% (−0.43% at d = 16, +0.38% at d = 128, −0.30% spherical), both beat Lloyd at 25
iterations, and the recipe matches for 220 shapes. On an Apple M5 Pro, `emeans()` clusters a million
128-d vectors into 256 clusters in 1.23 s on 18 threads (2.44 s on one), against 2.52 s for FAISS; on
macOS it uses Accelerate for the dot products. For embedding-sized vectors (100,000 of d = 768 or 1536)
it takes 0.67 s and 1.48 s, against 1.26 s and 2.48 s for FAISS's e-means, at the same objective. See
[docs/BENCHMARK.md](docs/BENCHMARK.md).

For nearest-neighbour search on 1536-d OpenAI embeddings, an IVF index built with `emeans` matches one
built with Lloyd's k-means in recall at half the build time, but both HNSW indexes, DuckDB's own (`vss`)
and Qdrant's, answer queries an order of magnitude faster at higher recall; `vss` is close to Qdrant at
100,000 vectors. See [docs/QDRANT.md](docs/QDRANT.md).

## Limits

Every function that fits stops when its query is interrupted (Ctrl-C in the CLI, `con.interrupt()` in
Python): the fit checks once per step. To also cap the work up front, in units of rows × k × d summed
over the passes:

```sql
SET emeans_max_work = 1e12;   -- default 0: no cap
```

A call splits a large batch across up to `emeans_threads` threads (default 0, meaning DuckDB's `threads`
setting). Small calls, such as one per group in a `GROUP BY`, stay on one thread. The result does not
depend on the thread count.

The large buffers come from DuckDB's buffer manager: the float copy of the input, the fit's workspace,
and `emeans_agg`'s points. They count against `memory_limit`. Between calls, an aggregate's points can
be written to `temp_directory` when they do not fit. A query that needs more than the limit at once
fails with DuckDB's out-of-memory error. In one test, 2,000 groups of 50,000 2-D points (1.6 GB of
aggregate state) ran under `memory_limit = '500MB'` at a peak RSS of 730–910 MB, against 2.9 GB for the
earlier malloc build, and took about 40% longer for the spilling. A single fit still needs its whole
input in memory, as float32 (n × d × 4 bytes), plus about 16 bytes per row of workspace.

## Building

```sh
git clone --recurse-submodules https://github.com/hugr-lab/duck-emeans
cd duck-emeans
GEN=ninja make release      # builds DuckDB v2.0.0 from the submodule, then the extension
make test_release test_core test_threads
make compare_faiss bench   # against faiss-cpu; needs uv
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## The browser demos

Both live in `web/`. `duckdb.js` starts DuckDB-WASM (the build of DuckDB v2.0.0) with unsigned extensions
allowed and loads `emeans.duckdb_extension.wasm`, the `wasm_eh` build from the distribution pipeline.

- `index.html` with `demo.js` is the simpler demo; `demo-core.js` holds its logic (`npm test`).
- `playground.html` is the playground: a Scala.js + Laminar app in `web/playground/` (Mill), which reaches
  DuckDB through `playground-bridge.js`. `make -C web/playground test` runs its tests on Node.

The Pages workflow builds both and publishes them after every successful build on `main`. To run them
locally (needs Java and Node):

```sh
gh run download <run-id> -n emeans-v2.0.0-extension-wasm_eh -D web/
make -C web/playground serve   # http://localhost:8099/playground.html
```

In the browser the extension runs on one thread, without Accelerate: 10,000 vectors of dimension 768
into 64 clusters take about 1 s in Chromium on an M5 Pro.

## Licence

MIT. The DuckDB headers are MIT (see [LICENSES](LICENSES/README.md)). The algorithm is reimplemented
from the MIT-licensed reference; no code is copied from it.
