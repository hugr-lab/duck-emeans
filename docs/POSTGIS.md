# emeans against PostGIS's ST_ClusterKMeans

Measured on 12 November 2026 on an Apple M5 Pro: PostgreSQL 18 with PostGIS 3.7.0 (GEOS 3.15, PROJ 9.9),
native arm64 from Homebrew, against DuckDB 2.0.0 on 8 threads. Reproduce with `make compare_postgis`
(`bench/compare_postgis.py`), which also writes the figure.

Each side labels every row. PostGIS: `ST_ClusterKMeans(geom, k) OVER ()`. DuckDB:
`emeans_assign_point(geom, (emeans_agg(geom, k) OVER ()).centroids)`. `emeans_agg` itself runs on one
thread (an aggregate has no bind step to read `emeans_threads`); the assignment uses DuckDB's threads.

![emeans against PostGIS](postgis_comparison.png)

## Behaviour

The same small inputs to both. The PostGIS documentation leaves most of these unstated, so this is
what each system did.

| case | PostGIS 3.7 | emeans | note |
|---|---|---|---|
| two separate pairs, k = 2 | the two pairs | the two pairs | same partition; PostGIS ids from 0, emeans from 1 |
| a NULL geometry | id -1 | NULL | PostGIS gives NULL and EMPTY the id -1 |
| POINT EMPTY | id -1 | NULL | |
| a polygon among points | clustered by its centroid | error: POINT geometries only | use `ST_Centroid(geom)` with emeans |
| k above the number of rows | error | error | both refuse |
| four identical points, k = 3 | one cluster | one cluster | |
| lon/lat points across 180°, SRID 4326 | split at 180° | one cluster across 180° | PostGIS treats lon/lat as planar; emeans clusters on the sphere |
| POINT Z, clusters apart only in z | uses z: two clusters | ignores z: one cluster | emeans is 2-D for geometries |
| POINT M, M as a weight | weighted | M ignored | emeans has no weights |
| the same input twice | same result | same result (same seed) | |
| the same rows in reverse order | a different partition | the same partition | emeans sorts the points first |

Not compared: PostGIS's `max_radius` (more clusters until none is wider than the radius), which emeans
does not have.

## Speed and quality

Gaussian blobs (128 centres in a 100 km square, 2 km spread) in projected metres (SWEREF99 TM range),
one partition, with `emeans_agg`'s defaults (k-means++ start, 5 epochs, seed 1234). WCSS is the
within-cluster sum of squares of each labelling, lower is better; the adjusted Rand index (ARI) is 1
when the two partitions are the same.

| n | k | PostGIS | DuckDB | speed-up | WCSS DuckDB / PostGIS | before k-means++ | ARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 8 | 0.02 s | 0.01 s | 2.3× | 1.063 | 1.063 | 0.592 |
| 10,000 | 64 | 0.06 s | 0.02 s | 3.6× | 1.030 | 1.253 | 0.765 |
| 100,000 | 8 | 0.16 s | 0.07 s | 2.3× | 0.962 | 0.983 | 0.678 |
| 100,000 | 64 | 0.52 s | 0.17 s | 3.1× | 1.018 | 1.071 | 0.831 |
| 100,000 | 256 | 5.49 s | 0.67 s | 8.2× | 1.009 | 1.013 | 0.591 |
| 1,000,000 | 64 | 26.9 s | 2.76 s | 9.8× | 1.020 | 1.078 | 0.746 |

The machine was busy with other work during this run: both sides took about three times as long as in
an earlier, quieter run (PostGIS 9.3 s and DuckDB 0.9 s at a million points), but the speed-up was the
same, about 10× at a million points.

DuckDB is 2× to 10× faster, more so as n and k grow, and its clusters are within 3% of PostGIS's in
within-cluster variance at every size here: 1% to 3% above, 4% below at n = 100,000, k = 8, equal at
n = 10,000, k = 8. The partitions differ (ARI 0.59 to 0.83) because on blobs that do not match k there
are many equally good local optima. In the figure's 20,000-point case the two are essentially the same.

`emeans_agg` starts from k-means++ seeds, unlike `emeans()` and the reference e-means, which start from
k random rows. With random rows the clusters on these maps were up to 25% worse than PostGIS's (the
"before" column), and more epochs did not help (5, 10, 20 and 50 epochs ended 7% to 12% above PostGIS at
n = 100,000, k = 64): the reference turns dead-cluster revival off for 2-D data, which leaves nothing
to repair a poor start. k-means++ costs one extra pass over the points.
