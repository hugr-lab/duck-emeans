---
title: Spatial clustering
sidebar_position: 4
---

# emeans_agg, emeans_cluster_id and emeans_assign_point

```sql
emeans_agg(points, k [, nepochs [, seed]])                        -- aggregate, also OVER (...)
emeans_cluster_id(points, k [, nepochs [, seed]]) OVER (...)      -- INTEGER, 1-based
emeans_assign_point(point, centroids)                             -- STRUCT(label INTEGER, distance DOUBLE)
```

Clustering of POINT geometries (and of vectors) as PostGIS's
[`ST_ClusterKMeans`](https://postgis.net/docs/ST_ClusterKMeans.html) does it, but on DuckDB's own
`GEOMETRY` type, without the spatial extension.

## Every row's cluster, in one call

```sql
-- PostGIS: ST_ClusterKMeans(geom, 8) OVER (PARTITION BY city), ids from 0
SELECT id, city, emeans_cluster_id(geom, 8) OVER (PARTITION BY city) AS cluster
FROM places;
```

`emeans_cluster_id` clusters the whole partition, like `ntile`, whatever the `ORDER BY` or frame, and gives
every row its 1-based cluster. NULL points and `POINT EMPTY` get NULL.

## Centroids per group

```sql
SELECT city, (emeans_agg(geom, 8)).centroids FROM places GROUP BY city;
```

`emeans_agg` returns

```
STRUCT(centroids DOUBLE[][], obj DOUBLE[], batch_size BIGINT, alpha0 DOUBLE, n_splits BIGINT, n BIGINT, geographic BOOLEAN)
```

where `n` is the number of points clustered and `geographic` says whether they were clustered on the
sphere. It is also a window function: `emeans_agg(geom, k) OVER (PARTITION BY ...)`.

The two agree: `emeans_cluster_id(geom, k) OVER w` is the label that
`emeans_assign_point(geom, (emeans_agg(geom, k) OVER w).centroids)` gives, the row's nearest final centroid.

## Coordinate systems

- **Geographic** (`OGC:CRS84`, `EPSG:4326`, `EPSG:4258`, `EPSG:4269` and a few others; x is longitude):
  the points are clustered on the sphere, so clusters across the antimeridian or around a pole are right.
  Centroids come back as `[longitude, latitude]`, and `emeans_assign_point` measures metres along the
  great circle.
- **Anything else** is planar: centroids and distances are in the coordinate system's units.

```sql
CREATE TABLE p AS SELECT * FROM (VALUES
  (1, 'Stockholm', 'POINT(18.06 59.33)'::GEOMETRY('OGC:CRS84')),
  (2, 'Stockholm', 'POINT(18.07 59.34)'::GEOMETRY('OGC:CRS84')),
  (3, 'Stockholm', 'POINT(17.90 59.40)'::GEOMETRY('OGC:CRS84')),
  (4, 'Stockholm', 'POINT(17.91 59.41)'::GEOMETRY('OGC:CRS84'))) t(id, city, geom);

SELECT (emeans_agg(geom, 2)).centroids FROM p;
-- two centroids near [18.065, 59.335] and [17.905, 59.405]

SELECT emeans_assign_point('POINT(18.0 59.35)'::GEOMETRY('OGC:CRS84'), [[18.065, 59.335], [17.905, 59.405]]);
-- {'label': 1, 'distance': 4045.28...}   (metres)
```

## Differences from PostGIS

- Cluster ids start at 1, not 0; NULL and `POINT EMPTY` get NULL, not -1.
- Only POINT geometries: use `ST_Centroid(geom)` for others.
- No `max_radius`.
- A partition's result does not depend on the order of its rows (the points are sorted before clustering).
- Both functions start from k-means++ seeds, unlike `emeans()`.

Against PostGIS 3.7 the DuckDB side is as fast on 10,000 points and up to 8.5× faster on more, with clusters
within 3% of PostGIS's in within-cluster variance. See the
[PostGIS comparison](https://github.com/hugr-lab/duck-emeans/blob/main/docs/POSTGIS.md).

Both functions also take vectors (`FLOAT[]`, `DOUBLE[]`, integer lists, fixed-size arrays), labelled as
`emeans_assign` labels them, and read `emeans_max_work` and `emeans_threads` as `emeans()` does.
