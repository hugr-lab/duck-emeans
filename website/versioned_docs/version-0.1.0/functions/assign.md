---
title: Assignment
sidebar_position: 2
---

# emeans_assign and emeans_assign_spherical

```sql
emeans_assign(vector, centroids)            -- STRUCT(label INTEGER, sq_distance DOUBLE)
emeans_assign_spherical(vector, centroids)  -- STRUCT(label INTEGER, similarity DOUBLE)
```

The nearest centroid of one vector, 1-based. `emeans_assign` measures the squared Euclidean distance;
`emeans_assign_spherical` takes the largest inner product with the centroid directions (the centroids are
normalised, the vector is not). Equal distances go to the lower label.

```sql
SELECT emeans_assign([9, 9], [[0, 0], [10, 10]]);
-- {'label': 2, 'sq_distance': 2.0}

SELECT emeans_assign_spherical([0.6, 0.8, 0], [[2, 0, 0], [0, 3, 0]]);
-- {'label': 2, 'similarity': 0.800000011920929}
```

The centroids can be the `centroids` of any fit, or a `list(centroid ORDER BY cluster)` of
[`emeans_stream`](./stream.md)'s rows. A NULL vector gives NULL.

## Labelling a table

```sql
SELECT id, (emeans_assign(embedding, c.centroids)).label AS cluster
FROM docs, (SELECT (emeans(list(embedding), 256)).centroids AS centroids FROM docs) c;
```

When the centroids are the same for every row (a literal, a parameter, a scalar subquery, a one-row cross
join as here) they are prepared once per 2048 rows: 100,000 rows against 256 × 128 centroids take 0.2 s.
When every row carries its own copy, as after a join with a table of per-group centroids, they are
prepared for every row, which is about 40 times slower.

## An IVF partition key

The label of a vector's nearest centroid is the partition key of an IVF index: cluster the base set once,
store each row's label, and search only the partitions nearest a query. The repository's
[Qdrant comparison](https://github.com/hugr-lab/duck-emeans/blob/main/docs/QDRANT.md) builds one in plain
SQL and compares it with DuckDB's HNSW (`vss`) and Qdrant.
