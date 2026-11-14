---
title: emeans and emeans_spherical
sidebar_position: 1
---

# emeans and emeans_spherical

```sql
emeans(vectors, k [, nepochs [, seed]])
emeans_spherical(vectors, k [, nepochs [, seed]])
```

Cluster a list of vectors into `k` clusters. Returns

```
STRUCT(centroids FLOAT[][], labels INTEGER[], obj DOUBLE[], batch_size BIGINT, alpha0 DOUBLE, n_splits BIGINT)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `vectors` | a list of vectors | | `FLOAT[]`, `DOUBLE[]` or integer lists, or fixed-size arrays such as `FLOAT[384]`; all of one length |
| `k` | `BIGINT` | | clusters, at most the number of vectors |
| `nepochs` | `DOUBLE` | 5 | passes over the data; fractional values are allowed |
| `seed` | `BIGINT` | 1234 | in [0, 4294967295]; the same seed gives the same result |

A NULL `nepochs` or `seed` means the default.

| Field | Meaning |
|---|---|
| `centroids` | k centroids. For a list of fixed-size arrays they keep that shape (`FLOAT[384][]`), otherwise `FLOAT[][]` |
| `labels` | each vector's cluster, 1-based, from the last pass: `centroids[labels[i]]` is row i's centroid |
| `obj` | the objective of each pass: the sum of squared distances (spherical: of 2 − 2 x·c), measured while the centroids move |
| `batch_size`, `alpha0` | the batch size and initial step the [recipe](./recipe-and-trace.md) picked |
| `n_splits` | dead clusters revived by splitting a live one |

```sql
SELECT list_sort((emeans([[0, 0], [0, 1], [1, 0], [1, 1],
                          [10, 10], [10, 11], [11, 10], [11, 11]], 2)).centroids);
-- [[0.5, 0.5], [10.5, 10.5]]
```

Build the list from a table with `list(embedding ORDER BY id)`; the order keeps `labels` aligned with your
rows. One call per group of a `GROUP BY` clusters many sets at once, each on its own thread.

## How a run goes

The start is k vectors drawn at random. Each pass visits the vectors in a new random order, in batches of
B: every batch assigns its vectors to the nearest centroid, and each centroid moves part of the way to the
mean of its vectors, by a step that follows a cosine schedule from `alpha0` down to `alpha0 / 10`. Every 5
steps a cluster whose mass fell below a tenth of the average takes half of a live cluster (not for
d ≤ 2). After the last pass every centroid becomes the exact mean of the vectors assigned to it in that
pass. [Algorithm](../reference/algorithm.md) has the formulas.

Labels are the assignments of the last pass, made while the centroids still moved, so a label is not
always the nearest final centroid. [`emeans_assign`](./assign.md) gives that.

## Spherical

`emeans_spherical` assigns each vector to the centroid direction with the largest inner product and
returns unit centroids, for embeddings compared by cosine similarity. Like the reference, it does not
normalise the vectors: pass unit vectors.

## Errors and limits

- A vector with NaN or infinity, vectors of different lengths, or `k` above the number of vectors fail
  the call. NULL `vectors` or `k` give NULL.
- `emeans_max_work` caps the work up front; a run also stops on an interrupt or `max_execution_time`.
  See [Settings and limits](../reference/settings.md).
- The input is copied once as float32 and kept for the whole fit, so a set must fit in memory twice. For
  more, use [`emeans_stream`](./stream.md).
