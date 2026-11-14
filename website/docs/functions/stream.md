---
title: Streaming (data larger than memory)
sidebar_position: 3
---

# emeans_stream and emeans_stream_spherical

```sql
SELECT * FROM emeans_stream(query, k [, nepochs := 5, seed := 1234, batch_size := 0]);
SELECT * FROM emeans_stream_spherical(query, k [, ...]);
```

`emeans()` takes the whole set as one list and copies it as float32 for the fit, so the data must fit in
memory twice. `emeans_stream` takes a **query** instead. It runs the query once per pass, ordered by a
hash of each vector, the seed and the pass, and feeds the rows to the fit as they arrive. It holds one
batch, the centroids and a small workspace; DuckDB's sort spills each pass to `temp_directory`.

```sql
CREATE TABLE topics AS SELECT * FROM emeans_stream('SELECT embedding FROM chunks', 1024, nepochs := 3);

-- Label every chunk with its topic, row by row
SELECT chunk_id, (emeans_assign(embedding, t.centroids)).label AS topic
FROM chunks, (SELECT list(centroid ORDER BY cluster) AS centroids FROM topics) t;
```

## Arguments

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `query` | `VARCHAR` | | a single `SELECT` returning one column of vectors (`FLOAT[d]`, `FLOAT[]`, `DOUBLE[]`, any numeric list or array) |
| `k` | `BIGINT` | | clusters |
| `nepochs` | `DOUBLE` | 5 | passes; fractional values are allowed |
| `seed` | `BIGINT` | 1234 | in [0, 4294967295] |
| `batch_size` | `BIGINT` | 0 | rows per step; 0 lets the batch law choose |

NULL vectors are left out. Every vector must have the same length; a list of another length fails.

## Result

One row per cluster:

| Column | Type | Meaning |
|---|---|---|
| `cluster` | `INTEGER` | 1 to k |
| `centroid` | `FLOAT[d]` | the closing mean of the cluster's rows (spherical: its direction) |
| `n` | `BIGINT` | the rows assigned to the cluster in the closing pass |
| `run` | `STRUCT(n, batch_size, alpha0, n_splits, obj DOUBLE[])` | the same in every row: rows, recipe, revivals and objective per pass |

## How a run goes

All of a run's queries go to a connection of the function's own, in one read-only transaction:

1. One `GROUP BY` over the vectors gives n and whether any vector repeats; n gives the recipe, as for
   `emeans()`.
2. The start: k distinct vectors, drawn in proportion to how many rows hold each.
3. Every pass: the rows ordered by `hash(v, copy, seed, pass)`, where copy numbers the copies of a
   repeated vector so they spread over the pass instead of sitting together. The fit steps every B rows.
4. The last pass is the closing one, as in `emeans()`: its sums give the exact means and its counts `n`.

## What it sees, and what it may run

- **One snapshot.** Every pass reads the same rows.
- **Committed data only.** It runs on its own connection, so it does not see the caller's temporary
  tables or uncommitted rows. It runs in the caller's default database and schema.
- **A single read-only SELECT.** The query must parse on its own as one `SELECT`; a second statement, a
  write, or SQL that does not parse alone is refused at bind. The transaction is read-only, so nothing the
  query calls can write (a `nextval` fails).

## Memory

The stream holds O(B·d + k·d). B comes from the batch law and grows with n, k and nepochs: 524,288 rows
for 10M × 768, k = 1024, 5 epochs, a 1.6 GB buffer. `batch_size := ...` sets B instead; `run.batch_size`
shows what was used.

DuckDB's sort needs some memory per thread, so under a tight `memory_limit` use fewer threads. In the
tests, 1M × 32 vectors (128 MB) cluster under a 60 MB limit on one thread, where `emeans()` runs out of
memory; keep such tables on disk, since an in-memory table competes with the sorts for the same limit.

## Interrupts

A run stops on Ctrl-C, `con.interrupt()` or the caller's `max_execution_time`, also while DuckDB is still
sorting a pass: a watcher interrupts the nested query. In the tests a run stops within 0.1 s. (In
DuckDB-WASM, which has no threads, it stops between tasks.)

## Results and speed

- The start and the order of the passes come from hashes, not from `emeans()`'s generator, so the result
  differs from `emeans()` for the same seed. In the tests the objective is within 1% of `emeans()`'s on a
  mixture without repeated vectors, and within 10% (inside `emeans()`'s own spread over seeds) on 50
  points repeated 200 times.
- The same seed gives the same result, on any number of threads.
- Where both fit, the stream is 1.7–2.6× slower than `emeans()` (the sorts). If any vector repeats, even
  once, every pass also groups the vectors, and it is 2.4–3.9× slower.

Not yet: k-means++ or k-means|| starts, and a step-size override.
