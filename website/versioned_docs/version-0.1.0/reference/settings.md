---
title: Settings and limits
sidebar_position: 1
---

# Settings and limits

## Settings

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `emeans_threads` | `BIGINT` | 0 | threads one call may use to assign a large batch; 0 means DuckDB's `threads` setting. The result does not depend on it |
| `emeans_max_work` | `DOUBLE` | 0 | the most work one call may do, in rows × k × d summed over all passes; 0 means no cap |

```sql
SET emeans_threads = 4;
SET emeans_max_work = 1e12;   -- fail up front instead of starting a fit that big
```

## Threads

A call splits a large batch into up to `emeans_threads` parts, run as tasks on DuckDB's own scheduler. A
batch is split only when it has at least 4 million units of work (rows × k × d) per thread, so the many
small calls of a `GROUP BY` stay on one thread each and do not oversubscribe the cores. The result is the
same bits on any number of threads.

## Interrupts and time limits

Every function that fits stops when its query is interrupted (Ctrl-C in the CLI, `con.interrupt()` in
Python) or passes `max_execution_time`, with DuckDB's own error. The fit checks once per step (and once per
k-means++ centroid); `emeans_stream` also stops while DuckDB is still sorting a pass.

```sql
SET max_execution_time = 60000;   -- milliseconds
```

## Memory

The large buffers come from DuckDB's buffer manager: the float copy of `emeans()`'s input, the fit's
workspace, and `emeans_agg`'s points. They count against `memory_limit`, and a query that needs more than
the limit at once fails with DuckDB's out-of-memory error instead of taking the memory anyway.

- **`emeans()` and `emeans_spherical()`** keep the whole input as float32 (n × d × 4 bytes) for the fit,
  plus about 16 bytes per row of workspace, so the set must fit in memory twice.
- **`emeans_agg`'s** points can be written to `temp_directory` between calls when they do not fit: 2,000
  groups of 50,000 2-D points (1.6 GB of state) ran under `memory_limit = '500MB'` at a peak RSS of
  730–910 MB.
- **`emeans_stream`** holds a batch (B × d floats) and the centroids, and DuckDB's sort spills each pass.
  See [Streaming](../functions/stream.md#memory).
- The window forms (`emeans_agg(...) OVER`, `emeans_cluster_id`) keep a partition's rows pinned, as
  DuckDB's window operator does, so they do not spill like the `GROUP BY` form.

## Inputs

- Vectors are `FLOAT[]`, `DOUBLE[]` or integer lists, or fixed-size numeric arrays (`FLOAT[384]`), all of
  one length within a call.
- NaN and infinity are rejected.
- `k` is at most the number of vectors and below 2^31; `nepochs` is positive and at most 10^6; `seed` is
  in [0, 4294967295].
- Labels and cluster ids are 1-based in SQL.
