---
title: Getting Started
sidebar_position: 2
---

# Getting Started

### Install

```sql
INSTALL emeans FROM community;
LOAD emeans;
```

:::note

The extension is being submitted to DuckDB's community extensions. Until it is published there, build it
from source ([Development](./development.md)) and load the file into a DuckDB 2.0.0 started with
`-unsigned`:

```sql
LOAD 'build/release/extension/emeans/emeans.duckdb_extension';
```

A binary loads only into the DuckDB release it was built for.

:::

### Cluster a list of vectors

`emeans(vectors, k [, nepochs [, seed]])` takes the whole set as one list and returns a struct:

```sql
SELECT emeans([[0, 0], [0, 1], [1, 0], [1, 1],
               [10, 10], [10, 11], [11, 10], [11, 11]], 2) AS r;
-- {'centroids': [[0.5, 0.5], [10.5, 10.5]], 'labels': [1, 1, 1, 1, 2, 2, 2, 2],
--  'obj': [...], 'batch_size': 1, 'alpha0': 0.198..., 'n_splits': 0}
```

`labels[i]` is row i's cluster, 1-based, so `centroids[labels[i]]` is its centroid. `obj` holds the
objective of each pass, `batch_size` and `alpha0` the recipe that was used, `n_splits` how many dead
clusters were revived. `nepochs` defaults to 5 and `seed` to 1234.

### Cluster a table

Build the list with `list()`. Ordering it keeps the labels aligned with your rows:

```sql
-- Cluster every document's chunks into 8 topics
CREATE TABLE topics AS
SELECT doc_id, emeans(list(embedding ORDER BY chunk_id), 8, 2.0) AS r
FROM chunks GROUP BY doc_id;

-- Then give each chunk its topic: the nearest final centroid
SELECT c.doc_id, c.chunk_id, (emeans_assign(c.embedding, t.r.centroids)).label AS topic
FROM chunks c JOIN topics t USING (doc_id);
```

Each group's call runs on its own thread, so a `GROUP BY` clusters many sets in parallel.

### Data larger than memory

`emeans()` holds the whole list, twice. For a table that does not fit, give `emeans_stream` a query instead:

```sql
CREATE TABLE topics AS SELECT * FROM emeans_stream('SELECT embedding FROM chunks', 1024, nepochs := 3);
```

See [Streaming](./functions/stream.md).

### Next

- [SQL functions](./functions/emeans.md): every function, its arguments and its result.
- [Settings and limits](./reference/settings.md): threads, the work limit, memory, interrupts.
- [Playground](https://hugr-lab.github.io/duck-emeans/demo/playground.html): Lloyd and e-means side by
  side, run by this extension in your browser.
