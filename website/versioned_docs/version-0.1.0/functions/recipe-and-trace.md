---
title: Recipe and trace
sidebar_position: 5
---

# emeans_recipe and emeans_trace

## emeans_recipe

```sql
emeans_recipe(n, k, d, nepochs [, batch_law])
-- STRUCT(batch_size BIGINT, alpha0 DOUBLE, batch_law VARCHAR, n_passes BIGINT, total_steps BIGINT)
```

The batch size and initial step that the scaling laws pick for n rows of dimension d in k clusters over
`nepochs` passes, without clustering anything. `batch_law` is `'auto'` (the default: `'quality'` for
d ≤ 16, else `'throughput'`), `'throughput'` or `'quality'`.

```sql
SELECT emeans_recipe(1000000, 256, 128, 2.0);
-- {'batch_size': 32768, 'alpha0': 1.0, 'batch_law': throughput, 'n_passes': 2, 'total_steps': 62}
```

For 220 shapes it gives the same batch size and number of steps as the reference in faiss-cpu, and the
same `alpha0` to within 1e-12 relative. [Algorithm](../reference/algorithm.md) has the laws.

## emeans_trace

```sql
emeans_trace(train, test, k, nepochs, seed, method, frames_per_epoch, batch_size, revival)
-- STRUCT(batch_size BIGINT, alpha0 DOUBLE,
--        frames STRUCT(epoch DOUBLE, step BIGINT, train_mse DOUBLE, test_mse DOUBLE, centroids FLOAT[][])[])
```

One run of e-means or of Lloyd's k-means, recorded: the centroids before the first step, every so many
steps, and after the closing means, each with the mean squared distance of the training rows (a sample of
at most 20,000) and of the test rows to their nearest centroid. It is for playgrounds and teaching; the
[playground](../demos.md) is built on it. Use `emeans()` to cluster.

| Argument | Default (NULL) | Meaning |
|---|---|---|
| `train` | | the vectors to fit, a list as for `emeans()` |
| `test` | none | held-out vectors, scored at every frame; may be NULL |
| `k`, `nepochs`, `seed` | | as for `emeans()` |
| `method` | `'emeans'` | `'emeans'` uses the recipe; `'lloyd'` uses full batches and a constant step of 1 |
| `frames_per_epoch` | 4 | how often e-means records a frame; Lloyd records every iteration |
| `batch_size` | the recipe's | e-means's batch size, if above 0 |
| `revival` | true | e-means's dead-cluster revival |

```sql
-- Lloyd on six points of a line: three iterations, then the closing means
SELECT f.epoch, f.step, round(f.train_mse, 6) AS train_mse, f.centroids
FROM (SELECT unnest(emeans_trace([[0, 0], [1, 0], [2, 0], [3, 0], [10, 0], [11, 0]]::DOUBLE[][],
                                 NULL, 2, 3.0, 1, 'lloyd', NULL, NULL, NULL).frames) AS f);
-- 0.0  0  19.166667  [[0.0, 0.0], [3.0, 0.0]]
-- 1.0  1   6.916667  [[0.5, 0.0], [6.5, 0.0]]
-- 2.0  2   0.916667  [[1.5, 0.0], [10.5, 0.0]]
-- ...
```
