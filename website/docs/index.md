---
title: Overview
sidebar_position: 1
slug: /
---

# DuckDB emeans Extension

[![GitHub stars](https://img.shields.io/github/stars/hugr-lab/duck-emeans?style=social)](https://github.com/hugr-lab/duck-emeans)
[![License](https://img.shields.io/github/license/hugr-lab/duck-emeans)](https://github.com/hugr-lab/duck-emeans/blob/main/LICENSE)

Cluster vectors, embeddings and points in SQL with **e-means**: mini-batch k-means whose centroids follow
an exponential moving average of their batch means, with the batch size and step set by fitted scaling
laws ([Bessa et al., arXiv 2611.09501](https://arxiv.org/abs/2611.09501); reference code in
[faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)). The paper
reports Lloyd's 25-epoch loss beaten in 1.4 to 2.6 epochs.

```sql
INSTALL emeans FROM community;
LOAD emeans;

-- Two unit squares; each centroid ends at the mean of its square's four corners
SELECT list_sort((emeans([[0, 0], [0, 1], [1, 0], [1, 1],
                          [10, 10], [10, 11], [11, 10], [11, 11]], 2)).centroids);
-- [[0.5, 0.5], [10.5, 10.5]]
```

→ **[Getting Started](./getting-started.md)** · **[SQL functions](./functions/emeans.md)** ·
**[Playground](https://hugr-lab.github.io/duck-emeans/demo/playground.html)**

## What it does

- **Cluster a list of vectors** with `emeans()`: centroids, 1-based labels and the objective per pass, for
  `FLOAT[]`, `DOUBLE[]`, integer lists or fixed-size arrays such as `FLOAT[384]`. One call per group in a
  `GROUP BY` clusters many sets in parallel.
- **Cluster by direction** with `emeans_spherical()`, for embeddings compared by cosine similarity.
- **Data larger than memory** with `emeans_stream()`: it takes a query, lets DuckDB order each pass
  (its sort spills to disk) and holds only a batch and the centroids.
- **Spatial clustering** of POINT geometries: `emeans_agg` (aggregate and window function) and
  `emeans_cluster_id`, a one-call `ST_ClusterKMeans`, on the sphere for geographic coordinate systems.
- **Assignment**: `emeans_assign`, `emeans_assign_spherical` and `emeans_assign_point` give any row its
  nearest centroid, which makes an IVF partition key in plain SQL.
- **Inspection**: `emeans_recipe` shows the batch size and step the scaling laws pick; `emeans_trace`
  records a run frame by frame, as the [playground](./demos.md) shows.

Every fit is interruptible (Ctrl-C, `con.interrupt()`, `max_execution_time`), takes its large buffers from
DuckDB's buffer manager (they count against `memory_limit`), and gives the same result for the same seed on
any number of threads.

## How it compares

Against faiss-cpu's `NumpyEmeans`, the reference implementation, the final objective is within 0.5% and
the recipe matches for 220 shapes; on an Apple M5 Pro `emeans()` clusters a million 128-d vectors into 256
clusters in 1.23 s on 18 threads, against 2.52 s for FAISS. Against PostGIS's `ST_ClusterKMeans` it is as
fast on 10,000 points and up to 8.5× faster on more. See [Performance](./performance.md).

## Platforms

Built for DuckDB 2.0 on Linux (x86-64 and arm64), macOS (arm64), Windows (MSVC and MinGW) and
DuckDB-WASM. On macOS the distance kernel uses Accelerate; elsewhere a portable blocked loop that the
compiler vectorises.
