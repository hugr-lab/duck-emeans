---
title: Performance
sidebar_position: 10
---

# Performance

Measured on 12 November 2026 on an Apple M5 Pro (18 cores) with DuckDB 2.0.0. The full write-ups, with
how to reproduce them, are in the repository:
[BENCHMARK.md](https://github.com/hugr-lab/duck-emeans/blob/main/docs/BENCHMARK.md),
[POSTGIS.md](https://github.com/hugr-lab/duck-emeans/blob/main/docs/POSTGIS.md) and
[QDRANT.md](https://github.com/hugr-lab/duck-emeans/blob/main/docs/QDRANT.md).

## Against the reference (faiss-cpu)

faiss-cpu 1.16.0's `NumpyEmeans` is the reference implementation. On Gaussian mixtures, n = 100,000,
k = 256, nepochs = 2, mean of seeds 1 to 3, the final objective:

| variant | d | emeans() | NumpyEmeans | difference | Lloyd, 25 iterations |
|---|---|---:|---:|---:|---:|
| Euclidean | 16 | 41.153 | 41.331 | −0.43% | 42.471 |
| Euclidean | 128 | 614.495 | 612.193 | +0.38% | 639.730 |
| spherical | 128 | 0.5572 | 0.5589 | −0.30% | |

Both beat Lloyd at 25 iterations after 2 epochs, the paper's claim. `emeans_assign` agrees with
`faiss.knn` for every one of 100,000 points. CI runs this comparison on every pull request.

## Speed

Median of 3 runs, nepochs = 2:

| n | d | k | emeans(), 1 thread | emeans(), 18 threads | NumpyEmeans | Lloyd, 25 iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 16 | 256 | 0.09 s | 0.09 s | 0.13 s | 0.51 s |
| 100,000 | 128 | 256 | 0.21 s | 0.12 s | 0.25 s | 0.29 s |
| 1,000,000 | 16 | 256 | 0.83 s | 0.35 s | 0.55 s | 5.10 s |
| 1,000,000 | 128 | 256 | 2.44 s | 1.23 s | 2.52 s | 2.68 s |

For embedding-sized vectors (n = 100,000, k = 256), d = 768 takes 0.67 s and d = 1536 1.48 s on 18
threads, against 1.26 s and 2.48 s for FAISS's e-means, at the same objective. On macOS the distance kernel
hands its dot products to Accelerate's `cblas_sgemm`; on Linux, Windows and in WASM it uses a portable
blocked loop, whose speed against FAISS has not been measured.

`emeans_stream` is 1.7–2.6× slower than `emeans()` where both fit, because every pass is a sort; see
[Streaming](./functions/stream.md#results-and-speed).

## Against PostGIS

`emeans_cluster_id(geom, k) OVER ()` against PostGIS 3.7's `ST_ClusterKMeans(geom, k) OVER ()`: as fast on
10,000 points and up to 8.5× faster on more, with clusters within 3% of PostGIS's in within-cluster
variance. The behaviour differs on NULLs, the antimeridian and row order; see
[Spatial clustering](./functions/spatial.md#differences-from-postgis).

## Nearest-neighbour search

e-means and a vector database are not compared directly; what is compared is approximate nearest-neighbour
search on 99,000 OpenAI embeddings (d = 1536). An IVF index whose partitions come from `emeans` matches one
built with Lloyd's k-means in recall at half the build time, but HNSW indexes, DuckDB's own (`vss`) and
Qdrant's, answer queries an order of magnitude faster at higher recall.
