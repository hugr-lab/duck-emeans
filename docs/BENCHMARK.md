# Validation and benchmark

Measured on 12 November 2026 on an Apple M5 Pro (18 cores, macOS 27.3) against faiss-cpu 1.16.0, whose
`faiss.contrib.e_means.NumpyEmeans` is the reference implementation, with DuckDB 2.0.0. Reproduce with
`make compare_faiss` and `make bench`; both run in a pinned Python 3.12 environment through uv, and
`make compare_faiss` also runs in CI on Ubuntu.

The data are Gaussian mixtures: 2k centres drawn from N(0, 3²) in d dimensions, and every point a centre
plus N(0, 1) noise; for the spherical variant every point is then normalised. The objective is the mean
squared distance of each point to its nearest final centroid (spherical: the mean of 2 − 2 x·c over the
best centroid), computed the same way, with `faiss.knn`, for every method.

## Validation (`make compare_faiss`)

Recipes. For 220 shapes (n from 100 to 50 million, k from 1 to 1024, d from 2 to 768, nepochs from 0.5
to 25), `emeans_recipe` gives the same batch size and number of steps as the reference, and the same
alpha0 to within 1e-12 relative.

Quality, n = 100,000, k = 256, nepochs = 2, mean over seeds 1 to 3:

| variant | d | emeans() | NumpyEmeans | difference | Lloyd, 25 iterations |
|---|---|---:|---:|---:|---:|
| Euclidean | 16 | 41.153 | 41.331 | −0.43% | 42.471 |
| Euclidean | 128 | 614.495 | 612.193 | +0.38% | 639.730 |
| spherical | 128 | 0.5572 | 0.5589 | −0.30% | |

The gate is 1%. Single seeds vary by up to 1.1% in either direction, because the two implementations
draw different random numbers (PCG32 against numpy). Both beat Lloyd at 25 iterations after 2 epochs,
which is the paper's claim.

Assignment. `emeans_assign` against `faiss.knn` (L2), and `emeans_assign_spherical` against `faiss.knn`
(inner product), for all 100,000 points of the d = 128 sets and their 256 centroids: the same centroid
for every point. Each took 0.2 s.

## Speed (`make bench`)

Median of 3 runs, nepochs = 2, seed 1. `emeans()` is one SQL call; `emeans_threads` sets how many threads
it may use for a large batch, and the result is the same bits on any number.

| n | d | k | emeans(), 1 thread | emeans(), 18 threads | NumpyEmeans | Lloyd, 25 iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 16 | 256 | 0.09 s | 0.09 s | 0.13 s | 0.51 s |
| 100,000 | 128 | 256 | 0.21 s | 0.12 s | 0.25 s | 0.29 s |
| 1,000,000 | 16 | 256 | 0.83 s | 0.35 s | 0.55 s | 5.10 s |
| 1,000,000 | 128 | 256 | 2.44 s | 1.23 s | 2.52 s | 2.68 s |

Embedding-sized vectors (sentence-transformer and OpenAI dimensions), n = 100,000, k = 256, with the
final objective of each:

| d | emeans(), 1 thread | emeans(), 18 threads | NumpyEmeans | Lloyd, 25 iterations | objective: emeans() / NumpyEmeans / Lloyd |
|---:|---:|---:|---:|---:|---:|
| 768 | 0.91 s | 0.67 s | 1.26 s | 0.38 s | 4055 / 4089 / 4320 |
| 1536 | 1.77 s | 1.48 s | 2.48 s | 0.86 s | 8209 / 8203 / 8583 |

At these dimensions FAISS's Lloyd is the fastest, because each of its iterations is one large sgemm over
all rows, but after 25 iterations its objective is 4% to 6% worse than either e-means after 2 epochs.
Threads help `emeans()` less here: the assignment is split, but the per-batch accumulation and the
centroid updates, which grow with d, run on one thread.

NumpyEmeans and Lloyd ran with FAISS's default of 18 OpenMP threads; with one OpenMP thread NumpyEmeans
was about as fast at d = 128 (2.50 s), because its distances go to Accelerate's sgemm, which runs on the
M5's matrix units whatever the OpenMP thread count.

On macOS `emeans()` now does the same: the kernel hands the dot products of 64 rows against 1024 centroids
to `cblas_sgemm` from Accelerate, a system framework, so the extension needs no extra library. On Linux,
Windows and in WASM it uses the portable blocked loop (8 rows × 256 centroids, centroids transposed,
vectorised by the compiler). The distance to the chosen centroid is recomputed exactly in double either
way, so the objectives are exact.

A batch is split across threads only when it has at least 4 million units of work (rows × k × d) per
thread, so the many small calls of a `GROUP BY` stay on DuckDB's threads and do not oversubscribe the
cores. At d ≤ 16 the quality law picks small batches (4,096 rows at n = 1,000,000), which limits what
threads can do; FAISS has the same limit.

History of the n = 1,000,000, d = 128 row on this machine: 33.8 s with the first plain loop, 6.30 s with
the portable kernel, 2.44 s with Accelerate, 1.23 s with Accelerate on 18 threads. The portable kernel's
speed on Linux has not been measured here.

`emeans_assign` prepares the centroids for the kernel once per distinct list in a chunk of rows. When
they are a constant for the query (a literal, a parameter, a scalar subquery, a one-row cross join),
that is once per 2048 rows, and 100,000 rows against 256 × 128 centroids take 0.2 s. When every row
carries its own copy, as after a join with a table of per-group centroids, they are prepared for every
row: 8 s for the same 100,000 rows.

What the extension does better already: no data leaves DuckDB, the vectors can be any list or array
column, and many small clusterings in one query (`GROUP BY`) run in parallel across DuckDB's threads.
