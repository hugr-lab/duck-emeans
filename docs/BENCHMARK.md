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
for every point. Each took 0.4 s.

## Speed (`make bench`)

Median of 3 runs, nepochs = 2, seed 1. `emeans()` is one SQL call, which DuckDB runs on one thread.

| n | d | k | emeans() | NumpyEmeans | Lloyd, 25 iterations | emeans() / NumpyEmeans |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 16 | 256 | 0.13 s | 0.12 s | 0.51 s | 1.1× |
| 100,000 | 128 | 256 | 0.60 s | 0.25 s | 0.29 s | 2.4× |
| 1,000,000 | 16 | 256 | 1.32 s | 0.52 s | 5.10 s | 2.5× |
| 1,000,000 | 128 | 256 | 6.30 s | 2.48 s | 2.68 s | 2.5× |

NumpyEmeans and Lloyd ran with FAISS's default of 18 OpenMP threads. With `omp_set_num_threads(1)`
NumpyEmeans took 1.31 s at n = 1,000,000, d = 16, the same as `emeans()`, and 2.50 s at d = 128: FAISS's
distances go to Accelerate's sgemm, which runs on the M5's matrix units whatever the OpenMP thread
count. So at d = 128 the gap is a vectorised C loop against a BLAS kernel on dedicated hardware.

The distances are a blocked kernel (`src/core/assign.c`): 8 rows against 256 centroids at a time, as
‖c‖² − 2 x·c with the centroids stored transposed, so the inner loop has independent accumulators and the
compiler vectorises it. The distance of the chosen centroid is then recomputed exactly in double, so the
objectives are exact. Before the kernel, `emeans()` took 0.30, 3.26, 3.02 and 33.8 s on the four rows
above.

`emeans_assign` prepares the centroids for the kernel once per distinct list in a chunk of rows. When
they are a constant for the query (a literal, a parameter, a scalar subquery, a one-row cross join),
that is once per 2048 rows, and 100,000 rows against 256 × 128 centroids take 0.4 s. When every row
carries its own copy, as after a join with a table of per-group centroids, they are prepared for every
row: 8 s for the same 100,000 rows.

What the extension does better already: no data leaves DuckDB, the vectors can be any list or array
column, and many small clusterings in one query (`GROUP BY`) run in parallel across DuckDB's threads.
