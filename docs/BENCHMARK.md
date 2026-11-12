# Validation and benchmark

Measured on 12 November 2026 on an Apple M5 Pro (18 cores, macOS 27.3) against faiss-cpu 1.16.0, whose
`faiss.contrib.e_means.NumpyEmeans` is the reference implementation, with DuckDB 2.0.0. Reproduce with
`make compare_faiss` and `make bench`; both run in a pinned Python 3.12 environment through uv.

The data are Gaussian mixtures: 2k centres drawn from N(0, 3²) in d dimensions, and every point a centre
plus N(0, 1) noise. The objective is the mean squared distance of each point to its nearest final
centroid, computed the same way (with `faiss.knn`) for every method.

## Validation (`make compare_faiss`)

Recipes. For 220 shapes (n from 100 to 50 million, k from 1 to 1024, d from 2 to 768, nepochs from 0.5
to 25), `emeans_recipe` gives the same batch size and number of steps as the reference, and the same
alpha0 to within 1e-12 relative.

Quality, n = 100,000, k = 256, nepochs = 2, seeds 1 to 3:

| d | emeans() | NumpyEmeans | difference | Lloyd, 25 iterations |
|---|---:|---:|---:|---:|
| 16 | 41.153 | 41.331 | −0.43% | 42.471 |
| 128 | 614.388 | 612.193 | +0.36% | 639.730 |

The gate is 1% on the mean over the three seeds. Single seeds vary by up to 1.1% in either direction,
because the two implementations draw different random numbers (PCG32 against numpy). Both beat Lloyd at
25 iterations after 2 epochs, which is the paper's claim.

Assignment. `emeans_assign` against `faiss.knn` for all 100,000 points of the d = 128 set and its 256
centroids: the same nearest centroid for every point.

## Speed (`make bench`)

Median of 3 runs, nepochs = 2, seed 1. `emeans()` is one SQL call, which DuckDB runs on one thread.

| n | d | k | emeans() | NumpyEmeans | Lloyd, 25 iterations | emeans() / NumpyEmeans |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 16 | 256 | 0.30 s | 0.12 s | 0.52 s | 2.5× |
| 100,000 | 128 | 256 | 3.26 s | 0.25 s | 0.30 s | 13× |
| 1,000,000 | 16 | 256 | 3.02 s | 0.69 s | 5.21 s | 4.4× |
| 1,000,000 | 128 | 256 | 33.8 s | 2.51 s | 2.66 s | 13× |

NumpyEmeans and Lloyd ran with FAISS's default of 18 OpenMP threads. With `omp_set_num_threads(1)`
they were at most 2× slower (1.33 s and 3.49 s for NumpyEmeans at n = 1,000,000), because the costly
part, the distances, goes to Accelerate's sgemm, which runs on the M5's matrix units and is not bound by
the OpenMP thread count. The comparison is therefore a plain C loop against a BLAS kernel, not one core
against eighteen.

Where the time goes: `emeans()` computes every distance as an exact sum of squared differences in
double, about n × k × d multiply-adds per pass. At d = 128 that is 6.6 × 10⁹ per pass for n = 100,000, or
about 4 GFLOP/s, which is what a scalar loop reaches. Phase 5 replaces it with ‖x‖² + ‖c‖² − 2 x·c in
blocked float32 loops that the compiler can vectorise, and splits a call's batches across threads.

`emeans_assign` reads the centroid list again for every row, so assigning 100,000 rows to 256 × 128
centroids took 4.7 s. Phase 5 converts a constant centroid argument once, at bind.

What the extension does better already: no data leaves DuckDB, the vectors can be any list or array
column, and many small clusterings in one query (`GROUP BY`) run in parallel across DuckDB's threads.
