#!/usr/bin/env python3
"""Benchmark: emeans() against faiss-cpu's NumpyEmeans and Lloyd (faiss.Kmeans, 25 iterations). make bench.

Gaussian mixtures as in test/python/compare_faiss.py, nepochs = 2, seed 1. Each time is the median of
REPEATS runs. NumpyEmeans and Lloyd run on all cores (FAISS uses OpenMP and BLAS) and on one; emeans()
is one call, which DuckDB runs on one thread. The objective is the mean squared distance of every point
to its nearest final centroid. Prints a Markdown table for docs/BENCHMARK.md.
"""

import os
import platform
import statistics
import sys
import time

import duckdb
import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test", "python"))
from compare_faiss import connect, gaussian_mixture, objective, register_points  # noqa: E402

SHAPES = [(100_000, 16, 256), (100_000, 128, 256), (1_000_000, 16, 256), (1_000_000, 128, 256)]
NEPOCHS = 2.0
REPEATS = 3


def timed(fn):
    """The median time of REPEATS runs, and the last result."""
    times, result = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def ours(con, k):
    sql = f"SELECT (emeans(list(v ORDER BY id), {k}, {NEPOCHS}, 1)).centroids FROM pts"
    return lambda: np.array(con.sql(sql).fetchall()[0][0], dtype="float32")


def reference(x, k):
    return lambda: np.asarray(NumpyEmeans(x.shape[1], k, nepochs=NEPOCHS, seed=1).train(x)[0], dtype="float32")


def lloyd(x, k):
    def run():
        km = faiss.Kmeans(x.shape[1], k, niter=25, seed=1, max_points_per_centroid=x.shape[0])
        km.train(x)
        return km.centroids
    return run


def row(label, x, seconds, centroids):
    return f"| {label} | {seconds:.2f} | {objective(x, centroids):.4f} |"


def bench_shape(con, n, d, k, cores):
    x = gaussian_mixture(n, d, 2 * k, seed=d)
    register_points(con, x)
    lines = [f"\n**n = {n:,}, d = {d}, k = {k}**\n", "| run | seconds | objective |", "|---|---:|---:|"]
    lines.append(row("emeans(), 1 thread", x, *timed(ours(con, k))))
    for threads in (cores, 1):
        faiss.omp_set_num_threads(threads)
        lines.append(row(f"NumpyEmeans, {threads} thread{'s' if threads > 1 else ''}", x, *timed(reference(x, k))))
        lines.append(row(f"Lloyd 25 iterations, {threads} thread{'s' if threads > 1 else ''}", x, *timed(lloyd(x, k))))
    faiss.omp_set_num_threads(cores)
    print("\n".join(lines), flush=True)


def main():
    con = connect(sys.argv[1])
    cores = faiss.omp_get_max_threads()
    print(f"{platform.machine()} {platform.system()} {platform.release()}, {cores} cores; faiss-cpu "
          f"{faiss.__version__}, DuckDB {duckdb.__version__}; nepochs {NEPOCHS}, median of {REPEATS}")
    for n, d, k in SHAPES:
        bench_shape(con, n, d, k, cores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
