#!/usr/bin/env python3
"""Benchmark: emeans() against faiss-cpu's NumpyEmeans and Lloyd (faiss.Kmeans, 25 iterations). make bench.

Gaussian mixtures as in test/python/compare_faiss.py, nepochs = 2, seed 1. Each time is the median of
REPEATS runs. emeans() runs with emeans_threads = 1 and = all cores; NumpyEmeans and Lloyd with FAISS's
OpenMP threads at all cores and at one (their BLAS ignores that count). The objective is the mean squared distance of every point
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


def thread_label(name, threads):
    return f"{name}, {threads} thread{'s' if threads > 1 else ''}"


def faiss_rows(x, k, threads):
    faiss.omp_set_num_threads(threads)
    return [row(thread_label("NumpyEmeans", threads), x, *timed(reference(x, k))),
            row(thread_label("Lloyd 25 iterations", threads), x, *timed(lloyd(x, k)))]


def bench_shape(con, n, d, k, cores):
    x = gaussian_mixture(n, d, 2 * k, seed=d)
    register_points(con, x)
    lines = [f"\n**n = {n:,}, d = {d}, k = {k}**\n", "| run | seconds | objective |", "|---|---:|---:|"]
    con.sql("SET emeans_threads = 1")
    lines.append(row(thread_label("emeans()", 1), x, *timed(ours(con, k))))
    con.sql(f"SET emeans_threads = {cores}")
    lines.append(row(thread_label("emeans()", cores), x, *timed(ours(con, k))))
    lines += faiss_rows(x, k, cores) + faiss_rows(x, k, 1)
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
