#!/usr/bin/env python3
"""Validation against the reference e-means in faiss-cpu (faiss.contrib.e_means.NumpyEmeans). make compare_faiss.

Not part of `make test`: it needs faiss-cpu, numpy, pyarrow and DuckDB 2.0 (the Makefile runs it with uv in
a pinned environment) and takes a few minutes. Exits 1 if any check fails.

1. Recipes: batch size, alpha0 and total steps equal the reference's over a grid of shapes.
2. Quality: on Gaussian mixtures, the mean final objective of emeans() over several seeds is within
   OBJ_TOLERANCE of NumpyEmeans's, both scored the same way (each point to its nearest final centroid,
   by faiss.knn). Lloyd with 25 iterations (faiss.Kmeans) is reported for context, as in the paper.
3. Assignment: emeans_assign() gives the same nearest centroid as faiss.knn for every point, up to
   float32 ties.
4. Spherical: emeans_spherical() against NumpyEmeans(spherical=True) on the same mixtures with every row
   normalised, scored by the mean of 2 - 2 x.c over the best centroid; and emeans_assign_spherical()
   against faiss.knn with the inner-product metric.

Random numbers differ (PCG32 against numpy), so runs agree in distribution, not in bits.
"""

import math
import sys
import time

import duckdb
import faiss
import numpy as np
import pyarrow as pa
from faiss.contrib.e_means import base as reference_base
from faiss.contrib.e_means import NumpyEmeans

OBJ_TOLERANCE = 0.01      # relative, on the mean over seeds
SEEDS = (1, 2, 3)
NEPOCHS = 2.0
ASSIGN_AGREEMENT = 0.9999  # share of points with the same nearest centroid as faiss.knn

failures = []


def check(label, ok, detail):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


def connect(extension):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.sql(f"LOAD '{extension}'")
    con.sql("SET emeans_max_work = 0")
    return con


def register_points(con, x, name="pts"):
    """x (n x d float32) as a table name(id BIGINT, v FLOAT[d])."""
    n, d = x.shape
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(x.ravel(), type=pa.float32()), d)
    con.register(name, pa.table({"id": pa.array(np.arange(n)), "v": vectors}))


def gaussian_mixture(n, d, centres, seed):
    """n points around `centres` centres drawn from N(0, 3^2), each point with N(0, 1) noise."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((centres, d)) * 3.0
    which = rng.integers(0, centres, n)
    return (means[which] + rng.standard_normal((n, d))).astype("float32")


def objective(x, centroids, spherical=False):
    """Mean squared distance of each point to its nearest centroid; spherical, the mean of 2 - 2 x.c."""
    c = np.ascontiguousarray(centroids, dtype="float32")
    if spherical:
        sim, _ = faiss.knn(x, c, 1, metric=faiss.METRIC_INNER_PRODUCT)
        return float((2 - 2 * sim).mean())
    dist, _ = faiss.knn(x, c, 1)
    return float(dist.mean())


def normalised(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("float32")


#===----------------------------------------------------------------------===#
# 1. Recipes
#===----------------------------------------------------------------------===#

def reference_recipe(n, k, d, nepochs):
    law, _ = reference_base._resolve_batch_law("auto", d)
    batch, alpha0 = reference_base._resolve_recipe(n, k, nepochs, law, None, None)
    steps = sum(math.ceil(rows / batch) for rows in reference_base._epoch_rows(n, nepochs))
    return batch, alpha0, steps


def check_recipes(con):
    print("1. recipes against faiss.contrib.e_means.base")
    shapes = [(n, k, d, t) for n in (100, 10_000, 1_000_000, 50_000_000) for k in (1, 16, 1024)
              for d in (2, 16, 17, 768) for t in (0.5, 1.0, 2.0, 2.5, 25.0) if k <= n]
    mismatches = []
    for n, k, d, t in shapes:
        ours = con.sql(f"SELECT r.batch_size, r.alpha0, r.total_steps FROM (SELECT emeans_recipe({n}, {k}, {d}, {t}) AS r)").fetchall()[0]
        theirs = reference_recipe(n, k, d, t)
        if ours[0] != theirs[0] or ours[2] != theirs[2] or abs(ours[1] - theirs[1]) > 1e-12 * theirs[1]:
            mismatches.append(((n, k, d, t), ours, theirs))
    check("batch size, alpha0 and steps equal", not mismatches,
          f"{len(shapes)} shapes" + (f"; first mismatch {mismatches[0]}" if mismatches else ""))


#===----------------------------------------------------------------------===#
# 2. Quality
#===----------------------------------------------------------------------===#

def run_ours(con, k, seed, spherical=False):
    fn = "emeans_spherical" if spherical else "emeans"
    start = time.perf_counter()
    centroids = con.sql(f"SELECT ({fn}(list(v ORDER BY id), {k}, {NEPOCHS}, {seed})).centroids FROM pts").fetchall()[0][0]
    return np.array(centroids, dtype="float32"), time.perf_counter() - start


def run_reference(x, k, seed, spherical=False):
    start = time.perf_counter()
    centroids, _ = NumpyEmeans(x.shape[1], k, nepochs=NEPOCHS, seed=seed, spherical=spherical).train(x)
    return np.asarray(centroids, dtype="float32"), time.perf_counter() - start


def run_lloyd(x, k):
    kmeans = faiss.Kmeans(x.shape[1], k, niter=25, seed=1, max_points_per_centroid=x.shape[0])
    start = time.perf_counter()
    kmeans.train(x)
    return kmeans.centroids, time.perf_counter() - start


def check_quality(con, n, d, k, spherical=False):
    kind = "spherical " if spherical else ""
    print(f"{'4' if spherical else '2'}. {kind}quality: n = {n}, d = {d}, k = {k}, nepochs = {NEPOCHS}, {len(SEEDS)} seeds")
    x = gaussian_mixture(n, d, 2 * k, seed=d)
    x = normalised(x) if spherical else x
    register_points(con, x)
    ours, theirs = [], []
    for seed in SEEDS:
        c, t_ours = run_ours(con, k, seed, spherical)
        ours.append(objective(x, c, spherical))
        c, t_ref = run_reference(x, k, seed, spherical)
        theirs.append(objective(x, c, spherical))
        print(f"     seed {seed}: emeans {ours[-1]:.5f} ({t_ours:.1f} s)   NumpyEmeans {theirs[-1]:.5f} ({t_ref:.1f} s)")
    if not spherical:
        lloyd, t_lloyd = run_lloyd(x, k)
        print(f"     Lloyd, 25 iterations: {objective(x, lloyd):.5f} ({t_lloyd:.1f} s)")
    mean_ours, mean_theirs = np.mean(ours), np.mean(theirs)
    gap = (mean_ours - mean_theirs) / mean_theirs
    check(f"{kind}d = {d}: mean objective within {OBJ_TOLERANCE:.0%} of NumpyEmeans", abs(gap) <= OBJ_TOLERANCE,
          f"{mean_ours:.5f} against {mean_theirs:.5f} ({gap:+.3%})")
    return x, run_ours(con, k, SEEDS[0], spherical)[0]


#===----------------------------------------------------------------------===#
# 3. Assignment
#===----------------------------------------------------------------------===#

def exact_score(x, centroids, spherical):
    """Exact (float64) distance, or minus the inner product, of row i to centroid j."""
    def score(i, j):
        a, b = x[i].astype("float64"), centroids[j].astype("float64")
        return -float(a @ b) if spherical else float(np.sum((a - b) ** 2))
    return score


def check_assign(con, x, centroids, spherical=False):
    fn = "emeans_assign_spherical" if spherical else "emeans_assign"
    print(f"{'4' if spherical else '3'}. {fn} against faiss.knn, n = {x.shape[0]}, k = {centroids.shape[0]}")
    con.execute("CREATE OR REPLACE TABLE cents AS SELECT $c::FLOAT[][] AS c", {"c": centroids.tolist()})
    start = time.perf_counter()
    rows = con.sql(f"SELECT ({fn}(p.v, cents.c)).label FROM pts p, cents ORDER BY p.id").fetchall()
    elapsed = time.perf_counter() - start
    ours = np.array([r[0] - 1 for r in rows])
    metric = faiss.METRIC_INNER_PRODUCT if spherical else faiss.METRIC_L2
    _, theirs = faiss.knn(x, centroids, 1, metric=metric)
    differ = np.flatnonzero(ours != theirs[:, 0])
    # A disagreement is a float32 tie when both centroids score the same in exact arithmetic.
    exact = exact_score(x, centroids, spherical)
    ties = sum(abs(exact(i, ours[i]) - exact(i, theirs[i, 0])) <= 1e-5 * abs(exact(i, theirs[i, 0])) for i in differ)
    agreement = 1 - (len(differ) - ties) / len(ours)
    check(f"same nearest centroid for at least {ASSIGN_AGREEMENT:.2%} (ties aside)", agreement >= ASSIGN_AGREEMENT,
          f"{len(differ)} differ, {ties} of them ties; {elapsed:.1f} s")


def main():
    extension = sys.argv[1]
    print(f"faiss-cpu {faiss.__version__}, DuckDB {duckdb.__version__}, numpy {np.__version__}")
    con = connect(extension)
    check_recipes(con)
    check_quality(con, 100_000, 16, 256)
    x, centroids = check_quality(con, 100_000, 128, 256)
    check_assign(con, x, centroids)
    x, centroids = check_quality(con, 100_000, 128, 256, spherical=True)
    check_assign(con, x, centroids, spherical=True)
    print("\nFAIL: " + ", ".join(failures) if failures else "\nPASS: every check")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
