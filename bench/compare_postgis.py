#!/usr/bin/env python3
"""emeans_agg + emeans_assign_point against PostGIS's ST_ClusterKMeans. make compare_postgis.

Two parts:

1. Behaviour. The same small inputs go to both: NULL and empty points, a polygon, k above the number
   of points, identical points, points across the antimeridian in lon/lat, Z and M values, row order,
   repeated runs. The PostGIS documentation leaves most of these unstated, so this records what each
   system does.
2. Speed and quality. Gaussian blobs in projected metres (n = 10,000 to 1,000,000, k = 8 to 256), one
   partition. Each side labels every row: PostGIS with ST_ClusterKMeans(geom, k) OVER (), DuckDB with
   emeans_assign_point(geom, (emeans_agg(geom, k) OVER ()).centroids). Reported: the time, the within-
   cluster sum of squares of each labelling (lower is better), and the adjusted Rand index between the
   two labellings (1 = the same partition).
3. A figure (PNG): clusters across the antimeridian in lon/lat from each system, the two labellings of
   the same projected data, and time against n.

Usage: compare_postgis.py <extension> <postgres DSN> [figure.png]
"""

import io
import sys
import time

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg

SIZES = [(10_000, 8), (10_000, 64), (100_000, 8), (100_000, 64), (100_000, 256), (1_000_000, 64)]
DUCK_THREADS = 8


def duck(extension):
    con = duckdb.connect(config={"allow_unsigned_extensions": "true", "threads": str(DUCK_THREADS)})
    con.execute(f"LOAD '{extension}'")
    return con


#===----------------------------------------------------------------------===#
# 1. Behaviour
#===----------------------------------------------------------------------===#

# (name, rows as WKT or None for NULL, k, lon/lat or planar)
CASES = [
    ("two clusters", ["POINT(0 0)", "POINT(0 1)", "POINT(10 10)", "POINT(10 11)"], 2, False),
    ("NULL point", ["POINT(0 0)", None, "POINT(10 10)", "POINT(10 11)"], 2, False),
    ("empty point", ["POINT(0 0)", "POINT EMPTY", "POINT(10 10)", "POINT(10 11)"], 2, False),
    ("polygon", ["POINT(0 0)", "POLYGON((9 9, 11 9, 11 11, 9 11, 9 9))", "POINT(10 10)", "POINT(10 11)"], 2, False),
    ("k above n", ["POINT(0 0)", "POINT(10 10)"], 3, False),
    ("identical points", ["POINT(5 5)"] * 4, 3, False),
    ("antimeridian, lon/lat", ["POINT(179.9 0)", "POINT(-179.9 0)", "POINT(179.8 0.1)", "POINT(-179.8 -0.1)",
                               "POINT(0 0)", "POINT(0.1 0)", "POINT(0 0.1)", "POINT(0.1 0.1)"], 2, True),
    ("Z ignored or used", ["POINT Z(0 0 0)", "POINT Z(0 0 100)", "POINT Z(0 0 1)", "POINT Z(0 0 101)"], 2, False),
    ("M as weight", ["POINT M(0 0 1)", "POINT M(10 0 1)", "POINT M(4 0 100)", "POINT M(6 0 1)"], 2, False),
]


def pg_case(pg, rows, k, geographic):
    srid = 4326 if geographic else 0
    values = ",".join(f"({i}, {'NULL' if w is None else f'ST_GeomFromText({w!r}, {srid})'})" for i, w in enumerate(rows))
    try:
        cur = pg.execute(f"SELECT i, ST_ClusterKMeans(g, {k}) OVER () FROM (VALUES {values}) t(i, g) ORDER BY i")
        return [r[1] for r in cur.fetchall()], None
    except psycopg.Error as e:
        pg.rollback()
        return None, str(e).splitlines()[0]


def duck_case(con, rows, k, geographic):
    cast = "GEOMETRY('OGC:CRS84')" if geographic else "GEOMETRY"
    values = ",".join(f"({i}, {'NULL' if w is None else f'{w!r}::{cast}'})" for i, w in enumerate(rows))
    try:
        rows_out = con.execute(
            f"SELECT i, (emeans_assign_point(g, (emeans_agg(g, {k}) OVER ()).centroids)).label "
            f"FROM (VALUES {values}) t(i, g) ORDER BY i").fetchall()
        return [r[1] for r in rows_out], None
    except duckdb.Error as e:
        return None, str(e).splitlines()[0]


def partition(labels):
    """Labels as a set of groups of row positions, so 0-based and 1-based ids compare equal."""
    if labels is None:
        return None
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    return sorted(groups.values())


def behaviour(pg, con):
    print("## Behaviour\n")
    print("| case | PostGIS ST_ClusterKMeans | DuckDB emeans | same partition |")
    print("|---|---|---|---|")
    for name, rows, k, geographic in CASES:
        p, perr = pg_case(pg, rows, k, geographic)
        d, derr = duck_case(con, rows, k, geographic)
        same = "yes" if p is not None and d is not None and partition(p) == partition(d) else "no"
        print(f"| {name} | {perr or p} | {derr or d} | {same} |")
    # Repeats and row order on the same random points.
    rng = np.random.default_rng(3)
    pts = [f"POINT({x:.4f} {y:.4f})" for x, y in rng.uniform(0, 100, (500, 2))]
    p1, _ = pg_case(pg, pts, 6, False)
    p2, _ = pg_case(pg, pts, 6, False)
    d1, _ = duck_case(con, pts, 6, False)
    p_rev, _ = pg_case(pg, pts[::-1], 6, False)
    d_rev, _ = duck_case(con, pts[::-1], 6, False)
    print(f"| same input twice | {'same' if p1 == p2 else 'different'} | same seed, same result | |")
    print(f"| rows reversed | {'same partition' if partition(p_rev[::-1]) == partition(p1) else 'different partition'} | "
          f"{'same partition' if partition(d_rev[::-1]) == partition(d1) else 'different partition'} | |")
    print()


#===----------------------------------------------------------------------===#
# 2. Speed and quality
#===----------------------------------------------------------------------===#

def blobs(n, seed):
    """n points around 2k... centres in a 100 km square, in projected metres near (674000, 6580000)."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 100_000, (128, 2))
    which = rng.integers(0, 128, n)
    return centres[which] + rng.normal(0, 2_000, (n, 2)) + np.array([674_000.0, 6_580_000.0])


def wcss(xy, labels):
    """Within-cluster sum of squares of a labelling: each point to its cluster's mean."""
    labels = np.asarray(labels)
    total = 0.0
    for l in np.unique(labels):
        pts = xy[labels == l]
        total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def adjusted_rand(a, b):
    a, b = np.unique(np.asarray(a), return_inverse=True)[1], np.unique(np.asarray(b), return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    comb = lambda x: (x * (x - 1) / 2).sum()
    index, ra, rb, n2 = comb(table), comb(table.sum(1)), comb(table.sum(0)), comb(np.array([len(a)]))
    expected = ra * rb / n2
    return float((index - expected) / ((ra + rb) / 2 - expected))


def load(pg, con, xy):
    pg.execute("DROP TABLE IF EXISTS pts")
    pg.execute("CREATE TABLE pts (id integer, x float8, y float8)")
    buf = io.StringIO("".join(f"{i}\t{x:.17g}\t{y:.17g}\n" for i, (x, y) in enumerate(xy)))
    with pg.cursor().copy("COPY pts FROM STDIN") as cp:
        cp.write(buf.getvalue())
    pg.execute("ALTER TABLE pts ADD COLUMN geom geometry(Point, 3006)")
    pg.execute("UPDATE pts SET geom = ST_SetSRID(ST_MakePoint(x, y), 3006)")
    pg.commit()
    con.execute("CREATE OR REPLACE TABLE pts AS SELECT * FROM (SELECT unnest(range($n)) AS id, unnest($x) AS x, "
                "unnest($y) AS y)", {"n": len(xy), "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist()})
    con.execute("CREATE OR REPLACE TABLE pts AS SELECT id, ('POINT(' || x || ' ' || y || ')')::GEOMETRY AS geom FROM pts")


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def speed(pg, con):
    results = []
    print("## Speed and quality\n")
    print("| n | k | PostGIS s | DuckDB s | WCSS PostGIS | WCSS DuckDB | DuckDB / PostGIS | ARI |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, k in SIZES:
        xy = blobs(n, seed=n + k)
        load(pg, con, xy)
        p, pt = timed(lambda: [r[0] for r in pg.execute(
            f"SELECT ST_ClusterKMeans(geom, {k}) OVER () FROM pts ORDER BY id").fetchall()])
        d, dt = timed(lambda: [r[0] for r in con.execute(
            f"SELECT (emeans_assign_point(geom, (emeans_agg(geom, {k}) OVER ()).centroids)).label "
            f"FROM pts ORDER BY id").fetchall()])
        wp, wd = wcss(xy, p), wcss(xy, d)
        ari = adjusted_rand(p, d)
        print(f"| {n:,} | {k} | {pt:.2f} | {dt:.2f} | {wp:.4g} | {wd:.4g} | {wd / wp:.4f} | {ari:.3f} |", flush=True)
        results.append({"n": n, "k": k, "pg_s": pt, "duck_s": dt, "ratio": wd / wp, "ari": ari})
    return results


#===----------------------------------------------------------------------===#
# 3. The figure
#===----------------------------------------------------------------------===#

PG_COLOUR, DUCK_COLOUR = "#c62828", "#1565c0"


def antimeridian_points(seed=7):
    """Three groups in lon/lat: one straddling 180 degrees, one at 150 E, one at 150 W."""
    rng = np.random.default_rng(seed)
    centres = [(180.0, 0.0), (150.0, -12.0), (-150.0, 12.0)]
    pts = []
    for lon, lat in centres:
        for _ in range(150):
            x = lon + rng.normal(0, 3)
            pts.append(((x + 180) % 360 - 180, lat + rng.normal(0, 3)))
    return np.array(pts)


def label_lonlat(pg, con, ll, k):
    wkt = [f"POINT({x:.5f} {y:.5f})" for x, y in ll]
    p, _ = pg_case(pg, wkt, k, True)
    d, _ = duck_case(con, wkt, k, True)
    return p, d


def scatter(ax, xy, labels, title, cmap="tab10", s=4):
    ax.scatter(xy[:, 0], xy[:, 1], c=np.asarray(labels) % 10, cmap=cmap, s=s, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def figure(pg, con, results, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    # Antimeridian: longitudes shown 0..360 so the straddling group is one blob on the page.
    ll = antimeridian_points()
    p, d = label_lonlat(pg, con, ll, 3)
    shown = np.column_stack([(ll[:, 0] + 360) % 360, ll[:, 1]])
    scatter(axes[0, 0], shown, p, "PostGIS ST_ClusterKMeans, SRID 4326, k = 3\n(planar on lon/lat: splits the group at 180°)")
    scatter(axes[1, 0], shown, d, "emeans_agg, OGC:CRS84, k = 3\n(on the sphere: one group across 180°)")
    for ax in axes[:, 0]:
        ax.axvline(180, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("longitude (shown 0 to 360)", fontsize=8)
        ax.set_ylabel("latitude", fontsize=8)
    # The same projected blobs, labelled by each.
    xy = blobs(20_000, seed=11)
    load(pg, con, xy)
    pl = [r[0] for r in pg.execute("SELECT ST_ClusterKMeans(geom, 8) OVER () FROM pts ORDER BY id").fetchall()]
    dl = [r[0] for r in con.execute("SELECT (emeans_assign_point(geom, (emeans_agg(geom, 8) OVER ()).centroids)).label "
                                    "FROM pts ORDER BY id").fetchall()]
    km = (xy - xy.min(axis=0)) / 1000
    scatter(axes[0, 1], km, pl, f"PostGIS, n = 20,000, k = 8\nWCSS {wcss(xy, pl):.4g}", s=2)
    scatter(axes[1, 1], km, dl, f"emeans_agg, n = 20,000, k = 8\nWCSS {wcss(xy, dl):.4g}, "
            f"adjusted Rand index to PostGIS {adjusted_rand(pl, dl):.3f}", s=2)
    for ax in axes[:, 1]:
        ax.set_xlabel("km east", fontsize=8)
        ax.set_ylabel("km north", fontsize=8)
    # Time against n at k = 64; quality ratio as labels.
    ax = axes[0, 2]
    rows = [r for r in results if r["k"] == 64]
    ns = [r["n"] for r in rows]
    ax.loglog(ns, [r["pg_s"] for r in rows], "o-", color=PG_COLOUR, label="PostGIS ST_ClusterKMeans")
    ax.loglog(ns, [r["duck_s"] for r in rows], "o-", color=DUCK_COLOUR, label="DuckDB emeans_agg + assign")
    for r in rows:
        ax.annotate(f"WCSS ratio {r['ratio']:.3f}", (r["n"], r["duck_s"]), textcoords="offset points", xytext=(4, -12),
                    fontsize=7, color=DUCK_COLOUR)
    ax.set_title("Time to label every point, k = 64", fontsize=10)
    ax.set_xlabel("points n", fontsize=8)
    ax.set_ylabel("seconds", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", lw=0.3)
    # Speed-up and quality for every size.
    ax = axes[1, 2]
    labels = [f"n {r['n'] // 1000}k\nk {r['k']}" for r in results]
    ax.bar(range(len(results)), [r["pg_s"] / r["duck_s"] for r in results], color=DUCK_COLOUR)
    ax.axhline(1, color="0.4", lw=0.8)
    ax.set_xticks(range(len(results)), labels, fontsize=7)
    ax.set_ylabel("PostGIS time / DuckDB time", fontsize=8)
    ax.set_title("Speed-up (above 1: DuckDB faster); bar labels: DuckDB WCSS / PostGIS WCSS", fontsize=9)
    for i, r in enumerate(results):
        ax.annotate(f"{r['ratio']:.3f}", (i, r["pg_s"] / r["duck_s"]), ha="center", va="bottom", fontsize=7)
    fig.suptitle("emeans (DuckDB) against PostGIS ST_ClusterKMeans", fontsize=13)
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


def main():
    extension, dsn = sys.argv[1], sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else "postgis_comparison.png"
    pg = psycopg.connect(dsn)
    con = duck(extension)
    version = pg.execute("SELECT postgis_full_version()").fetchone()[0].split()[0:2]
    print(f"PostgreSQL {pg.info.server_version}, {' '.join(version)}; DuckDB {duckdb.__version__} "
          f"({DUCK_THREADS} threads)\n")
    behaviour(pg, con)
    results = speed(pg, con)
    figure(pg, con, results, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
