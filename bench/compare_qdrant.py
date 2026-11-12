#!/usr/bin/env python3
"""Nearest-neighbour search on real embeddings: an IVF index in DuckDB built with emeans, against Lloyd's
k-means for the same index, DuckDB's own HNSW (vss) and Qdrant's HNSW. make compare_qdrant.

Data: Qdrant's DBpedia set embedded with OpenAI text-embedding-3-small (d = 1536, 100,000 rows; Hugging
Face Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K). The first 99,000 rows are the
base, the last 1,000 the queries. Ground truth is the exact top 10 by inner product (numpy).

What is measured, per method and setting: index build time, recall@10 (the share of the true top 10
found), and the mean time per query, one query at a time from Python (each query's round trip, so
Qdrant's includes gRPC to a container and DuckDB's the Python API).

  IVF (emeans / Lloyd)  NLIST cells; centroids from emeans() (2 epochs) or emeans_trace(..., 'lloyd')
                        (25 iterations); every vector assigned with emeans_assign; stored sorted by cell
                        in 2,048-row row groups so the cells not probed are skipped. A query finds the
                        nprobe nearest centroids, then ranks the vectors of those cells by inner product.
  DuckDB HNSW (vss)     CREATE INDEX ... USING HNSW (metric 'ip'), defaults; hnsw_ef_search varied.
  Qdrant HNSW           cosine distance, defaults (m 16, ef_construct 100); hnsw_ef varied.
  Exact (DuckDB)        ORDER BY array_inner_product DESC LIMIT 10 over the whole base.

Run Qdrant first:  docker run -d --name emeans-qdrant -p 127.0.0.1:16333:6333 -p 127.0.0.1:16334:6334 qdrant/qdrant:v1.20.0
Usage: compare_qdrant.py <extension> <parquet glob> <scratch dir> [qdrant host]
"""

import os
import sys
import time

import duckdb
import numpy as np
from qdrant_client import QdrantClient, models

D = 1536
N_BASE = 99_000
N_QUERIES = 1_000
NLIST = 512
THREADS = 8  # the CPUs Docker gives Qdrant on this machine
NPROBES = (1, 2, 4, 8, 16, 32)
EFS = (16, 32, 64, 128, 256)
COLUMN = '"text-embedding-3-small-1536-embedding"'


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def recall(found, truth):
    return float(np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)]))


def row(method, setting, build_s, found, truth, seconds):
    ms = 1000 * seconds / len(found)
    print(f"| {method} | {setting} | {build_s} | {recall(found, truth):.3f} | {ms:.2f} | {1000 / ms:.0f} |", flush=True)


#===----------------------------------------------------------------------===#
# Data and ground truth
#===----------------------------------------------------------------------===#

def load(con, parquet):
    con.sql(f"""CREATE TABLE emb AS SELECT row_number() OVER () - 1 AS id, {COLUMN}::FLOAT[{D}] AS v
                FROM read_parquet('{parquet}')""")
    con.sql(f"CREATE TABLE base AS SELECT id, v FROM emb WHERE id < {N_BASE}")
    con.sql(f"CREATE TABLE queries AS SELECT id, v FROM emb WHERE id >= {N_BASE} ORDER BY id")


def matrix(con, table):
    vectors = con.sql(f"SELECT v FROM {table} ORDER BY id").arrow().read_all().column("v").combine_chunks()
    return vectors.values.to_numpy().reshape(-1, D)


def ground_truth(base, queries):
    scores = queries @ base.T
    top = np.argpartition(-scores, 10, axis=1)[:, :10]
    return [list(t) for t in top]


#===----------------------------------------------------------------------===#
# IVF in DuckDB
#===----------------------------------------------------------------------===#

def build_ivf(con, name, centroids_sql):
    """Centroids from the query, then every base vector assigned to its cell, sorted by cell."""
    def build():
        con.sql(f"CREATE OR REPLACE TABLE ivf.{name}_c AS {centroids_sql}")
        con.sql(f"""CREATE OR REPLACE TABLE ivf.{name}_cells AS
                    SELECT i AS cell, c[i]::FLOAT[{D}] AS cv FROM ivf.{name}_c, range(1, {NLIST} + 1) t(i)""")
        con.sql(f"""CREATE OR REPLACE TABLE ivf.{name} AS
                    SELECT id, v, (emeans_assign(v, (SELECT c FROM ivf.{name}_c))).label AS cell
                    FROM base ORDER BY cell""")
        con.sql("CHECKPOINT ivf")
    _, seconds = timed(build)
    return seconds


def query_ivf(con, name, queries, nprobe):
    """Two statements per query: the nearest cells, then a scan of only those (row groups of other cells
    are skipped by their min/max)."""
    probe = con.cursor()
    found = []
    def run():
        for q in queries:
            ql = q.tolist()
            cells = [r[0] for r in probe.execute(
                f"SELECT cell FROM ivf.{name}_cells ORDER BY array_distance(cv, $1::FLOAT[{D}]) LIMIT {nprobe}",
                [ql]).fetchall()]
            found.append([r[0] for r in probe.execute(
                f"SELECT id FROM ivf.{name} WHERE cell IN ({','.join(map(str, cells))}) "
                f"ORDER BY array_inner_product(v, $1::FLOAT[{D}]) DESC LIMIT 10", [ql]).fetchall()])
    _, seconds = timed(run)
    return found, seconds


#===----------------------------------------------------------------------===#
# DuckDB exact and HNSW
#===----------------------------------------------------------------------===#

def query_duckdb(con, queries, sql):
    cur = con.cursor()
    found = []
    def run():
        for q in queries:
            found.append([r[0] for r in cur.execute(sql, [q.tolist()]).fetchall()])
    _, seconds = timed(run)
    return found, seconds


def build_hnsw(con):
    con.sql("INSTALL vss")
    con.sql("LOAD vss")
    con.sql(f"CREATE TABLE hbase AS SELECT id, v FROM base")
    _, seconds = timed(lambda: con.sql("CREATE INDEX hnsw_ip ON hbase USING HNSW (v) WITH (metric = 'ip')"))
    return seconds


#===----------------------------------------------------------------------===#
# Qdrant
#===----------------------------------------------------------------------===#

def build_qdrant(client, base):
    name = "dbpedia"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(size=D, distance=models.Distance.COSINE))
    def upload_and_index():
        client.upload_collection(name, vectors=base, ids=list(range(len(base))), batch_size=512, parallel=1)  # forked workers and gRPC do not mix reliably
        while True:
            info = client.get_collection(name)
            if info.status == models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= len(base):
                return
            time.sleep(0.5)
    _, seconds = timed(upload_and_index)
    return name, seconds


def query_qdrant(client, name, queries, ef):
    found = []
    params = models.SearchParams(hnsw_ef=ef)
    def run():
        for q in queries:
            hits = client.query_points(name, query=q.tolist(), limit=10, search_params=params, with_payload=False)
            found.append([p.id for p in hits.points])
    _, seconds = timed(run)
    return found, seconds


def main():
    extension, parquet, scratch = sys.argv[1], sys.argv[2], sys.argv[3]
    host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
    con = duckdb.connect(config={"allow_unsigned_extensions": "true", "threads": str(THREADS)})
    con.sql(f"LOAD '{extension}'")
    con.sql("SET emeans_max_work = 0")
    ivf_path = os.path.join(scratch, "ivf.duckdb")
    for f in (ivf_path, ivf_path + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    con.sql(f"ATTACH '{ivf_path}' AS ivf (ROW_GROUP_SIZE 2048)")

    load(con, parquet)
    base, queries = matrix(con, "base"), matrix(con, "queries")
    print(f"base {base.shape}, queries {queries.shape}, norms {np.linalg.norm(base, axis=1).min():.4f} to "
          f"{np.linalg.norm(base, axis=1).max():.4f}; DuckDB {duckdb.__version__}, {THREADS} threads\n")
    truth, gt_s = timed(lambda: ground_truth(base, queries))
    print(f"ground truth (numpy, exact): {gt_s:.1f} s\n")
    print("| method | setting | build s | recall@10 | ms/query | queries/s |")
    print("|---|---|---:|---:|---:|---:|")

    found, s = query_duckdb(con, queries[:200],
                            f"SELECT id FROM base ORDER BY array_inner_product(v, $1::FLOAT[{D}]) DESC LIMIT 10")
    row("exact, DuckDB", "full scan (200 queries)", "0", found, truth[:200], s)

    for name, sql in (
        ("emeans", f"SELECT (emeans(list(v ORDER BY id), {NLIST}, 2.0, 1)).centroids AS c FROM base"),
        ("lloyd", f"SELECT (emeans_trace(list(v ORDER BY id), NULL, {NLIST}, 25.0, 1, 'lloyd', 0.04, NULL, NULL))"
                  f".frames[-1].centroids AS c FROM base"),
    ):
        build_s = build_ivf(con, name, sql)
        for nprobe in NPROBES:
            found, s = query_ivf(con, name, queries, nprobe)
            row(f"IVF {NLIST}, {'e-means 2 epochs' if name == 'emeans' else 'Lloyd 25 iterations'}",
                f"nprobe {nprobe}", f"{build_s:.1f}", found, truth, s)

    build_s = build_hnsw(con)
    for ef in EFS:
        con.sql(f"SET GLOBAL hnsw_ef_search = {ef}")  # global: query_duckdb runs on a cursor, a separate connection
        found, s = query_duckdb(con, queries,
                                f"SELECT id FROM hbase ORDER BY array_negative_inner_product(v, $1::FLOAT[{D}]) LIMIT 10")
        row("HNSW, DuckDB vss", f"ef_search {ef}", f"{build_s:.1f}", found, truth, s)

    client = QdrantClient(host=host, port=16333, grpc_port=16334, prefer_grpc=True, timeout=600)
    name, build_s = build_qdrant(client, base)
    for ef in EFS:
        found, s = query_qdrant(client, name, queries, ef)
        row("HNSW, Qdrant 1.20", f"hnsw_ef {ef}", f"{build_s:.1f}", found, truth, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
