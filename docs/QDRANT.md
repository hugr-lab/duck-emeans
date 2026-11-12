# Nearest-neighbour search: IVF with emeans, DuckDB's HNSW and Qdrant

e-means is a clustering algorithm and Qdrant a vector database, so they are not compared directly. What
can be compared is what both are used for: approximate nearest-neighbour search on embeddings. The
extension makes the coarse quantiser of an IVF index in plain SQL; DuckDB's `vss` extension and Qdrant
offer HNSW indexes.

Measured on 12 November 2026 on an Apple M5 Pro with `make compare_qdrant` (`bench/compare_qdrant.py`).

**Data.** Qdrant's DBpedia set embedded with OpenAI `text-embedding-3-small`, d = 1536, unit length
([Hugging Face](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K),
revision `0d5fd00fbc5a`). The first 99,000 rows are the base, the last 1,000 the queries. Ground truth is
the exact top 10 by inner product.

**Setup.** DuckDB 2.0.0 on 8 threads; Qdrant 1.20.0 in Docker, which has 8 CPUs here, reached over gRPC
on localhost. Queries run one at a time from Python, so each time includes the client round trip.

| method | setting | build s | recall@10 | ms/query |
|---|---|---:|---:|---:|
| exact, DuckDB | full scan | 0 | 1.000 | 155 |
| IVF 512 cells, e-means 2 epochs | nprobe 4 | 7.1 | 0.762 | 9.0 |
| | nprobe 8 | | 0.836 | 11.3 |
| | nprobe 16 | | 0.893 | 18.5 |
| | nprobe 32 | | 0.940 | 31.5 |
| IVF 512 cells, Lloyd 25 iterations | nprobe 4 | 13.5 | 0.757 | 9.2 |
| | nprobe 8 | | 0.834 | 11.6 |
| | nprobe 16 | | 0.895 | 16.9 |
| | nprobe 32 | | 0.934 | 24.2 |
| HNSW, DuckDB `vss` | ef_search 32 | 28.1 | 0.926 | 2.5 |
| | ef_search 64 | | 0.969 | 2.8 |
| | ef_search 128 | | 0.987 | 3.2 |
| | ef_search 256 | | 0.995 | 3.9 |
| HNSW, Qdrant 1.20 | hnsw_ef 16 | 28.9 | 0.911 | 1.7 |
| | hnsw_ef 64 | | 0.987 | 2.3 |
| | hnsw_ef 128 | | 0.997 | 3.1 |
| | hnsw_ef 256 | | 0.999 | 4.1 |

nprobe 1 and 2, and the other ef settings, are in the script's full output.

**e-means as an IVF quantiser.** At every nprobe the e-means index finds the same neighbours as the Lloyd
index, within 0.01 recall either way, and builds in half the time (7.1 s against 13.5 s, for the
centroids and the assignment of every vector). This is the comparison that tests e-means itself.

**IVF in SQL against HNSW.** Both HNSW indexes are far ahead of the SQL IVF: about 0.97 to 0.99 recall at
2 to 3 ms per query, where the IVF needs 31 ms for 0.94. The IVF query is two statements and a scan of
the probed cells through DuckDB's general query machinery; an IVF needs many cells probed for high recall
on 1536-dimensional text embeddings. The IVF builds four times faster.

**DuckDB `vss` against Qdrant.** At this size the two HNSW indexes are close: the same build time, and
similar latency; Qdrant reaches a given recall at a smaller `ef` (0.987 at 64 against 0.969). For data
already in DuckDB, `vss` is a working alternative to running Qdrant at 100,000 vectors. Qdrant offers
what neither DuckDB index here does: payload filtering, updates while serving, persistence and scaling
out; none of that was measured.

**Limits.** One data set and one size; queries one at a time; Qdrant behind Docker's VM and gRPC, DuckDB
in the process. `vss` keeps its index in memory. To reproduce:

```sh
docker run -d --name emeans-qdrant -p 127.0.0.1:16333:6333 -p 127.0.0.1:16334:6334 qdrant/qdrant:v1.20.0
# download the four Parquet files of the data set into a directory, then
make compare_qdrant QDRANT_DATA=/path/to/dir
```
