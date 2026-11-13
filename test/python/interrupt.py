"""emeans_stream stops when its query is interrupted (make test_interrupt).

A long run (2M x 64, k = 256, 50 epochs) is interrupted from another thread with con.interrupt(), early (while
the count and the start run) and late (in the passes), on one thread and on eight, and on four under a
memory_limit below the data (512 MB on disk under 300 MB), where every pass's sort spills and this thread waits in Fetch
while the workers sort (src/cpp/emeans_stream.cpp, InterruptWatcher). Each must stop within LIMIT_S seconds of
the interrupt with DuckDB's InterruptException, and the connection must work afterwards. Run in CI (core-tests.yml).
"""

import os
import sys
import tempfile
import threading
import time

import duckdb

LIMIT_S = 2.0
# 50 epochs: the batch law gives B = 524,288 rows, a 134 MB batch buffer at d = 64; batch_size keeps it small
# under the memory_limit case, so the sorts have the memory.
RUN = "SELECT count(*) FROM emeans_stream('SELECT v FROM t', 256, nepochs := 50, batch_size := {batch_size})"


def interrupted_after(con: duckdb.DuckDBPyConnection, delay: float, batch_size: int) -> float | None:
    """Seconds from the interrupt to the end of the query, or None if it was not interrupted."""
    timer = threading.Timer(delay, con.interrupt)
    start = time.time()
    timer.start()
    try:
        con.execute(RUN.format(batch_size=batch_size)).fetchall()
    except duckdb.InterruptException:
        return time.time() - start - delay
    finally:
        timer.cancel()
    return None


def check(extension: str, threads: int, memory_limit: str | None = None) -> list[str]:
    # Under a memory_limit the table is on disk, as data larger than memory would be: in an in-memory database it
    # would compete with the sorts for the same memory, and how far a run gets before running out varies. The
    # directory (the 512 MB database and any spill files) goes when the check ends.
    with tempfile.TemporaryDirectory() as tmp:
        database = os.path.join(tmp, "interrupt.duckdb") if memory_limit else ":memory:"
        con = duckdb.connect(database, config={"allow_unsigned_extensions": "true"})
        try:
            return check_on(con, extension, threads, memory_limit)
        finally:
            con.close()


def check_on(con: duckdb.DuckDBPyConnection, extension: str, threads: int, memory_limit: str | None) -> list[str]:
    con.execute(f"LOAD '{extension}'")
    con.execute(f"SET threads = {threads}")
    con.execute(
        "CREATE TABLE t AS SELECT list_transform(range(64), lambda j: (hash(i, j) % 1000)::FLOAT)::FLOAT[64] AS v "
        "FROM range(2000000) r(i)"
    )
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    label = f"threads {threads}" + (f", memory_limit {memory_limit}" if memory_limit else "")
    failures = []
    for delay in (0.05, 1.0, 2.5):
        late = interrupted_after(con, delay, 16384 if memory_limit else 0)
        print(f"{label}, interrupt at {delay} s: " + ("not interrupted" if late is None else f"stopped {late:.2f} s later"))
        if late is None or late > LIMIT_S:
            failures.append(f"{label}, interrupt at {delay} s")
    after = con.execute("SELECT count(*) FROM emeans_stream('SELECT v FROM t LIMIT 1000', 4)").fetchone()
    if after != (4,):
        failures.append(f"{label}: the connection after an interrupt returned {after}")
    return failures


def main() -> int:
    failures = check(sys.argv[1], 1) + check(sys.argv[1], 8) + check(sys.argv[1], 4, "300MB")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
