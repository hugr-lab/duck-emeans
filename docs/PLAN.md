# Plan

e-means has no C or C++ source of its own: the reference is Python, with FAISS doing the nearest-centroid
search on CPU and Triton/CuTe kernels on GPU. This extension is a reimplementation in C11 of the
algorithm in `base.py` and `e_means.py` (about 1,100 lines together), against the stable DuckDB C
extension API, for DuckDB 2.0.

## SQL interface

The data goes through a LIST, as in `cpt`: the C API has no table-in/table-out functions, and e-means
needs the whole dataset for its random permutations anyway.

```sql
SELECT emeans(list(embedding ORDER BY id), 1000) FROM docs;
-- STRUCT(centroids LIST<LIST<FLOAT>>, labels LIST<INTEGER>, obj LIST<DOUBLE>,
--        batch_size BIGINT, alpha0 DOUBLE, n_splits INTEGER)

SELECT id, emeans_assign(embedding, c.centroids) FROM docs, (SELECT emeans(...) AS c FROM docs);
-- STRUCT(label INTEGER, distance DOUBLE)

SELECT emeans_recipe(1000000, 1000, 128, 2.0);
-- STRUCT(batch_size, alpha0, batch_law, n_passes, total_steps)
```

Overloads: `emeans(x, k)`, `emeans(x, k, nepochs)`, `emeans(x, k, nepochs, seed)`, and
`emeans_spherical(...)`. Vectors are accepted as `LIST<DOUBLE>`, `LIST<FLOAT>` and the fixed-size
`FLOAT[d]` array that the `vss` extension uses. Centroids come back as `LIST<LIST<FLOAT>>` first;
returning `FLOAT[d]` needs the bind callback to see `d`. `emeans_assign` converts a constant centroid
argument once in bind and caches the norms, so the hot path is a plain n × k × d loop.

## Core (`src/core/`, no DuckDB dependency)

- `recipe.c` (done): batch laws B = c · k^0.134749 · n^0.876473 · T^0.813344 with c = 0.035134
  (throughput) or 0.00439175 (quality; `auto` picks it for d ≤ 16), rounded to the nearest power of
  two and capped; step law α0 = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057 clamped to
  [1e-4, 1]; pass schedule; cosine schedule α = α0 · (0.1 + 0.9 · ½(1 + cos(π · step/total))).
- `rng.c` (done): PCG32 and a partial Fisher–Yates shuffle. Numpy's permutation cannot be matched
  bit for bit, so tests compare objectives and assignments, not centroid bits.
- `assign.c` (done, plain loop): exact nearest centroid, squared differences in double. Phase 5 replaces
  it with ‖x‖² + ‖c‖² − 2 x·c in blocked loops (64 rows × 64 centroids × d) so the compiler can
  vectorise. No BLAS.
- `emeans.c` (done): state is the centroid c_j and mass N_j (initially 2^-16). Per batch: assign, accumulate
  n_j and s_j; N_j ← (1 − α) N_j for all j; for n_j > 0, N_j += α n_j and
  c_j += (α n_j / N_j)(s_j / n_j − c_j). α = 1 is one Lloyd step. Every 5 steps: revival, with
  threshold τ ΣN_j / k (τ = 0.1, none for d ≤ 2), donors sampled with Gumbel keys
  log N_j − log(−log u), split at ±1e-4 relative offset with the mass halved. After the last pass the
  closing mean c_j = pass_sums_j / pass_counts_j; labels are the last pass's assignments. Spherical:
  normalised centroids, argmax dot product, objective Σ(2 − 2·sim).
- Budget: a scalar function cannot be interrupted, so the assignment loop charges a budget and a
  setting bounds it. Threads inside one call come later: DuckDB gives one `emeans()` call one thread,
  and the assignment is where the time goes.

## Phases

1. Scaffold (done): repository, build, CI, test gates, `emeans_recipe`.
2. Core (done): rng, assign kernel, the loop; C tests, Lloyd equivalence (B = n, α0 = 1 reproduces two
   Lloyd iterations computed by hand on a 6-point set), revival on a set with two dead initial centroids,
   closing means, determinism, fault injection, 8 threads under TSan.
3. SQL (done): `emeans`, `emeans_assign`, LIST and ARRAY inputs (arrays through the implicit cast),
   NULL handling, errors, the `emeans_max_work` setting, sqllogictests, a threads test.
4. Validation (done): the recipe matches faiss-cpu 1.16.0 for 220 shapes; the final objective is within
   1% of `NumpyEmeans` on Gaussian mixtures (n = 100k, d = 16 and 128, k = 256); `emeans_assign` agrees
   with `faiss.knn`; benchmark in docs/BENCHMARK.md (2.5× slower than FAISS at d = 16, 13× at d = 128).
5. Done: the blocked kernel, spherical, `emeans_assign` reusing prepared centroids within a chunk, the
   budget setting (phase 3). Not done: `FLOAT[d]` output, because the stable C API cannot set a return
   type at bind (cast in SQL instead); folding constant centroids at bind, because asking for a bind
   argument fails for scalar subqueries.
6. Done: threads inside one call (`src/core/parallel.c`: POSIX threads, Win32 threads, one thread in
   WASM), used only for batches with at least 4 million units of work per thread, and Accelerate's sgemm
   for the kernel on macOS. `emeans()` now beats NumpyEmeans on an M5 Pro (docs/BENCHMARK.md).
7. GPU, as a separate build that users choose: the community extension stays CPU-only, and a second
   build adds a GPU backend for the kernel (MLX or Metal on Apple silicon, CUDA elsewhere). e-means makes
   many sequential steps with small batches, so per-step launch overhead matters; the reference needed
   custom Triton and CuTe kernels for it.
8. Done: the DuckDB-WASM demo (`web/`, GitHub Pages).
9. Done: the playground in the style of the e-means site's (Lloyd and e-means side by side with Voronoi
   cells, train and held-out test MSE over epochs, playback), in Scala.js and Laminar, on `emeans_trace`
   and the core's constant-step (Lloyd) mode.
10. Done: spatial clustering. `emeans_agg` (aggregate and window function over POINT geometries or vectors)
    and `emeans_assign_point`, after PostGIS's `ST_ClusterKMeans`; geographic CRSs on the sphere.
11. Done: the move from the stable C API to DuckDB's C++ API, keeping the numerical core in C (below).
12. Submission to `duckdb/community-extensions` (in progress): a documentation site (`website/`, Docusaurus,
    as the mssql extension has; published at the root of hugr-lab.github.io/duck-emeans/ with the demos
    under /demo/), `description.yml` brought up to date, then the v0.1.0 release and the pull request to
    community-extensions pinning that release's commit.
13. Done (merged in #22): streaming e-means for data larger than memory, as separate functions (below).

## Moving to the C++ API

The stable C API was chosen so one binary would load in every DuckDB release. emeans targets DuckDB 2.0
only, so that advantage is small, and the C API keeps blocking things the C++ API has:

- Memory that spills. The C API's only allocator is `duckdb_malloc`, a heap wrapper: emeans's buffers
  (a float copy of the input, the aggregate's points) are invisible to `memory_limit` and cannot go to
  disk. The C++ API's `BufferManager` gives spillable, accounted blocks.
- A real `ST_ClusterKMeans`. A C++ window function (`ExtensionLoader::RegisterFunction(WindowFunction)`
  in v2.0.0), or an aggregate's window callback, which receives the current row (`idx_t rid`), can
  return each row's cluster id in one call, where the C API needs `emeans_assign_point` as a second.
- Argument types at bind (a GEOMETRY's CRS without the ANY parameter), return types at bind (`FLOAT[d]`
  centroids), interruptible queries (the work limit becomes optional), DuckDB's task scheduler (in place
  of the thread shim), and bind-time constant folding without the scalar-subquery failure.

The cost: the extension builds against DuckDB's C++ headers (the C++ extension template, DuckDB as a
submodule, compiled in CI), each DuckDB release needs a rebuild (the community pipeline does this), and
internal APIs can change between releases.

Stages, each with every SQL test passing:

1. Scaffold (done): the C++ template's build, DuckDB v2.0.0 as a submodule, and
   a C++ entry point that registers the existing C functions unchanged. Inside DuckDB a
   `duckdb_connection` is a `Connection *` (see `duckdb_connect` in `src/main/capi/v1/duckdb-c.cpp`), so
   the C++ `Load` hands the C registrations a `Connection` on the loader's database. The C sources
   include a shim `duckdb_extension.h`, so their calls bind to the C API inside DuckDB. The CRAP gate
   takes its SQL coverage from a DuckDB build with only the extension instrumented, so it scores the
   C++ as well as the C. The C API build stayed available (`make -f Makefile.capi`) until step 4.2.
2. Spilling memory (done): the input copy, the fit's workspace and the aggregate's points in
   `BufferManager` blocks (`src/include/emeans_mem.h`, `src/cpp/emeans_mem.cpp`). An aggregate state's
   block is pinned only inside a callback, so between callbacks it can go to `temp_directory`.
   `test/sql/emeans_memory.test` runs a `GROUP BY` whose states exceed `memory_limit` and checks that a
   block above the limit fails with DuckDB's error.
3. `emeans_cluster_id(geom, k) OVER (...)` (done): the per-row id in one call, compared again with
   PostGIS (docs/POSTGIS.md). It is a DuckDB `WindowFunction` (`src/cpp/emeans_window.cpp`), like
   `ntile`: it asks for the partition bounds, puts its arguments in the window's collection, and fits
   each partition once, on the first row evaluated, behind a lock that the threads of a hash group
   share. The fit and the labels are emeans_agg's C code (`emeans_agg.h`), so the id equals
   `emeans_assign_point(geom, (emeans_agg(geom, k) OVER w).centroids)`, which the tests check row by row.
   An aggregate with only window callbacks (`window_init` and the batched `window` callback, as
   quantile uses) was tried first and does not work for `OVER (PARTITION BY ...)`: DuckDB's window
   self-join optimizer turns a partition-only window aggregate into a `GROUP BY` and a join without
   checking that the function can aggregate, and the plan fails with "missing a combine method".
4. Native C++ for the rest, in steps, each its own pull request:
   1. Interrupts (done): `emeans_params.interrupted`, asked once per step and per k-means++ centroid;
      the SQL functions pass the query's `ClientContext::IsInterrupted` (`emeans_query.h`), taken at
      bind. `emeans()`, `emeans_spherical()`, `emeans_trace()` and `emeans_cluster_id()` stop within a
      step of Ctrl-C or `con.interrupt()`, so `emeans_max_work` now defaults to 0 (no cap). A C API
      aggregate has no query, so `emeans_agg` kept a fixed limit until step 3.
   2. Remove the C API build (done): `Makefile.capi`, `duckdb_capi/`, `src/functions/emeans_capi.c`
      and the `duckdb_extension.h` shim are gone; the C glue includes `duckdb.h` directly, and
      CMakeLists.txt builds only inside DuckDB's build.
   3. `emeans_agg` as a C++ aggregate with a bind (done, `src/cpp/emeans_agg.cpp`): the callbacks pass
      DuckDB's vectors to the C state code as the C API's types, as DuckDB's own C API adapter does. The
      bind checks the points' type (a wrong one fails at bind, a fixed-size array is cast to a list) and
      takes `emeans_max_work`, `emeans_threads` and the query, so `emeans_agg` is limited, threaded and
      interruptible like `emeans()`. The parameter stays ANY: a GEOMETRY parameter would drop the CRS in
      the implicit cast, and the bind is what now rejects the types ANY lets through.
   4. DuckDB's task scheduler (done): `emeans_params.parallel` runs a batch's parts; the SQL functions
      make them tasks of a `TaskExecutor` that the calling thread works on too, so a fit inside a
      pipeline task never waits on a busy worker. The core keeps its pthreads fork-join for callers
      without a scheduler (the C tests). 200,000 × 64, k = 256: 0.22 s on one part, 0.07 s on eight,
      the same result.
   5. `FLOAT[d][]` centroids (done): `emeans()` and `emeans_spherical()` are registered in C++
      (`src/cpp/emeans_cluster.cpp`) with x as `LIST(ANY)`, so the bind sees the vectors' own type; a
      `FLOAT[]` parameter would have the binder cast the arrays to lists first and lose d. The bind casts x
      to what the C code reads and, for arrays, returns `FLOAT[d][]` centroids, cast from the `FLOAT[][]`
      the C code writes. A wrong x fails at bind.

   Still on the C API inside the extension: `emeans_assign`, `emeans_assign_spherical`,
   `emeans_assign_point`, `emeans_recipe`, `emeans_trace` and the settings. They work as they are; they
   would move to C++ only for a reason of their own.

## Streaming e-means: data larger than memory (done, #22)

`emeans()` takes the whole data set as one value, `list(embedding)`, and the fit keeps a float32 copy
pinned until it ends, because it reads rows at random. The data must therefore fit in memory twice:
10M × 768 is 30 GB as FLOAT plus 30 GB pinned. Stage 2 made that copy count against `memory_limit`; it
cannot make it spill.

e-means reads the data in a few passes (5 by default), each over a random permutation of the rows, and
each step needs only its batch. DuckDB can produce a shuffled pass out of core: its sort spills to
`temp_directory`. So the streaming functions let DuckDB order each pass and feed the core batch by batch,
holding one batch, the centroids and the workspace: O(B·d + k·d) memory instead of O(n·d).

They are separate functions, so `emeans()` and its results stay as they are
(`src/cpp/emeans_stream.cpp`, `test/sql/emeans_stream.test`, `test/sql/emeans_stream_memory.test`):

```sql
-- One row per cluster: cluster INTEGER, centroid FLOAT[d], n BIGINT (rows in the closing pass), and
-- run STRUCT(n, batch_size, alpha0, n_splits, obj DOUBLE[]), the same in every row.
SELECT * FROM emeans_stream('SELECT embedding FROM docs', 256, nepochs := 5, seed := 1234);
SELECT * FROM emeans_stream_spherical('SELECT embedding FROM docs', 256);

-- Labels are already a streaming operation: emeans_assign row by row.
SELECT id, (emeans_assign(embedding, c.centroids)).label
FROM docs, (SELECT list(centroid ORDER BY cluster) AS centroids FROM emeans_stream('SELECT embedding FROM docs', 256)) c;
```

The argument is a query that returns one column of vectors (`FLOAT[d]`, `FLOAT[]`, `DOUBLE[]`, any
numeric list or array); NULL vectors are left out. The bind prepares the query for its type and, for
lists, reads d from the first vector. A run, all on a connection of the function's own, in one read
transaction:

1. n and the number of distinct vectors, from one `GROUP BY v` (which spills; `count(DISTINCT v)` does
   not), and from n the recipe, as `emeans()` resolves it.
2. The start: k distinct vectors drawn without replacement in proportion to how many rows hold each, by
   Efraimidis–Spirakis keys ln(u) / copies with u from `hash(v, seed, -1)`, a top-N of k rows. Without
   repeated vectors this is a uniform draw of k rows, like the reference's random start; with them, a
   repeated row is drawn once, so no two centroids start equal. With fewer than k distinct vectors the
   start repeats them. "Distinct" is in the query's own type: two `DOUBLE[]` vectors that round to the same
   floats are two groups and can both be drawn (grouping on `v::FLOAT[d]` would change the keys of every
   non-FLOAT input). It is deterministic on any number of threads, which
   `USING SAMPLE reservoir(k ROWS) REPEATABLE (seed)` (the plan's first idea) is not.
3. Pass p: the rows ordered by `hash(v, j, seed, p)`, where j numbers the copies of a repeated vector (the
   `GROUP BY v` again, each group expanded to its copies), cast to `FLOAT[d]` (a vector of another length
   fails the cast) and streamed into the core. Without repeated vectors every row is copy 0 and the query
   is a plain sort with the same keys, so the result does not depend on which form ran. A single repeated
   vector puts every pass on the grouping form (2.4–3.9× slower than `emeans()` against 1.7–2.6×; see
   Speed). Expanding only the groups with more than one copy and passing the rest through as copy 0 would
   keep the keys and cut that cost; not measured, not done. A fractional pass
   stops reading once it has its rows. The pass query has no `LIMIT`: a `LIMIT` makes the sort a top-N,
   which keeps its rows in memory.
4. No extra closing pass: as in `emeans_fit`, the last pass is the closing one (its sums give the exact
   means and its counts the column `n`), so with emeans_fit's start and permutations the stream is
   emeans_fit bit for bit, which `test/c/test_core.c` checks (Euclidean and spherical, partial batches,
   chunks crossing batch boundaries).

The core's stepper (`emeans_stream_begin`, `_add`, `_end_pass`, `_finish` in `emeans.h`) runs the same
`run_batch` as `emeans_fit`: the batch's rows are copied into a B × d buffer in the workspace, and
`perm` holds 0 … B − 1, so the kernel indexes the buffer as it indexes the matrix. Rows can come in chunks
of any size; the stepper steps every B rows and on `_end_pass`. The workspace drops `perm` and the
k-means++ distances (2n × 8 bytes) for the buffer (B × d × 4).

What the prototype settled:

- **Running the passes.** A `Connection` on the caller's `DatabaseInstance` from inside the table
  function works, with no deadlock on the task scheduler. It is drained through DuckDB 2.0's
  `QueryResultStream`: while the query is not ready the calling thread runs its tasks one by one and
  checks the caller's interrupt between them; once it waits on workers or on its consumer, `Fetch`
  refills the result. (Waiting in `WaitForTask` in that state spins forever with `threads = 1`.) The
  connection does `USE` of the caller's default catalog and schema. It cannot see the caller's
  temporary tables or uncommitted rows; the tests show this.
- **The query as SQL text.** The query is spliced into subqueries (the source, the passes, the start), so
  it must parse on its own as a single `SELECT`, checked at bind: such a string has balanced parentheses,
  quotes and comments and cannot close the subquery around it or add a statement, and the newline the
  splices add ends a line comment at its end. The nested connection runs every query, the bind's
  included, in one `READ ONLY` transaction, so nothing it calls can write (a `nextval` in the query
  fails; before the bind was covered, it advanced a sequence even inside the caller's read-only
  transaction, found in the security review). The `USE` target quotes the caller's catalog and schema as
  identifiers. Every security setting DuckDB has (`enable_external_access`, `allowed_directories`,
  `lock_configuration`, ...) is global, so the nested connection is bound by them as the caller is.
- **Interrupts.** While this thread waits in `Fetch` for workers to sort a pass, it checks nothing, and a
  spilled sort can take minutes (found in review). A watcher thread for the length of the fit asks every
  50 ms whether the caller was interrupted or passed `max_execution_time` (`BlockingInterruptCheck`;
  `IsInterrupted` misses the deadline, found in review too, and in `emeans()` as well) and interrupts the
  nested connection. `con.interrupt()` from Python stops a long run (2M × 64, k = 256, 50 epochs) within
  0.1 s on one and eight threads, and on four with the table on disk under a 300 MB `memory_limit`, where
  every pass spills; the connection works afterwards (`make test_interrupt`, run in CI). WASM has no
  threads and keeps the checks between tasks.
- **The permutation key.** A hash of the vector needs no row identity, which a view or a query does not
  have. But keyed by `hash(v, seed, p)` alone, the copies of a repeated vector sort together: on data with
  few distinct vectors the start becomes k copies of one vector and batches hold a few vectors many
  times (found in review). Numbering the copies fixes that and keeps runs reproducible, since it only
  tells identical rows apart. `row_number() OVER (PARTITION BY v)` does the same but the window does not
  spill (the memory test needed over 80 MB); the `GROUP BY` and `unnest(range(copies))` form does.
  On 10,000 rows over 50 distinct points, k = 16, the objective is within 10% of `emeans()`'s, inside
  the spread of `emeans()` over seeds (tested). The same seed gives the same result on 1 and 4 threads
  (tested).
- **Speed.** On an M5 Pro, 18 threads, vectors without repeats: 200,000 × 64, k = 256: 0.41 s against
  0.16 s for `emeans()` (2.6×); 1M × 64, k = 256: 1.32 s against 0.77 s (1.7×); 1M × 128, k = 1024:
  3.9 s against 2.0 s (1.9×). With repeated vectors every pass also groups them, which made the same runs
  2.4–3.9× slower than `emeans()` when all passes grouped.
- **Memory.** The stream holds O(B·d + k·d), and B is not small: the batch law gives 524,288 rows for
  10M × 768, k = 1024, 5 epochs (a 1.6 GB buffer, 5% of the data), and 50 epochs over 2M × 64 give the
  same B (134 MB at d = 64), which crowded the sorts out of a 300 MB limit. `batch_size := ...` sets B
  and so bounds the buffer. `test/sql/emeans_stream_memory.test`: 1M × 32 (128 MB of FLOAT) in a
  database file, under `memory_limit = '60MB'` with one thread: `emeans()` fails with out-of-memory,
  `emeans_stream` completes. The floor is between 30 and 40 MB, so the test has 20 MB of margin. In an
  in-memory database the table itself competes for the limit: there the same run needed 60 MB, and how far
  a run gets near the floor varied from run to run, so the tests keep the table on disk. DuckDB's sort
  needs memory per thread: with four threads a plain sort of the same data fails at 50 and 80 MB, before
  the extension allocates anything large. So the floor under `memory_limit` is DuckDB's, and fewer threads
  lower it.
- **Results.** For the same seed the results differ from `emeans()` (another start, other permutations).
  On a 16-group mixture (20,000 × 8, k = 16) the objective measured after the fit is within 1% of
  `emeans()`'s (tested); the spherical variant finds the same centroids as `emeans_spherical()` on the
  axis test.

Not done: k-means++ or k-means|| starts (one pass per centroid, or a few with k-means||, Bahmani et al.
2012), the batch-law and step overrides (`batch_size` is there), and comparing the stream with faiss-cpu
directly (it is compared with `emeans()`, which is).

Later, if wanted: an "IVF-Flat in SQL" example where `emeans_assign` gives the partition key (see
docs/QDRANT.md for how such an index compares).

## Risks

- Speed against FAISS: settled on macOS (docs/BENCHMARK.md). On Linux the portable kernel has not been
  measured against FAISS with OpenBLAS or MKL.
- Memory: the `list(embedding)` path holds the data once in DuckDB and once as float32 in the
  function; 10M × 128 is 10 GB + 5 GB. Since stage 2 the float32 copy is counted against
  `memory_limit`, but it is pinned for the whole fit, so it cannot spill. The window operator also
  keeps a partition's rows pinned, so `emeans_agg(...) OVER (PARTITION BY ...)` does not spill as the
  `GROUP BY` form does. `emeans_stream` (above) is the way past this for data larger than memory.
- Reference drift: blog, abstract and code disagree on the speed-up figures and on the scaling-law
  coefficients. Pin the FAISS commit the tests compare against.
