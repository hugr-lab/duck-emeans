PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# The C++ extension, built inside DuckDB v2.0.0 (the duckdb submodule) as the extension template does.
# Makefile.capi is the stable C API build it replaces, kept until stage 4 of docs/PLAN.md.
EXT_NAME=emeans
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# DuckDB 2.0 takes its version from this, not from git; without it a build stamps the commit hash and
# only loads into a DuckDB built from the same tree. The distribution CI sets it the same way.
OVERRIDE_GIT_DESCRIBE ?= v2.0.0

# Compatible with DuckDB Community Extensions CI
# (provides: release, debug, reldebug, test_release, test_debug, wasm_*, format, clean, ...)
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: test_core test_threads crap compare_faiss bench compare_postgis compare_qdrant help

EXTENSION_BINARY := build/release/extension/$(EXT_NAME)/$(EXT_NAME).duckdb_extension

CORE_SOURCES := src/core/recipe.c src/core/rng.c src/core/assign.c src/core/parallel.c src/core/emeans.c src/core/geo.c

# On macOS the kernel's dot products go to Accelerate (src/core/assign.c), as in the extension build.
ifeq ($(shell uname -s),Darwin)
CORE_FLAGS := -DEMEANS_USE_ACCELERATE -framework Accelerate
endif

# Standalone C tests of src/core under AddressSanitizer + UndefinedBehaviorSanitizer (no DuckDB needed).
# EMEANS_FAULT_INJECTION lets the test fail every allocation in turn (src/include/emeans_alloc.h).
CORE_TEST_BIN := build/test_core
test_core:
	@mkdir -p build
	$(CC) -std=c11 -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer \
		-Wall -Wextra -pthread -DEMEANS_FAULT_INJECTION -Isrc/include test/c/test_core.c $(CORE_SOURCES) $(CORE_FLAGS) \
		-lm -o $(CORE_TEST_BIN)
	$(CORE_TEST_BIN)

# Concurrency test of src/core under ThreadSanitizer: 8 threads resolve recipes from one shared parameter
# table and compare with a single-threaded reference (no DuckDB needed).
THREADS_TEST_BIN := build/test_threads
test_threads:
	@mkdir -p build
	$(CC) -std=c11 -g -O1 -fsanitize=thread -fno-omit-frame-pointer -Wall -Wextra -pthread \
		-Isrc/include test/c/test_threads.c $(CORE_SOURCES) $(CORE_FLAGS) -lm -o $(THREADS_TEST_BIN)
	TSAN_OPTIONS=halt_on_error=1 $(THREADS_TEST_BIN)

# Validation against the reference e-means in faiss-cpu, and the benchmark (manual, not CI gating).
# A pinned Python 3.12 environment via uv: faiss-cpu ships faiss.contrib.e_means from 1.16.0.
REFERENCE_PYTHON := uv run --quiet --no-project --python 3.12 --with faiss-cpu==1.16.0 --with numpy==2.3.4 \
	--with pyarrow==22.0.0 --with duckdb==2.0.0 python -I
compare_faiss: release
	$(REFERENCE_PYTHON) test/python/compare_faiss.py $(EXTENSION_BINARY)

bench: release
	$(REFERENCE_PYTHON) bench/run_bench.py $(EXTENSION_BINARY)

# emeans_agg against PostGIS's ST_ClusterKMeans: behaviour, speed, quality and docs/postgis_comparison.png
# (see docs/POSTGIS.md). Needs a PostgreSQL with PostGIS reachable at POSTGIS_DSN.
POSTGIS_DSN ?= host=127.0.0.1 port=5432 user=postgres dbname=postgres
compare_postgis: release
	uv run --quiet --no-project --python 3.12 --with duckdb==2.0.0 --with numpy==2.3.4 \
		--with "psycopg[binary]==3.2.10" --with matplotlib==3.10.7 python -I bench/compare_postgis.py \
		$(EXTENSION_BINARY) "$(POSTGIS_DSN)" docs/postgis_comparison.png

# Nearest-neighbour search against DuckDB's HNSW (vss) and Qdrant; see docs/QDRANT.md. Needs a running
# Qdrant on 127.0.0.1:16333/16334 and the data set's Parquet files in QDRANT_DATA.
QDRANT_DATA ?= qdrant-data
compare_qdrant: release
	uv run --quiet --no-project --python 3.12 --with duckdb==2.0.0 --with numpy==2.3.4 --with pyarrow==22.0.0 \
		--with qdrant-client==1.20.0 python -I bench/compare_qdrant.py $(EXTENSION_BINARY) \
		"$(QDRANT_DATA)/*.parquet" build

# CRAP score of every C and C++ function in src/ (complexity from lizard, coverage from llvm-cov over
# test_core, test_threads and the SQL tests on a coverage build). Exit 2 when any function scores CRAP >= 10.
crap:
	python3 tools/crap/crap.py

help:
	@echo "make release | debug  - build DuckDB and build/{release,debug}/extension/emeans/emeans.duckdb_extension"
	@echo "make test_release     - SQL logic tests (DuckDB's unittest over test/sql) against the release build"
	@echo "make test_core        - C tests of src/core under ASan/UBSan (no DuckDB needed)"
	@echo "make test_threads     - 8 threads on a shared parameter table under ThreadSanitizer"
	@echo "make crap             - CRAP score per C/C++ function (lizard + llvm-cov); fails at CRAP >= 10"
	@echo "make compare_faiss    - recipe, objective and assignment against faiss-cpu's NumpyEmeans (manual)"
	@echo "make bench            - emeans() against NumpyEmeans and Lloyd, timings (manual)"
	@echo "make -f Makefile.capi - the stable C API build (until stage 4 of docs/PLAN.md)"
