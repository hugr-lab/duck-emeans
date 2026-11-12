.PHONY: clean clean_all test_core test_threads crap compare_faiss bench compare_qdrant help

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Main extension configuration
EXTENSION_NAME=emeans

# Stable C API only: the binary carries the C_STRUCT ABI footer. Never set this to 1.
USE_UNSTABLE_C_API=0

# The extension API version the binary declares. DuckDB 2.0.0 ships extension API v1.5.6 (its
# duckdb_extension.h accepts no other major version and adds no functions over 1.5.6), so this stays
# at v1.5.6 while the extension is built, tested and released for DuckDB 2.0 only. duckdb_capi/ holds
# the headers from exactly this tag (refresh with `make update_duckdb_headers`).
TARGET_DUCKDB_VERSION=v1.5.6

all: configure release

# Compatible with DuckDB Community Extensions CI
# (provides: configure, platform, venv, test_extension_*, update_duckdb_headers, ...)
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/c_cpp.Makefile

configure: venv platform extension_version

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_debug
test_debug: test_extension_debug
test_release: test_extension_release

CORE_SOURCES := src/core/recipe.c src/core/rng.c src/core/assign.c src/core/parallel.c src/core/emeans.c

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
	$(REFERENCE_PYTHON) test/python/compare_faiss.py build/release/$(EXTENSION_NAME).duckdb_extension

bench: release
	$(REFERENCE_PYTHON) bench/run_bench.py build/release/$(EXTENSION_NAME).duckdb_extension

# Nearest-neighbour search against DuckDB's HNSW (vss) and Qdrant; see docs/QDRANT.md. Needs a running
# Qdrant on 127.0.0.1:16333/16334 and the data set's Parquet files in QDRANT_DATA.
QDRANT_DATA ?= qdrant-data
compare_qdrant: release
	uv run --quiet --no-project --python 3.12 --with duckdb==2.0.0 --with numpy==2.3.4 --with pyarrow==22.0.0 \
		--with qdrant-client==1.20.0 python -I bench/compare_qdrant.py build/release/$(EXTENSION_NAME).duckdb_extension \
		"$(QDRANT_DATA)/*.parquet" build

# CRAP score of every C function in src/ (complexity from lizard, coverage from llvm-cov over test_core,
# test_threads and the SQL tests). Exit 2 when any function scores CRAP >= 10.
crap:
	$(PYTHON_BIN) tools/crap/crap.py

clean: clean_build clean_cmake
clean_all: clean clean_configure

help:
	@echo "make configure        - create configure/venv (test runner) and detect platform"
	@echo "make release | debug  - build build/{release,debug}/emeans.duckdb_extension"
	@echo "make test             - SQL logic tests against the debug build"
	@echo "make test_release     - SQL logic tests against the release build"
	@echo "make test_core        - C tests of src/core under ASan/UBSan (no DuckDB needed)"
	@echo "make test_threads     - 8 threads on a shared parameter table under ThreadSanitizer"
	@echo "make crap             - CRAP score per C function (lizard + llvm-cov); fails at CRAP >= 10"
	@echo "make compare_faiss    - recipe, objective and assignment against faiss-cpu's NumpyEmeans (manual)"
	@echo "make bench            - emeans() against NumpyEmeans and Lloyd, timings (manual)"
