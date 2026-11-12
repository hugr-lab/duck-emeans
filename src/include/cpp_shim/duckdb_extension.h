#pragma once

// Stands in for duckdb_capi/duckdb_extension.h in the C++ build (EMEANS_CPP_BUILD). There the extension is
// part of the DuckDB build: its C files call the C API directly through DuckDB's own duckdb.h, instead of
// through the function-pointer table a C API extension receives at load time.

#include "duckdb.h"

#define DUCKDB_EXTENSION_EXTERN
