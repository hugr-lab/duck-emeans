// emeans_query.h: inside DuckDB a duckdb_client_context is a CClientContextWrapper around the ClientContext
// (src/main/capi/v1/duckdb-c.cpp), which outlives the wrapper the bind destroys.

#include "duckdb.hpp"
#include "duckdb/main/capi/capi_internal.hpp"

extern "C" {
#include "emeans_query.h"
}

extern "C" void *emeans_query_context(duckdb_client_context context) {
	return context ? &reinterpret_cast<duckdb::CClientContextWrapper *>(context)->context : nullptr;
}

extern "C" int emeans_query_interrupted(void *query) {
	return query && static_cast<duckdb::ClientContext *>(query)->IsInterrupted();
}
