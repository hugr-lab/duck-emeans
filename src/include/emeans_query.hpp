#pragma once

// The C++ side of emeans_query.h.

#include <string>

namespace duckdb {

// Throws the query's own InterruptException: "Interrupted!" after Ctrl-C or con.interrupt(), DuckDB's
// "Query exceeded maximum execution time" after max_execution_time. For a fit that stopped because
// emeans_query_interrupted said so. A NULL query throws the plain one.
[[noreturn]] void ThrowInterrupted(void *query);

// The message ThrowInterrupted would throw, for a caller that keeps it to throw again later.
std::string InterruptMessage(void *query);

} // namespace duckdb
