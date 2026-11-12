#pragma once

// The query a function runs in, so a long fit can stop when the query is interrupted (Ctrl-C in the CLI,
// con.interrupt() in Python). In the C++ build (src/cpp/emeans_query.cpp) it is the ClientContext behind a
// C API client context; it lives as long as the connection, so a bind may keep it for the query. In the C
// API build (src/functions/emeans_capi.c) there is none and nothing is ever interrupted.

#include "duckdb_extension.h"

// The query behind a client context from a bind callback, or NULL.
void *emeans_query_context(duckdb_client_context context);

// Whether the query has been interrupted: emeans_params.interrupted, with the context as interrupt_ctx.
// NULL is never interrupted.
int emeans_query_interrupted(void *query);
