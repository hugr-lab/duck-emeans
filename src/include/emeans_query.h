#pragma once

// The query a function runs in, so a long fit can stop when the query is interrupted (Ctrl-C in the CLI,
// con.interrupt() in Python): the ClientContext behind a C API client context (src/cpp/emeans_query.cpp).
// It lives as long as the connection, so a bind may keep it for the query.

#include "duckdb.h"

// The query behind a client context from a bind callback, or NULL.
void *emeans_query_context(duckdb_client_context context);

// Whether the query has been interrupted: emeans_params.interrupted, with the context as interrupt_ctx.
// NULL is never interrupted.
int emeans_query_interrupted(void *query);
