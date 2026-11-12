#pragma once

// SQL surface: registration of every function set, and the helpers they share.

#include "duckdb_extension.h"

#include <stdbool.h>

// Adds one overload to a set, with an optional bind callback. NULL arguments reach the callback (special
// handling), so optional parameters can treat NULL as "use the default".
bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn,
                         duckdb_scalar_function_bind_t bind);

bool emeans_register_settings(duckdb_connection connection);

// emeans_max_work and the thread count (emeans_threads, 0 meaning DuckDB's threads), captured at bind as
// the function's bind data by emeans_bind_settings.
typedef struct {
	double max_work;
	int threads;
} emeans_settings;
void emeans_bind_settings(duckdb_bind_info info);
bool emeans_register_recipe(duckdb_connection connection);
bool emeans_register_cluster(duckdb_connection connection);
bool emeans_register_assign(duckdb_connection connection);
bool emeans_register_trace(duckdb_connection connection);
