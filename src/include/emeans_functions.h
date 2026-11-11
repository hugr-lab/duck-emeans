#pragma once

// SQL surface: registration of every function set, and the helpers they share.

#include "duckdb_extension.h"

#include <stdbool.h>

// Adds one overload to a set. NULL arguments reach the callback (special handling), so optional
// parameters can treat NULL as "use the default".
bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn);

bool emeans_register_recipe(duckdb_connection connection);
