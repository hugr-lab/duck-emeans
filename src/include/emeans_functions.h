#pragma once

// SQL surface: registration of every function set, and the helpers they share.

#include "duckdb.h"
#include "emeans_mem.h"

#include <stdbool.h>

// Adds one overload to a set, with an optional bind callback and extra info (the emeans_mem, for functions
// with large buffers). NULL arguments reach the callback (special handling), so optional parameters can
// treat NULL as "use the default".
bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn,
                         duckdb_scalar_function_bind_t bind, void *extra_info);

// Every function and setting of the extension; false if one failed to register. mem is the database's
// buffer manager (emeans_mem.h).
bool emeans_register_all(duckdb_connection connection, emeans_mem *mem);

bool emeans_register_settings(duckdb_connection connection, emeans_mem *mem);

// emeans_max_work and the thread count (emeans_threads, 0 meaning DuckDB's threads), captured at bind as
// the function's bind data by emeans_bind_settings.
typedef struct {
	double max_work;
	int threads;
	void *query; // emeans_query.h, for interrupts
} emeans_settings;
void emeans_bind_settings(duckdb_bind_info info);
// The same from a client context, for binds on the C++ side.
void emeans_settings_read(duckdb_client_context context, emeans_settings *s);
bool emeans_register_recipe(duckdb_connection connection, emeans_mem *mem);
bool emeans_register_cluster(duckdb_connection connection, emeans_mem *mem);
bool emeans_register_assign(duckdb_connection connection, emeans_mem *mem);
bool emeans_register_trace(duckdb_connection connection, emeans_mem *mem);

// emeans_assign_point(GEOMETRY, DOUBLE[][]): the scalar callback, registered with emeans_assign.
void emeans_assign_geometry(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output);
