#include "duckdb.h"

#include "emeans_functions.h"

bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn,
                         duckdb_scalar_function_bind_t bind, void *extra_info) {
	duckdb_scalar_function f = duckdb_create_scalar_function();
	duckdb_scalar_function_set_name(f, name);
	for (idx_t i = 0; i < n_params; i++) {
		duckdb_scalar_function_add_parameter(f, params[i]);
	}
	duckdb_scalar_function_set_return_type(f, ret);
	duckdb_scalar_function_set_function(f, fn);
	if (bind) {
		duckdb_scalar_function_set_bind(f, bind);
	}
	duckdb_scalar_function_set_extra_info(f, extra_info, NULL);
	duckdb_scalar_function_set_special_handling(f);
	bool ok = duckdb_add_scalar_function_to_set(set, f) == DuckDBSuccess;
	duckdb_destroy_scalar_function(&f);
	return ok;
}

static bool (*const REGISTRATIONS[])(duckdb_connection, emeans_mem *) = {
    emeans_register_settings,
    emeans_register_recipe,
    emeans_register_assign,
    emeans_register_trace,
};

bool emeans_register_all(duckdb_connection connection, emeans_mem *mem) {
	for (size_t i = 0; i < sizeof(REGISTRATIONS) / sizeof(REGISTRATIONS[0]); i++) {
		if (!REGISTRATIONS[i](connection, mem)) {
			return false;
		}
	}
	return true;
}
