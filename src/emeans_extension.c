#include "duckdb_extension.h"

#include "emeans_functions.h"

DUCKDB_EXTENSION_EXTERN

bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn) {
	duckdb_scalar_function f = duckdb_create_scalar_function();
	duckdb_scalar_function_set_name(f, name);
	for (idx_t i = 0; i < n_params; i++) {
		duckdb_scalar_function_add_parameter(f, params[i]);
	}
	duckdb_scalar_function_set_return_type(f, ret);
	duckdb_scalar_function_set_function(f, fn);
	duckdb_scalar_function_set_special_handling(f);
	bool ok = duckdb_add_scalar_function_to_set(set, f) == DuckDBSuccess;
	duckdb_destroy_scalar_function(&f);
	return ok;
}

static bool (*const REGISTRATIONS[])(duckdb_connection) = {
    emeans_register_recipe,
};

DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection, duckdb_extension_info info,
                            struct duckdb_extension_access *access) {
	for (size_t i = 0; i < sizeof(REGISTRATIONS) / sizeof(REGISTRATIONS[0]); i++) {
		if (!REGISTRATIONS[i](connection)) {
			access->set_error(info, "emeans: failed to register functions");
			return false;
		}
	}
	return true;
}
