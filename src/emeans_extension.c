#include "duckdb_extension.h"

#include "emeans_functions.h"

DUCKDB_EXTENSION_EXTERN

bool emeans_add_overload(duckdb_scalar_function_set set, const char *name, duckdb_logical_type *params,
                         idx_t n_params, duckdb_logical_type ret, duckdb_scalar_function_t fn,
                         duckdb_scalar_function_bind_t bind) {
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
	duckdb_scalar_function_set_special_handling(f);
	bool ok = duckdb_add_scalar_function_to_set(set, f) == DuckDBSuccess;
	duckdb_destroy_scalar_function(&f);
	return ok;
}

static bool (*const REGISTRATIONS[])(duckdb_connection) = {
    emeans_register_settings,
    emeans_register_recipe,
    emeans_register_cluster,
    emeans_register_assign,
    emeans_register_trace,
    emeans_register_agg,
};

bool emeans_register_all(duckdb_connection connection) {
	for (size_t i = 0; i < sizeof(REGISTRATIONS) / sizeof(REGISTRATIONS[0]); i++) {
		if (!REGISTRATIONS[i](connection)) {
			return false;
		}
	}
	return true;
}

// The C API entry point. The C++ build has its own (src/cpp/emeans_extension.cpp), which calls
// emeans_register_all with a connection on the loader's database.
#ifndef EMEANS_CPP_BUILD
DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection, duckdb_extension_info info,
                            struct duckdb_extension_access *access) {
	if (!emeans_register_all(connection)) {
		access->set_error(info, "emeans: failed to register functions");
		return false;
	}
	return true;
}
#endif
