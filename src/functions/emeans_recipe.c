// emeans_recipe(n, k, d, nepochs [, batch_law]) -> STRUCT(batch_size, alpha0, batch_law, n_passes, total_steps)
//
// The recipe e-means would use for n rows of dimension d into k clusters over nepochs passes. NULL n, k
// or d gives NULL; NULL nepochs means 5 and NULL batch_law means 'auto'.

#include "duckdb_extension.h"

#include "emeans_functions.h"
#include "recipe.h"

#include <stdio.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

#define DEFAULT_NEPOCHS 5.0
#define N_FIELDS 5

typedef struct {
	int64_t *batch_size;
	double *alpha0;
	duckdb_vector law;
	int64_t *n_passes;
	int64_t *total_steps;
} recipe_columns;

static bool row_is_valid(duckdb_vector vec, idx_t row) {
	return duckdb_validity_row_is_valid(duckdb_vector_get_validity(vec), row);
}

static void set_null(duckdb_vector vec, idx_t row) {
	duckdb_vector_ensure_validity_writable(vec);
	duckdb_validity_set_row_invalid(duckdb_vector_get_validity(vec), row);
}

// A NULL struct row: the parent and, as DuckDB's own FlatVector::SetNull does, every child. Readers
// that materialise the result copy each child by the child's own validity, so a child left valid
// would hand them uninitialised bytes (a string_t, for batch_law).
static void set_struct_null(duckdb_vector output, idx_t row) {
	set_null(output, row);
	for (idx_t i = 0; i < N_FIELDS; i++) {
		set_null(duckdb_struct_vector_get_child(output, i), row);
	}
}

static recipe_columns output_columns(duckdb_vector output) {
	recipe_columns c;
	c.batch_size = (int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 0));
	c.alpha0 = (double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 1));
	c.law = duckdb_struct_vector_get_child(output, 2);
	c.n_passes = (int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 3));
	c.total_steps = (int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 4));
	return c;
}

static void write_recipe(const recipe_columns *c, idx_t row, const emeans_recipe *r) {
	c->batch_size[row] = r->batch_size;
	c->alpha0[row] = r->alpha0;
	duckdb_vector_assign_string_element(c->law, row, emeans_batch_law_name(r->law));
	c->n_passes[row] = r->n_passes;
	c->total_steps[row] = r->total_steps;
}

// The optional batch_law argument: 'auto' when absent or NULL. Returns 0, or 1 with a message in err.
static int read_law(duckdb_data_chunk input, idx_t row, emeans_batch_law *law, char *err) {
	*law = EMEANS_LAW_AUTO;
	if (duckdb_data_chunk_get_column_count(input) < 5) {
		return 0;
	}
	duckdb_vector vec = duckdb_data_chunk_get_vector(input, 4);
	if (!row_is_valid(vec, row)) {
		return 0;
	}
	duckdb_string_t s = ((duckdb_string_t *)duckdb_vector_get_data(vec))[row];
	const char *data = duckdb_string_t_data(&s);
	uint32_t len = duckdb_string_t_length(s);
	char name[16]; // longer than every law name, so anything that does not fit is unknown
	if (len < sizeof(name)) {
		memcpy(name, data, len);
		name[len] = '\0';
		if (emeans_parse_batch_law(name, law) == 0) {
			return 0;
		}
	}
	snprintf(err, EMEANS_ERR_LEN, "emeans_recipe: unknown batch_law '%.*s'; use 'auto', 'throughput' or 'quality'",
	         (int)len, data);
	return 1;
}

static double read_nepochs(duckdb_data_chunk input, idx_t row) {
	duckdb_vector vec = duckdb_data_chunk_get_vector(input, 3);
	if (!row_is_valid(vec, row)) {
		return DEFAULT_NEPOCHS;
	}
	return ((const double *)duckdb_vector_get_data(vec))[row];
}

// Returns 0 with the row's recipe, 1 for a NULL result, or 2 with an error in err.
static int resolve_row(duckdb_data_chunk input, idx_t row, emeans_recipe *r, char *err) {
	duckdb_vector n_vec = duckdb_data_chunk_get_vector(input, 0);
	duckdb_vector k_vec = duckdb_data_chunk_get_vector(input, 1);
	duckdb_vector d_vec = duckdb_data_chunk_get_vector(input, 2);
	if (!row_is_valid(n_vec, row) || !row_is_valid(k_vec, row) || !row_is_valid(d_vec, row)) {
		return 1;
	}
	emeans_batch_law law;
	if (read_law(input, row, &law, err)) {
		return 2;
	}
	int64_t n = ((const int64_t *)duckdb_vector_get_data(n_vec))[row];
	int64_t k = ((const int64_t *)duckdb_vector_get_data(k_vec))[row];
	int64_t d = ((const int64_t *)duckdb_vector_get_data(d_vec))[row];
	char core_err[EMEANS_ERR_LEN];
	if (emeans_resolve_recipe(n, k, d, read_nepochs(input, row), law, 0, 0, r, core_err)) {
		snprintf(err, EMEANS_ERR_LEN, "emeans_recipe: %s", core_err);
		return 2;
	}
	return 0;
}

static void RecipeFunction(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	idx_t count = duckdb_data_chunk_get_size(input);
	recipe_columns columns = output_columns(output);
	for (idx_t row = 0; row < count; row++) {
		emeans_recipe r;
		char err[EMEANS_ERR_LEN];
		int rc = resolve_row(input, row, &r, err);
		if (rc == 2) {
			duckdb_scalar_function_set_error(info, err);
			return;
		}
		if (rc == 1) {
			set_struct_null(output, row);
			continue;
		}
		write_recipe(&columns, row, &r);
	}
}

static duckdb_logical_type recipe_type(void) {
	duckdb_logical_type members[N_FIELDS] = {
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT), duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	    duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR), duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT)};
	const char *names[N_FIELDS] = {"batch_size", "alpha0", "batch_law", "n_passes", "total_steps"};
	duckdb_logical_type type = duckdb_create_struct_type(members, names, N_FIELDS);
	for (int i = 0; i < N_FIELDS; i++) {
		duckdb_destroy_logical_type(&members[i]);
	}
	return type;
}

bool emeans_register_recipe(duckdb_connection connection) {
	duckdb_logical_type params[5] = {
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT), duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT), duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	    duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR)};
	duckdb_logical_type ret = recipe_type();
	duckdb_scalar_function_set set = duckdb_create_scalar_function_set("emeans_recipe");
	bool ok = emeans_add_overload(set, "emeans_recipe", params, 4, ret, RecipeFunction, NULL) &&
	          emeans_add_overload(set, "emeans_recipe", params, 5, ret, RecipeFunction, NULL) &&
	          duckdb_register_scalar_function_set(connection, set) == DuckDBSuccess;
	duckdb_destroy_scalar_function_set(&set);
	duckdb_destroy_logical_type(&ret);
	for (int i = 0; i < 5; i++) {
		duckdb_destroy_logical_type(&params[i]);
	}
	return ok;
}
