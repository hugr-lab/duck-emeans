// emeans_assign(vector, centroids) -> STRUCT(label INTEGER, sq_distance DOUBLE)
//
// The nearest of the centroids (a LIST of vectors, such as emeans(...).centroids) to one vector, as a
// 1-based label, and the squared Euclidean distance to it. Ties go to the lowest label. NULL in either
// argument gives NULL.

#include "duckdb_extension.h"

#include "assign.h"
#include "emeans_functions.h"
#include "recipe.h"
#include "vec_io.h"

#include <stdio.h>

DUCKDB_EXTENSION_EXTERN

#define N_FIELDS 2

enum { ROW_OK = 0, ROW_NULL = 1, ROW_ERROR = 2 };

typedef struct {
	emeans_buffer vector;
	emeans_buffer centroids;
} assign_buffers;

static int assign_row(duckdb_data_chunk input, idx_t row, assign_buffers *b, int32_t *label, double *dist,
                      char *err) {
	duckdb_vector vec = duckdb_data_chunk_get_vector(input, 0);
	duckdb_vector cents = duckdb_data_chunk_get_vector(input, 1);
	if (!emeans_row_is_valid(vec, row) || !emeans_row_is_valid(cents, row)) {
		return ROW_NULL;
	}
	int64_t d, k, cd;
	if (emeans_read_vector(vec, row, &b->vector, &d, err) ||
	    emeans_read_matrix(cents, row, "centroids", &b->centroids, &k, &cd, err)) {
		return ROW_ERROR;
	}
	if (cd != d) {
		snprintf(err, EMEANS_ERR_LEN, "the vector has %lld elements, the centroids %lld", (long long)d, (long long)cd);
		return ROW_ERROR;
	}
	const int64_t first = 0;
	emeans_assign(b->vector.data, d, &first, 1, b->centroids.data, k, label, dist);
	return ROW_OK;
}

static void AssignFunction(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	idx_t count = duckdb_data_chunk_get_size(input);
	int32_t *labels = (int32_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 0));
	double *dists = (double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 1));
	assign_buffers b = {{NULL, 0}, {NULL, 0}};
	char err[EMEANS_ERR_LEN];
	for (idx_t row = 0; row < count; row++) {
		int rc = assign_row(input, row, &b, &labels[row], &dists[row], err);
		if (rc == ROW_ERROR) {
			char msg[EMEANS_ERR_LEN + 24];
			snprintf(msg, sizeof(msg), "emeans_assign: %s", err);
			duckdb_scalar_function_set_error(info, msg);
			break;
		}
		if (rc == ROW_NULL) {
			emeans_set_struct_null(output, row, N_FIELDS);
		}
		labels[row] += rc == ROW_OK;
	}
	emeans_buffer_free(&b.vector);
	emeans_buffer_free(&b.centroids);
}

static duckdb_logical_type result_type(void) {
	duckdb_logical_type members[N_FIELDS] = {duckdb_create_logical_type(DUCKDB_TYPE_INTEGER),
	                                         duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE)};
	const char *names[N_FIELDS] = {"label", "sq_distance"};
	duckdb_logical_type type = duckdb_create_struct_type(members, names, N_FIELDS);
	for (int i = 0; i < N_FIELDS; i++) {
		duckdb_destroy_logical_type(&members[i]);
	}
	return type;
}

static bool add_overload(duckdb_scalar_function_set set, duckdb_type element, duckdb_logical_type ret) {
	duckdb_logical_type params[2] = {emeans_list_type(element), emeans_matrix_type(element)};
	bool ok = emeans_add_overload(set, "emeans_assign", params, 2, ret, AssignFunction, NULL);
	duckdb_destroy_logical_type(&params[0]);
	duckdb_destroy_logical_type(&params[1]);
	return ok;
}

bool emeans_register_assign(duckdb_connection connection) {
	duckdb_logical_type ret = result_type();
	duckdb_scalar_function_set set = duckdb_create_scalar_function_set("emeans_assign");
	bool ok = add_overload(set, DUCKDB_TYPE_FLOAT, ret) && add_overload(set, DUCKDB_TYPE_DOUBLE, ret) &&
	          duckdb_register_scalar_function_set(connection, set) == DuckDBSuccess;
	duckdb_destroy_scalar_function_set(&set);
	duckdb_destroy_logical_type(&ret);
	return ok;
}
