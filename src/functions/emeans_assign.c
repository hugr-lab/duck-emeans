// emeans_assign(vector, centroids) -> STRUCT(label INTEGER, sq_distance DOUBLE)
// emeans_assign_spherical(vector, centroids) -> STRUCT(label INTEGER, similarity DOUBLE)
//
// The nearest of the centroids (a LIST of vectors, such as emeans(...).centroids) to one vector, as a
// 1-based label, and the squared Euclidean distance to it; or, spherical, the centroid direction with
// the largest inner product and that inner product (the vector should be unit length). NULL in either
// argument gives NULL.
//
// The centroids are read and prepared for the kernel once per distinct list in a chunk: a constant (a
// literal, a parameter, a scalar subquery, a one-row cross join) reaches every row of a chunk as the same
// list entry, so it is prepared once per chunk of up to 2048 rows. (Folding a constant at bind would
// save that too, but asking for a bind argument fails for scalar subqueries.)

#include "duckdb_extension.h"

#include "assign.h"
#include "emeans_functions.h"
#include "recipe.h"
#include "vec_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

#define N_FIELDS 2

enum { ROW_OK = 0, ROW_NULL = 1, ROW_ERROR = 2 };

// Centroids ready for the kernel: as given (or as directions, spherical), transposed, with the bias.
typedef struct {
	emeans_buffer centroids; // k x d
	emeans_buffer ct;        // d x k
	emeans_buffer bias;      // k
	int64_t k;
	int64_t d;
	int inner_product;
} prepared;

static void prepared_free(prepared *pc) {
	emeans_buffer_free(&pc->centroids);
	emeans_buffer_free(&pc->ct);
	emeans_buffer_free(&pc->bias);
}

// Fills ct and bias from pc->centroids (k x d), normalising the centroids first for inner product.
static int prepare_kernel(prepared *pc, emeans_kernel *kn, char *err) {
	if (emeans_buffer_reserve(&pc->ct, (size_t)(pc->k * pc->d), err) ||
	    emeans_buffer_reserve(&pc->bias, (size_t)pc->k, err)) {
		return 1;
	}
	*kn = (emeans_kernel) {pc->ct.data, pc->bias.data, pc->k, pc->d, pc->inner_product};
	for (int64_t j = 0; j < pc->k; j++) {
		float *c = pc->centroids.data + j * pc->d;
		if (pc->inner_product) {
			emeans_normalize(c, c, pc->d);
		}
		emeans_kernel_set(kn, j, c);
	}
	return 0;
}

//===--------------------------------------------------------------------===//
// Rows
//===--------------------------------------------------------------------===//

typedef struct {
	emeans_buffer vector;
	prepared pc;           // the centroids last prepared
	emeans_kernel kn;
	duckdb_list_entry key; // the list entry they came from
	int have;
	float scratch[EMEANS_KERNEL_SCRATCH];
} assign_work;

// The row's centroids, prepared for the kernel; reused while rows point at the same list entry.
static int row_centroids(duckdb_vector cents, idx_t row, assign_work *w, char *err) {
	duckdb_list_entry e = ((duckdb_list_entry *)duckdb_vector_get_data(cents))[row];
	if (w->have && e.offset == w->key.offset && e.length == w->key.length) {
		return 0;
	}
	w->have = 0;
	if (emeans_read_matrix(cents, row, "centroids", &w->pc.centroids, &w->pc.k, &w->pc.d, err) ||
	    prepare_kernel(&w->pc, &w->kn, err)) {
		return 1;
	}
	w->key = e;
	w->have = 1;
	return 0;
}

static int assign_row(duckdb_data_chunk input, idx_t row, assign_work *w, int32_t *label, double *score,
                      char *err) {
	duckdb_vector vec = duckdb_data_chunk_get_vector(input, 0);
	duckdb_vector cents = duckdb_data_chunk_get_vector(input, 1);
	if (!emeans_row_is_valid(vec, row) || !emeans_row_is_valid(cents, row)) {
		return ROW_NULL;
	}
	int64_t d;
	const prepared *pc = &w->pc;
	if (emeans_read_vector(vec, row, &w->vector, &d, err) || row_centroids(cents, row, w, err)) {
		return ROW_ERROR;
	}
	if (pc->d != d) {
		snprintf(err, EMEANS_ERR_LEN, "the vector has %lld elements, the centroids %lld", (long long)d, (long long)pc->d);
		return ROW_ERROR;
	}
	const int64_t first = 0;
	emeans_kernel_nearest(&w->kn, w->vector.data, &first, 1, label, w->scratch);
	const float *c = pc->centroids.data + *label * d;
	*score = pc->inner_product ? emeans_dot(w->vector.data, c, d) : emeans_sq_dist(w->vector.data, c, d);
	return ROW_OK;
}

static void assign_chunk(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output, const char *name,
                         int inner_product) {
	idx_t count = duckdb_data_chunk_get_size(input);
	int32_t *labels = (int32_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 0));
	double *scores = (double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 1));
	assign_work *w = calloc(1, sizeof(assign_work));
	char err[EMEANS_ERR_LEN] = "out of memory";
	if (w) {
		w->pc.inner_product = inner_product;
	}
	for (idx_t row = 0; row < count; row++) {
		int rc = w ? assign_row(input, row, w, &labels[row], &scores[row], err) : ROW_ERROR;
		if (rc == ROW_ERROR) {
			char msg[EMEANS_ERR_LEN + 32];
			snprintf(msg, sizeof(msg), "%s: %s", name, err);
			duckdb_scalar_function_set_error(info, msg);
			break;
		}
		if (rc == ROW_NULL) {
			emeans_set_struct_null(output, row, N_FIELDS);
		}
		labels[row] += rc == ROW_OK;
	}
	if (w) {
		emeans_buffer_free(&w->vector);
		prepared_free(&w->pc);
		free(w);
	}
}

static void AssignFunction(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	assign_chunk(info, input, output, "emeans_assign", 0);
}

static void AssignSphericalFunction(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	assign_chunk(info, input, output, "emeans_assign_spherical", 1);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

static duckdb_logical_type result_type(const char *score) {
	duckdb_logical_type members[N_FIELDS] = {duckdb_create_logical_type(DUCKDB_TYPE_INTEGER),
	                                         duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE)};
	const char *names[N_FIELDS] = {"label", score};
	duckdb_logical_type type = duckdb_create_struct_type(members, names, N_FIELDS);
	for (int i = 0; i < N_FIELDS; i++) {
		duckdb_destroy_logical_type(&members[i]);
	}
	return type;
}

typedef struct {
	const char *name;
	const char *score;
	duckdb_scalar_function_t fn;
} assign_variant;

static bool add_overload(duckdb_scalar_function_set set, const assign_variant *v, duckdb_type element,
                         duckdb_logical_type ret) {
	duckdb_logical_type params[2] = {emeans_list_type(element), emeans_matrix_type(element)};
	bool ok = emeans_add_overload(set, v->name, params, 2, ret, v->fn, NULL);
	duckdb_destroy_logical_type(&params[0]);
	duckdb_destroy_logical_type(&params[1]);
	return ok;
}

static bool register_variant(duckdb_connection connection, const assign_variant *v) {
	duckdb_logical_type ret = result_type(v->score);
	duckdb_scalar_function_set set = duckdb_create_scalar_function_set(v->name);
	bool ok = add_overload(set, v, DUCKDB_TYPE_FLOAT, ret) && add_overload(set, v, DUCKDB_TYPE_DOUBLE, ret) &&
	          duckdb_register_scalar_function_set(connection, set) == DuckDBSuccess;
	duckdb_destroy_scalar_function_set(&set);
	duckdb_destroy_logical_type(&ret);
	return ok;
}

bool emeans_register_assign(duckdb_connection connection) {
	static const assign_variant euclidean = {"emeans_assign", "sq_distance", AssignFunction};
	static const assign_variant spherical = {"emeans_assign_spherical", "similarity", AssignSphericalFunction};
	return register_variant(connection, &euclidean) && register_variant(connection, &spherical);
}
