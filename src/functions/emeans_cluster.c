// emeans(x, k [, nepochs [, seed]]) -> STRUCT(centroids FLOAT[][], labels INTEGER[], obj DOUBLE[],
//                                            batch_size BIGINT, alpha0 DOUBLE, n_splits BIGINT)
// (centroids FLOAT[d][] when x is a list of FLOAT[d] or other fixed-size arrays)
// emeans_spherical(x, k [, nepochs [, seed]]) -> the same, for unit-length vectors: assignment by the
// largest inner product with the centroid directions, unit centroids, obj the sum of 2 - 2 x.c.
//
// Clusters the vectors in x (LIST(LIST(FLOAT)) or LIST(LIST(DOUBLE)), one list per row, typically built
// with list(embedding ORDER BY id)) into k clusters. labels are 1-based, so centroids[labels[i]] is row
// i's centroid; a row the passes never reached (nepochs < 1) has a NULL label. NULL x or k gives NULL;
// NULL nepochs means 5, NULL seed 1234. A running call stops when the query is interrupted; SET
// emeans_max_work caps the work up front (0, the default, means no cap). A large batch is assigned on up
// to emeans_threads threads (0, the default, means DuckDB's threads setting).
//
// This file holds the per-row work (emeans_cluster_chunk) and the settings; src/cpp/emeans_cluster.cpp
// registers the two functions, with the bind that picks the centroids' type.

#include "duckdb.h"

#include "emeans.h"
#include "emeans_functions.h"
#include "emeans_query.h"
#include "vec_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORK_SETTING "emeans_max_work"
#define MAX_WORK_DEFAULT 0.0
#define THREADS_SETTING "emeans_threads"
#define N_FIELDS 6

enum { ROW_OK = 0, ROW_NULL = 1, ROW_ERROR = 2 };

//===--------------------------------------------------------------------===//
// emeans_max_work, emeans_threads
//===--------------------------------------------------------------------===//

static bool register_option(duckdb_connection connection, const char *name, duckdb_type type_id, duckdb_value def,
                            const char *description) {
	duckdb_config_option option = duckdb_create_config_option();
	duckdb_config_option_set_name(option, name);
	duckdb_logical_type type = duckdb_create_logical_type(type_id);
	duckdb_config_option_set_type(option, type);
	duckdb_destroy_logical_type(&type);
	duckdb_config_option_set_default_value(option, def);
	duckdb_destroy_value(&def);
	duckdb_config_option_set_description(option, description);
	bool ok = duckdb_register_config_option(connection, option) == DuckDBSuccess;
	duckdb_destroy_config_option(&option);
	return ok;
}

bool emeans_register_settings(duckdb_connection connection, emeans_mem *mem) {
	(void)mem;
	return register_option(connection, MAX_WORK_SETTING, DUCKDB_TYPE_DOUBLE, duckdb_create_double(MAX_WORK_DEFAULT),
	                       "Maximum work per emeans() call, in rows x k x d summed over all passes; 0 = no limit. "
	                       "A running call also stops when the query is interrupted.") &&
	       register_option(connection, THREADS_SETTING, DUCKDB_TYPE_BIGINT, duckdb_create_int64(0),
	                       "Threads one emeans() call may use to assign a large batch; 0 = the threads setting. "
	                       "The result does not depend on it.");
}

static void *copy_settings(void *data) {
	emeans_settings *copy = (emeans_settings *)malloc(sizeof(emeans_settings));
	if (copy) {
		*copy = *(emeans_settings *)data;
	}
	return copy;
}

// A setting's value as a double, or def if it is not set or there is no context.
static double setting_double(duckdb_client_context context, const char *name, double def) {
	duckdb_config_option_scope scope;
	duckdb_value value = context ? duckdb_client_context_get_config_option(context, name, &scope) : NULL;
	double result = value ? duckdb_get_double(value) : def;
	duckdb_destroy_value(&value);
	return result;
}

// emeans_threads, 0 meaning DuckDB's threads setting, clamped to [1, 64].
static int thread_count(duckdb_client_context context) {
	double threads = setting_double(context, THREADS_SETTING, 0);
	threads = threads > 0 ? threads : setting_double(context, "threads", 1);
	return threads < 1 ? 1 : (threads > 64 ? 64 : (int)threads);
}

void emeans_settings_read(duckdb_client_context context, emeans_settings *s) {
	s->max_work = setting_double(context, MAX_WORK_SETTING, MAX_WORK_DEFAULT);
	s->threads = thread_count(context);
	s->query = emeans_query_context(context);
}

// Bind: capture emeans_max_work, the thread count and the query.
void emeans_bind_settings(duckdb_bind_info info) {
	emeans_settings *data = (emeans_settings *)malloc(sizeof(emeans_settings));
	if (!data) {
		duckdb_scalar_function_bind_set_error(info, "emeans: out of memory");
		return;
	}
	duckdb_client_context context = NULL;
	duckdb_scalar_function_get_client_context(info, &context);
	emeans_settings_read(context, data);
	duckdb_destroy_client_context(&context);
	duckdb_scalar_function_set_bind_data(info, data, free);
	duckdb_scalar_function_set_bind_data_copy(info, copy_settings);
}

//===--------------------------------------------------------------------===//
// One row
//===--------------------------------------------------------------------===//

static bool arg_present(duckdb_data_chunk input, idx_t col, idx_t row) {
	return col < duckdb_data_chunk_get_column_count(input) &&
	       emeans_row_is_valid(duckdb_data_chunk_get_vector(input, col), row);
}

static void *arg_data(duckdb_data_chunk input, idx_t col) {
	return duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, col));
}

typedef struct {
	emeans_settings settings;
	int spherical;
	emeans_mem *mem; // the input matrix and the fit's workspace come from here
} cluster_mode;

static int read_params(duckdb_data_chunk input, idx_t row, const cluster_mode *mode, emeans_params *p, char *err) {
	emeans_default_params(p);
	p->max_work = mode->settings.max_work;
	p->threads = mode->settings.threads;
	p->spherical = mode->spherical;
	p->allocator = emeans_mem_allocator(mode->mem);
	emeans_query_attach(p, mode->settings.query);
	if (arg_present(input, 2, row)) {
		p->nepochs = ((const double *)arg_data(input, 2))[row];
	}
	if (!arg_present(input, 3, row)) {
		return 0;
	}
	int64_t seed = ((const int64_t *)arg_data(input, 3))[row];
	if (seed < 0 || seed > (int64_t)UINT32_MAX) {
		snprintf(err, EMEANS_ERR_LEN, "seed must be in [0, 4294967295], got %lld", (long long)seed);
		return 1;
	}
	p->seed = (uint32_t)seed;
	return 0;
}

typedef struct {
	float *centroids;
	int32_t *labels;
	double *obj;
} result_buffers;

static int alloc_result(result_buffers *r, int64_t n, int64_t d, int64_t k, int64_t passes, char *err) {
	r->centroids = malloc((size_t)(k * d) * sizeof(float));
	r->labels = malloc((size_t)n * sizeof(int32_t));
	r->obj = malloc((size_t)passes * sizeof(double));
	if (!r->centroids || !r->labels || !r->obj) {
		return emeans_error(err, "out of memory");
	}
	return 0;
}

static void free_result(result_buffers *r) {
	free(r->centroids);
	free(r->labels);
	free(r->obj);
}

static int write_result(duckdb_vector output, idx_t row, const emeans_output *o, int64_t n, int64_t d, int64_t k,
                        char *err) {
	if (emeans_write_matrix(duckdb_struct_vector_get_child(output, 0), row, o->centroids, k, d, err) ||
	    emeans_write_labels(duckdb_struct_vector_get_child(output, 1), row, o->labels, n, err) ||
	    emeans_write_doubles(duckdb_struct_vector_get_child(output, 2), row, o->obj, o->recipe.n_passes, err)) {
		return 1;
	}
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 3)))[row] = o->recipe.batch_size;
	((double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 4)))[row] = o->recipe.alpha0;
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 5)))[row] = o->n_splits;
	return 0;
}

static void add_budget_hint(char *err) {
	size_t len = strlen(err);
	snprintf(err + len, EMEANS_ERR_LEN - len, "; raise the limit with SET emeans_max_work (0 = no limit)");
}

// The recipe first, for the number of passes the objective list needs (and its argument checks).
static int fit_row(const float *x, int64_t n, int64_t d, int64_t k, const emeans_params *p, duckdb_vector output,
                   idx_t row, char *err) {
	emeans_recipe recipe;
	if (emeans_resolve_recipe(n, k, d, p->nepochs, p->law, 0, 0, &recipe, err)) {
		return 1;
	}
	result_buffers r = {NULL, NULL, NULL};
	int rc = alloc_result(&r, n, d, k, recipe.n_passes, err);
	emeans_output out = {r.centroids, r.labels, r.obj, 0, recipe};
	int fit_rc = rc ? EMEANS_OOM : emeans_fit(x, n, d, k, p, &out, err);
	if (fit_rc == EMEANS_BUDGET_EXCEEDED) {
		add_budget_hint(err);
	}
	rc = fit_rc != EMEANS_OK || write_result(output, row, &out, n, d, k, err);
	free_result(&r);
	return rc;
}

static int cluster_row(duckdb_data_chunk input, idx_t row, const cluster_mode *mode, duckdb_vector output,
                       emeans_buffer *x, char *err) {
	if (!arg_present(input, 0, row) || !arg_present(input, 1, row)) {
		return ROW_NULL;
	}
	int64_t n, d;
	emeans_params p;
	if (emeans_read_matrix(duckdb_data_chunk_get_vector(input, 0), row, "x", x, &n, &d, err) ||
	    read_params(input, row, mode, &p, err)) {
		return ROW_ERROR;
	}
	int64_t k = ((const int64_t *)arg_data(input, 1))[row];
	return fit_row(x->data, n, d, k, &p, output, row, err) ? ROW_ERROR : ROW_OK;
}

int emeans_cluster_chunk(duckdb_data_chunk input, duckdb_vector output, const emeans_settings *settings,
                         emeans_mem *mem, int spherical, char *err) {
	idx_t count = duckdb_data_chunk_get_size(input);
	cluster_mode mode = {*settings, spherical, mem};
	emeans_buffer x = emeans_managed_buffer(mem);
	int rc = ROW_OK;
	for (idx_t row = 0; rc != ROW_ERROR && row < count; row++) {
		rc = cluster_row(input, row, &mode, output, &x, err);
		if (rc == ROW_NULL) {
			emeans_set_struct_null(output, row, N_FIELDS);
		}
	}
	emeans_buffer_free(&x);
	return rc == ROW_ERROR;
}
