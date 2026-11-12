// emeans_trace(train, test, k, nepochs, seed, method, frames_per_epoch, batch_size, revival)
//   -> STRUCT(batch_size BIGINT, alpha0 DOUBLE,
//             frames STRUCT(epoch DOUBLE, step BIGINT, train_mse DOUBLE, test_mse DOUBLE, centroids FLOAT[][])[])
//
// One run of e-means or Lloyd's k-means, recorded: the centroids before the first step, every so many steps,
// and after the closing means, each with the mean squared distance of the training rows (a sample of at
// most 20,000, every n/20,000th row) and of the test rows to their nearest centroid. For playgrounds and
// teaching; emeans() is the function to cluster with.
//
// method 'emeans' (the default) uses the recipe, or batch_size if it is above 0; 'lloyd' uses full
// batches and a constant step of 1. frames_per_epoch (default 4) spaces e-means frames; Lloyd records
// every iteration. revival (default true) is e-means's dead-cluster revival. test may be NULL. NULL
// train or k gives NULL; other NULLs mean the defaults.

#include "duckdb.h"

#include "assign.h"
#include "emeans.h"
#include "emeans_functions.h"
#include "emeans_query.h"
#include "vec_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ARGS 9
#define N_FRAME_FIELDS 5
#define SCORE_ROWS 20000
#define DEFAULT_FRAMES_PER_EPOCH 4.0

enum { ROW_OK = 0, ROW_NULL = 1, ROW_ERROR = 2 };

//===--------------------------------------------------------------------===//
// Arguments
//===--------------------------------------------------------------------===//

static bool arg_present(duckdb_data_chunk input, idx_t col, idx_t row) {
	return emeans_row_is_valid(duckdb_data_chunk_get_vector(input, col), row);
}

static void *arg_data(duckdb_data_chunk input, idx_t col) {
	return duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, col));
}

static double arg_double(duckdb_data_chunk input, idx_t col, idx_t row, double def) {
	return arg_present(input, col, row) ? ((const double *)arg_data(input, col))[row] : def;
}

static int64_t arg_bigint(duckdb_data_chunk input, idx_t col, idx_t row, int64_t def) {
	return arg_present(input, col, row) ? ((const int64_t *)arg_data(input, col))[row] : def;
}

static bool arg_bool(duckdb_data_chunk input, idx_t col, idx_t row, bool def) {
	return arg_present(input, col, row) ? ((const bool *)arg_data(input, col))[row] : def;
}

// 1 for 'lloyd', 0 for 'emeans' or NULL; -1 for anything else.
static int arg_method(duckdb_data_chunk input, idx_t row) {
	if (!arg_present(input, 5, row)) {
		return 0;
	}
	duckdb_string_t s = ((duckdb_string_t *)arg_data(input, 5))[row];
	const char *data = duckdb_string_t_data(&s);
	uint32_t len = duckdb_string_t_length(s);
	int lloyd = len == 5 && memcmp(data, "lloyd", 5) == 0;
	int emeans = len == 6 && memcmp(data, "emeans", 6) == 0;
	return lloyd ? 1 : (emeans ? 0 : -1);
}

typedef struct {
	int64_t k;
	int64_t batch_size;
	double frames_per_epoch;
	int lloyd;
	emeans_params p;
} trace_args;

static int read_seed(duckdb_data_chunk input, idx_t row, uint32_t *seed, char *err) {
	int64_t value = arg_bigint(input, 4, row, 1234);
	if (value < 0 || value > (int64_t)UINT32_MAX) {
		snprintf(err, EMEANS_ERR_LEN, "seed must be in [0, 4294967295], got %lld", (long long)value);
		return 1;
	}
	*seed = (uint32_t)value;
	return 0;
}

static int read_args(duckdb_data_chunk input, idx_t row, const emeans_settings *settings, trace_args *a, char *err) {
	emeans_default_params(&a->p);
	a->p.max_work = settings->max_work;
	a->p.threads = settings->threads;
	a->p.interrupted = emeans_query_interrupted;
	a->p.interrupt_ctx = settings->query;
	a->k = ((const int64_t *)arg_data(input, 2))[row];
	a->p.nepochs = arg_double(input, 3, row, 5.0);
	a->lloyd = arg_method(input, row);
	a->frames_per_epoch = arg_double(input, 6, row, DEFAULT_FRAMES_PER_EPOCH);
	a->batch_size = arg_bigint(input, 7, row, 0);
	a->p.unused_threshold = arg_bool(input, 8, row, true) ? -1 : 0;
	if (a->lloyd < 0) {
		return emeans_error(err, "method must be 'emeans' or 'lloyd'");
	}
	if (!(a->frames_per_epoch > 0 && a->frames_per_epoch <= 1000)) {
		return emeans_error(err, "frames_per_epoch must be above 0 and at most 1000");
	}
	return read_seed(input, row, &a->p.seed, err);
}

// Lloyd: full batches, step 1 at every step, no revival. e-means: the recipe, or the given batch.
static void apply_method(trace_args *a, int64_t n) {
	a->p.batch_size = a->lloyd ? n : (a->batch_size > 0 ? a->batch_size : 0);
	a->p.alpha0 = a->lloyd ? 1.0 : 0.0;
	a->p.constant_step = a->lloyd;
	a->p.unused_threshold = a->lloyd ? 0.0 : a->p.unused_threshold;
}

//===--------------------------------------------------------------------===//
// Frames
//===--------------------------------------------------------------------===//

typedef struct {
	double epoch;
	int64_t step;
	double train_mse;
	double test_mse;
	float *centroids; // k x d
} trace_frame;

typedef struct {
	const float *train, *test;
	int64_t n, n_test, d, k;
	trace_frame *frames;
	int64_t count, cap;
	// The kernel for scoring, and its buffers.
	emeans_kernel kn;
	float *ct, *bias, *scratch;
	int64_t *rows;
	int32_t *labels;
	int failed;
} trace_log;

// The mean, over rows[0..count) of x, of the squared distance to the nearest centroid.
static double score_rows(trace_log *t, const float *x, const int64_t *rows, int64_t count, const float *c) {
	double sum = 0;
	for (int64_t r0 = 0; r0 < count; r0 += SCORE_ROWS) {
		int64_t nr = count - r0 < SCORE_ROWS ? count - r0 : SCORE_ROWS;
		emeans_kernel_nearest(&t->kn, x, rows + r0, nr, t->labels, t->scratch);
		for (int64_t i = 0; i < nr; i++) {
			sum += emeans_sq_dist(x + rows[r0 + i] * t->d, c + t->labels[i] * t->d, t->d);
		}
	}
	return count ? sum / (double)count : NAN;
}

// Every stride-th row of n, at most SCORE_ROWS of them; returns how many.
static int64_t sample_rows(int64_t *rows, int64_t n) {
	int64_t stride = n > SCORE_ROWS ? (n + SCORE_ROWS - 1) / SCORE_ROWS : 1;
	int64_t count = 0;
	for (int64_t i = 0; i < n; i += stride) {
		rows[count++] = i;
	}
	return count;
}

static void identity_rows(int64_t *rows, int64_t n) {
	for (int64_t i = 0; i < n; i++) {
		rows[i] = i;
	}
}

static int grow_frames(trace_log *t) {
	if (t->count < t->cap) {
		return 0;
	}
	int64_t cap = t->cap ? 2 * t->cap : 64;
	trace_frame *grown = realloc(t->frames, (size_t)cap * sizeof(trace_frame));
	if (!grown) {
		return 1;
	}
	t->frames = grown;
	t->cap = cap;
	return 0;
}

static void score_frame(trace_log *t, trace_frame *f) {
	for (int64_t j = 0; j < t->k; j++) {
		emeans_kernel_set(&t->kn, j, f->centroids + j * t->d);
	}
	int64_t sampled = sample_rows(t->rows, t->n);
	f->train_mse = score_rows(t, t->train, t->rows, sampled, f->centroids);
	identity_rows(t->rows, t->n_test);
	f->test_mse = t->test ? score_rows(t, t->test, t->rows, t->n_test, f->centroids) : NAN;
}

static void on_frame(void *ctx, const emeans_frame *frame) {
	trace_log *t = ctx;
	size_t kd = (size_t)(t->k * t->d);
	float *copy = t->failed || grow_frames(t) ? NULL : malloc(kd * sizeof(float));
	if (!copy) {
		t->failed = 1;
		return;
	}
	memcpy(copy, frame->centroids, kd * sizeof(float));
	trace_frame *f = &t->frames[t->count++];
	*f = (trace_frame) {frame->epoch, frame->step, 0, 0, copy};
	score_frame(t, f);
}

static int log_init(trace_log *t) {
	int64_t row_cap = t->n > t->n_test ? t->n : t->n_test;
	row_cap = row_cap < SCORE_ROWS ? SCORE_ROWS : row_cap;
	t->ct = malloc((size_t)(t->k * t->d) * sizeof(float));
	t->bias = malloc((size_t)t->k * sizeof(float));
	t->scratch = malloc((size_t)emeans_kernel_scratch(t->d) * sizeof(float));
	t->rows = malloc((size_t)row_cap * sizeof(int64_t));
	t->labels = malloc((size_t)SCORE_ROWS * sizeof(int32_t));
	emeans_kernel_init(&t->kn, t->ct, t->bias, t->k, t->d, 0);
	return !t->ct || !t->bias || !t->scratch || !t->rows || !t->labels;
}

static void log_free(trace_log *t) {
	for (int64_t i = 0; i < t->count; i++) {
		free(t->frames[i].centroids);
	}
	free(t->frames);
	free(t->ct);
	free(t->bias);
	free(t->scratch);
	free(t->rows);
	free(t->labels);
}

//===--------------------------------------------------------------------===//
// Output
//===--------------------------------------------------------------------===//

static void write_score(duckdb_vector vec, idx_t at, double value) {
	((double *)duckdb_vector_get_data(vec))[at] = value;
	if (isnan(value)) {
		emeans_set_row_null(vec, at);
	}
}

static int write_frames(duckdb_vector list, idx_t row, const trace_log *t, char *err) {
	idx_t start;
	if (emeans_list_append(list, row, (idx_t)t->count, &start, err)) {
		return 1;
	}
	duckdb_vector frames = duckdb_list_vector_get_child(list);
	for (int64_t i = 0; i < t->count; i++) {
		const trace_frame *f = &t->frames[i];
		idx_t at = start + (idx_t)i;
		((double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(frames, 0)))[at] = f->epoch;
		((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(frames, 1)))[at] = f->step;
		write_score(duckdb_struct_vector_get_child(frames, 2), at, f->train_mse);
		write_score(duckdb_struct_vector_get_child(frames, 3), at, f->test_mse);
		if (emeans_write_matrix(duckdb_struct_vector_get_child(frames, 4), at, f->centroids, t->k, t->d, err)) {
			return 1;
		}
	}
	return 0;
}

static int write_result(duckdb_vector output, idx_t row, const emeans_recipe *recipe, const trace_log *t, char *err) {
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 0)))[row] = recipe->batch_size;
	((double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 1)))[row] = recipe->alpha0;
	return write_frames(duckdb_struct_vector_get_child(output, 2), row, t, err);
}

//===--------------------------------------------------------------------===//
// One row
//===--------------------------------------------------------------------===//

typedef struct {
	emeans_buffer train;
	emeans_buffer test;
} trace_buffers;

static int read_data(duckdb_data_chunk input, idx_t row, trace_buffers *b, trace_log *t, char *err) {
	if (emeans_read_matrix(duckdb_data_chunk_get_vector(input, 0), row, "train", &b->train, &t->n, &t->d, err)) {
		return 1;
	}
	t->train = b->train.data;
	if (!arg_present(input, 1, row)) {
		return 0;
	}
	int64_t dt;
	if (emeans_read_matrix(duckdb_data_chunk_get_vector(input, 1), row, "test", &b->test, &t->n_test, &dt, err)) {
		return 1;
	}
	t->test = b->test.data;
	if (dt != t->d) {
		snprintf(err, EMEANS_ERR_LEN, "test vectors have %lld elements, train vectors %lld", (long long)dt,
		         (long long)t->d);
		return 1;
	}
	return 0;
}

// Frames every steps_per_pass / frames_per_epoch steps (at least every step); Lloyd's every step.
static int64_t frame_every(const emeans_recipe *recipe, int64_t n, double frames_per_epoch) {
	double steps_per_pass = ceil((double)n / (double)recipe->batch_size);
	int64_t every = (int64_t)llround(steps_per_pass / frames_per_epoch);
	return every < 1 ? 1 : every;
}

static int run_trace(trace_args *a, trace_log *t, emeans_recipe *recipe, char *err) {
	apply_method(a, t->n);
	t->k = a->k;
	if (emeans_resolve_recipe(t->n, a->k, t->d, a->p.nepochs, a->p.law, a->p.batch_size, a->p.alpha0, recipe, err)) {
		return 1;
	}
	if (log_init(t)) {
		return emeans_error(err, "out of memory");
	}
	a->p.on_frame = on_frame;
	a->p.frame_ctx = t;
	a->p.frame_every = frame_every(recipe, t->n, a->frames_per_epoch);
	float *centroids = malloc((size_t)(a->k * t->d) * sizeof(float));
	int32_t *labels = malloc((size_t)t->n * sizeof(int32_t));
	double *obj = malloc((size_t)recipe->n_passes * sizeof(double));
	emeans_output out = {centroids, labels, obj, 0, *recipe};
	int rc = !centroids || !labels || !obj ? emeans_error(err, "out of memory")
	                                       : emeans_fit(t->train, t->n, t->d, a->k, &a->p, &out, err) != EMEANS_OK;
	free(centroids);
	free(labels);
	free(obj);
	return rc || (t->failed && emeans_error(err, "out of memory"));
}

static int trace_row(duckdb_data_chunk input, idx_t row, const emeans_settings *settings, duckdb_vector output,
                     trace_buffers *b, char *err) {
	if (!arg_present(input, 0, row) || !arg_present(input, 2, row)) {
		return ROW_NULL;
	}
	trace_args a;
	trace_log t = {0};
	emeans_recipe recipe;
	int rc = read_args(input, row, settings, &a, err) || read_data(input, row, b, &t, err);
	a.p.allocator = emeans_mem_allocator(b->train.mem);
	rc = rc || run_trace(&a, &t, &recipe, err) || write_result(output, row, &recipe, &t, err);
	log_free(&t);
	return rc ? ROW_ERROR : ROW_OK;
}

static void TraceFunction(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	idx_t count = duckdb_data_chunk_get_size(input);
	const emeans_settings *settings = duckdb_scalar_function_get_bind_data(info);
	emeans_mem *mem = duckdb_scalar_function_get_extra_info(info);
	trace_buffers b = {emeans_managed_buffer(mem), emeans_managed_buffer(mem)};
	char err[EMEANS_ERR_LEN];
	for (idx_t row = 0; row < count; row++) {
		int rc = trace_row(input, row, settings, output, &b, err);
		if (rc == ROW_ERROR) {
			char msg[EMEANS_ERR_LEN + 24];
			snprintf(msg, sizeof(msg), "emeans_trace: %s", err);
			duckdb_scalar_function_set_error(info, msg);
			break;
		}
		if (rc == ROW_NULL) {
			emeans_set_struct_null(output, row, 3);
		}
	}
	emeans_buffer_free(&b.train);
	emeans_buffer_free(&b.test);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

static duckdb_logical_type struct_of(duckdb_logical_type *members, const char **names, idx_t n) {
	duckdb_logical_type type = duckdb_create_struct_type(members, names, n);
	for (idx_t i = 0; i < n; i++) {
		duckdb_destroy_logical_type(&members[i]);
	}
	return type;
}

static duckdb_logical_type result_type(void) {
	duckdb_logical_type frame_members[N_FRAME_FIELDS] = {
	    duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE), duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	    duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE), duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	    emeans_matrix_type(DUCKDB_TYPE_FLOAT)};
	const char *frame_names[N_FRAME_FIELDS] = {"epoch", "step", "train_mse", "test_mse", "centroids"};
	duckdb_logical_type frame = struct_of(frame_members, frame_names, N_FRAME_FIELDS);
	duckdb_logical_type members[3] = {duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	                                  duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE), duckdb_create_list_type(frame)};
	duckdb_destroy_logical_type(&frame);
	const char *names[3] = {"batch_size", "alpha0", "frames"};
	return struct_of(members, names, 3);
}

static bool add_overload(duckdb_scalar_function_set set, duckdb_type element, duckdb_logical_type ret,
                         emeans_mem *mem) {
	duckdb_logical_type params[N_ARGS] = {
	    emeans_matrix_type(element),                     emeans_matrix_type(element),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),  duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),  duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR),
	    duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),  duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	    duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN)};
	bool ok = emeans_add_overload(set, "emeans_trace", params, N_ARGS, ret, TraceFunction, emeans_bind_settings, mem);
	for (int i = 0; i < N_ARGS; i++) {
		duckdb_destroy_logical_type(&params[i]);
	}
	return ok;
}

bool emeans_register_trace(duckdb_connection connection, emeans_mem *mem) {
	duckdb_logical_type ret = result_type();
	duckdb_scalar_function_set set = duckdb_create_scalar_function_set("emeans_trace");
	bool ok = add_overload(set, DUCKDB_TYPE_FLOAT, ret, mem) && add_overload(set, DUCKDB_TYPE_DOUBLE, ret, mem) &&
	          duckdb_register_scalar_function_set(connection, set) == DuckDBSuccess;
	duckdb_destroy_scalar_function_set(&set);
	duckdb_destroy_logical_type(&ret);
	return ok;
}
