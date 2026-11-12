// emeans_agg(points, k [, nepochs [, seed]]) -> STRUCT(centroids DOUBLE[][], obj DOUBLE[], batch_size BIGINT,
//                                                   alpha0 DOUBLE, n_splits BIGINT, n BIGINT, geographic BOOLEAN)
//
// An aggregate: the e-means clustering of a group's points, for GROUP BY and as a window function over a
// partition (... OVER (PARTITION BY region)). points is a POINT GEOMETRY or a vector (FLOAT[] or DOUBLE[]).
// With a geographic CRS (OGC:CRS84, EPSG:4326 and others, see geo.h; x is longitude) the points are
// clustered on the sphere and the centroids are [longitude, latitude]; otherwise on (x, y), or on the
// vector. NULL points are skipped, and a group of none gives NULL. The points are sorted before
// clustering, so the result does not depend on the order rows arrive in. Label rows with
// emeans_assign_point(point, centroids) for geometries, emeans_assign(vector, centroids) for vectors.
//
// An aggregate gets no bind step in the C API, so emeans_agg cannot read settings: it runs on one thread
// and stops above the default work limit (1e11 units of rows x k x d), as emeans() does by default.

#include "duckdb_extension.h"

#include "emeans.h"
#include "emeans_functions.h"
#include "geo.h"
#include "vec_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

#define N_FIELDS 7
#define MAX_WORK 1e11

typedef struct {
	double *rows; // n x dim, as given: (x, y), (longitude, latitude) or the vector
	int64_t n, cap, dim;
	int64_t k;
	double nepochs;
	uint32_t seed;
	int geographic;
	int geometry; // the points were POINT geometries
	int has_args;
} agg_state;

static idx_t StateSize(duckdb_function_info info) {
	(void)info;
	return sizeof(agg_state);
}

static void StateInit(duckdb_function_info info, duckdb_aggregate_state state) {
	(void)info;
	memset(state, 0, sizeof(agg_state));
}

static void StateDestroy(duckdb_aggregate_state *states, idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		free(((agg_state *)states[i])->rows);
	}
}

//===--------------------------------------------------------------------===//
// Building a state
//===--------------------------------------------------------------------===//

// The arguments of a row or of another state: the first fixes them, and they may not change.
static int set_args(agg_state *s, int64_t k, double nepochs, uint32_t seed, int geographic, char *err) {
	if (!s->has_args) {
		s->k = k;
		s->nepochs = nepochs;
		s->seed = seed;
		s->geographic = geographic;
		s->has_args = 1;
		return 0;
	}
	if (s->k != k || s->nepochs != nepochs || s->seed != seed || s->geographic != geographic) {
		return emeans_error(err, "k, nepochs and seed must be the same for every row of a group");
	}
	return 0;
}

static int append_rows(agg_state *s, const double *rows, int64_t count, int64_t dim, char *err) {
	if (s->dim != 0 && s->dim != dim) {
		snprintf(err, EMEANS_ERR_LEN, "points have %lld dimensions, earlier ones %lld", (long long)dim,
		         (long long)s->dim);
		return 1;
	}
	s->dim = dim;
	int64_t need = s->n + count;
	if (need > s->cap) {
		int64_t cap = need > 2 * s->cap ? need : 2 * s->cap;
		double *grown = realloc(s->rows, (size_t)(cap * dim) * sizeof(double));
		if (!grown) {
			return emeans_error(err, "out of memory");
		}
		s->rows = grown;
		s->cap = cap;
	}
	memcpy(s->rows + s->n * dim, rows, (size_t)(count * dim) * sizeof(double));
	s->n = need;
	return 0;
}

// The GEOMETRY argument's CRS: geographic or not.
static int column_is_geographic(duckdb_vector vec) {
	duckdb_logical_type type = duckdb_vector_get_column_type(vec);
	char *crs = duckdb_get_type_id(type) == DUCKDB_TYPE_GEOMETRY ? duckdb_geometry_type_get_crs(type) : NULL;
	int geographic = emeans_crs_is_geographic(crs);
	duckdb_free(crs);
	duckdb_destroy_logical_type(&type);
	return geographic;
}

static int is_geometry(duckdb_vector vec) {
	duckdb_logical_type type = duckdb_vector_get_column_type(vec);
	int geometry = duckdb_get_type_id(type) == DUCKDB_TYPE_GEOMETRY;
	duckdb_destroy_logical_type(&type);
	return geometry;
}

// What reaches the ANY overload must be a GEOMETRY or a list of numbers (an INTEGER[] literal, say).
static int check_points_type(duckdb_vector vec, char *err) {
	if (is_geometry(vec) || emeans_is_numeric_list(vec)) {
		return 0;
	}
	return emeans_error(err, "points must be POINT geometries or lists of numbers");
}

// One POINT geometry's coordinates into xy. Returns 0, 1 to skip an empty point, or 2 with an error.
static int read_geometry(duckdb_vector vec, idx_t row, int geographic, double *xy, char *err) {
	duckdb_string_t s = ((duckdb_string_t *)duckdb_vector_get_data(vec))[row];
	int kind = emeans_wkb_point((const uint8_t *)duckdb_string_t_data(&s), duckdb_string_t_length(s), &xy[0], &xy[1]);
	if (kind != EMEANS_WKB_POINT) {
		return kind == EMEANS_WKB_EMPTY ? 1
		                                : 2 * emeans_error(err, "takes POINT geometries; use ST_Centroid for others");
	}
	if (geographic && (xy[1] < -90 || xy[1] > 90)) {
		snprintf(err, EMEANS_ERR_LEN, "latitude %g is outside [-90, 90]; geographic points are (longitude, latitude)",
		         xy[1]);
		return 2;
	}
	return 0;
}

static bool arg_valid(duckdb_data_chunk input, idx_t col, idx_t row) {
	return col < duckdb_data_chunk_get_column_count(input) &&
	       emeans_row_is_valid(duckdb_data_chunk_get_vector(input, col), row);
}

static void *arg_data(duckdb_data_chunk input, idx_t col) {
	return duckdb_vector_get_data(duckdb_data_chunk_get_vector(input, col));
}

static int read_args(duckdb_data_chunk input, idx_t row, int64_t *k, double *nepochs, uint32_t *seed, char *err) {
	*k = ((const int64_t *)arg_data(input, 1))[row];
	*nepochs = arg_valid(input, 2, row) ? ((const double *)arg_data(input, 2))[row] : 5.0;
	int64_t s = arg_valid(input, 3, row) ? ((const int64_t *)arg_data(input, 3))[row] : 1234;
	if (s < 0 || s > (int64_t)UINT32_MAX) {
		snprintf(err, EMEANS_ERR_LEN, "seed must be in [0, 4294967295], got %lld", (long long)s);
		return 1;
	}
	*seed = (uint32_t)s;
	return 0;
}

typedef struct {
	duckdb_vector points;
	int geometry;
	int geographic;
	emeans_buffer buf; // a vector point, as read
	double *row;       // the point as stored
	int64_t row_cap;
} update_ctx;

static int reserve_row(update_ctx *ctx, int64_t dim, char *err) {
	if (dim <= ctx->row_cap) {
		return 0;
	}
	double *grown = realloc(ctx->row, (size_t)dim * sizeof(double));
	if (!grown) {
		return emeans_error(err, "out of memory");
	}
	ctx->row = grown;
	ctx->row_cap = dim;
	return 0;
}

// The row's point into ctx->row. Returns 0, 1 to skip it, or 2 with an error.
static int read_point(update_ctx *ctx, idx_t row, int64_t *dim, char *err) {
	if (ctx->geometry) {
		*dim = 2;
		return reserve_row(ctx, 2, err) ? 2 : read_geometry(ctx->points, row, ctx->geographic, ctx->row, err);
	}
	if (emeans_read_vector(ctx->points, row, &ctx->buf, dim, err) || reserve_row(ctx, *dim, err)) {
		return 2;
	}
	for (int64_t t = 0; t < *dim; t++) {
		ctx->row[t] = ctx->buf.data[t];
	}
	return 0;
}

static int update_row(duckdb_data_chunk input, idx_t row, update_ctx *ctx, agg_state *s, char *err) {
	if (!arg_valid(input, 0, row) || !arg_valid(input, 1, row)) {
		return 0;
	}
	int64_t k, dim = 0;
	double nepochs;
	uint32_t seed;
	int rc = read_point(ctx, row, &dim, err);
	if (rc != 0) {
		return rc == 2;
	}
	s->geometry = ctx->geometry;
	return read_args(input, row, &k, &nepochs, &seed, err) || set_args(s, k, nepochs, seed, ctx->geographic, err) ||
	       append_rows(s, ctx->row, 1, dim, err);
}

static void report(duckdb_function_info info, const char *err) {
	char msg[EMEANS_ERR_LEN + 16];
	snprintf(msg, sizeof(msg), "emeans_agg: %s", err);
	duckdb_aggregate_function_set_error(info, msg);
}

static void Update(duckdb_function_info info, duckdb_data_chunk input, duckdb_aggregate_state *states) {
	idx_t count = duckdb_data_chunk_get_size(input);
	duckdb_vector points = duckdb_data_chunk_get_vector(input, 0);
	update_ctx ctx = {points, is_geometry(points), column_is_geographic(points), {NULL, 0}, NULL, 0};
	char err[EMEANS_ERR_LEN];
	if (check_points_type(points, err)) {
		report(info, err);
		return;
	}
	for (idx_t row = 0; row < count; row++) {
		if (update_row(input, row, &ctx, (agg_state *)states[row], err)) {
			report(info, err);
			break;
		}
	}
	emeans_buffer_free(&ctx.buf);
	free(ctx.row);
}

static int combine_one(const agg_state *src, agg_state *dst, char *err) {
	if (src->n == 0) {
		return 0;
	}
	dst->geometry = src->geometry;
	return set_args(dst, src->k, src->nepochs, src->seed, src->geographic, err) ||
	       append_rows(dst, src->rows, src->n, src->dim, err);
}

static void Combine(duckdb_function_info info, duckdb_aggregate_state *source, duckdb_aggregate_state *target,
                    idx_t count) {
	char err[EMEANS_ERR_LEN];
	for (idx_t i = 0; i < count; i++) {
		if (combine_one((const agg_state *)source[i], (agg_state *)target[i], err)) {
			report(info, err);
			return;
		}
	}
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

// How the stored points become the float32 matrix the core clusters: unit vectors for a geographic CRS;
// otherwise offsets from the smallest coordinates, so that projected coordinates keep their precision (a
// northing of 6,500,000 m is only good to 0.5 m as a float) and the result does not depend on row order.
typedef struct {
	int64_t dim;      // of the clustered matrix
	double origin[2]; // planar points: subtracted before, added back after
	int planar;
} layout;

static layout make_layout(const agg_state *s, int geometry) {
	layout l = {s->geographic ? 3 : s->dim, {0, 0}, geometry && !s->geographic};
	for (int64_t i = 0; l.planar && i < s->n; i++) {
		l.origin[0] = i == 0 || s->rows[2 * i] < l.origin[0] ? s->rows[2 * i] : l.origin[0];
		l.origin[1] = i == 0 || s->rows[2 * i + 1] < l.origin[1] ? s->rows[2 * i + 1] : l.origin[1];
	}
	return l;
}

static void to_matrix(const agg_state *s, const layout *l, float *x) {
	for (int64_t i = 0; i < s->n; i++) {
		const double *r = s->rows + i * s->dim;
		if (s->geographic) {
			emeans_lonlat_to_unit(r[0], r[1], x + i * 3);
			continue;
		}
		for (int64_t t = 0; t < s->dim; t++) {
			x[i * s->dim + t] = (float)(r[t] - (l->planar && t < 2 ? l->origin[t] : 0.0));
		}
	}
}

// The centroids for output: [longitude, latitude] for geographic runs, (x, y) or the vector otherwise.
static double *output_centroids(const agg_state *s, const layout *l, const float *c, int64_t *cols) {
	*cols = s->geographic ? 2 : s->dim;
	double *out = malloc((size_t)(s->k * *cols) * sizeof(double));
	for (int64_t j = 0; out && j < s->k; j++) {
		if (s->geographic) {
			emeans_unit_to_lonlat(c + j * 3, &out[j * 2], &out[j * 2 + 1]);
			continue;
		}
		for (int64_t t = 0; t < s->dim; t++) {
			out[j * s->dim + t] = (double)c[j * s->dim + t] + (l->planar && t < 2 ? l->origin[t] : 0.0);
		}
	}
	return out;
}

static int write_result(duckdb_vector result, idx_t row, const agg_state *s, const layout *l, const emeans_output *o,
                        char *err) {
	int64_t cols;
	double *centroids = output_centroids(s, l, o->centroids, &cols);
	int rc = !centroids ? emeans_error(err, "out of memory")
	                    : emeans_write_matrix_f64(duckdb_struct_vector_get_child(result, 0), row, centroids, s->k, cols,
	                                              err) ||
	                          emeans_write_doubles(duckdb_struct_vector_get_child(result, 1), row, o->obj,
	                                               o->recipe.n_passes, err);
	free(centroids);
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(result, 2)))[row] = o->recipe.batch_size;
	((double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(result, 3)))[row] = o->recipe.alpha0;
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(result, 4)))[row] = o->n_splits;
	((int64_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(result, 5)))[row] = s->n;
	((bool *)duckdb_vector_get_data(duckdb_struct_vector_get_child(result, 6)))[row] = s->geographic != 0;
	return rc;
}

typedef struct {
	float *x, *tmp, *c;
	int32_t *labels;
	double *obj;
} fit_buffers;

static int alloc_fit(fit_buffers *b, const agg_state *s, int64_t dim, int64_t passes) {
	b->x = malloc((size_t)(s->n * dim) * sizeof(float));
	b->tmp = malloc((size_t)(s->n * dim) * sizeof(float));
	b->c = malloc((size_t)(s->k * dim) * sizeof(float));
	b->labels = malloc((size_t)s->n * sizeof(int32_t));
	b->obj = malloc((size_t)passes * sizeof(double));
	return !b->x || !b->tmp || !b->c || !b->labels || !b->obj;
}

static void free_fit(fit_buffers *b) {
	free(b->x);
	free(b->tmp);
	free(b->c);
	free(b->labels);
	free(b->obj);
}

// The points in canonical form and sorted, so the result does not depend on the order rows arrived in.
static int fit_state(const agg_state *s, int geometry, duckdb_vector result, idx_t row, char *err) {
	layout l = make_layout(s, geometry);
	emeans_params p;
	emeans_default_params(&p);
	p.nepochs = s->nepochs;
	p.seed = s->seed;
	p.spherical = s->geographic;
	p.max_work = MAX_WORK;
	// Unlike emeans() and the reference, k-means++ seeding: on 2-D maps, where the reference turns revival
	// off, a random start left clusters up to 25% worse than PostGIS (docs/POSTGIS.md).
	p.kmeans_plus_plus = 1;
	emeans_recipe recipe;
	if (emeans_resolve_recipe(s->n, s->k, l.dim, p.nepochs, p.law, 0, 0, &recipe, err)) {
		return 1;
	}
	fit_buffers b = {NULL, NULL, NULL, NULL, NULL};
	emeans_output out = {NULL, NULL, NULL, 0, recipe};
	int rc = alloc_fit(&b, s, l.dim, recipe.n_passes) ? emeans_error(err, "out of memory") : 0;
	if (rc == 0) {
		to_matrix(s, &l, b.x);
		emeans_sort_rows(b.x, b.tmp, s->n, l.dim);
		out = (emeans_output) {b.c, b.labels, b.obj, 0, recipe};
		rc = emeans_fit(b.x, s->n, l.dim, s->k, &p, &out, err) != EMEANS_OK ||
		     write_result(result, row, s, &l, &out, err);
	}
	free_fit(&b);
	return rc;
}

static void Finalize(duckdb_function_info info, duckdb_aggregate_state *source, duckdb_vector result, idx_t count,
                     idx_t offset) {
	char err[EMEANS_ERR_LEN];
	for (idx_t i = 0; i < count; i++) {
		agg_state *s = (agg_state *)source[i];
		if (s->n == 0) {
			emeans_set_struct_null(result, offset + i, N_FIELDS);
			continue;
		}
		if (fit_state(s, s->geometry, result, offset + i, err)) {
			report(info, err);
			return;
		}
	}
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

static duckdb_logical_type result_type(void) {
	duckdb_logical_type members[N_FIELDS] = {
	    emeans_matrix_type(DUCKDB_TYPE_DOUBLE),         emeans_list_type(DUCKDB_TYPE_DOUBLE),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT), duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	    duckdb_create_logical_type(DUCKDB_TYPE_BIGINT), duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	    duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN)};
	const char *names[N_FIELDS] = {"centroids", "obj", "batch_size", "alpha0", "n_splits", "n", "geographic"};
	duckdb_logical_type type = duckdb_create_struct_type(members, names, N_FIELDS);
	for (int i = 0; i < N_FIELDS; i++) {
		duckdb_destroy_logical_type(&members[i]);
	}
	return type;
}

static bool add_overload(duckdb_aggregate_function_set set, duckdb_logical_type points, idx_t n_params,
                         duckdb_logical_type ret) {
	duckdb_logical_type extra[3] = {duckdb_create_logical_type(DUCKDB_TYPE_BIGINT),
	                                duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE),
	                                duckdb_create_logical_type(DUCKDB_TYPE_BIGINT)};
	duckdb_aggregate_function f = duckdb_create_aggregate_function();
	duckdb_aggregate_function_set_name(f, "emeans_agg");
	duckdb_aggregate_function_add_parameter(f, points);
	for (idx_t i = 0; i + 1 < n_params; i++) {
		duckdb_aggregate_function_add_parameter(f, extra[i]);
	}
	duckdb_aggregate_function_set_return_type(f, ret);
	duckdb_aggregate_function_set_functions(f, StateSize, StateInit, Update, Combine, Finalize);
	duckdb_aggregate_function_set_destructor(f, StateDestroy);
	// NULL nepochs and seed mean the defaults, so the callbacks must see NULLs.
	duckdb_aggregate_function_set_special_handling(f);
	bool ok = duckdb_add_aggregate_function_to_set(set, f) == DuckDBSuccess;
	duckdb_destroy_aggregate_function(&f);
	for (int i = 0; i < 3; i++) {
		duckdb_destroy_logical_type(&extra[i]);
	}
	return ok;
}

bool emeans_register_agg(duckdb_connection connection) {
	// GEOMETRY arrives as ANY: a GEOMETRY parameter would cast GEOMETRY('OGC:CRS84') to plain GEOMETRY and
	// drop the CRS before the function could see it. ANY also takes lists of other numeric types, such as
	// INTEGER[] literals, which Update reads like the others; anything else is an error.
	duckdb_logical_type points[3] = {duckdb_create_logical_type(DUCKDB_TYPE_ANY),
	                                 emeans_list_type(DUCKDB_TYPE_FLOAT), emeans_list_type(DUCKDB_TYPE_DOUBLE)};
	duckdb_logical_type ret = result_type();
	duckdb_aggregate_function_set set = duckdb_create_aggregate_function_set("emeans_agg");
	bool ok = true;
	for (int t = 0; t < 3; t++) {
		for (idx_t n_params = 2; n_params <= 4; n_params++) {
			ok = ok && add_overload(set, points[t], n_params, ret);
		}
	}
	ok = ok && duckdb_register_aggregate_function_set(connection, set) == DuckDBSuccess;
	duckdb_destroy_aggregate_function_set(&set);
	duckdb_destroy_logical_type(&ret);
	for (int t = 0; t < 3; t++) {
		duckdb_destroy_logical_type(&points[t]);
	}
	return ok;
}
