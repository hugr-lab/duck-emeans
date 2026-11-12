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
// The state and the callbacks; src/cpp/emeans_agg.cpp registers the aggregate, with a bind that takes
// emeans_max_work, emeans_threads and the query (emeans_agg_context), so a fit is limited, threaded and
// interruptible as emeans() is.

#include "duckdb.h"

#include "assign.h"
#include "emeans_agg.h"
#include "emeans.h"
#include "emeans_functions.h"
#include "emeans_query.h"
#include "geo.h"
#include "vec_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FIELDS 7

// A group's points live in an emeans_mem.h block, cap x dim doubles as given: (x, y), (longitude,
// latitude) or the vector. A callback pins the states it touches and unpins them before it returns, so
// between callbacks the buffer manager may write the points to temp_directory (pin_list).
typedef struct {
	emeans_block *block;
	double *rows; // the block's data while pinned, NULL otherwise
	int64_t n, cap, dim;
	int64_t k;
	double nepochs;
	uint32_t seed;
	int geographic;
	int geometry; // the points were POINT geometries
	int has_args;
} agg_state;

//===--------------------------------------------------------------------===//
// Pinning
//===--------------------------------------------------------------------===//

// The states one callback has pinned; a state is pinned exactly when it is on the list.
typedef struct {
	emeans_mem *mem;
	agg_state **states;
	idx_t n;
} pin_list;

static int pin_state(agg_state *s, pin_list *pins, char *err) {
	if (!s->block || s->rows) {
		return 0;
	}
	if (emeans_block_pin(s->block, err)) {
		return 1;
	}
	s->rows = emeans_block_data(s->block);
	pins->states[pins->n++] = s;
	return 0;
}

static void unpin_all(pin_list *pins) {
	for (idx_t i = 0; i < pins->n; i++) {
		emeans_block_unpin(pins->states[i]->block);
		pins->states[i]->rows = NULL;
	}
	pins->n = 0;
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

// A block for at least `need` points, doubling, with the points so far copied over. s is pinned (or has
// no block yet), and the new block is pinned and on the list.
static int grow(agg_state *s, int64_t need, pin_list *pins, char *err) {
	int64_t cap = need > 2 * s->cap ? need : 2 * s->cap;
	emeans_block *grown = emeans_block_alloc(pins->mem, (size_t)(cap * s->dim) * sizeof(double), err);
	if (!grown) {
		return 1;
	}
	double *rows = emeans_block_data(grown);
	if (s->n) {
		memcpy(rows, s->rows, (size_t)(s->n * s->dim) * sizeof(double));
	}
	if (!s->block) {
		pins->states[pins->n++] = s;
	}
	emeans_block_free(s->block);
	s->block = grown;
	s->rows = rows;
	s->cap = cap;
	return 0;
}

static int append_rows(agg_state *s, const double *rows, int64_t count, int64_t dim, pin_list *pins, char *err) {
	if (s->dim != 0 && s->dim != dim) {
		snprintf(err, EMEANS_ERR_LEN, "points have %lld dimensions, earlier ones %lld", (long long)dim,
		         (long long)s->dim);
		return 1;
	}
	s->dim = dim;
	int64_t need = s->n + count;
	if (pin_state(s, pins, err) || (need > s->cap && grow(s, need, pins, err))) {
		return 1;
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
	pin_list pins;
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
	       append_rows(s, ctx->row, 1, dim, &ctx->pins, err);
}

int emeans_agg_update(duckdb_data_chunk input, void **states, const emeans_agg_context *c, char *err) {
	idx_t count = duckdb_data_chunk_get_size(input);
	duckdb_vector points = duckdb_data_chunk_get_vector(input, 0);
	pin_list pins = {c->mem, malloc((count ? count : 1) * sizeof(agg_state *)), 0};
	if (!pins.states) {
		return emeans_error(err, "out of memory");
	}
	update_ctx ctx = {points, is_geometry(points), column_is_geographic(points), {NULL, 0, 0, NULL, NULL}, NULL, 0, pins};
	int rc = check_points_type(points, err);
	for (idx_t row = 0; rc == 0 && row < count; row++) {
		rc = update_row(input, row, &ctx, (agg_state *)states[row], err);
	}
	unpin_all(&ctx.pins);
	free(pins.states);
	emeans_buffer_free(&ctx.buf);
	free(ctx.row);
	return rc;
}

static int combine_one(agg_state *src, agg_state *dst, pin_list *pins, char *err) {
	if (src->n == 0) {
		return 0;
	}
	dst->geometry = src->geometry;
	return pin_state(src, pins, err) || set_args(dst, src->k, src->nepochs, src->seed, src->geographic, err) ||
	       append_rows(dst, src->rows, src->n, src->dim, pins, err);
}

int emeans_agg_combine(void **sources, void **targets, idx_t count, const emeans_agg_context *c, char *err) {
	agg_state *pinned[2];
	pin_list pins = {c->mem, pinned, 0};
	int rc = 0;
	for (idx_t i = 0; rc == 0 && i < count; i++) {
		rc = combine_one((agg_state *)sources[i], (agg_state *)targets[i], &pins, err);
		unpin_all(&pins);
	}
	return rc;
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

// The float32 matrix, its sort buffer and the labels are blocks, pinned throughout; the small ones malloc.
typedef struct {
	emeans_block *x, *tmp, *labels;
	float *c;
	double *obj;
} fit_buffers;

static int alloc_fit(fit_buffers *b, const agg_state *s, int64_t dim, int64_t passes, emeans_mem *mem, char *err) {
	size_t floats = (size_t)(s->n * dim);
	b->x = emeans_block_alloc(mem, floats * sizeof(float), err);
	b->tmp = b->x ? emeans_block_alloc(mem, floats * sizeof(float), err) : NULL;
	b->labels = b->tmp ? emeans_block_alloc(mem, (size_t)s->n * sizeof(int32_t), err) : NULL;
	if (!b->labels) {
		return 1;
	}
	b->c = malloc((size_t)(s->k * dim) * sizeof(float));
	b->obj = malloc((size_t)passes * sizeof(double));
	return !b->c || !b->obj ? emeans_error(err, "out of memory") : 0;
}

static void free_fit(fit_buffers *b) {
	emeans_block_free(b->x);
	emeans_block_free(b->tmp);
	emeans_block_free(b->labels);
	free(b->c);
	free(b->obj);
}

static void fit_params(const agg_state *s, const emeans_agg_context *c, emeans_params *p) {
	emeans_default_params(p);
	p->nepochs = s->nepochs;
	p->seed = s->seed;
	p->spherical = s->geographic;
	p->max_work = c->max_work;
	p->threads = c->threads;
	// Unlike emeans() and the reference, k-means++ seeding: on 2-D maps, where the reference turns revival
	// off, a random start left clusters up to 25% worse than PostGIS (docs/POSTGIS.md).
	p->kmeans_plus_plus = 1;
	p->allocator = emeans_mem_allocator(c->mem);
	emeans_query_attach(p, c->query);
}

// The fit of a state's points: in canonical form and sorted, so the result does not depend on the order
// rows arrived in. The sort buffer goes back before the fit, which takes its own workspace. s is pinned;
// on success b holds the centroids in b->c and out describes the run.
static int fit_model(const agg_state *s, const emeans_agg_context *c, const layout *l, fit_buffers *b,
                     emeans_output *out, char *err) {
	emeans_params p;
	fit_params(s, c, &p);
	emeans_recipe recipe;
	if (emeans_resolve_recipe(s->n, s->k, l->dim, p.nepochs, p.law, 0, 0, &recipe, err) ||
	    alloc_fit(b, s, l->dim, recipe.n_passes, c->mem, err)) {
		return 1;
	}
	float *x = emeans_block_data(b->x);
	to_matrix(s, l, x);
	emeans_sort_rows(x, emeans_block_data(b->tmp), s->n, l->dim);
	emeans_block_free(b->tmp);
	b->tmp = NULL;
	*out = (emeans_output) {b->c, emeans_block_data(b->labels), b->obj, 0, recipe};
	return emeans_fit(x, s->n, l->dim, s->k, &p, out, err) != EMEANS_OK;
}

static int fit_state(const agg_state *s, const emeans_agg_context *c, duckdb_vector result, idx_t row, char *err) {
	layout l = make_layout(s, s->geometry);
	fit_buffers b = {NULL, NULL, NULL, NULL, NULL};
	emeans_output out;
	int rc = fit_model(s, c, &l, &b, &out, err) || write_result(result, row, s, &l, &out, err);
	free_fit(&b);
	return rc;
}

int emeans_agg_finalize(void **states, idx_t count, duckdb_vector result, idx_t offset,
                        const emeans_agg_context *c, char *err) {
	agg_state *pinned[1];
	pin_list pins = {c->mem, pinned, 0};
	int rc = 0;
	for (idx_t i = 0; rc == 0 && i < count; i++) {
		agg_state *s = (agg_state *)states[i];
		if (s->n == 0) {
			emeans_set_struct_null(result, offset + i, N_FIELDS);
			continue;
		}
		rc = pin_state(s, &pins, err) || fit_state(s, c, result, offset + i, err);
		unpin_all(&pins);
	}
	return rc;
}

//===--------------------------------------------------------------------===//
// emeans_cluster_id: one state per window frame (emeans_agg.h)
//===--------------------------------------------------------------------===//

size_t emeans_points_size(void) {
	return sizeof(agg_state);
}

void emeans_points_init(void *state) {
	memset(state, 0, sizeof(agg_state));
}

void emeans_points_free(void *state) {
	emeans_block_free(((agg_state *)state)->block);
}

int emeans_points_add(void *state, duckdb_data_chunk input, idx_t first, idx_t end, emeans_mem *mem, bool *kept,
                      char *err) {
	agg_state *s = state;
	duckdb_vector points = duckdb_data_chunk_get_vector(input, 0);
	agg_state *pinned[1];
	update_ctx ctx = {points,   is_geometry(points), column_is_geographic(points), {NULL, 0, 0, NULL, NULL},
	                  NULL,     0,                   {mem, pinned, 0}};
	int rc = check_points_type(points, err);
	for (idx_t row = first; rc == 0 && row < end; row++) {
		int64_t before = s->n;
		rc = update_row(input, row, &ctx, s, err);
		kept[row - first] = s->n > before;
	}
	unpin_all(&ctx.pins);
	emeans_buffer_free(&ctx.buf);
	free(ctx.row);
	return rc;
}

// Labels as emeans_assign_point gives them for emeans_agg's centroids: Euclidean on (x, y), in double.
static void label_planar(const agg_state *s, const double *c, int32_t *labels) {
	for (int64_t i = 0; i < s->n; i++) {
		double best_d = INFINITY;
		for (int64_t j = 0; j < s->k; j++) {
			double dx = s->rows[2 * i] - c[2 * j], dy = s->rows[2 * i + 1] - c[2 * j + 1];
			double d = dx * dx + dy * dy;
			if (d < best_d) {
				best_d = d;
				labels[i] = (int32_t)j + 1;
			}
		}
	}
}

// The same along the sphere, with the great-circle distance in double, for [longitude, latitude] centroids.
static int label_geographic(const agg_state *s, const double *c, int32_t *labels, char *err) {
	double *q = malloc((size_t)(3 * s->k) * sizeof(double));
	if (!q) {
		return emeans_error(err, "out of memory");
	}
	for (int64_t j = 0; j < s->k; j++) {
		emeans_lonlat_to_unit_d(c[2 * j], c[2 * j + 1], q + 3 * j);
	}
	for (int64_t i = 0; i < s->n; i++) {
		double p[3], best_d = INFINITY;
		emeans_lonlat_to_unit_d(s->rows[2 * i], s->rows[2 * i + 1], p);
		for (int64_t j = 0; j < s->k; j++) {
			double d = emeans_great_circle_m_d(p, q + 3 * j);
			if (d < best_d) {
				best_d = d;
				labels[i] = (int32_t)j + 1;
			}
		}
	}
	free(q);
	return 0;
}

// Vectors as emeans_assign labels them: the float32 kernel against the float centroids.
typedef struct {
	float *ct, *bias, *scratch, *x;
} kernel_buffers;

static int label_vectors(const agg_state *s, const float *c, int32_t *labels, char *err) {
	int64_t d = s->dim;
	kernel_buffers kb = {malloc((size_t)(s->k * d) * sizeof(float)), malloc((size_t)s->k * sizeof(float)),
	                     malloc((size_t)emeans_kernel_scratch(d) * sizeof(float)), malloc((size_t)d * sizeof(float))};
	int rc = !kb.ct || !kb.bias || !kb.scratch || !kb.x ? emeans_error(err, "out of memory") : 0;
	if (rc == 0) {
		emeans_kernel kn;
		emeans_kernel_init(&kn, kb.ct, kb.bias, s->k, d, 0);
		for (int64_t j = 0; j < s->k; j++) {
			emeans_kernel_set(&kn, j, c + j * d);
		}
		const int64_t first = 0;
		for (int64_t i = 0; i < s->n; i++) {
			for (int64_t t = 0; t < d; t++) {
				kb.x[t] = (float)s->rows[i * d + t];
			}
			emeans_kernel_nearest(&kn, kb.x, &first, 1, &labels[i], kb.scratch);
			labels[i]++;
		}
	}
	free(kb.ct);
	free(kb.bias);
	free(kb.scratch);
	free(kb.x);
	return rc;
}

static int label_state(const agg_state *s, const layout *l, const float *c, int32_t *labels, char *err) {
	if (!s->geometry) {
		return label_vectors(s, c, labels, err);
	}
	int64_t cols;
	double *centroids = output_centroids(s, l, c, &cols);
	int rc = !centroids ? emeans_error(err, "out of memory")
	         : s->geographic ? label_geographic(s, centroids, labels, err)
	                         : (label_planar(s, centroids, labels), 0);
	free(centroids);
	return rc;
}

int64_t emeans_points_count(const void *state) {
	return ((const agg_state *)state)->n;
}

int emeans_points_label(void *state, const emeans_agg_context *c, int32_t *labels, char *err) {
	agg_state *s = state;
	agg_state *pinned[1];
	pin_list pins = {c->mem, pinned, 0};
	if (pin_state(s, &pins, err)) {
		return 1;
	}
	layout l = make_layout(s, s->geometry); // reads the points, so after the pin
	fit_buffers b = {NULL, NULL, NULL, NULL, NULL};
	emeans_output out;
	int rc = fit_model(s, c, &l, &b, &out, err) || label_state(s, &l, b.c, labels, err);
	free_fit(&b);
	unpin_all(&pins);
	return rc;
}
