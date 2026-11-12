// emeans_assign_point(point GEOMETRY, centroids DOUBLE[][]) -> STRUCT(label INTEGER, distance DOUBLE)
//
// The nearest centroid of a POINT, as a 1-based label, with emeans_agg's centroids. With a geographic CRS
// (see geo.h) nearest is along the sphere and distance is the great-circle distance in metres; otherwise
// it is the Euclidean distance in the CRS's units. NULL, or POINT EMPTY, gives NULL.

#include "duckdb.h"

#include "emeans_functions.h"
#include "geo.h"
#include "recipe.h"
#include "vec_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { ROW_OK = 0, ROW_NULL = 1, ROW_ERROR = 2 };

typedef struct {
	double *xy; // k x 2
	int64_t k, cap;
} centroid_list;

static int grow(centroid_list *c, int64_t k, char *err) {
	if (k <= c->cap) {
		return 0;
	}
	double *xy = realloc(c->xy, (size_t)(2 * k) * sizeof(double));
	if (!xy) {
		return emeans_error(err, "out of memory");
	}
	c->xy = xy;
	c->cap = k;
	return 0;
}

// Centroid j of the row: a list of two finite numbers.
static int read_centroid(duckdb_vector inner, idx_t at, double *out, int64_t j, char *err) {
	duckdb_list_entry e = ((duckdb_list_entry *)duckdb_vector_get_data(inner))[at];
	const double *v = (const double *)duckdb_vector_get_data(duckdb_list_vector_get_child(inner)) + e.offset;
	uint64_t *valid = duckdb_vector_get_validity(duckdb_list_vector_get_child(inner));
	int ok = emeans_row_is_valid(inner, at) && e.length == 2 &&
	         (!valid || (duckdb_validity_row_is_valid(valid, e.offset) && duckdb_validity_row_is_valid(valid, e.offset + 1)));
	if (!ok || !isfinite(v[0]) || !isfinite(v[1])) {
		snprintf(err, EMEANS_ERR_LEN, "centroid %lld must be two finite numbers, x and y", (long long)j + 1);
		return 1;
	}
	out[0] = v[0];
	out[1] = v[1];
	return 0;
}

static int read_centroids(duckdb_vector cents, idx_t row, centroid_list *c, char *err) {
	duckdb_list_entry e = ((duckdb_list_entry *)duckdb_vector_get_data(cents))[row];
	if (e.length == 0) {
		return emeans_error(err, "no centroids");
	}
	c->k = (int64_t)e.length;
	if (grow(c, c->k, err)) {
		return 1;
	}
	duckdb_vector inner = duckdb_list_vector_get_child(cents);
	for (int64_t j = 0; j < c->k; j++) {
		if (read_centroid(inner, e.offset + (idx_t)j, c->xy + 2 * j, j, err)) {
			return 1;
		}
	}
	return 0;
}

static int32_t nearest_planar(const centroid_list *c, double x, double y, double *distance) {
	int32_t best = 0;
	double best_d = INFINITY;
	for (int64_t j = 0; j < c->k; j++) {
		double dx = x - c->xy[2 * j], dy = y - c->xy[2 * j + 1];
		double d = dx * dx + dy * dy;
		if (d < best_d) {
			best_d = d;
			best = (int32_t)j;
		}
	}
	*distance = sqrt(best_d);
	return best;
}

// In double precision throughout: the distance is reported, and float32 unit vectors would put ~0.2 m of
// rounding into it.
static int32_t nearest_geographic(const centroid_list *c, double lon, double lat, double *distance) {
	double p[3], q[3];
	emeans_lonlat_to_unit_d(lon, lat, p);
	int32_t best = 0;
	double best_d = INFINITY;
	for (int64_t j = 0; j < c->k; j++) {
		emeans_lonlat_to_unit_d(c->xy[2 * j], c->xy[2 * j + 1], q);
		double d = emeans_great_circle_m_d(p, q);
		if (d < best_d) {
			best_d = d;
			best = (int32_t)j;
		}
	}
	*distance = best_d;
	return best;
}

static int read_point(duckdb_vector vec, idx_t row, double *x, double *y, char *err) {
	duckdb_string_t s = ((duckdb_string_t *)duckdb_vector_get_data(vec))[row];
	int kind = emeans_wkb_point((const uint8_t *)duckdb_string_t_data(&s), duckdb_string_t_length(s), x, y);
	if (kind == EMEANS_WKB_EMPTY) {
		return ROW_NULL;
	}
	return kind == EMEANS_WKB_POINT ? ROW_OK
	                                : ROW_ERROR * emeans_error(err, "takes POINT geometries; use ST_Centroid for others");
}

typedef struct {
	int geographic;
	centroid_list c;
} geo_ctx;

static int assign_row(duckdb_data_chunk input, idx_t row, geo_ctx *ctx, int32_t *label, double *distance,
                      char *err) {
	duckdb_vector points = duckdb_data_chunk_get_vector(input, 0);
	duckdb_vector cents = duckdb_data_chunk_get_vector(input, 1);
	if (!emeans_row_is_valid(points, row) || !emeans_row_is_valid(cents, row)) {
		return ROW_NULL;
	}
	double x, y;
	int rc = read_point(points, row, &x, &y, err);
	if (rc != ROW_OK) {
		return rc;
	}
	if (read_centroids(cents, row, &ctx->c, err)) {
		return ROW_ERROR;
	}
	*label = 1 + (ctx->geographic ? nearest_geographic(&ctx->c, x, y, distance)
	                              : nearest_planar(&ctx->c, x, y, distance));
	return ROW_OK;
}

// -1 if the argument is not a GEOMETRY; else whether its CRS is geographic.
static int column_kind(duckdb_vector vec) {
	duckdb_logical_type type = duckdb_vector_get_column_type(vec);
	int geometry = duckdb_get_type_id(type) == DUCKDB_TYPE_GEOMETRY;
	char *crs = geometry ? duckdb_geometry_type_get_crs(type) : NULL;
	int kind = geometry ? emeans_crs_is_geographic(crs) : -1;
	duckdb_free(crs);
	duckdb_destroy_logical_type(&type);
	return kind;
}

void emeans_assign_geometry(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
	idx_t count = duckdb_data_chunk_get_size(input);
	int32_t *labels = (int32_t *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 0));
	double *distances = (double *)duckdb_vector_get_data(duckdb_struct_vector_get_child(output, 1));
	geo_ctx ctx = {column_kind(duckdb_data_chunk_get_vector(input, 0)), {NULL, 0, 0}};
	if (ctx.geographic < 0) {
		duckdb_scalar_function_set_error(info, "emeans_assign_point: takes a POINT geometry; for vectors use emeans_assign");
		return;
	}
	char err[EMEANS_ERR_LEN];
	for (idx_t row = 0; row < count; row++) {
		int rc = assign_row(input, row, &ctx, &labels[row], &distances[row], err);
		if (rc == ROW_ERROR) {
			char msg[EMEANS_ERR_LEN + 24];
			snprintf(msg, sizeof(msg), "emeans_assign_point: %s", err);
			duckdb_scalar_function_set_error(info, msg);
			break;
		}
		if (rc == ROW_NULL) {
			emeans_set_struct_null(output, row, 2);
		}
	}
	free(ctx.c.xy);
}
