#pragma once

// A window frame's points as one emeans_agg state, for emeans_cluster_id (src/cpp/emeans_window.cpp): the
// frame's rows are added chunk by chunk, fitted as emeans_agg fits a group, and every point gets the label
// the two-call form would give it, emeans_assign_point(point, centroids) for geometries and
// emeans_assign(vector, centroids) for vectors.

#include "duckdb_extension.h"
#include "emeans_mem.h"

#include <stdbool.h>
#include <stdint.h>

size_t emeans_points_size(void);
void emeans_points_init(void *state);
void emeans_points_free(void *state);

// Adds rows [first, end) of a chunk whose columns are emeans_agg's arguments (points, k[, nepochs[, seed]]).
// kept[row - first] says whether the row gave a point: NULL arguments and POINT EMPTY do not. Returns 0, or
// 1 with a message in err (EMEANS_ERR_LEN bytes).
int emeans_points_add(void *state, duckdb_data_chunk input, idx_t first, idx_t end, emeans_mem *mem, bool *kept,
                      char *err);

// The number of points added.
int64_t emeans_points_count(const void *state);

// Fits the points and writes each one's 1-based label, in the order they were added, to labels
// (emeans_points_count entries); the fit stops if query (emeans_query.h, may be NULL) is interrupted.
// Returns 0, or 1 with a message in err.
int emeans_points_label(void *state, emeans_mem *mem, void *query, int32_t *labels, char *err);
