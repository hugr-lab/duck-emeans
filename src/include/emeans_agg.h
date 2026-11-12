#pragma once

// emeans_agg's state and callbacks (src/functions/emeans_agg.c), for the C++ side: the aggregate
// emeans_agg (src/cpp/emeans_agg.cpp) and the window function emeans_cluster_id
// (src/cpp/emeans_window.cpp). A state holds a group's points; the aggregate adds a chunk's rows to one
// state per row, the window function a partition's rows to one state, which it fits and labels: each
// point gets the label the two-call form would give it, emeans_assign_point(point, centroids) for
// geometries and emeans_assign(vector, centroids) for vectors.
//
// Functions returning int give 0, or 1 with a message in err (EMEANS_ERR_LEN bytes).

#include "duckdb.h"
#include "emeans_mem.h"

#include <stdbool.h>
#include <stdint.h>

// What a query gives every state it fits: the buffer manager, the query to stop with (emeans_query.h),
// the work limit (0: none) and the threads one fit may use.
typedef struct {
	emeans_mem *mem;
	void *query;
	double max_work;
	int threads;
} emeans_agg_context;

size_t emeans_points_size(void);
void emeans_points_init(void *state);
void emeans_points_free(void *state);

// The aggregate's callbacks over arrays of state pointers: update adds row i of the chunk (emeans_agg's
// arguments) to states[i]; combine adds each source to its target; finalize writes each state's result
// to rows offset.. of result, a NULL struct for a state without points.
int emeans_agg_update(duckdb_data_chunk input, void **states, const emeans_agg_context *ctx, char *err);
int emeans_agg_combine(void **sources, void **targets, idx_t count, const emeans_agg_context *ctx, char *err);
int emeans_agg_finalize(void **states, idx_t count, duckdb_vector result, idx_t offset,
                        const emeans_agg_context *ctx, char *err);

// Adds rows [first, end) of a chunk whose columns are emeans_agg's arguments (points, k[, nepochs[, seed]]).
// kept[row - first] says whether the row gave a point: NULL arguments and POINT EMPTY do not.
int emeans_points_add(void *state, duckdb_data_chunk input, idx_t first, idx_t end, emeans_mem *mem, bool *kept,
                      char *err);

// The number of points added.
int64_t emeans_points_count(const void *state);

// Fits the points and writes each one's 1-based label, in the order they were added, to labels
// (emeans_points_count entries).
int emeans_points_label(void *state, const emeans_agg_context *ctx, int32_t *labels, char *err);
