#pragma once

// The e-means loop (Bessa et al., arXiv 2611.09501; reference faiss/contrib/e_means): mini-batch k-means
// whose centroids follow an exponential moving average of their batch means, with the batch size and
// step from the recipe (recipe.h), dead clusters revived by splitting a live one, and a closing pass
// that sets every centroid to the exact mean of the rows it was assigned in that pass. The spherical
// variant assigns each row to the centroid direction with the largest inner product and returns unit
// centroids; like the reference it does not normalise the rows, which should be unit length already.
// No DuckDB dependency; docs/ALGORITHM.md has the formulas.

#include "parallel.h"
#include "recipe.h"

#include <stddef.h>
#include <stdint.h>

#define EMEANS_OK 0
#define EMEANS_INVALID 1
#define EMEANS_OOM 2
#define EMEANS_BUDGET_EXCEEDED 3
#define EMEANS_INTERRUPTED 4

// A snapshot of the centroids during a run, for tracing (emeans_trace in SQL). centroids is k x d and valid
// only during the callback.
typedef struct {
	double epoch;           // rows processed so far / n
	int64_t step;           // updates so far
	const float *centroids; // the moving centroids, or the closing means when final is set
	int final;
} emeans_frame;

typedef void (*emeans_frame_fn)(void *ctx, const emeans_frame *frame);

// Where emeans_fit's workspace comes from. alloc returns `bytes` bytes and sets *handle, which release
// later gets, or returns NULL with a message in err (EMEANS_ERR_LEN bytes). A NULL alloc means calloc.
// The SQL functions pass DuckDB's buffer manager (emeans_mem.h).
typedef struct {
	void *(*alloc)(void *ctx, size_t bytes, void **handle, char *err);
	void (*release)(void *ctx, void *handle);
	void *ctx;
} emeans_allocator;

typedef struct {
	double nepochs;          // passes over the data, fractional allowed (default 5)
	uint32_t seed;           // initial centroids, pass permutations and donor sampling (default 1234)
	emeans_batch_law law;    // default EMEANS_LAW_AUTO
	int64_t batch_size;      // > 0 overrides the batch law
	double alpha0;           // > 0 overrides the step law
	double unused_threshold; // tau for revival; < 0 (default) means 0.1 above d = 2 and 0 (off) otherwise
	const float *init;       // k x d initial centroids, or NULL (default) to choose them from the rows
	int kmeans_plus_plus;    // choose them by k-means++ seeding rather than at random (default 0)
	double max_work;         // limit on rows x k x d summed over all passes; 0 (default) = none
	int spherical;           // assign by the largest inner product with the centroid directions (default 0)
	int threads;             // threads for the assignment of a large batch (default 1); the result does not
	                         // depend on it
	int constant_step;       // alpha0 at every step instead of the cosine schedule (default 0). With
	                         // batch_size = n, alpha0 = 1 and unused_threshold = 0 the run is Lloyd's k-means.
	emeans_frame_fn on_frame; // called before the first step, every frame_every steps, and after closing
	void *frame_ctx;
	int64_t frame_every;     // 0 (default): only the first and the final frame
	emeans_allocator allocator; // the workspace's; zeroed (default) for calloc
	int (*interrupted)(void *ctx); // asked once per step and per k-means++ centroid; non-zero stops the
	void *interrupt_ctx;           // fit with EMEANS_INTERRUPTED. NULL (default): never
	// Runs fn(arg, part, parts) for every part and returns when all have finished, as emeans_parallel_for
	// does; for a caller with its own threads (the SQL functions use DuckDB's scheduler). NULL (default):
	// emeans_parallel_for.
	void (*parallel)(void *ctx, emeans_task_fn fn, void *arg, int parts);
	void *parallel_ctx;
} emeans_params;

void emeans_default_params(emeans_params *p);

// Caller-allocated output. centroids holds k x d floats, labels n, obj emeans_n_passes(n, nepochs).
typedef struct {
	float *centroids;
	int32_t *labels;     // each row's cluster in the last pass; -1 for rows a fractional single pass skipped
	double *obj;         // per pass: the sum of squared distances (spherical: of 2 - 2 x.c), while the centroids move
	int64_t n_splits;    // dead clusters revived
	emeans_recipe recipe; // what was used
} emeans_output;

// Clusters the n x d row-major matrix x into k clusters. Returns EMEANS_OK, or another EMEANS_* code with
// a message in err (EMEANS_ERR_LEN bytes). Uses one allocation and no global state.
int emeans_fit(const float *x, int64_t n, int64_t d, int64_t k, const emeans_params *p, emeans_output *out,
               char *err);

// A fit fed row by row: the same run as emeans_fit, for data that arrives in passes instead of sitting in
// memory (emeans_stream in SQL, where DuckDB orders each pass). The caller hands in every pass's rows in
// that pass's order, emeans_pass_rows(n, nepochs, pass) of them, and the stream steps every B rows. Memory
// is O(B d + k d), not O(n d). Given rows in emeans_fit's permutation and its start, the result is
// emeans_fit's.
typedef struct emeans_stream emeans_stream;

// Begins a stream over n rows of dimension d. p->init must hold the k x d start (the stream cannot draw it
// from rows it has not seen); p->kmeans_plus_plus is ignored. out->centroids (k x d) and out->obj
// (emeans_n_passes(n, nepochs)) are the caller's and must live until emeans_stream_finish; out->labels is
// not used. p is copied. Returns EMEANS_OK with *stream set, or an EMEANS_* code with a message in err.
int emeans_stream_begin(int64_t n, int64_t d, int64_t k, const emeans_params *p, emeans_output *out,
                        emeans_stream **stream, char *err);

// Rows the current pass still takes (0: end it with emeans_stream_end_pass), or -1 after the last pass.
int64_t emeans_stream_wanted(const emeans_stream *stream);
// The current pass, from 0; the number of passes once all are done.
int64_t emeans_stream_pass(const emeans_stream *stream);
const emeans_recipe *emeans_stream_recipe(const emeans_stream *stream);

// Hands in count rows (count x d, row-major) of the current pass; rows beyond what it takes are ignored.
// Returns EMEANS_OK, EMEANS_INVALID (a NaN or infinity, or no pass left) or EMEANS_INTERRUPTED.
int emeans_stream_add(emeans_stream *stream, const float *rows, int64_t count, char *err);

// Ends the current pass: steps on its last, partial batch and stores its objective in out->obj. Fails with
// EMEANS_INVALID if the pass had fewer rows than it takes.
int emeans_stream_end_pass(emeans_stream *stream, char *err);

// After the last pass: sets out->centroids to the closing means and, unless counts is NULL, counts (k) to
// the rows each cluster was assigned in the closing pass.
int emeans_stream_finish(emeans_stream *stream, int64_t *counts, char *err);

// Releases the stream and its workspace. NULL is a no-op.
void emeans_stream_free(emeans_stream *stream);
