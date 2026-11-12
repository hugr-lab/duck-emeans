#pragma once

// The e-means loop (Bessa et al., arXiv 2611.09501; reference faiss/contrib/e_means): mini-batch k-means
// whose centroids follow an exponential moving average of their batch means, with the batch size and
// step from the recipe (recipe.h), dead clusters revived by splitting a live one, and a closing pass
// that sets every centroid to the exact mean of the rows it was assigned in that pass. The spherical
// variant assigns each row to the centroid direction with the largest inner product and returns unit
// centroids; like the reference it does not normalise the rows, which should be unit length already.
// No DuckDB dependency; docs/ALGORITHM.md has the formulas.

#include "recipe.h"

#include <stdint.h>

#define EMEANS_OK 0
#define EMEANS_INVALID 1
#define EMEANS_OOM 2
#define EMEANS_BUDGET_EXCEEDED 3

// A snapshot of the centroids during a run, for tracing (emeans_trace in SQL). centroids is k x d and valid
// only during the callback.
typedef struct {
	double epoch;           // rows processed so far / n
	int64_t step;           // updates so far
	const float *centroids; // the moving centroids, or the closing means when final is set
	int final;
} emeans_frame;

typedef void (*emeans_frame_fn)(void *ctx, const emeans_frame *frame);

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
