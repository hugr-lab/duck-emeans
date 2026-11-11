#pragma once

// The e-means recipe: the batch size B and the initial EMA step alpha0 from the fitted scaling laws,
// the pass schedule and the cosine step schedule (Bessa et al., arXiv 2611.09501). Pure arithmetic,
// no DuckDB dependency. It follows faiss/contrib/e_means/base.py (MIT), whose shipped coefficients
// differ from Table 2 of the paper; the code is the reference here.

#include <stdint.h>

#define EMEANS_ERR_LEN 256
#define EMEANS_MAX_NEPOCHS 1e6 // largest accepted nepochs; keeps the pass arithmetic inside int64

typedef enum { EMEANS_LAW_AUTO = 0, EMEANS_LAW_THROUGHPUT = 1, EMEANS_LAW_QUALITY = 2 } emeans_batch_law;

typedef struct {
	int64_t batch_size;      // B: rows per EMA update
	double alpha0;           // initial EMA step, in [1e-4, 1]
	emeans_batch_law law;    // the law that chose B; never AUTO after resolving
	int64_t n_passes;        // length of the pass schedule
	int64_t total_steps;     // sum over the passes of ceil(rows / B)
	double unused_threshold; // tau for dead-cluster revival; 0 disables it
} emeans_recipe;

// Resolves B and alpha0 for n rows of dimension d into k clusters over nepochs passes. batch_size > 0
// overrides the batch law (capped at n); alpha0 > 0 overrides the step law. Returns 0, or 1 with a
// message in err (EMEANS_ERR_LEN bytes).
int emeans_resolve_recipe(int64_t n, int64_t k, int64_t d, double nepochs, emeans_batch_law law, int64_t batch_size,
                          double alpha0, emeans_recipe *out, char *err);

// Nearest power of two in the geometric sense; 1 for values <= 1.
int64_t emeans_round_pow2(double value);

// Pass schedule: [n] * (floor(T) - 1), then round((T - floor(T)) * n) rows if non-zero, then a full
// closing pass; below one pass only the rounded prefix. Rounding is ties-to-even like Python's round().
int64_t emeans_n_passes(int64_t n, double nepochs);
int64_t emeans_pass_rows(int64_t n, double nepochs, int64_t pass); // -1 past the end

// Cosine anneal from alpha0 to alpha0 / 10 over total_steps.
double emeans_scheduled_alpha(double alpha0, int64_t step, int64_t total_steps);

// Permutation seed of one pass: (seed * 1000003 + pass) mod 2^32.
uint32_t emeans_epoch_seed(uint32_t seed, int64_t pass);

// "auto", "throughput" or "quality"; returns 0 on success.
int emeans_parse_batch_law(const char *name, emeans_batch_law *law);
const char *emeans_batch_law_name(emeans_batch_law law);
