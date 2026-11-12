// Standalone tests for the numerical core (src/core), built with ASan/UBSan by `make test_core`.
// No DuckDB involved. Every expected value is derived from the formulas in recipe.h, with the
// arithmetic in a comment; none is copied from the program's output.

#include "assign.h"
#include "emeans.h"
#include "recipe.h"
#include "rng.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#ifdef EMEANS_FAULT_INJECTION
long emeans_fail_countdown = 0;
#endif

#define CHECK(cond, ...)                                                                                               \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			failures++;                                                                                                \
			fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                       \
			fprintf(stderr, __VA_ARGS__);                                                                              \
			fprintf(stderr, "\n");                                                                                     \
		}                                                                                                              \
	} while (0)

#define CHECK_NEAR(a, b, tol) CHECK(fabs((a) - (b)) <= (tol), "%s = %.12g, expected %.12g", #a, (double)(a), (double)(b))

static void test_round_pow2(void) {
	// Geometric nearest: for 3, 3/2 = 1.5 is not below 4/3 = 1.33, so 4; for 5, 5/4 = 1.25 is below 8/5 = 1.6, so 4;
	// for 6, 6/4 = 1.5 is not below 8/6 = 1.33, so 8. 4*sqrt(2) is the tie, which rounds up.
	CHECK(emeans_round_pow2(0.5) == 1, "0.5");
	CHECK(emeans_round_pow2(1) == 1, "1");
	CHECK(emeans_round_pow2(3) == 4, "3");
	CHECK(emeans_round_pow2(5) == 4, "5");
	CHECK(emeans_round_pow2(6) == 8, "6");
	CHECK(emeans_round_pow2(4 * sqrt(2.0)) == 8, "tie");
	CHECK(emeans_round_pow2(1024) == 1024, "1024");
	// 1024 * sqrt(2) = 1448.15: 1448 rounds down, 1449 up.
	CHECK(emeans_round_pow2(1448) == 1024, "1448");
	CHECK(emeans_round_pow2(1449) == 2048, "1449");
}

static void test_passes(void) {
	// T = 2.5, n = 10: [10, 5, 10]. T = 0.5: [5]. T = 1.25: 2.5 rounds to 2 (ties-to-even): [2, 10].
	// T = 1.35: 3.5 rounds to 4: [4, 10]. T = 3: [10, 10, 10].
	CHECK(emeans_n_passes(10, 2.5) == 3, "2.5 passes");
	CHECK(emeans_pass_rows(10, 2.5, 0) == 10 && emeans_pass_rows(10, 2.5, 1) == 5 && emeans_pass_rows(10, 2.5, 2) == 10,
	      "2.5 rows");
	CHECK(emeans_pass_rows(10, 2.5, 3) == -1 && emeans_pass_rows(10, 2.5, -1) == -1, "out of range");
	CHECK(emeans_n_passes(10, 0.5) == 1 && emeans_pass_rows(10, 0.5, 0) == 5, "0.5");
	CHECK(emeans_n_passes(10, 1.25) == 2 && emeans_pass_rows(10, 1.25, 0) == 2 && emeans_pass_rows(10, 1.25, 1) == 10,
	      "1.25");
	CHECK(emeans_pass_rows(10, 1.35, 0) == 4, "1.35");
	CHECK(emeans_n_passes(10, 3.0) == 3 && emeans_pass_rows(10, 3.0, 1) == 10, "3.0");
}

static void test_schedule(void) {
	// alpha0 * (0.1 + 0.9 * cosine): step 0 gives alpha0, the middle gives 0.55 alpha0, the end 0.1 alpha0.
	CHECK_NEAR(emeans_scheduled_alpha(0.5, 0, 10), 0.5, 1e-15);
	CHECK_NEAR(emeans_scheduled_alpha(0.5, 5, 10), 0.275, 1e-15);
	CHECK_NEAR(emeans_scheduled_alpha(0.5, 10, 10), 0.05, 1e-15);
	CHECK_NEAR(emeans_scheduled_alpha(0.5, 12, 10), 0.05, 1e-15); // clamped to total_steps
	CHECK_NEAR(emeans_scheduled_alpha(0.5, 0, 0), 0.5, 1e-15);   // total_steps 0 behaves as 1
	// (1234 * 1000003 + 3) = 1234003705 < 2^32; (2^32 - 1) * 1000003 + 7 mod 2^32 = 4293967300.
	CHECK(emeans_epoch_seed(1234, 3) == 1234003705u, "seed");
	CHECK(emeans_epoch_seed(4294967295u, 7) == 4293967300u, "seed wrap");
}

static void test_law_names(void) {
	emeans_batch_law law;
	CHECK(emeans_parse_batch_law("quality", &law) == 0 && law == EMEANS_LAW_QUALITY, "quality");
	CHECK(emeans_parse_batch_law("throughput", &law) == 0 && law == EMEANS_LAW_THROUGHPUT, "throughput");
	CHECK(emeans_parse_batch_law("auto", &law) == 0 && law == EMEANS_LAW_AUTO, "auto");
	CHECK(emeans_parse_batch_law("Quality", &law) == 1, "case sensitive");
	CHECK(strcmp(emeans_batch_law_name(EMEANS_LAW_THROUGHPUT), "throughput") == 0, "name");
}

static void test_recipe_laws(void) {
	emeans_recipe r;
	char err[EMEANS_ERR_LEN];
	// n = 1e6, k = 1000, d = 128, T = 2: auto picks throughput (d > 16).
	//   raw B = 0.035134 * 1000^0.134749 * 1e6^0.876473 * 2^0.813344 = 28421.93 -> 32768 (cap round_pow2(5e5) = 524288)
	//   alpha0 = 0.8244 * 1000^-0.4617 * 32768^0.9003 * 1e6^-0.4498 * 2^-0.1057 = 0.73397064
	//   2 passes of ceil(1e6 / 32768) = 31 steps: 62
	CHECK(emeans_resolve_recipe(1000000, 1000, 128, 2.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.law == EMEANS_LAW_THROUGHPUT && r.batch_size == 32768, "B = %lld", (long long)r.batch_size);
	CHECK_NEAR(r.alpha0, 0.7339706421, 1e-9);
	CHECK(r.n_passes == 2 && r.total_steps == 62, "steps %lld", (long long)r.total_steps);
	CHECK_NEAR(r.unused_threshold, 0.1, 0);
	// Same with d = 16: auto picks quality. raw B = 0.00439175 * (same factors) = 3552.74 -> 4096 (cap n/2 = 5e5)
	//   alpha0 = 0.8244 * 1000^-0.4617 * 4096^0.9003 * 1e6^-0.4498 * 2^-0.1057 = 0.11288254; 2 * ceil(1e6 / 4096) = 490
	CHECK(emeans_resolve_recipe(1000000, 1000, 16, 2.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.law == EMEANS_LAW_QUALITY && r.batch_size == 4096 && r.total_steps == 490, "quality");
	CHECK_NEAR(r.alpha0, 0.1128825402, 1e-9);
	// Forcing throughput at d = 16 gives the d = 128 recipe.
	CHECK(emeans_resolve_recipe(1000000, 1000, 16, 2.0, EMEANS_LAW_THROUGHPUT, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 32768, "forced throughput");
	CHECK_NEAR(r.alpha0, 0.7339706421, 1e-9);
}

static void test_recipe_small(void) {
	emeans_recipe r;
	char err[EMEANS_ERR_LEN];
	// n = 100, k = 10, d = 3, T = 5 (quality): raw B = 0.00439175 * 10^0.134749 * 100^0.876473 * 5^0.813344 = 1.2556,
	//   1.2556 / 1 < 2 / 1.2556 so B = 1. alpha0 = 0.8244 * 10^-0.4617 * 1 * 100^-0.4498 * 5^-0.1057 = 0.030266127.
	//   5 passes of 100 steps.
	CHECK(emeans_resolve_recipe(100, 10, 3, 5.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 1 && r.n_passes == 5 && r.total_steps == 500, "small");
	CHECK_NEAR(r.alpha0, 0.0302661272, 1e-9);
	// n = 1000, k = 1, d = 2, T = 5: raw B = 0.00439175 * 1 * 1000^0.876473 * 5^0.813344 = 6.9272 -> 8 (6.93/4 = 1.73
	//   is not below 8/6.93 = 1.15); cap n/2 = 500. alpha0 = 0.8244 * 8^0.9003 * 1000^-0.4498 * 5^-0.1057 = 0.20225943.
	//   5 * ceil(1000 / 8) = 625 steps. d = 2: no revival.
	CHECK(emeans_resolve_recipe(1000, 1, 2, 5.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 8 && r.total_steps == 625, "k = 1");
	CHECK_NEAR(r.alpha0, 0.2022594253, 1e-9);
	CHECK_NEAR(r.unused_threshold, 0.0, 0);
	// n = 10, k = 2, d = 3, T = 2.5: raw B = 0.0764 -> 1; passes [10, 5, 10] = 25 steps;
	//   alpha0 = 0.8244 * 2^-0.4617 * 10^-0.4498 * 2.5^-0.1057 = 0.19288132.
	CHECK(emeans_resolve_recipe(10, 2, 3, 2.5, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 1 && r.n_passes == 3 && r.total_steps == 25, "fractional");
	CHECK_NEAR(r.alpha0, 0.1928813225, 1e-9);
	// Below one pass: T = 0.5 on 10 rows is one pass of 5 rows, B = 1, so 5 steps.
	CHECK(emeans_resolve_recipe(10, 2, 3, 0.5, EMEANS_LAW_AUTO, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.n_passes == 1 && r.total_steps == 5, "half a pass: %lld steps", (long long)r.total_steps);
	// At the bound, T = 1e6 on 10 rows with B = 1: 1e6 passes of 10 steps, in constant time.
	CHECK(emeans_resolve_recipe(10, 2, 3, 1e6, EMEANS_LAW_AUTO, 1, 0, &r, err) == 0, "%s", err);
	CHECK(r.n_passes == 1000000 && r.total_steps == 10000000, "1e6 passes");
}

static void test_recipe_caps_and_clamps(void) {
	emeans_recipe r;
	char err[EMEANS_ERR_LEN];
	// Throughput cap: n = 7, k = 1, T = 100: raw B = 0.035134 * 7^0.876473 * 100^0.813344 = 8.14 -> 8, but the cap is
	//   round_pow2(3.5) = 4 (3.5 / 2 = 1.75 is not below 4 / 3.5 = 1.14).
	CHECK(emeans_resolve_recipe(7, 1, 100, 100.0, EMEANS_LAW_THROUGHPUT, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 4, "throughput cap gives %lld", (long long)r.batch_size);
	// Quality cap: same shape, raw B = 1.02 -> 1 is below n / 2 = 3; with k = 7 it is still 1.
	CHECK(emeans_resolve_recipe(7, 1, 100, 100.0, EMEANS_LAW_QUALITY, 0, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 1, "quality cap");
	// A given batch size is capped at n; alpha0 then comes from the step law at that batch:
	//   n = 100, k = 10, T = 5, B = 50: 0.8244 * 10^-0.4617 * 50^0.9003 * 100^-0.4498 * 5^-0.1057 = 1.03 -> clamped to 1.
	CHECK(emeans_resolve_recipe(100, 10, 3, 5.0, EMEANS_LAW_AUTO, 50, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 50 && r.alpha0 == 1.0 && r.total_steps == 10, "given B");
	CHECK(emeans_resolve_recipe(100, 10, 3, 5.0, EMEANS_LAW_AUTO, 1000, 0, &r, err) == 0, "%s", err);
	CHECK(r.batch_size == 100 && r.total_steps == 5, "B capped at n");
	// Lower clamp: n = 1e9, k = 1e6, T = 50, B = 1: 0.8244 * 1e6^-0.4617 * 1e9^-0.4498 * 50^-0.1057 = 9.4e-8 -> 1e-4.
	CHECK(emeans_resolve_recipe(1000000000, 1000000, 3, 50.0, EMEANS_LAW_AUTO, 1, 0, &r, err) == 0, "%s", err);
	CHECK(r.alpha0 == 1e-4, "alpha0 lower clamp %g", r.alpha0);
	// A given alpha0 is used as is.
	CHECK(emeans_resolve_recipe(100, 10, 3, 5.0, EMEANS_LAW_AUTO, 0, 0.42, &r, err) == 0, "%s", err);
	CHECK(r.alpha0 == 0.42 && r.batch_size == 1, "given alpha0");
}

static void test_recipe_errors(void) {
	emeans_recipe r;
	char err[EMEANS_ERR_LEN];
	CHECK(emeans_resolve_recipe(0, 1, 1, 1.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "n must"), "n");
	CHECK(emeans_resolve_recipe(10, 0, 1, 1.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "k must be"), "k");
	CHECK(emeans_resolve_recipe(10, 11, 1, 1.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "exceed"), "k > n");
	CHECK(emeans_resolve_recipe(10, 2, 0, 1.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "d must"), "d");
	CHECK(emeans_resolve_recipe(10, 2, 1, 0.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "nepochs"), "T = 0");
	CHECK(emeans_resolve_recipe(10, 2, 1, -1.0, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1, "T < 0");
	CHECK(emeans_resolve_recipe(10, 2, 1, NAN, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1, "T nan");
	CHECK(emeans_resolve_recipe(10, 2, 1, INFINITY, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1, "T inf");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1e6 + 1, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "at most"),
	      "T above the bound");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1e19, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1, "T beyond int64");
	// 0.01 * 10 = 0.1 rounds to zero rows.
	CHECK(emeans_resolve_recipe(10, 2, 1, 0.01, EMEANS_LAW_AUTO, 0, 0, &r, err) == 1 && strstr(err, "zero rows"), "T tiny");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1.0, (emeans_batch_law)7, 0, 0, &r, err) == 1 && strstr(err, "batch_law"),
	      "law");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1.0, EMEANS_LAW_AUTO, -1, 0, &r, err) == 1 && strstr(err, "batch_size"),
	      "B < 0");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1.0, EMEANS_LAW_AUTO, 0, 1.5, &r, err) == 1 && strstr(err, "alpha0"),
	      "alpha0 > 1");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1.0, EMEANS_LAW_AUTO, 0, -0.1, &r, err) == 1, "alpha0 < 0");
	CHECK(emeans_resolve_recipe(10, 2, 1, 1.0, EMEANS_LAW_AUTO, 0, NAN, &r, err) == 1, "alpha0 nan");
}

//===--------------------------------------------------------------------===//
// rng, assign
//===--------------------------------------------------------------------===//

static int is_permutation(const int64_t *perm, int64_t n) {
	int seen[64] = {0};
	for (int64_t i = 0; i < n; i++) {
		if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]++) {
			return 0;
		}
	}
	return 1;
}

static void test_rng(void) {
	emeans_rng a, b, c;
	emeans_rng_seed(&a, 42, 1);
	emeans_rng_seed(&b, 42, 1);
	emeans_rng_seed(&c, 42, 2);
	int same = 1, differs = 0;
	for (int i = 0; i < 100; i++) {
		uint32_t x = emeans_rng_next(&a);
		same &= x == emeans_rng_next(&b);
		differs |= x != emeans_rng_next(&c);
	}
	CHECK(same && differs, "same seed and stream repeat, another stream differs");
	// bound 1 is always 0; bound 2^63 + 1 rejects about half the draws (2^64 mod bound = 2^63 - 1).
	CHECK(emeans_rng_below(&a, 1) == 0, "below 1");
	int hits[5] = {0};
	for (int i = 0; i < 500; i++) {
		uint64_t big = emeans_rng_below(&a, (1ULL << 63) + 1);
		CHECK(big <= (1ULL << 63), "below 2^63 + 1");
		hits[emeans_rng_below(&a, 5)]++;
	}
	CHECK(hits[0] && hits[1] && hits[2] && hits[3] && hits[4], "every value below 5 is drawn");
	double lo = 1, hi = 0;
	for (int i = 0; i < 1000; i++) {
		double u = emeans_rng_open01(&a);
		lo = fmin(lo, u);
		hi = fmax(hi, u);
	}
	CHECK(lo > 0 && hi < 1 && lo < 0.01 && hi > 0.99, "open01 in (0, 1): %g .. %g", lo, hi);
	int64_t perm[10];
	emeans_shuffle_prefix(&a, perm, 10, 10);
	CHECK(is_permutation(perm, 10), "full shuffle");
	emeans_shuffle_prefix(&a, perm, 10, 3);
	CHECK(is_permutation(perm, 10), "a prefix shuffle leaves a permutation");
	emeans_shuffle_prefix(&a, perm, 10, 0);
	CHECK(perm[0] == 0 && perm[9] == 9, "m = 0 is the identity");
}

static void test_assign(void) {
	// Rows (0,0), (2,0), (1,0) against centroids (0,0), (2,0): row 2 is 1 from both, the tie goes to 0.
	const float x[] = {0, 0, 2, 0, 1, 0};
	const float c[] = {0, 0, 2, 0};
	const int64_t rows[] = {2, 0, 1};
	int32_t labels[3];
	double dist[3];
	emeans_assign(x, 2, rows, 3, c, 2, labels, dist);
	CHECK(labels[0] == 0 && dist[0] == 1.0, "tie: label %d dist %g", labels[0], dist[0]);
	CHECK(labels[1] == 0 && dist[1] == 0.0 && labels[2] == 1 && dist[2] == 0.0, "exact hits");
	// (3, 4) to the origin: 9 + 16.
	const float p[] = {3, 4};
	CHECK(emeans_sq_dist(p, c, 2) == 25.0, "sq_dist");
}

//===--------------------------------------------------------------------===//
// emeans_fit
//===--------------------------------------------------------------------===//

typedef struct {
	float centroids[64 * 8];
	int32_t labels[512];
	double obj[16];
	emeans_output out;
	char err[EMEANS_ERR_LEN];
} fit_result;

static int fit(const float *x, int64_t n, int64_t d, int64_t k, const emeans_params *p, fit_result *r) {
	r->out.centroids = r->centroids;
	r->out.labels = r->labels;
	r->out.obj = r->obj;
	r->err[0] = '\0';
	return emeans_fit(x, n, d, k, p, &r->out, r->err);
}

// Six points on the x axis, 0 1 2 3 10 11, from centroids 0 and 1.
static const float LINE[] = {0, 0, 1, 0, 2, 0, 3, 0, 10, 0, 11, 0};
static const float LINE_INIT[] = {0, 0, 1, 0};

static void line_params(emeans_params *p, double nepochs) {
	emeans_default_params(p);
	p->init = LINE_INIT;
	p->batch_size = 6;
	p->alpha0 = 1.0;
	p->nepochs = nepochs;
}

static void test_fit_one_lloyd_step(void) {
	// One pass, one batch of all rows, alpha = alpha0 = 1 at step 0: one Lloyd step and then the closing
	// mean, which is the same thing. Assignment: 0 -> c0; 1, 2, 3, 10, 11 -> c1, mean 27 / 5 = 5.4.
	// Objective against the initial centroids: 0 + 0 + 1 + 4 + 81 + 100 = 186.
	emeans_params p;
	fit_result r;
	line_params(&p, 1.0);
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK_NEAR(r.centroids[0], 0.0, 0);
	CHECK_NEAR(r.centroids[2], 5.4, 1e-6);
	CHECK(r.labels[0] == 0 && r.labels[1] == 1 && r.labels[5] == 1, "labels");
	CHECK_NEAR(r.obj[0], 186.0, 1e-9);
	CHECK(r.out.recipe.total_steps == 1 && r.out.n_splits == 0, "one step, no revival (d = 2)");
}

static void test_fit_two_lloyd_steps(void) {
	// Two passes of one batch each. Step 0 has alpha = 1: c = (0, 5.4), as above. The closing pass assigns
	// against those (0, 1, 2 -> c0, since 2 is 2 from 0 and 3.4 from 5.4; 3, 10, 11 -> c1) and its closing
	// means replace whatever the alpha = 0.55 step did: c = (1, 8), two Lloyd iterations exactly.
	// Objective of pass 1: 0 + 1 + 4 + 2.4^2 + 4.6^2 + 5.6^2 = 63.28.
	emeans_params p;
	fit_result r;
	line_params(&p, 2.0);
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK_NEAR(r.centroids[0], 1.0, 1e-6);
	CHECK_NEAR(r.centroids[1], 0.0, 0);
	CHECK_NEAR(r.centroids[2], 8.0, 1e-6);
	int expected[] = {0, 0, 0, 1, 1, 1};
	for (int i = 0; i < 6; i++) {
		CHECK(r.labels[i] == expected[i], "label %d is %d", i, r.labels[i]);
	}
	CHECK_NEAR(r.obj[0], 186.0, 1e-9);
	CHECK_NEAR(r.obj[1], 63.28, 1e-4);
}

static void test_fit_fractional_pass(void) {
	// nepochs = 0.5 on 6 rows: one pass of 3 rows; the other 3 keep label -1.
	emeans_params p;
	fit_result r;
	line_params(&p, 0.5);
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_OK, "%s", r.err);
	int labelled = 0;
	for (int i = 0; i < 6; i++) {
		labelled += r.labels[i] >= 0;
	}
	CHECK(labelled == 3 && r.out.recipe.n_passes == 1, "%d rows labelled", labelled);
}

// n points around 3 centres (0, 0, 0, 0), (20, 0, 0, 0) and (0, 20, 0, 0), with deterministic noise in
// [-1, 1]. Row i belongs to centre i % 3.
static void make_blobs(float *x, int64_t n, int64_t d) {
	for (int64_t i = 0; i < n; i++) {
		for (int64_t t = 0; t < d; t++) {
			x[i * d + t] = (float)sin((double)(i * 7 + t * 13) * 1.37);
		}
		x[i * d] += (i % 3 == 1) ? 20.0f : 0.0f;
		x[i * d + 1] += (i % 3 == 2) ? 20.0f : 0.0f;
	}
}

static int pure_blobs(const int32_t *labels, int64_t n) {
	for (int64_t i = 3; i < n; i++) {
		if (labels[i] != labels[i % 3]) {
			return 0;
		}
	}
	return labels[0] != labels[1] && labels[1] != labels[2] && labels[0] != labels[2];
}

// After the closing pass every centroid with rows is their mean.
static double closing_error(const float *x, int64_t n, int64_t d, int64_t k, const fit_result *r) {
	double worst = 0;
	for (int64_t j = 0; j < k; j++) {
		for (int64_t t = 0; t < d; t++) {
			double sum = 0;
			int64_t count = 0;
			for (int64_t i = 0; i < n; i++) {
				sum += r->labels[i] == j ? x[i * d + t] : 0;
				count += r->labels[i] == j;
			}
			worst = count ? fmax(worst, fabs(sum / (double)count - r->centroids[j * d + t])) : worst;
		}
	}
	return worst;
}

static void test_fit_blobs(void) {
	enum { N = 300, D = 4, K = 3 };
	float x[N * D];
	make_blobs(x, N, D);
	emeans_params p;
	emeans_default_params(&p);
	p.nepochs = 3;
	fit_result r, again;
	CHECK(fit(x, N, D, K, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK(pure_blobs(r.labels, N), "each blob is one cluster");
	CHECK(r.obj[2] < r.obj[0], "the objective falls: %g -> %g", r.obj[0], r.obj[2]);
	CHECK(closing_error(x, N, D, K, &r) < 1e-5, "closing means, error %g", closing_error(x, N, D, K, &r));
	CHECK(fit(x, N, D, K, &p, &again) == EMEANS_OK, "%s", again.err);
	CHECK(memcmp(r.centroids, again.centroids, sizeof(float) * K * D) == 0 &&
	          memcmp(r.labels, again.labels, sizeof(int32_t) * N) == 0,
	      "the same seed gives the same result");
}

// 10 points near (0, 0, 0) and 10 near (10, 0, 0), alternating, and four initial centroids, two of them
// far from every point: those two are never assigned, their mass decays from 2^-16, and every 5 steps
// revival hands them half of a live cluster. The first cluster has x = 0 exactly, so splitting it gives
// two centroids with the same coordinate 0, and the child is moved one ULP.
static void make_dead_start(float *x, float *init) {
	for (int i = 0; i < 20; i++) {
		x[i * 3] = i % 2 ? 10.0f + 0.1f * (float)(i % 5) : 0.0f;
		x[i * 3 + 1] = 0.05f * (float)(i % 3);
		x[i * 3 + 2] = 0.0f;
	}
	const float start[] = {0, 0, 0, 10, 0, 0, 1000, 1000, 1000, -1000, -1000, -1000};
	memcpy(init, start, sizeof(start));
}

static void test_fit_revival(void) {
	float x[60], init[12];
	make_dead_start(x, init);
	emeans_params p;
	emeans_default_params(&p);
	p.init = init;
	p.nepochs = 1;
	fit_result r;
	CHECK(fit(x, 20, 3, 4, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK(r.out.n_splits >= 2, "%lld splits", (long long)r.out.n_splits);
	for (int t = 0; t < 12; t++) {
		CHECK(fabsf(r.centroids[t]) < 20.0f, "centroid coordinate %d is %g", t, r.centroids[t]);
	}
	// tau = 0 turns revival off: the far centroids stay exactly where they started.
	p.unused_threshold = 0;
	CHECK(fit(x, 20, 3, 4, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK(r.out.n_splits == 0 && r.centroids[6] == 1000.0f && r.centroids[11] == -1000.0f, "no revival");
}

static void test_fit_errors(void) {
	emeans_params p;
	fit_result r;
	line_params(&p, 2.0);
	CHECK(fit(LINE, 6, 2, 7, &p, &r) == EMEANS_INVALID && strstr(r.err, "exceed"), "k > n: %s", r.err);
	r.out.centroids = r.centroids;
	CHECK(emeans_fit(NULL, 6, 2, 2, &p, &r.out, r.err) == EMEANS_INVALID && strstr(r.err, "NULL"), "NULL x");
	float bad[12];
	memcpy(bad, LINE, sizeof(bad));
	bad[7] = NAN;
	CHECK(fit(bad, 6, 2, 2, &p, &r) == EMEANS_INVALID && strstr(r.err, "x contains"), "NaN: %s", r.err);
	bad[7] = 0;
	bad[0] = INFINITY;
	CHECK(fit(bad, 6, 2, 2, &p, &r) == EMEANS_INVALID, "infinity");
	const float bad_init[] = {0, 0, NAN, 0};
	p.init = bad_init;
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_INVALID && strstr(r.err, "init contains"), "%s", r.err);
	// k = 2^31 is rejected before x is read: n = k so the shape is valid, and LINE is far too short.
	line_params(&p, 1.0);
	CHECK(fit(LINE, 2147483648LL, 1, 2147483648LL, &p, &r) == EMEANS_INVALID && strstr(r.err, "2^31"), "%s", r.err);
	// 2 passes of 6 rows x k 2 x d 2 = 48 units of work.
	line_params(&p, 2.0);
	p.max_work = 47;
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_BUDGET_EXCEEDED && strstr(r.err, "is 48"), "%s", r.err);
	p.max_work = 48;
	CHECK(fit(LINE, 6, 2, 2, &p, &r) == EMEANS_OK, "at the limit: %s", r.err);
}

static void test_fit_out_of_memory(void) {
	enum { N = 30, D = 4 };
	float x[N * D];
	make_blobs(x, N, D);
	emeans_params p;
	emeans_default_params(&p);
	fit_result r;
	long fail_at = 1;
	for (; fail_at < 10; fail_at++) {
		emeans_fail_countdown = fail_at;
		int rc = fit(x, N, D, 3, &p, &r);
		if (rc == EMEANS_OK) {
			break;
		}
		CHECK(rc == EMEANS_OOM && strstr(r.err, "out of memory"), "allocation %ld: %s", fail_at, r.err);
	}
	emeans_fail_countdown = 0;
	CHECK(fail_at == 2, "one allocation per call, succeeded at %ld", fail_at);
}

//===--------------------------------------------------------------------===//
// The blocked kernel, normalisation, the spherical variant
//===--------------------------------------------------------------------===//

// k = 300 crosses the 256-centroid block; 61 rows leave a partial 8-row block.
static void test_kernel_matches_exact(void) {
	enum { N = 61, D = 7, K = 300 };
	static float x[N * D], c[K * D], ct[D * K], bias[K], scratch[EMEANS_KERNEL_SCRATCH];
	int64_t rows[N];
	for (int i = 0; i < N * D; i++) {
		x[i] = (float)sin(i * 0.37) * 3.0f;
	}
	for (int i = 0; i < K * D; i++) {
		c[i] = (float)cos(i * 0.91) * 3.0f;
	}
	emeans_kernel kn = {ct, bias, K, D, 0};
	for (int j = 0; j < K; j++) {
		emeans_kernel_set(&kn, j, c + j * D);
	}
	for (int i = 0; i < N; i++) {
		rows[i] = N - 1 - i; // out of order, as a permutation would be
	}
	int32_t fast[N], exact[N];
	double dist[N];
	emeans_kernel_nearest(&kn, x, rows, N, fast, scratch);
	emeans_assign(x, D, rows, N, c, K, exact, dist);
	int differ = 0;
	for (int i = 0; i < N; i++) {
		// A disagreement is allowed only for a float32 near-tie.
		double gap = emeans_sq_dist(x + rows[i] * D, c + fast[i] * D, D) - dist[i];
		differ += fast[i] != exact[i] && gap > 1e-4 * dist[i];
	}
	CHECK(differ == 0, "%d rows assigned differently from the exact loop", differ);
	// The last centroid wins for a row equal to it: bias 1, dots 1 and 0 against a = (1, 0) and (0, 1).
	float two_c[] = {0, 1, 1, 0}, two_ct[4], two_bias[2], row[] = {1, 0};
	emeans_kernel two = {two_ct, two_bias, 2, 2, 0};
	emeans_kernel_set(&two, 0, two_c);
	emeans_kernel_set(&two, 1, two_c + 2);
	int64_t first = 0;
	int32_t label;
	emeans_kernel_nearest(&two, row, &first, 1, &label, scratch);
	CHECK(label == 1 && two_bias[0] == 1.0f, "label %d", label);
	// Inner product: bias 0, (1, 0) prefers (1, 0) to (0, 1).
	two.inner_product = 1;
	emeans_kernel_set(&two, 0, two_c);
	emeans_kernel_set(&two, 1, two_c + 2);
	emeans_kernel_nearest(&two, row, &first, 1, &label, scratch);
	CHECK(label == 1 && two_bias[0] == 0.0f && two_bias[1] == 0.0f, "inner product label %d", label);
}

static void test_normalize(void) {
	// (3, 4) has norm 5; (1e30, 1e30) would overflow a float norm, but scales first: each becomes 1/sqrt(2).
	float v[] = {3, 4}, big[] = {1e30f, -1e30f}, zero[] = {0, 0}, out[2];
	emeans_normalize(v, out, 2);
	CHECK_NEAR(out[0], 0.6, 1e-7);
	CHECK_NEAR(out[1], 0.8, 1e-7);
	emeans_normalize(big, big, 2);
	CHECK_NEAR(big[0], 1 / sqrt(2.0), 1e-7);
	CHECK_NEAR(big[1], -1 / sqrt(2.0), 1e-7);
	emeans_normalize(zero, out, 2);
	CHECK(out[0] == 0 && out[1] == 0, "zero stays zero");
	// (1, 2, 2) . (2, 0, 1) = 4.
	const float a[] = {1, 2, 2}, b[] = {2, 0, 1};
	CHECK(emeans_dot(a, b, 3) == 4.0, "dot");
}

// Unit vectors near three directions in 3-D: e1, e2 and e3, tilted by small angles. Row i belongs to
// direction i % 3.
static void make_directions(float *x, int64_t n) {
	for (int64_t i = 0; i < n; i++) {
		float v[3] = {0.1f * (float)sin(i * 1.7), 0.1f * (float)cos(i * 2.3), 0.1f * (float)sin(i * 0.9)};
		v[i % 3] += 1.0f;
		emeans_normalize(v, x + i * 3, 3);
	}
}

static void test_fit_spherical(void) {
	enum { N = 150, D = 3, K = 3 };
	float x[N * D];
	make_directions(x, N);
	emeans_params p;
	emeans_default_params(&p);
	p.spherical = 1;
	p.nepochs = 3;
	fit_result r;
	CHECK(fit(x, N, D, K, &p, &r) == EMEANS_OK, "%s", r.err);
	CHECK(pure_blobs(r.labels, N), "each direction is one cluster");
	for (int j = 0; j < K; j++) {
		CHECK_NEAR(emeans_dot(r.centroids + j * D, r.centroids + j * D, D), 1.0, 1e-6);
	}
	// The objective is the sum of 2 - 2 x.c: between 0 and 4 per row, small for tight directions.
	CHECK(r.obj[2] >= 0 && r.obj[2] < 0.1 * N, "obj %g", r.obj[2]);
	// Each centroid is the normalised mean of its rows: the mean's direction, so x.c is largest for its own.
	double own = emeans_dot(x, r.centroids + r.labels[0] * D, D);
	CHECK(own > 0.95, "row 0 to its centroid: %g", own);
}

int main(void) {
	test_round_pow2();
	test_passes();
	test_schedule();
	test_law_names();
	test_recipe_laws();
	test_recipe_small();
	test_recipe_caps_and_clamps();
	test_recipe_errors();
	test_rng();
	test_assign();
	test_fit_one_lloyd_step();
	test_fit_two_lloyd_steps();
	test_fit_fractional_pass();
	test_fit_blobs();
	test_fit_revival();
	test_fit_errors();
	test_fit_out_of_memory();
	test_kernel_matches_exact();
	test_normalize();
	test_fit_spherical();
	if (failures) {
		fprintf(stderr, "%d checks failed\n", failures);
		return 1;
	}
	printf("test_core: all checks passed\n");
	return 0;
}
