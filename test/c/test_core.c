// Standalone tests for the numerical core (src/core), built with ASan/UBSan by `make test_core`.
// No DuckDB involved. Every expected value is derived from the formulas in recipe.h, with the
// arithmetic in a comment; none is copied from the program's output.

#include "recipe.h"

#include <math.h>
#include <stdio.h>
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

int main(void) {
	test_round_pow2();
	test_passes();
	test_schedule();
	test_law_names();
	test_recipe_laws();
	test_recipe_small();
	test_recipe_caps_and_clamps();
	test_recipe_errors();
	if (failures) {
		fprintf(stderr, "%d checks failed\n", failures);
		return 1;
	}
	printf("test_core: all checks passed\n");
	return 0;
}
