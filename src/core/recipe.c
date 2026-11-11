#include "recipe.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define EMEANS_PI 3.14159265358979323846

// Batch laws (Equation 5): B = c * k^b_k * n^b_n * T^b_T, indexed by emeans_batch_law.
static const double BATCH_LAW[3][4] = {
    {0, 0, 0, 0},                               // AUTO is resolved before use
    {0.035134, 0.134749, 0.876473, 0.813344},   // throughput
    {0.00439175, 0.134749, 0.876473, 0.813344}, // quality
};

// Step law (Equation 6): alpha0 = c * k^a_k * B^a_B * n^a_n * T^a_T, clamped to [1e-4, 1].
static const double STEP_LAW[5] = {0.8244, -0.4617, 0.9003, -0.4498, -0.1057};

#define ALPHA_MIN_FACTOR 0.1 // the cosine schedule ends at alpha0 * ALPHA_MIN_FACTOR
#define AUTO_QUALITY_MAX_D 16 // "auto" picks the quality law up to this dimension
#define NO_REVIVAL_MAX_D 2    // no dead-cluster revival up to this dimension
#define UNUSED_THRESHOLD 0.1  // tau: a cluster with mass below tau * mean mass is dead

static const char *const LAW_NAMES[3] = {"auto", "throughput", "quality"};

int64_t emeans_round_pow2(double value) {
	if (value <= 1) {
		return 1;
	}
	double low = exp2(floor(log2(value)));
	double high = 2 * low;
	return (int64_t)(value / low < high / value ? low : high);
}

static double step_law(int64_t n, int64_t k, double nepochs, int64_t batch_size) {
	double alpha0 = STEP_LAW[0] * pow((double)k, STEP_LAW[1]) * pow((double)batch_size, STEP_LAW[2]) *
	                pow((double)n, STEP_LAW[3]) * pow(nepochs, STEP_LAW[4]);
	return fmin(fmax(alpha0, 1e-4), 1.0);
}

static int64_t law_batch(int64_t n, int64_t k, double nepochs, emeans_batch_law law) {
	const double *c = BATCH_LAW[law];
	double raw = c[0] * pow((double)k, c[1]) * pow((double)n, c[2]) * pow(nepochs, c[3]);
	int64_t batch = emeans_round_pow2(raw);
	// The two fitted recipes cap differently, on purpose.
	int64_t cap = law == EMEANS_LAW_THROUGHPUT ? emeans_round_pow2((double)n / 2) : n / 2;
	if (batch > cap) {
		batch = cap;
	}
	if (batch > n) {
		batch = n;
	}
	return batch < 1 ? 1 : batch;
}

// Python's round() is ties-to-even; nearbyint() is the same in the default rounding mode.
static int64_t fractional_rows(int64_t n, double nepochs) {
	return (int64_t)nearbyint((nepochs - floor(nepochs)) * (double)n);
}

int64_t emeans_n_passes(int64_t n, double nepochs) {
	int64_t whole = (int64_t)floor(nepochs);
	if (whole == 0) {
		return 1;
	}
	return whole + (fractional_rows(n, nepochs) ? 1 : 0);
}

int64_t emeans_pass_rows(int64_t n, double nepochs, int64_t pass) {
	int64_t whole = (int64_t)floor(nepochs);
	int64_t passes = emeans_n_passes(n, nepochs);
	if (pass < 0 || pass >= passes) {
		return -1;
	}
	if (whole == 0) {
		return fractional_rows(n, nepochs);
	}
	// [n] * (whole - 1), the rounded fraction if any, then the closing full pass.
	if (pass < whole - 1 || pass == passes - 1) {
		return n;
	}
	return fractional_rows(n, nepochs);
}

// Closed form of the sum over the pass schedule, so a large nepochs costs nothing here: whole - 1 full
// passes plus the closing one, and the rounded fraction if any; below one pass only the fraction.
static int64_t total_steps(int64_t n, double nepochs, int64_t batch_size) {
	int64_t whole = (int64_t)floor(nepochs);
	int64_t full = (n + batch_size - 1) / batch_size;
	int64_t partial = (fractional_rows(n, nepochs) + batch_size - 1) / batch_size;
	return whole == 0 ? partial : whole * full + partial;
}

double emeans_scheduled_alpha(double alpha0, int64_t step, int64_t total_steps) {
	int64_t clamped = step < total_steps ? step : total_steps;
	double fraction = (double)clamped / (double)(total_steps > 1 ? total_steps : 1);
	double cosine = 0.5 * (1.0 + cos(EMEANS_PI * fraction));
	return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine);
}

uint32_t emeans_epoch_seed(uint32_t seed, int64_t pass) {
	// seed * 1000003 < 2^52 and pass is small, so the sum fits; the cast reduces mod 2^32.
	return (uint32_t)((uint64_t)seed * 1000003u + (uint64_t)pass);
}

int emeans_parse_batch_law(const char *name, emeans_batch_law *law) {
	for (int i = 0; i < 3; i++) {
		if (strcmp(name, LAW_NAMES[i]) == 0) {
			*law = (emeans_batch_law)i;
			return 0;
		}
	}
	return 1;
}

const char *emeans_batch_law_name(emeans_batch_law law) {
	return LAW_NAMES[law];
}

int emeans_error(char *err, const char *msg) {
	snprintf(err, EMEANS_ERR_LEN, "%s", msg);
	return 1;
}

static int check_shape(int64_t n, int64_t k, int64_t d, char *err) {
	if (n < 1) {
		return emeans_error(err, "n must be at least 1");
	}
	if (k < 1) {
		return emeans_error(err, "k must be at least 1");
	}
	if (k > n) {
		return emeans_error(err, "k must not exceed n");
	}
	if (d < 1) {
		return emeans_error(err, "d must be at least 1");
	}
	return 0;
}

static int check_schedule(int64_t n, double nepochs, emeans_batch_law law, char *err) {
	// The bound keeps floor(nepochs) and the pass count inside int64 (NaN and infinity fail it too).
	if (!(nepochs > 0) || !(nepochs <= EMEANS_MAX_NEPOCHS)) {
		return emeans_error(err, "nepochs must be a positive number, at most 1e6");
	}
	if (floor(nepochs) == 0 && fractional_rows(n, nepochs) == 0) {
		return emeans_error(err, "nepochs * n rounds to zero rows");
	}
	if (law < EMEANS_LAW_AUTO || law > EMEANS_LAW_QUALITY) {
		return emeans_error(err, "batch_law must be 'auto', 'throughput' or 'quality'");
	}
	return 0;
}

static int check_overrides(int64_t batch_size, double alpha0, char *err) {
	if (batch_size < 0) {
		return emeans_error(err, "batch_size must be at least 1 (0 = from the batch law)");
	}
	if (alpha0 < 0 || alpha0 > 1 || !isfinite(alpha0)) {
		return emeans_error(err, "alpha0 must be in (0, 1] (0 = from the step law)");
	}
	return 0;
}

// B from the override (capped at n) or the law, then alpha0 from the override or the step law at that B.
static void resolve_sizes(int64_t n, int64_t k, int64_t d, double nepochs, emeans_batch_law law, int64_t batch_size,
                          double alpha0, emeans_recipe *out) {
	if (law == EMEANS_LAW_AUTO) {
		law = d <= AUTO_QUALITY_MAX_D ? EMEANS_LAW_QUALITY : EMEANS_LAW_THROUGHPUT;
	}
	out->law = law;
	out->batch_size = batch_size > 0 ? (batch_size < n ? batch_size : n) : law_batch(n, k, nepochs, law);
	out->alpha0 = alpha0 > 0 ? alpha0 : step_law(n, k, nepochs, out->batch_size);
}

int emeans_resolve_recipe(int64_t n, int64_t k, int64_t d, double nepochs, emeans_batch_law law, int64_t batch_size,
                          double alpha0, emeans_recipe *out, char *err) {
	if (check_shape(n, k, d, err) || check_schedule(n, nepochs, law, err) || check_overrides(batch_size, alpha0, err)) {
		return 1;
	}
	resolve_sizes(n, k, d, nepochs, law, batch_size, alpha0, out);
	out->n_passes = emeans_n_passes(n, nepochs);
	out->total_steps = total_steps(n, nepochs, out->batch_size);
	out->unused_threshold = d <= NO_REVIVAL_MAX_D ? 0.0 : UNUSED_THRESHOLD;
	return 0;
}
