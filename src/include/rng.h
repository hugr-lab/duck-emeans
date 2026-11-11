#pragma once

// PCG32 (O'Neill, pcg-random.org): a small, fast generator whose state lives on the stack of the call
// that uses it, so concurrent calls share nothing. It cannot reproduce numpy's permutations, which the
// reference uses; tests compare objectives and assignments, never centroid bits.

#include <stdint.h>

typedef struct {
	uint64_t state;
	uint64_t inc; // odd; selects one of 2^63 independent streams
} emeans_rng;

void emeans_rng_seed(emeans_rng *rng, uint64_t seed, uint64_t stream);
uint32_t emeans_rng_next(emeans_rng *rng);

// Uniform in [0, bound) without modulo bias; bound >= 1.
uint64_t emeans_rng_below(emeans_rng *rng, uint64_t bound);

// Uniform in the open interval (0, 1), so log(u) and log(-log(u)) are finite.
double emeans_rng_open01(emeans_rng *rng);

// Sets perm to 0..n-1, then makes its first m entries a uniformly random ordered sample of m distinct
// indices (a partial Fisher-Yates shuffle, O(n) to reset and O(m) to draw). perm stays a permutation.
void emeans_shuffle_prefix(emeans_rng *rng, int64_t *perm, int64_t n, int64_t m);
