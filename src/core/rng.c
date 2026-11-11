#include "rng.h"

#define PCG_MULTIPLIER 6364136223846793005ULL

uint32_t emeans_rng_next(emeans_rng *rng) {
	uint64_t old = rng->state;
	rng->state = old * PCG_MULTIPLIER + rng->inc;
	uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
	uint32_t rot = (uint32_t)(old >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

void emeans_rng_seed(emeans_rng *rng, uint64_t seed, uint64_t stream) {
	rng->state = 0;
	rng->inc = (stream << 1u) | 1u;
	emeans_rng_next(rng);
	rng->state += seed;
	emeans_rng_next(rng);
}

// Two draws, in a fixed order (the order of evaluation of operands is unspecified in C).
static uint64_t next64(emeans_rng *rng) {
	uint64_t high = emeans_rng_next(rng);
	uint64_t low = emeans_rng_next(rng);
	return (high << 32u) | low;
}

uint64_t emeans_rng_below(emeans_rng *rng, uint64_t bound) {
	// Values below 2^64 mod bound would make the low residues more likely; draw again.
	uint64_t threshold = (0u - bound) % bound;
	uint64_t x = next64(rng);
	while (x < threshold) {
		x = next64(rng);
	}
	return x % bound;
}

double emeans_rng_open01(emeans_rng *rng) {
	return ((double)emeans_rng_next(rng) + 0.5) / 4294967296.0;
}

void emeans_shuffle_prefix(emeans_rng *rng, int64_t *perm, int64_t n, int64_t m) {
	for (int64_t i = 0; i < n; i++) {
		perm[i] = i;
	}
	for (int64_t i = 0; i < m; i++) {
		int64_t j = i + (int64_t)emeans_rng_below(rng, (uint64_t)(n - i));
		int64_t t = perm[i];
		perm[i] = perm[j];
		perm[j] = t;
	}
}
