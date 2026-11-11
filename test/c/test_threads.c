// Concurrency test of src/core, built with ThreadSanitizer by `make test_threads`.
//
// DuckDB calls a scalar function from many threads at once. Eight threads resolve recipes from one
// shared, read-only parameter table and compare every result with a single-threaded reference. The
// core must have no mutable global state for this to pass.

#include "recipe.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_THREADS 8
#define N_CASES 4096

typedef struct {
	int64_t n, k, d;
	double nepochs;
} recipe_case;

typedef struct {
	const recipe_case *cases;
	const emeans_recipe *reference;
	int offset;
	int mismatches;
} worker_args;

static int same(const emeans_recipe *a, const emeans_recipe *b) {
	return a->batch_size == b->batch_size && a->alpha0 == b->alpha0 && a->law == b->law &&
	       a->n_passes == b->n_passes && a->total_steps == b->total_steps &&
	       a->unused_threshold == b->unused_threshold;
}

static int resolve(const recipe_case *c, emeans_recipe *r) {
	char err[EMEANS_ERR_LEN];
	return emeans_resolve_recipe(c->n, c->k, c->d, c->nepochs, EMEANS_LAW_AUTO, 0, 0, r, err);
}

static void *worker(void *arg) {
	worker_args *w = (worker_args *)arg;
	for (int i = w->offset; i < N_CASES; i += N_THREADS) {
		emeans_recipe r;
		if (resolve(&w->cases[i], &r) || !same(&r, &w->reference[i])) {
			w->mismatches++;
		}
	}
	return NULL;
}

// Deterministic parameter table: n from 2 to ~2^20, k up to n, d from 1 to 256, T from 0.5 to 8.
static void fill_cases(recipe_case *cases) {
	uint32_t state = 12345;
	for (int i = 0; i < N_CASES; i++) {
		state = state * 1664525u + 1013904223u;
		cases[i].n = 2 + (int64_t)(state >> 12);
		state = state * 1664525u + 1013904223u;
		cases[i].k = 1 + (int64_t)(state % (uint32_t)cases[i].n);
		state = state * 1664525u + 1013904223u;
		cases[i].d = 1 + (int64_t)(state % 256u);
		state = state * 1664525u + 1013904223u;
		cases[i].nepochs = 0.5 + (state % 16u) * 0.5;
	}
}

int main(void) {
	recipe_case *cases = malloc(N_CASES * sizeof(recipe_case));
	emeans_recipe *reference = malloc(N_CASES * sizeof(emeans_recipe));
	if (!cases || !reference) {
		return 2;
	}
	fill_cases(cases);
	for (int i = 0; i < N_CASES; i++) {
		if (resolve(&cases[i], &reference[i])) {
			fprintf(stderr, "reference case %d failed to resolve\n", i);
			return 1;
		}
	}
	pthread_t threads[N_THREADS];
	worker_args args[N_THREADS];
	for (int t = 0; t < N_THREADS; t++) {
		args[t] = (worker_args) {cases, reference, t, 0};
		pthread_create(&threads[t], NULL, worker, &args[t]);
	}
	int mismatches = 0;
	for (int t = 0; t < N_THREADS; t++) {
		pthread_join(threads[t], NULL);
		mismatches += args[t].mismatches;
	}
	free(cases);
	free(reference);
	if (mismatches) {
		fprintf(stderr, "%d recipes differed between threads and the reference\n", mismatches);
		return 1;
	}
	printf("test_threads: %d recipes on %d threads match the reference\n", N_CASES, N_THREADS);
	return 0;
}
