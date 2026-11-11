#include "assign.h"

double emeans_sq_dist(const float *a, const float *b, int64_t d) {
	double sum = 0;
	for (int64_t t = 0; t < d; t++) {
		double diff = (double)a[t] - (double)b[t];
		sum += diff * diff;
	}
	return sum;
}

static int32_t nearest(const float *row, const float *c, int64_t k, int64_t d, double *dist) {
	int32_t best = 0;
	double best_dist = emeans_sq_dist(row, c, d);
	for (int64_t j = 1; j < k; j++) {
		double dj = emeans_sq_dist(row, c + j * d, d);
		if (dj < best_dist) {
			best = (int32_t)j;
			best_dist = dj;
		}
	}
	*dist = best_dist;
	return best;
}

void emeans_assign(const float *x, int64_t d, const int64_t *rows, int64_t count, const float *c, int64_t k,
                   int32_t *labels, double *dist) {
	for (int64_t i = 0; i < count; i++) {
		labels[i] = nearest(x + rows[i] * d, c, k, d, &dist[i]);
	}
}
