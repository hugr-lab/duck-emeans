#include "assign.h"

#include <math.h>
#include <string.h>

double emeans_sq_dist(const float *a, const float *b, int64_t d) {
	double sum = 0;
	for (int64_t t = 0; t < d; t++) {
		double diff = (double)a[t] - (double)b[t];
		sum += diff * diff;
	}
	return sum;
}

double emeans_dot(const float *a, const float *b, int64_t d) {
	double sum = 0;
	for (int64_t t = 0; t < d; t++) {
		sum += (double)a[t] * (double)b[t];
	}
	return sum;
}

static double max_abs(const float *v, int64_t d) {
	double m = 0;
	for (int64_t t = 0; t < d; t++) {
		m = fmax(m, fabs((double)v[t]));
	}
	return m;
}

void emeans_normalize(const float *src, float *dst, int64_t d) {
	double scale = max_abs(src, d);
	double norm = 0;
	for (int64_t t = 0; t < d && scale > 0; t++) {
		double v = (double)src[t] / scale;
		norm += v * v;
	}
	norm = sqrt(norm) * scale;
	for (int64_t t = 0; t < d; t++) {
		dst[t] = norm > 0 ? (float)((double)src[t] / norm) : 0.0f;
	}
}

//===--------------------------------------------------------------------===//
// Blocked kernel
//===--------------------------------------------------------------------===//

void emeans_kernel_set(emeans_kernel *kn, int64_t j, const float *a) {
	for (int64_t t = 0; t < kn->d; t++) {
		kn->ct[t * kn->k + j] = a[t];
	}
	kn->bias[j] = kn->inner_product ? 0.0f : (float)emeans_dot(a, a, kn->d);
}

// acc[r][j] = x_r . a_(c0 + j) for nr rows and nc centroids. The inner loop runs over j: independent sums,
// contiguous in ct, so it vectorises without reassociating a reduction.
static void block_dots(const emeans_kernel *kn, const float *const *xr, int64_t nr, int64_t c0, int64_t nc,
                       float *acc) {
	memset(acc, 0, (size_t)(nr * EMEANS_COL_BLOCK) * sizeof(float));
	for (int64_t t = 0; t < kn->d; t++) {
		const float *col = kn->ct + t * kn->k + c0;
		for (int64_t r = 0; r < nr; r++) {
			float xv = xr[r][t];
			float *a = acc + r * EMEANS_COL_BLOCK;
			for (int64_t j = 0; j < nc; j++) {
				a[j] += xv * col[j];
			}
		}
	}
}

// Keeps, per row, the lowest score so far; a later block wins only with a strictly lower score.
static void pick_best(const emeans_kernel *kn, const float *acc, int64_t nr, int64_t c0, int64_t nc, float *best,
                      int32_t *labels) {
	for (int64_t r = 0; r < nr; r++) {
		const float *a = acc + r * EMEANS_COL_BLOCK;
		for (int64_t j = 0; j < nc; j++) {
			float score = kn->bias[c0 + j] - 2.0f * a[j];
			if (score < best[r]) {
				best[r] = score;
				labels[r] = (int32_t)(c0 + j);
			}
		}
	}
}

static void nearest_block(const emeans_kernel *kn, const float *const *xr, int64_t nr, int32_t *labels,
                          float *scratch) {
	float best[EMEANS_ROW_BLOCK];
	for (int64_t r = 0; r < nr; r++) {
		best[r] = INFINITY;
		labels[r] = 0;
	}
	for (int64_t c0 = 0; c0 < kn->k; c0 += EMEANS_COL_BLOCK) {
		int64_t nc = kn->k - c0 < EMEANS_COL_BLOCK ? kn->k - c0 : EMEANS_COL_BLOCK;
		block_dots(kn, xr, nr, c0, nc, scratch);
		pick_best(kn, scratch, nr, c0, nc, best, labels);
	}
}

void emeans_kernel_nearest(const emeans_kernel *kn, const float *x, const int64_t *rows, int64_t count,
                           int32_t *labels, float *scratch) {
	const float *xr[EMEANS_ROW_BLOCK];
	for (int64_t r0 = 0; r0 < count; r0 += EMEANS_ROW_BLOCK) {
		int64_t nr = count - r0 < EMEANS_ROW_BLOCK ? count - r0 : EMEANS_ROW_BLOCK;
		for (int64_t r = 0; r < nr; r++) {
			xr[r] = x + rows[r0 + r] * kn->d;
		}
		nearest_block(kn, xr, nr, labels + r0, scratch);
	}
}

//===--------------------------------------------------------------------===//
// Exact reference
//===--------------------------------------------------------------------===//

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
