#include "assign.h"

#include <math.h>
#include <string.h>

#if defined(__APPLE__) && defined(EMEANS_USE_ACCELERATE)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAVE_BLAS 1
#else
#define HAVE_BLAS 0
#endif

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

int emeans_kernel_has_blas(void) {
	return HAVE_BLAS;
}

void emeans_kernel_init(emeans_kernel *kn, float *ct, float *bias, int64_t k, int64_t d, int inner_product) {
	*kn = (emeans_kernel) {ct, bias, k, d, inner_product, HAVE_BLAS};
}

int64_t emeans_kernel_scratch(int64_t d) {
	int64_t portable = EMEANS_ROW_BLOCK * EMEANS_COL_BLOCK;
	int64_t blas = HAVE_BLAS ? EMEANS_BLAS_ROWS * (EMEANS_BLAS_COLS + d) : 0;
	return portable > blas ? portable : blas;
}

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

// Keeps, per row, the lowest score so far; a later block wins only with a strictly lower score. Row r of
// acc starts at r * stride.
static void pick_best(const emeans_kernel *kn, const float *acc, int64_t stride, int64_t nr, int64_t c0, int64_t nc,
                      float *best, int32_t *labels) {
	for (int64_t r = 0; r < nr; r++) {
		const float *a = acc + r * stride;
		for (int64_t j = 0; j < nc; j++) {
			float score = kn->bias[c0 + j] - 2.0f * a[j];
			if (score < best[r]) {
				best[r] = score;
				labels[r] = (int32_t)(c0 + j);
			}
		}
	}
}

static void reset_best(float *best, int32_t *labels, int64_t nr) {
	for (int64_t r = 0; r < nr; r++) {
		best[r] = INFINITY;
		labels[r] = 0;
	}
}

static int64_t block_len(int64_t total, int64_t start, int64_t block) {
	return total - start < block ? total - start : block;
}

static void nearest_block(const emeans_kernel *kn, const float *const *xr, int64_t nr, int32_t *labels,
                          float *scratch) {
	float best[EMEANS_ROW_BLOCK];
	reset_best(best, labels, nr);
	for (int64_t c0 = 0; c0 < kn->k; c0 += EMEANS_COL_BLOCK) {
		int64_t nc = block_len(kn->k, c0, EMEANS_COL_BLOCK);
		block_dots(kn, xr, nr, c0, nc, scratch);
		pick_best(kn, scratch, EMEANS_COL_BLOCK, nr, c0, nc, best, labels);
	}
}

static void nearest_portable(const emeans_kernel *kn, const float *x, const int64_t *rows, int64_t count,
                             int32_t *labels, float *scratch) {
	const float *xr[EMEANS_ROW_BLOCK];
	for (int64_t r0 = 0; r0 < count; r0 += EMEANS_ROW_BLOCK) {
		int64_t nr = block_len(count, r0, EMEANS_ROW_BLOCK);
		for (int64_t r = 0; r < nr; r++) {
			xr[r] = x + rows[r0 + r] * kn->d;
		}
		nearest_block(kn, xr, nr, labels + r0, scratch);
	}
}

#if HAVE_BLAS
// Up to 64 rows gathered into one contiguous block, then one sgemm per 1024 centroids:
// acc (nr x nc) = rows (nr x d) . ct[:, c0 .. c0 + nc) (d x nc, row stride k).
static void nearest_blas_block(const emeans_kernel *kn, const float *xb, int64_t nr, int32_t *labels, float *acc) {
	float best[EMEANS_BLAS_ROWS];
	reset_best(best, labels, nr);
	for (int64_t c0 = 0; c0 < kn->k; c0 += EMEANS_BLAS_COLS) {
		int64_t nc = block_len(kn->k, c0, EMEANS_BLAS_COLS);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (__LAPACK_int)nr, (__LAPACK_int)nc,
		            (__LAPACK_int)kn->d, 1.0f, xb, (__LAPACK_int)kn->d, kn->ct + c0, (__LAPACK_int)kn->k, 0.0f, acc,
		            EMEANS_BLAS_COLS);
		pick_best(kn, acc, EMEANS_BLAS_COLS, nr, c0, nc, best, labels);
	}
}

static void nearest_blas(const emeans_kernel *kn, const float *x, const int64_t *rows, int64_t count,
                         int32_t *labels, float *scratch) {
	float *acc = scratch;
	float *xb = scratch + EMEANS_BLAS_ROWS * EMEANS_BLAS_COLS;
	for (int64_t r0 = 0; r0 < count; r0 += EMEANS_BLAS_ROWS) {
		int64_t nr = block_len(count, r0, EMEANS_BLAS_ROWS);
		for (int64_t r = 0; r < nr; r++) {
			memcpy(xb + r * kn->d, x + rows[r0 + r] * kn->d, (size_t)kn->d * sizeof(float));
		}
		nearest_blas_block(kn, xb, nr, labels + r0, acc);
	}
}
#endif

void emeans_kernel_nearest(const emeans_kernel *kn, const float *x, const int64_t *rows, int64_t count,
                           int32_t *labels, float *scratch) {
#if HAVE_BLAS
	if (kn->use_blas) {
		nearest_blas(kn, x, rows, count, labels, scratch);
		return;
	}
#endif
	nearest_portable(kn, x, rows, count, labels, scratch);
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
