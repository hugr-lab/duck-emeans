#pragma once

// Nearest-centroid assignment.
//
// emeans_kernel_nearest is the fast path: centroid j is scored as bias_j - 2 x.a_j, where a_j is the
// centroid (Euclidean, bias = ||a_j||^2, the argmin of ||x - a_j||^2 less the constant ||x||^2) or its
// direction (inner product, bias = 0, the argmax of x.a_j). The dot products are computed in float32
// for blocks of rows against blocks of centroids stored transposed (d x k), so the innermost loop runs
// over centroids with independent accumulators and the compiler vectorises it without reassociating.
// Like FAISS, it can pick a different centroid than exact arithmetic when two are within float32
// rounding of each other, or for points far from the origin relative to their spread.
//
// emeans_assign is the plain exact loop, kept as the reference the kernel is tested against.

#include <stdint.h>

#define EMEANS_ROW_BLOCK 8
#define EMEANS_COL_BLOCK 256
#define EMEANS_KERNEL_SCRATCH (EMEANS_ROW_BLOCK * EMEANS_COL_BLOCK) // floats

typedef struct {
	float *ct;   // d x k: column j is centroid j (or its direction)
	float *bias; // k
	int64_t k;
	int64_t d;
	int inner_product;
} emeans_kernel;

// Sets column j from the d floats of a (the centroid, or its direction for an inner-product kernel).
void emeans_kernel_set(emeans_kernel *kn, int64_t j, const float *a);

// The best centroid for rows[0..count) of x (row-major, kn->d columns). scratch holds
// EMEANS_KERNEL_SCRATCH floats. Ties go to the lowest index.
void emeans_kernel_nearest(const emeans_kernel *kn, const float *x, const int64_t *rows, int64_t count,
                           int32_t *labels, float *scratch);

// Squared Euclidean distance and dot product of two d-vectors, in double.
double emeans_sq_dist(const float *a, const float *b, int64_t d);
double emeans_dot(const float *a, const float *b, int64_t d);

// dst = src / ||src||, scaled by the largest magnitude first so the norm cannot overflow; zero stays zero.
// src and dst may be the same.
void emeans_normalize(const float *src, float *dst, int64_t d);

// For rows[0..count) of x (n x d), the index of the nearest of the k centroids c (k x d) and the squared
// distance to it, exactly. Ties go to the lowest centroid index.
void emeans_assign(const float *x, int64_t d, const int64_t *rows, int64_t count, const float *c, int64_t k,
                   int32_t *labels, double *dist);
