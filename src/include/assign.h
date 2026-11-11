#pragma once

// Exact nearest-centroid assignment. Row-major float32 data, distances accumulated in double. This is
// the plain O(count * k * d) loop; a blocked ||x||^2 + ||c||^2 - 2 x.c kernel comes with the performance
// work (docs/PLAN.md, phase 5).

#include <stdint.h>

// Squared Euclidean distance between two d-vectors.
double emeans_sq_dist(const float *a, const float *b, int64_t d);

// For rows[0..count) of x (n x d), the index of the nearest of the k centroids c (k x d) and the squared
// distance to it. Ties go to the lowest centroid index.
void emeans_assign(const float *x, int64_t d, const int64_t *rows, int64_t count, const float *c, int64_t k,
                   int32_t *labels, double *dist);
