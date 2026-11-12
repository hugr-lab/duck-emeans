#include "geo.h"

#include <math.h>
#include <string.h>

#define DEG (3.14159265358979323846 / 180.0)

//===--------------------------------------------------------------------===//
// WKB
//===--------------------------------------------------------------------===//

static uint32_t read_u32(const uint8_t *p, int little) {
	return little ? (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24
	              : (uint32_t)p[3] | (uint32_t)p[2] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[0] << 24;
}

static double read_f64(const uint8_t *p, int little) {
	uint8_t b[8];
	for (int i = 0; i < 8; i++) {
		b[i] = little ? p[i] : p[7 - i];
	}
	// The host is little-endian on every platform DuckDB builds for.
	double v;
	memcpy(&v, b, 8);
	return v;
}

// The geometry kind without dimension flags: ISO adds 1000 (Z), 2000 (M) or 3000 (ZM); extended WKB sets
// the high bits 0x80000000 (Z), 0x40000000 (M) and 0x20000000 (an SRID follows the type).
static uint32_t base_type(uint32_t type) {
	return (type & 0x0fffffffu) % 1000u;
}

int emeans_wkb_point(const uint8_t *wkb, size_t len, double *x, double *y) {
	if (len < 5 || wkb[0] > 1) {
		return EMEANS_WKB_MALFORMED;
	}
	int little = wkb[0] == 1;
	uint32_t type = read_u32(wkb + 1, little);
	if (base_type(type) != 1) {
		return EMEANS_WKB_NOT_POINT;
	}
	size_t at = 5 + ((type & 0x20000000u) ? 4 : 0);
	if (len < at + 16) {
		return EMEANS_WKB_MALFORMED;
	}
	*x = read_f64(wkb + at, little);
	*y = read_f64(wkb + at + 8, little);
	return isnan(*x) || isnan(*y) ? EMEANS_WKB_EMPTY : EMEANS_WKB_POINT;
}

//===--------------------------------------------------------------------===//
// Coordinate systems and the sphere
//===--------------------------------------------------------------------===//

int emeans_crs_is_geographic(const char *crs) {
	static const char *const GEOGRAPHIC[] = {"OGC:CRS84", "OGC:CRS83", "EPSG:4326", "EPSG:4258",
	                                         "EPSG:4269", "EPSG:4283", "EPSG:4619", "EPSG:7844"};
	for (size_t i = 0; crs && i < sizeof(GEOGRAPHIC) / sizeof(GEOGRAPHIC[0]); i++) {
		if (strcmp(crs, GEOGRAPHIC[i]) == 0) {
			return 1;
		}
	}
	return 0;
}

void emeans_lonlat_to_unit(double lon, double lat, float out[3]) {
	double cl = cos(lat * DEG);
	out[0] = (float)(cl * cos(lon * DEG));
	out[1] = (float)(cl * sin(lon * DEG));
	out[2] = (float)sin(lat * DEG);
}

void emeans_unit_to_lonlat(const float v[3], double *lon, double *lat) {
	*lon = atan2((double)v[1], (double)v[0]) / DEG;
	*lat = atan2((double)v[2], hypot((double)v[0], (double)v[1])) / DEG;
}

// atan2 of the cross and dot products is accurate for small and large angles alike, where acos is not.
double emeans_great_circle_m(const float a[3], const float b[3]) {
	double cx = (double)a[1] * b[2] - (double)a[2] * b[1];
	double cy = (double)a[2] * b[0] - (double)a[0] * b[2];
	double cz = (double)a[0] * b[1] - (double)a[1] * b[0];
	double dot = (double)a[0] * b[0] + (double)a[1] * b[1] + (double)a[2] * b[2];
	return EMEANS_EARTH_RADIUS_M * atan2(sqrt(cx * cx + cy * cy + cz * cz), dot);
}

//===--------------------------------------------------------------------===//
// Sorting rows
//===--------------------------------------------------------------------===//

static int row_less_equal(const float *a, const float *b, int64_t d) {
	for (int64_t t = 0; t < d; t++) {
		if (a[t] != b[t]) {
			return a[t] < b[t];
		}
	}
	return 1;
}

// Merges the sorted runs [lo, mid) and [mid, hi) of src into dst.
static void merge(const float *src, float *dst, int64_t lo, int64_t mid, int64_t hi, int64_t d) {
	int64_t i = lo, j = mid;
	for (int64_t out = lo; out < hi; out++) {
		int take_left = i < mid && (j >= hi || row_less_equal(src + i * d, src + j * d, d));
		const float *from = take_left ? src + (i++) * d : src + (j++) * d;
		memcpy(dst + out * d, from, (size_t)d * sizeof(float));
	}
}

// Bottom-up: runs of width 1, 2, 4, ... merged between rows and tmp; the result is copied back if it
// ended in tmp.
void emeans_sort_rows(float *rows, float *tmp, int64_t n, int64_t d) {
	float *src = rows, *dst = tmp;
	for (int64_t width = 1; width < n; width *= 2) {
		for (int64_t lo = 0; lo < n; lo += 2 * width) {
			int64_t mid = lo + width < n ? lo + width : n;
			int64_t hi = lo + 2 * width < n ? lo + 2 * width : n;
			merge(src, dst, lo, mid, hi, d);
		}
		float *t = src;
		src = dst;
		dst = t;
	}
	if (src != rows) {
		memcpy(rows, src, (size_t)(n * d) * sizeof(float));
	}
}
