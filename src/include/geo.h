#pragma once

// Points for spatial clustering: POINT geometries from WKB, and the move between longitude/latitude and
// unit vectors on the sphere. No DuckDB dependency.
//
// A geographic CRS (longitude and latitude in degrees, x first) is clustered on the sphere: each point
// becomes the unit vector (cos lat cos lon, cos lat sin lon, sin lat) and the spherical variant of e-means
// clusters those by direction, so clusters cross the antimeridian and the poles correctly. Anything else is
// treated as planar and clustered on (x, y).

#include <stddef.h>
#include <stdint.h>

#define EMEANS_EARTH_RADIUS_M 6371008.8 // mean Earth radius (IUGG)

enum { EMEANS_WKB_POINT = 0, EMEANS_WKB_EMPTY = 1, EMEANS_WKB_NOT_POINT = 2, EMEANS_WKB_MALFORMED = 3 };

// x and y of a WKB POINT (ISO or extended: XY, XYZ, XYM, XYZM; either byte order). Returns
// EMEANS_WKB_POINT, EMEANS_WKB_EMPTY for POINT EMPTY (NaN coordinates), or an error kind.
int emeans_wkb_point(const uint8_t *wkb, size_t len, double *x, double *y);

// Whether a CRS identifier is a geographic (longitude, latitude) one this extension knows: OGC:CRS84,
// OGC:CRS83, and EPSG 4326 (WGS 84), 4258 (ETRS89), 4269 (NAD83), 4283 (GDA94), 4619 (SWEREF99),
// 7844 (GDA2020). NULL or anything else is planar.
int emeans_crs_is_geographic(const char *crs);

void emeans_lonlat_to_unit(double lon, double lat, float out[3]);
void emeans_unit_to_lonlat(const float v[3], double *lon, double *lat);

// Great-circle distance in metres between two unit vectors.
double emeans_great_circle_m(const float a[3], const float b[3]);

// Sorts n rows of d floats in place, lexicographically (a stable merge sort through tmp, n x d floats).
// The aggregate sorts its points so that the result does not depend on the order rows arrive in.
void emeans_sort_rows(float *rows, float *tmp, int64_t n, int64_t d);
