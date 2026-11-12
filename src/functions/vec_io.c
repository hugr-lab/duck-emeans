#include "vec_io.h"

#include "recipe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DUCKDB_EXTENSION_EXTERN

void emeans_buffer_free(emeans_buffer *b) {
	free(b->data);
	b->data = NULL;
	b->cap = 0;
}

int emeans_buffer_reserve(emeans_buffer *b, size_t count, char *err) {
	if (count <= b->cap) {
		return 0;
	}
	float *grown = realloc(b->data, count * sizeof(float));
	if (!grown) {
		return emeans_error(err, "out of memory");
	}
	b->data = grown;
	b->cap = count;
	return 0;
}

bool emeans_row_is_valid(duckdb_vector vec, idx_t row) {
	return duckdb_validity_row_is_valid(duckdb_vector_get_validity(vec), row);
}

void emeans_set_row_null(duckdb_vector vec, idx_t row) {
	duckdb_vector_ensure_validity_writable(vec);
	duckdb_validity_set_row_invalid(duckdb_vector_get_validity(vec), row);
}

void emeans_set_struct_null(duckdb_vector out, idx_t row, idx_t n_fields) {
	emeans_set_row_null(out, row);
	for (idx_t i = 0; i < n_fields; i++) {
		emeans_set_row_null(duckdb_struct_vector_get_child(out, i), row);
	}
}

//===--------------------------------------------------------------------===//
// Reading
//===--------------------------------------------------------------------===//

static bool element_is_double(duckdb_vector list_vec) {
	duckdb_logical_type type = duckdb_vector_get_column_type(duckdb_list_vector_get_child(list_vec));
	bool is_double = duckdb_get_type_id(type) == DUCKDB_TYPE_DOUBLE;
	duckdb_destroy_logical_type(&type);
	return is_double;
}

static duckdb_list_entry list_entry(duckdb_vector list_vec, idx_t row) {
	return ((duckdb_list_entry *)duckdb_vector_get_data(list_vec))[row];
}

static int64_t first_null(duckdb_vector child, uint64_t offset, uint64_t length) {
	uint64_t *validity = duckdb_vector_get_validity(child);
	for (uint64_t i = 0; validity && i < length; i++) {
		if (!duckdb_validity_row_is_valid(validity, offset + i)) {
			return (int64_t)i;
		}
	}
	return -1;
}

// Copies length elements of the list child, as float, into dst; returns the first non-finite index or -1.
static int64_t copy_elements(duckdb_vector list_vec, uint64_t offset, uint64_t length, float *dst) {
	void *data = duckdb_vector_get_data(duckdb_list_vector_get_child(list_vec));
	bool is_double = element_is_double(list_vec);
	int64_t bad = -1;
	for (uint64_t i = 0; i < length; i++) {
		dst[i] = is_double ? (float)((const double *)data)[offset + i] : ((const float *)data)[offset + i];
		bad = bad < 0 && !isfinite(dst[i]) ? (int64_t)i : bad;
	}
	return bad;
}

// Vector `index` of a LIST(T) vector into dst, which holds d floats. `pos` (1-based) names it in errors.
static int read_one(duckdb_vector list_vec, idx_t index, int64_t d, float *dst, int64_t pos, char *err) {
	if (!emeans_row_is_valid(list_vec, index)) {
		snprintf(err, EMEANS_ERR_LEN, "vector %lld is NULL", (long long)pos);
		return 1;
	}
	duckdb_list_entry e = list_entry(list_vec, index);
	if ((int64_t)e.length != d) {
		snprintf(err, EMEANS_ERR_LEN, "vector %lld has %llu elements, expected %lld", (long long)pos,
		         (unsigned long long)e.length, (long long)d);
		return 1;
	}
	int64_t null_at = first_null(duckdb_list_vector_get_child(list_vec), e.offset, e.length);
	if (null_at >= 0) {
		snprintf(err, EMEANS_ERR_LEN, "vector %lld has a NULL at element %lld", (long long)pos, (long long)null_at + 1);
		return 1;
	}
	if (copy_elements(list_vec, e.offset, e.length, dst) >= 0) {
		snprintf(err, EMEANS_ERR_LEN, "vector %lld contains NaN or infinity", (long long)pos);
		return 1;
	}
	return 0;
}

// The dimension: the length of the first vector, which must be valid and non-empty.
static int first_length(duckdb_vector inner, uint64_t offset, const char *what, int64_t *d, char *err) {
	*d = emeans_row_is_valid(inner, offset) ? (int64_t)list_entry(inner, offset).length : 0;
	if (*d == 0) {
		snprintf(err, EMEANS_ERR_LEN, "the first vector of %s is NULL or empty", what);
		return 1;
	}
	return 0;
}

int emeans_read_matrix(duckdb_vector outer, idx_t row, const char *what, emeans_buffer *buf, int64_t *n,
                       int64_t *d, char *err) {
	duckdb_list_entry e = list_entry(outer, row);
	duckdb_vector inner = duckdb_list_vector_get_child(outer);
	*n = (int64_t)e.length;
	if (*n == 0) {
		snprintf(err, EMEANS_ERR_LEN, "%s has no vectors", what);
		return 1;
	}
	if (first_length(inner, e.offset, what, d, err) || emeans_buffer_reserve(buf, (size_t)(*n * *d), err)) {
		return 1;
	}
	for (int64_t i = 0; i < *n; i++) {
		if (read_one(inner, e.offset + (idx_t)i, *d, buf->data + i * *d, i + 1, err)) {
			return 1;
		}
	}
	return 0;
}

int emeans_read_vector(duckdb_vector vec, idx_t row, emeans_buffer *buf, int64_t *d, char *err) {
	*d = (int64_t)list_entry(vec, row).length;
	if (*d == 0) {
		return emeans_error(err, "the vector is empty");
	}
	if (emeans_buffer_reserve(buf, (size_t)*d, err)) {
		return 1;
	}
	return read_one(vec, row, *d, buf->data, 1, err);
}

//===--------------------------------------------------------------------===//
// Writing
//===--------------------------------------------------------------------===//

// Makes room for count more elements in the list vector's child and points row at them.
static int append(duckdb_vector out, idx_t row, idx_t count, idx_t *start, char *err) {
	*start = duckdb_list_vector_get_size(out);
	if (duckdb_list_vector_reserve(out, *start + count) != DuckDBSuccess) {
		return emeans_error(err, "out of memory");
	}
	duckdb_list_vector_set_size(out, *start + count);
	duckdb_list_entry *entries = (duckdb_list_entry *)duckdb_vector_get_data(out);
	entries[row].offset = *start;
	entries[row].length = count;
	return 0;
}

static void *child_data(duckdb_vector out) {
	return duckdb_vector_get_data(duckdb_list_vector_get_child(out));
}

int emeans_write_matrix(duckdb_vector out, idx_t row, const float *m, int64_t rows, int64_t cols, char *err) {
	idx_t first, start;
	if (append(out, row, (idx_t)rows, &first, err)) {
		return 1;
	}
	duckdb_vector inner = duckdb_list_vector_get_child(out);
	for (int64_t r = 0; r < rows; r++) {
		if (append(inner, first + (idx_t)r, (idx_t)cols, &start, err)) {
			return 1;
		}
		memcpy((float *)child_data(inner) + start, m + r * cols, (size_t)cols * sizeof(float));
	}
	return 0;
}

// Labels as 1-based cluster numbers, so centroids[label] works in SQL; -1 (not seen) becomes NULL.
int emeans_write_labels(duckdb_vector out, idx_t row, const int32_t *labels, int64_t n, char *err) {
	idx_t start;
	if (append(out, row, (idx_t)n, &start, err)) {
		return 1;
	}
	duckdb_vector child = duckdb_list_vector_get_child(out);
	int32_t *data = (int32_t *)duckdb_vector_get_data(child);
	for (int64_t i = 0; i < n; i++) {
		data[start + (idx_t)i] = labels[i] + 1;
		if (labels[i] < 0) {
			emeans_set_row_null(child, start + (idx_t)i);
		}
	}
	return 0;
}

int emeans_write_doubles(duckdb_vector out, idx_t row, const double *v, int64_t n, char *err) {
	idx_t start;
	if (append(out, row, (idx_t)n, &start, err)) {
		return 1;
	}
	memcpy((double *)child_data(out) + start, v, (size_t)n * sizeof(double));
	return 0;
}

duckdb_logical_type emeans_list_type(duckdb_type element) {
	duckdb_logical_type child = duckdb_create_logical_type(element);
	duckdb_logical_type list = duckdb_create_list_type(child);
	duckdb_destroy_logical_type(&child);
	return list;
}

duckdb_logical_type emeans_matrix_type(duckdb_type element) {
	duckdb_logical_type inner = emeans_list_type(element);
	duckdb_logical_type outer = duckdb_create_list_type(inner);
	duckdb_destroy_logical_type(&inner);
	return outer;
}
