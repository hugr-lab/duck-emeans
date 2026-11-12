#pragma once

// Vector I/O between DuckDB lists and the core's row-major float matrices.
//
// Input vectors are LIST(FLOAT) or LIST(DOUBLE) (FLOAT[d] and other numeric lists reach these through
// DuckDB's implicit casts). They are copied into a float buffer, checked for NULLs, equal lengths and
// finite values. Output lists are appended to the list vector's child, which may move on reserve.

#include "duckdb_extension.h"
#include "emeans_mem.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
	float *data;
	size_t cap; // floats
	int managed; // in an emeans_mem.h block (pinned throughout), not on the heap
	emeans_mem *mem;
	emeans_block *block;
} emeans_buffer;

// A buffer for a whole input matrix, in a block from mem. {NULL, 0} is a heap buffer, for small things.
emeans_buffer emeans_managed_buffer(emeans_mem *mem);

// Grows b to hold at least count floats, keeping its contents. Returns 0, or 1 with a message in err.
int emeans_buffer_reserve(emeans_buffer *b, size_t count, char *err);
void emeans_buffer_free(emeans_buffer *b);

bool emeans_row_is_valid(duckdb_vector vec, idx_t row);

// Whether a vector is a LIST of a numeric type the readers take: FLOAT, DOUBLE, the signed and unsigned
// integers, HUGEINT and DECIMAL.
bool emeans_is_numeric_list(duckdb_vector vec);
void emeans_set_row_null(duckdb_vector vec, idx_t row);

// A NULL struct row: the parent and every child, so materialisation (which copies children by their own
// validity) never reads an unwritten child.
void emeans_set_struct_null(duckdb_vector out, idx_t row, idx_t n_fields);

// The vectors of row `row` of a LIST(LIST(T)) vector, as an n x d matrix in buf. Returns 0, or 1 with
// a message in err (EMEANS_ERR_LEN bytes) naming the 1-based vector at fault. `what` names the argument.
int emeans_read_matrix(duckdb_vector outer, idx_t row, const char *what, emeans_buffer *buf, int64_t *n,
                       int64_t *d, char *err);

// One LIST(T) vector at row `row`, as d floats in buf.
int emeans_read_vector(duckdb_vector vec, idx_t row, emeans_buffer *buf, int64_t *d, char *err);

// Makes room for count more elements in the list vector's child, points row at them and returns where they
// start. The child's buffers may move: fetch their pointers afterwards. Returns 0, or 1 with err.
int emeans_list_append(duckdb_vector out, idx_t row, idx_t count, idx_t *start, char *err);

// Writers for row `row` of a list vector. Return 0, or 1 with a message in err.
int emeans_write_matrix(duckdb_vector out, idx_t row, const float *m, int64_t rows, int64_t cols, char *err);
int emeans_write_labels(duckdb_vector out, idx_t row, const int32_t *labels, int64_t n, char *err);
int emeans_write_doubles(duckdb_vector out, idx_t row, const double *v, int64_t n, char *err);
// A rows x cols matrix of doubles into a LIST(LIST(DOUBLE)) vector.
int emeans_write_matrix_f64(duckdb_vector out, idx_t row, const double *m, int64_t rows, int64_t cols, char *err);

duckdb_logical_type emeans_list_type(duckdb_type element);
duckdb_logical_type emeans_matrix_type(duckdb_type element); // LIST(LIST(element))
