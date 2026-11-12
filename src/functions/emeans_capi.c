// The C API build's stand-ins for src/cpp (until stage 4 of docs/PLAN.md removes that build):
// emeans_mem.h on malloc, where pinning does nothing, and emeans_query.h with no query to interrupt.

#ifndef EMEANS_CPP_BUILD

#include "emeans_mem.h"
#include "emeans_query.h"

#include "recipe.h"

#include <stdlib.h>

struct emeans_block {
	void *data;
};

// One allocation: the data follows the header, which is one pointer, so it is aligned for doubles.
emeans_block *emeans_block_alloc(emeans_mem *mem, size_t bytes, char *err) {
	(void)mem;
	emeans_block *b = malloc(sizeof(emeans_block) + bytes);
	if (!b) {
		emeans_error(err, "out of memory");
		return NULL;
	}
	b->data = b + 1;
	return b;
}

void *emeans_block_data(emeans_block *b) {
	return b->data;
}

int emeans_block_pin(emeans_block *b, char *err) {
	(void)b;
	(void)err;
	return 0;
}

void emeans_block_unpin(emeans_block *b) {
	(void)b;
}

void emeans_block_free(emeans_block *b) {
	free(b);
}

emeans_allocator emeans_mem_allocator(emeans_mem *mem) {
	(void)mem;
	return (emeans_allocator) {NULL, NULL, NULL};
}

void *emeans_query_context(duckdb_client_context context) {
	(void)context;
	return NULL;
}

int emeans_query_interrupted(void *query) {
	(void)query;
	return 0;
}

#endif
