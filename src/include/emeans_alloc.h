#pragma once

// Allocation in src/core goes through these wrappers. In normal builds they are plain libc calls.
// Test builds define EMEANS_FAULT_INJECTION: the n-th allocation after emeans_fail_countdown = n then
// fails, so test/c/test_core.c can exercise every out-of-memory path and check (under ASan/LSan) that
// each one reports an error and frees what it had allocated. The countdown is the only global mutable
// state, and it exists only in those single-threaded test builds.

#include <stddef.h>
#include <stdlib.h>

#ifdef EMEANS_FAULT_INJECTION
extern long emeans_fail_countdown;
static inline int emeans_alloc_fails(void) {
	return emeans_fail_countdown > 0 && --emeans_fail_countdown == 0;
}
#else
static inline int emeans_alloc_fails(void) {
	return 0;
}
#endif

static inline void *emeans_malloc(size_t size) {
	return emeans_alloc_fails() ? NULL : malloc(size);
}

static inline void *emeans_calloc(size_t count, size_t size) {
	return emeans_alloc_fails() ? NULL : calloc(count, size);
}

static inline void *emeans_realloc(void *ptr, size_t size) {
	return emeans_alloc_fails() ? NULL : realloc(ptr, size);
}
