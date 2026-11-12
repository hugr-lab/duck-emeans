#pragma once

// Large buffers: the input matrices, the aggregate's points and the core's workspace.
//
// A block comes from the database's buffer manager (src/cpp/emeans_mem.cpp): it counts against
// memory_limit, an allocation past the limit fails with DuckDB's out-of-memory message instead of taking
// the process down, and a block that is not pinned may be written to temp_directory when memory runs short
// and read back on the next pin.
//
// A block is pinned when allocated. Its data pointer is valid only while it is pinned; a pin may move it.
// Blocks are not thread safe: one thread uses a block at a time.

#include "emeans.h"

#include <stddef.h>

typedef struct emeans_mem emeans_mem; // a database's buffer manager
typedef struct emeans_block emeans_block;

// A pinned block of `bytes` bytes (contents undefined), or NULL with a message in err (EMEANS_ERR_LEN).
emeans_block *emeans_block_alloc(emeans_mem *mem, size_t bytes, char *err);
void *emeans_block_data(emeans_block *b); // NULL while unpinned
// Pins an unpinned block, reading it back if it was spilled. Returns 0, or 1 with a message in err.
int emeans_block_pin(emeans_block *b, char *err);
void emeans_block_unpin(emeans_block *b);
void emeans_block_free(emeans_block *b); // NULL is ignored

// An allocator for emeans_fit's workspace (emeans_params.allocator) that takes it from mem.
emeans_allocator emeans_mem_allocator(emeans_mem *mem);
