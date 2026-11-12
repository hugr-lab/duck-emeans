// emeans_mem.h on DuckDB's buffer manager. A block is temporary memory (MemoryTag::EXTENSION) registered
// with can_destroy = false, so while unpinned the buffer manager may write it to temp_directory rather
// than drop it. Exceptions stay on this side: the C callers get NULL or 1 and DuckDB's message.

#include "duckdb.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <cstdio>

extern "C" {
#include "emeans_mem.h"
}

struct emeans_block {
	duckdb::BufferManager &manager;
	duckdb::shared_ptr<duckdb::BlockHandle> block;
	duckdb::BufferHandle pin;
};

static void set_error(char *err, std::exception &e) {
	std::snprintf(err, EMEANS_ERR_LEN, "%s", duckdb::ErrorData(e).RawMessage().c_str());
}

extern "C" emeans_block *emeans_block_alloc(emeans_mem *mem, size_t bytes, char *err) {
	auto &manager = *reinterpret_cast<duckdb::BufferManager *>(mem);
	try {
		auto pin = manager.Allocate(duckdb::MemoryTag::EXTENSION, bytes ? bytes : 1, false);
		auto block = pin.GetBlockHandle();
		return new emeans_block {manager, std::move(block), std::move(pin)};
	} catch (std::exception &e) {
		set_error(err, e);
		return nullptr;
	}
}

extern "C" void *emeans_block_data(emeans_block *b) {
	return b->pin.IsValid() ? b->pin.GetDataMutable() : nullptr;
}

extern "C" int emeans_block_pin(emeans_block *b, char *err) {
	if (b->pin.IsValid()) {
		return 0;
	}
	try {
		b->pin = b->manager.Pin(b->block);
		return 0;
	} catch (std::exception &e) {
		set_error(err, e);
		return 1;
	}
}

extern "C" void emeans_block_unpin(emeans_block *b) {
	b->pin.Destroy();
}

extern "C" void emeans_block_free(emeans_block *b) {
	delete b;
}

static void *alloc_workspace(void *ctx, size_t bytes, void **handle, char *err) {
	emeans_block *b = emeans_block_alloc(static_cast<emeans_mem *>(ctx), bytes, err);
	*handle = b;
	return b ? emeans_block_data(b) : nullptr;
}

static void release_workspace(void *ctx, void *handle) {
	emeans_block_free(static_cast<emeans_block *>(handle));
}

extern "C" emeans_allocator emeans_mem_allocator(emeans_mem *mem) {
	return emeans_allocator {alloc_workspace, release_workspace, mem};
}
