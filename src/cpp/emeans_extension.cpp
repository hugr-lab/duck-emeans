// The entry point. Most functions are the C implementations in src/functions, registered through
// DuckDB's C API on a connection to the loader's database: inside DuckDB a duckdb_connection is a
// Connection * (duckdb_connect in src/main/capi/v1/duckdb-c.cpp). emeans, emeans_spherical, emeans_agg
// and emeans_cluster_id are registered here in C++ (src/cpp/emeans_cluster.cpp, emeans_agg.cpp,
// emeans_window.cpp). docs/PLAN.md, "Moving to the C++ API", says which move next.

#define DUCKDB_EXTENSION_MAIN

#include "emeans_extension.hpp"

#include "duckdb.h"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/storage/buffer_manager.hpp"

extern "C" {
#include "emeans_functions.h"
}

namespace duckdb {

// The functions take their large buffers from the database's buffer manager (src/cpp/emeans_mem.cpp).
static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	Connection connection(db);
	auto mem = reinterpret_cast<emeans_mem *>(&BufferManager::GetBufferManager(db));
	if (!emeans_register_all(reinterpret_cast<duckdb_connection>(&connection), mem)) {
		throw InternalException("emeans: failed to register functions");
	}
	RegisterCluster(loader);
	RegisterAgg(loader);
	RegisterClusterId(loader);
}

void EmeansExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string EmeansExtension::Name() {
	return "emeans";
}

std::string EmeansExtension::Version() const {
#ifdef EXT_VERSION_EMEANS
	return EXT_VERSION_EMEANS;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(emeans, loader) {
	duckdb::LoadInternal(loader);
}
}
