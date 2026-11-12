#pragma once

#include "duckdb.hpp"

namespace duckdb {

class EmeansExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

// emeans_cluster_id, the window function (src/cpp/emeans_window.cpp).
void RegisterClusterId(ExtensionLoader &loader);
// emeans_agg, the aggregate (src/cpp/emeans_agg.cpp).
void RegisterAgg(ExtensionLoader &loader);
// emeans and emeans_spherical (src/cpp/emeans_cluster.cpp).
void RegisterCluster(ExtensionLoader &loader);

} // namespace duckdb
