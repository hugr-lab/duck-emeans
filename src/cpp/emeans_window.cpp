// emeans_cluster_id(points, k [, nepochs [, seed]]) OVER (PARTITION BY ...) -> INTEGER: each row's 1-based
// cluster within its partition, in one call, as PostGIS's ST_ClusterKMeans gives it (0-based there).
//
// A window function, like ntile: it clusters the whole partition and ignores ORDER BY and the frame. The
// arguments go into the window's collection; the first row of a partition that is evaluated fits it, as
// emeans_agg fits a group (src/functions/emeans_agg.c, emeans_points.h), and every row then gets the label
// emeans_assign_point (or emeans_assign) gives it with those centroids. Partitions are shared between the
// threads evaluating one hash group, behind a lock, so none is fitted twice.
//
// Not an aggregate with window callbacks: DuckDB's window self-join optimizer turns a partition-only
// window aggregate into a GROUP BY and a join without checking that the aggregate can aggregate.

#include "duckdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/function/window/window_collection.hpp"
#include "duckdb/function/window/window_executor.hpp"
#include "duckdb/function/window/window_shared_expressions.hpp"
#include "duckdb/function/window_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <map>
#include <mutex>

extern "C" {
#include "emeans_points.h"
#include "recipe.h"
}

namespace duckdb {

namespace {

// One partition's labels: 0 for rows that gave no point, which come out NULL.
struct PartitionLabels {
	std::mutex lock;
	bool done = false;
	std::string error;
	vector<int32_t> labels;
};

class ClusterIdGlobalState : public WindowExecutorGlobalState {
public:
	ClusterIdGlobalState(ClientContext &client, const WindowExecutor &executor, const idx_t payload_count,
	                     const ValidityMask &partition_mask, const ValidityMask &order_mask)
	    : WindowExecutorGlobalState(client, executor, payload_count, partition_mask, order_mask) {
	}

	shared_ptr<PartitionLabels> Get(idx_t begin) {
		std::lock_guard<std::mutex> guard(lock);
		auto &entry = partitions[begin];
		if (!entry) {
			entry = make_shared_ptr<PartitionLabels>();
		}
		return entry;
	}

	std::mutex lock;
	//! The hash group's arguments, from the first thread to finalize
	optional_ptr<WindowCollection> collection;
	//! By the partition's first row
	std::map<idx_t, shared_ptr<PartitionLabels>> partitions;
};

// A partition's points, as one emeans_agg state, freed on every path.
class PartitionPoints {
public:
	PartitionPoints() : state(emeans_points_size()) {
		emeans_points_init(state.data());
	}
	~PartitionPoints() {
		emeans_points_free(state.data());
	}
	void *Get() {
		return state.data();
	}

private:
	vector<data_t> state;
};

void Check(int rc, const char *err) {
	if (rc) {
		throw InvalidInputException("emeans_cluster_id: %s", err);
	}
}

// Rows [begin, end) of the arguments, chunk by chunk, into points; kept says which rows gave one.
void AddRows(ColumnDataCollection &inputs, const vector<column_t> &columns, idx_t begin, idx_t end, emeans_mem *mem,
             PartitionPoints &points, vector<bool> &kept) {
	ColumnDataScanState scan;
	DataChunk page;
	inputs.InitializeScan(scan, columns);
	inputs.InitializeScanChunk(scan, page);
	auto flags = make_unsafe_uniq_array<bool>(end - begin);
	char err[EMEANS_ERR_LEN];
	for (idx_t row = begin; row < end;) {
		inputs.Seek(row, scan, page);
		page.Flatten();
		const idx_t base = scan.current_row_index;
		const idx_t last = MinValue(end, scan.next_row_index);
		Check(emeans_points_add(points.Get(), reinterpret_cast<duckdb_data_chunk>(&page), row - base, last - base, mem,
		                        flags.get() + (row - begin), err),
		      err);
		row = last;
	}
	kept.assign(flags.get(), flags.get() + (end - begin));
}

void FitPartition(ClusterIdGlobalState &gstate, idx_t begin, idx_t end, PartitionLabels &out) {
	auto mem = reinterpret_cast<emeans_mem *>(&BufferManager::GetBufferManager(gstate.client));
	PartitionPoints points;
	vector<bool> kept;
	AddRows(*gstate.collection->inputs, gstate.executor.child_idx, begin, end, mem, points, kept);
	vector<int32_t> labels(NumericCast<idx_t>(emeans_points_count(points.Get())));
	char err[EMEANS_ERR_LEN];
	if (!labels.empty()) {
		Check(emeans_points_label(points.Get(), mem, &gstate.client, labels.data(), err), err);
	}
	out.labels.assign(kept.size(), 0);
	for (idx_t i = 0, j = 0; i < kept.size(); i++) {
		out.labels[i] = kept[i] ? labels[j++] : 0;
	}
}

// The partition's labels, fitting it if no thread has yet. A failure is kept and thrown to every row asking.
PartitionLabels &Labels(ClusterIdGlobalState &gstate, idx_t begin, idx_t end) {
	auto entry = gstate.Get(begin);
	std::lock_guard<std::mutex> guard(entry->lock);
	if (!entry->done) {
		try {
			FitPartition(gstate, begin, end, *entry);
		} catch (std::exception &e) {
			entry->error = ErrorData(e).RawMessage();
		}
		entry->done = true;
	}
	if (!entry->error.empty()) {
		throw InvalidInputException(entry->error);
	}
	return *entry;
}

void GetBounds(WindowBoundsSet &required, const BoundWindowExpression &) {
	required.insert(PARTITION_BEGIN);
	required.insert(PARTITION_END);
}

// Every argument is read across the partition, so all of them go into the collection.
void GetSharing(WindowExecutor &executor, WindowSharedExpressions &shared) {
	for (auto &child : executor.wexpr.GetChildren()) {
		executor.child_idx.emplace_back(shared.RegisterCollection(child, false));
	}
}

unique_ptr<GlobalSinkState> GetGlobal(ClientContext &client, const WindowExecutor &executor, const idx_t payload_count,
                                      const ValidityMask &partition_mask, const ValidityMask &order_mask) {
	return make_uniq<ClusterIdGlobalState>(client, executor, payload_count, partition_mask, order_mask);
}

void Finalize(ExecutionContext &, optional_ptr<WindowCollection> collection, OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<ClusterIdGlobalState>();
	std::lock_guard<std::mutex> guard(gstate.lock);
	gstate.collection = collection;
}

void Evaluate(ExecutionContext &, DataChunk &, DataChunk &bounds, Vector &result, idx_t row_idx,
              OperatorSinkInput &sink) {
	auto &gstate = sink.global_state.Cast<ClusterIdGlobalState>();
	auto partition_begin = FlatVector::GetData<const idx_t>(bounds.data[PARTITION_BEGIN]);
	auto partition_end = FlatVector::GetData<const idx_t>(bounds.data[PARTITION_END]);
	auto rdata = FlatVector::GetDataMutable<int32_t>(result);
	auto &rmask = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < bounds.size(); i++, row_idx++) {
		const int32_t label = Labels(gstate, partition_begin[i], partition_end[i]).labels[row_idx - partition_begin[i]];
		rdata[i] = label;
		if (label == 0) {
			rmask.SetInvalid(i);
		}
	}
}

WindowFunction Overload(const LogicalType &points, idx_t n_args) {
	vector<LogicalType> args {points, LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::BIGINT};
	args.resize(n_args);
	return WindowFunction("emeans_cluster_id", args, LogicalType::INTEGER, ExpressionType::WINDOW_FUNCTION, nullptr,
	                      GetBounds, GetSharing, GetGlobal, nullptr, nullptr, Finalize, Evaluate);
}

} // namespace

// As emeans_agg: points arrive as ANY so a GEOMETRY keeps its CRS, with FLOAT[] and DOUBLE[] for vectors.
void RegisterClusterId(ExtensionLoader &loader) {
	WindowFunctionSet set("emeans_cluster_id");
	const vector<LogicalType> point_types {LogicalType::ANY, LogicalType::LIST(LogicalType::FLOAT),
	                                       LogicalType::LIST(LogicalType::DOUBLE)};
	for (auto &points : point_types) {
		for (idx_t n_args = 2; n_args <= 4; n_args++) {
			set.AddFunction(Overload(points, n_args));
		}
	}
	loader.RegisterFunction(set);
}

} // namespace duckdb
