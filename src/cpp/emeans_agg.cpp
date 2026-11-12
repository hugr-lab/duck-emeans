// emeans_agg(points, k [, nepochs [, seed]]): the aggregate, registered in C++ for its bind. The state and
// the work are emeans_agg's C code (src/functions/emeans_agg.c, emeans_agg.h); these callbacks pass it
// DuckDB's vectors as the C API's types, as DuckDB's own C API adapter does
// (src/main/capi/v1/aggregate_function-c.cpp), and turn its errors into exceptions.
//
// The bind checks the points' type, so a wrong argument fails before the query runs, and captures what
// the C API could not give an aggregate: emeans_max_work, emeans_threads, the buffer manager and the
// query, so a fit is limited, threaded and interruptible as emeans() is.

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

extern "C" {
#include "emeans_agg.h"
#include "emeans_functions.h"
#include "emeans_query.h"
#include "recipe.h"
}

namespace duckdb {

namespace {

struct AggBindData : public FunctionData {
	explicit AggBindData(const emeans_agg_context &context) : context(context) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AggBindData>(context);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AggBindData>();
		return context.mem == other.context.mem && context.query == other.context.query &&
		       context.max_work == other.context.max_work && context.threads == other.context.threads;
	}

	emeans_agg_context context;
};

const emeans_agg_context &Context(const AggregateInputData &input) {
	return input.bind_data->Cast<AggBindData>().context;
}

// An error from the C code; DuckDB's own InterruptException when the fit stopped for the query.
void Check(int rc, const char *err, const emeans_agg_context &context) {
	if (rc && emeans_query_interrupted(context.query)) {
		throw InterruptException();
	}
	if (rc) {
		throw InvalidInputException("emeans_agg: %s", err);
	}
}

// Points arrive as ANY so that a GEOMETRY keeps its CRS (a GEOMETRY parameter would cast
// GEOMETRY('OGC:CRS84') to plain GEOMETRY); what ANY lets through is checked here instead.
bool IsPoints(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::GEOMETRY:
	case LogicalTypeId::SQLNULL:
		return true;
	case LogicalTypeId::LIST:
		return ListType::GetChildType(type).IsNumeric() || ListType::GetChildType(type).id() == LogicalTypeId::SQLNULL;
	default:
		return false;
	}
}

// A fixed-size numeric array, such as FLOAT[384], is read as a list: the C reader takes lists.
void ArrayToList(ClientContext &context, unique_ptr<Expression> &points) {
	auto type = points->GetReturnType();
	if (type.id() == LogicalTypeId::ARRAY && ArrayType::GetChildType(type).IsNumeric()) {
		points = BoundCastExpression::AddCastToType(context, std::move(points),
		                                            LogicalType::LIST(ArrayType::GetChildType(type)));
	}
}

unique_ptr<FunctionData> Bind(BindAggregateFunctionInput &input) {
	auto &context = input.GetClientContext();
	ArrayToList(context, input.GetArguments()[0]);
	auto &type = input.GetArguments()[0]->GetReturnType();
	if (!IsPoints(type)) {
		throw BinderException("emeans_agg: points must be POINT geometries or lists of numbers, not %s",
		                      type.ToString());
	}
	CClientContextWrapper wrapper(context);
	emeans_settings settings;
	emeans_settings_read(reinterpret_cast<duckdb_client_context>(&wrapper), &settings);
	emeans_agg_context agg {reinterpret_cast<emeans_mem *>(&BufferManager::GetBufferManager(context)), settings.query,
	                        settings.max_work, settings.threads};
	return make_uniq<AggBindData>(agg);
}

idx_t StateSize(AggregateStateInput &) {
	return emeans_points_size();
}

void Initialize(AggregateStateInput &, data_ptr_t *states, idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		emeans_points_init(states[i]);
	}
}

// The state pointers of a state vector, flat.
void **States(Vector &state, idx_t count) {
	if (state.GetVectorType() != VectorType::FLAT_VECTOR) {
		state.Flatten(*FlatVector::IncrementalSelectionVector(), count);
	}
	return reinterpret_cast<void **>(FlatVector::GetDataMutableUnsafe<data_ptr_t>(state));
}

void Update(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state, idx_t count) {
	DataChunk chunk;
	for (idx_t c = 0; c < input_count; c++) {
		inputs[c].Flatten();
		chunk.data.emplace_back(Vector::Ref(inputs[c]));
	}
	chunk.SetCardinalityUnsafe(count); // only the count: the vectors are already there
	auto states = Vector::Ref(state);
	char err[EMEANS_ERR_LEN];
	auto &context = Context(aggr_input_data);
	Check(emeans_agg_update(reinterpret_cast<duckdb_data_chunk>(&chunk), States(states, count), &context, err), err,
	      context);
}

void Combine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
	char err[EMEANS_ERR_LEN];
	auto &context = Context(aggr_input_data);
	Check(emeans_agg_combine(States(source, count), States(target, count), count, &context, err), err, context);
}

void Finalize(Vector &state, AggregateFinalizeInputData &finalize_data, Vector &result, idx_t count, idx_t offset) {
	char err[EMEANS_ERR_LEN];
	auto &context = finalize_data.bind_data->Cast<AggBindData>().context;
	Check(emeans_agg_finalize(States(state, count), count, reinterpret_cast<duckdb_vector>(&result), offset, &context,
	                          err),
	      err, context);
}

void Destroy(Vector &state, AggregateInputData &, idx_t count) {
	auto states = States(state, count);
	for (idx_t i = 0; i < count; i++) {
		emeans_points_free(states[i]);
	}
}

LogicalType ResultType() {
	child_list_t<LogicalType> members {{"centroids", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
	                                   {"obj", LogicalType::LIST(LogicalType::DOUBLE)},
	                                   {"batch_size", LogicalType::BIGINT},
	                                   {"alpha0", LogicalType::DOUBLE},
	                                   {"n_splits", LogicalType::BIGINT},
	                                   {"n", LogicalType::BIGINT},
	                                   {"geographic", LogicalType::BOOLEAN}};
	return LogicalType::STRUCT(members);
}

AggregateFunction Overload(const LogicalType &points, idx_t n_args) {
	vector<LogicalType> args {points, LogicalType::BIGINT, LogicalType::DOUBLE, LogicalType::BIGINT};
	args.resize(n_args);
	// NULL nepochs and seed mean the defaults, so the C code sees the NULLs.
	return AggregateFunction("emeans_agg", args, ResultType(), StateSize, Initialize, Update, Combine, Finalize,
	                         FunctionNullHandling::SPECIAL_HANDLING, nullptr, Bind, Destroy);
}

} // namespace

void RegisterAgg(ExtensionLoader &loader) {
	AggregateFunctionSet set("emeans_agg");
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
