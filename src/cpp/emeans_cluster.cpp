// emeans(x, k [, nepochs [, seed]]) and emeans_spherical(...): registered in C++ for their bind, which picks
// the centroids' type. The per-row work is the C code in src/functions/emeans_cluster.c.
//
// When x is a list of fixed-size arrays, FLOAT[384][] say, the centroids come back as FLOAT[384][], the
// type the vectors had; otherwise FLOAT[][]. The C code writes FLOAT[][] in either case, and an array result
// is cast from it. The bind also takes emeans_max_work, emeans_threads, the query and the buffer manager.

#include "duckdb.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

extern "C" {
#include "emeans_functions.h"
#include "emeans_query.h"
#include "recipe.h"
}

namespace duckdb {

namespace {

LogicalType ResultType(const LogicalType &centroids) {
	child_list_t<LogicalType> members {{"centroids", LogicalType::LIST(centroids)},
	                                   {"labels", LogicalType::LIST(LogicalType::INTEGER)},
	                                   {"obj", LogicalType::LIST(LogicalType::DOUBLE)},
	                                   {"batch_size", LogicalType::BIGINT},
	                                   {"alpha0", LogicalType::DOUBLE},
	                                   {"n_splits", LogicalType::BIGINT}};
	return LogicalType::STRUCT(members);
}

// The result as the C code writes it.
const LogicalType &ListResultType() {
	static const LogicalType type = ResultType(LogicalType::LIST(LogicalType::FLOAT));
	return type;
}

struct ClusterBindData : public FunctionData {
	ClusterBindData(const emeans_settings &settings, emeans_mem *mem, bool cast_result)
	    : settings(settings), mem(mem), cast_result(cast_result) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ClusterBindData>(settings, mem, cast_result);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<ClusterBindData>();
		return settings.max_work == other.settings.max_work && settings.threads == other.settings.threads &&
		       settings.query == other.settings.query && mem == other.mem && cast_result == other.cast_result;
	}

	emeans_settings settings;
	emeans_mem *mem;
	bool cast_result; // centroids are FLOAT[d][]: write FLOAT[][] and cast
};

// The vectors of x: their element type and, for fixed-size arrays, their size d (0 for lists). The element
// is SQLNULL when x or its vectors are NULL literals, INVALID when x is not a list of vectors.
LogicalType VectorElement(const LogicalType &x, idx_t &d) {
	d = 0;
	if (x.id() == LogicalTypeId::SQLNULL) {
		return LogicalType::SQLNULL;
	}
	auto vector = x.id() == LogicalTypeId::LIST ? ListType::GetChildType(x) : LogicalType::INVALID;
	switch (vector.id()) {
	case LogicalTypeId::SQLNULL:
		return LogicalType::SQLNULL;
	case LogicalTypeId::ARRAY:
		d = ArrayType::GetSize(vector);
		return ArrayType::GetChildType(vector);
	case LogicalTypeId::LIST:
		return ListType::GetChildType(vector);
	default:
		return LogicalType::INVALID;
	}
}

// x is LIST(ANY) in the signature, so the bind sees the vectors' own type: a list or a fixed-size array of
// numbers. It casts x to what the C code reads, a list of FLOAT or DOUBLE lists, and returns the arrays'
// size d (0 for lists).
idx_t CastVectors(ClientContext &context, unique_ptr<Expression> &x, LogicalType &target, const char *name) {
	auto type = x->GetReturnType();
	idx_t d;
	auto element = VectorElement(type, d);
	if (!element.IsNumeric() && element.id() != LogicalTypeId::SQLNULL) {
		throw BinderException("%s: x must be a list of vectors, lists or arrays of numbers, not %s", name,
		                      type.ToString());
	}
	target = LogicalType::LIST(LogicalType::LIST(element.id() == LogicalTypeId::FLOAT ? LogicalType::FLOAT
	                                                                                    : LogicalType::DOUBLE));
	if (type != target) {
		x = BoundCastExpression::AddCastToType(context, std::move(x), target);
	}
	return d;
}

unique_ptr<FunctionData> Bind(BindScalarFunctionInput &input) {
	auto &context = input.GetClientContext();
	auto &function = input.GetBoundFunction();
	LogicalType target;
	const idx_t d = CastVectors(context, input.GetArguments()[0], target, function.GetName().c_str());
	function.GetArguments()[0] = target;
	if (d > 0) {
		input.GetBoundFunction().SetReturnType(ResultType(LogicalType::ARRAY(LogicalType::FLOAT, d)));
	}
	CClientContextWrapper wrapper(context);
	emeans_settings settings;
	emeans_settings_read(reinterpret_cast<duckdb_client_context>(&wrapper), &settings);
	return make_uniq<ClusterBindData>(settings, reinterpret_cast<emeans_mem *>(&BufferManager::GetBufferManager(context)),
	                                  d > 0);
}

void Run(DataChunk &args, ExpressionState &state, Vector &result, int spherical) {
	auto &info = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<ClusterBindData>();
	args.Flatten();
	Vector list_result(ListResultType(), args.size());
	Vector &out = info.cast_result ? list_result : result;
	char err[EMEANS_ERR_LEN];
	const int rc = emeans_cluster_chunk(reinterpret_cast<duckdb_data_chunk>(&args), reinterpret_cast<duckdb_vector>(&out),
	                                    &info.settings, info.mem, spherical, err);
	if (rc && emeans_query_interrupted(info.settings.query)) {
		throw InterruptException();
	}
	if (rc) {
		throw InvalidInputException("%s: %s", spherical ? "emeans_spherical" : "emeans", err);
	}
	if (info.cast_result) {
		VectorOperations::DefaultCast(list_result, result, args.size());
	}
}

void Cluster(DataChunk &args, ExpressionState &state, Vector &result) {
	Run(args, state, result, 0);
}

void Spherical(DataChunk &args, ExpressionState &state, Vector &result) {
	Run(args, state, result, 1);
}

// name(x, k), name(x, k, nepochs) and name(x, k, nepochs, seed).
ScalarFunctionSet Set(const char *name, scalar_function_t fn) {
	ScalarFunctionSet set(name);
	vector<LogicalType> args {LogicalType::LIST(LogicalType::ANY), LogicalType::BIGINT, LogicalType::DOUBLE,
	                          LogicalType::BIGINT};
	for (idx_t n_args = 2; n_args <= 4; n_args++) {
		ScalarFunction f(name, vector<LogicalType>(args.begin(), args.begin() + NumericCast<int64_t>(n_args)),
		                 ListResultType(), fn, Bind);
		// NULL nepochs and seed mean the defaults, so the C code sees the NULLs.
		f.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		f.SetFallible();
		set.AddFunction(f);
	}
	return set;
}

} // namespace

void RegisterCluster(ExtensionLoader &loader) {
	loader.RegisterFunction(Set("emeans", Cluster));
	loader.RegisterFunction(Set("emeans_spherical", Spherical));
}

} // namespace duckdb
