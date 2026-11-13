// emeans_stream(query, k, nepochs := 5, seed := 1234, batch_size := 0) and emeans_stream_spherical(...): e-means over
// data that need not fit in memory. The query returns one column of vectors; the function runs it again, on a
// connection of its own, once per pass, ordered by a hash of the vector, its copy number among identical vectors, the
// seed and the pass (PassQuery), and feeds the rows to the core's stream (emeans_stream_begin in emeans.h) as they
// arrive. It holds one batch, the centroids and the workspace: O(B d + k d) memory where emeans() needs O(n d). B
// comes from the batch law and grows with n, k and nepochs (524,288 rows for 10M x 768, k = 1024, 5 epochs: 1.6 GB
// of FLOAT); batch_size > 0 sets it instead. The start is k distinct vectors drawn in proportion to their rows
// (StartQuery).
//
// Every query runs in one read transaction of that connection, so all passes read one snapshot; it is a separate
// connection, so it sees committed data only (not the caller's temporary tables or uncommitted rows). One row per
// cluster: cluster INTEGER (from 1), centroid FLOAT[d], n BIGINT (its rows in the closing pass) and run, the same in
// every row: STRUCT(n BIGINT, batch_size BIGINT, alpha0 DOUBLE, n_splits BIGINT, obj DOUBLE[]).

#include "duckdb.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/query_result_stream.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "emeans_query.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>

extern "C" {
#include "emeans.h"
#include "emeans_functions.h"
#include "emeans_mem.h"
#include "emeans_query.h"
}

namespace duckdb {

namespace {

constexpr int64_t INIT_PASS = -1; // the hash salt of the start's order

struct StreamBindData : public TableFunctionData {
	string name;
	string source; // the vectors, non-NULL, as a subquery with one column v
	string search_path;
	int64_t k = 0;
	idx_t d = 0;
	double nepochs = 5;
	uint32_t seed = 1234;
	int64_t batch_size = 0; // > 0 overrides the batch law, and so bounds the batch buffer (B x d floats)
	bool spherical = false;
	emeans_settings settings {};
};

struct StreamState : public GlobalTableFunctionState {
	bool fitted = false;
	idx_t emitted = 0;
	int64_t n = 0;
	vector<float> centroids;
	vector<int64_t> counts;
	vector<double> obj;
	emeans_output out {};
};

// While this thread blocks in Fetch, worker threads may sort a whole pass, which on spilled data takes minutes; the
// checks in NextChunk run only between chunks. The watcher asks every WATCH_MS whether the caller was interrupted or
// passed its max_execution_time and then interrupts the nested connection (Connection::Interrupt is safe from any
// thread), so the wait ends at once. WASM has no threads; there the checks between tasks are all there is.
class InterruptWatcher {
public:
	InterruptWatcher(ClientContext &caller, Connection &con) {
#ifndef __EMSCRIPTEN__
		thread = std::thread([this, &caller, &con]() { Watch(caller, con); });
#endif
	}
	~InterruptWatcher() {
#ifndef __EMSCRIPTEN__
		{
			std::lock_guard<std::mutex> guard(lock);
			done = true;
		}
		wake.notify_one();
		thread.join();
#endif
	}

private:
	static constexpr int WATCH_MS = 50;

	void Watch(ClientContext &caller, Connection &con) {
		std::unique_lock<std::mutex> guard(lock);
		while (!wake.wait_for(guard, std::chrono::milliseconds(WATCH_MS), [this]() { return done; })) {
			if (emeans_query_interrupted(&caller)) {
				con.Interrupt();
				return;
			}
		}
	}

	std::mutex lock;
	std::condition_variable wake;
	bool done = false;
#ifndef __EMSCRIPTEN__
	std::thread thread;
#endif
};

// One read-only transaction for every query of the run: one snapshot, and no writes whatever the query holds.
void ReadOnlyTransaction(Connection &con) {
	auto begin = con.Query("BEGIN TRANSACTION READ ONLY");
	if (begin->HasError()) {
		begin->ThrowError();
	}
}

// The connection the queries run on: the caller's database and default schema, in one read-only transaction from
// the first query on, the bind's included (a side effect such as nextval must not escape the caller's context).
unique_ptr<Connection> OpenConnection(ClientContext &context, const string &search_path) {
	auto con = make_uniq<Connection>(DatabaseInstance::GetDatabase(context));
	auto use = con->Query("USE " + search_path);
	if (use->HasError()) {
		use->ThrowError();
	}
	ReadOnlyTransaction(*con);
	return con;
}

// The caller's default catalog and schema as a USE target, each part quoted as an SQL identifier.
string SearchPath(ClientContext &context) {
	auto &entry = ClientData::Get(context).catalog_search_path->GetDefault();
	auto &catalog = entry.GetCatalog().GetIdentifierName();
	auto schema = KeywordHelper::WriteQuotedAndEscaped(entry.GetSchema().GetIdentifierName(), '"');
	// An empty catalog is the default database.
	return catalog.empty() ? schema : KeywordHelper::WriteQuotedAndEscaped(catalog, '"') + "." + schema;
}

// Stops for the caller: an interrupt, or its max_execution_time (the nested query has no deadline of its own).
void CheckCaller(ClientContext &caller, Connection &con) {
	if (emeans_query_interrupted(&caller)) {
		con.Interrupt();
		ThrowInterrupted(&caller);
	}
}

// The stream's error, as the caller's interrupt when the watcher interrupted the nested query for it.
void ThrowIfFailed(QueryResultStream<> &stream, ClientContext &caller, Connection &con) {
	if (stream.HasError()) {
		CheckCaller(caller, con);
		stream.GetErrorObject().Throw();
	}
}

// The next chunk of a streamed query, nullptr at its end. While the query is not ready this thread runs its tasks
// one at a time and the caller's interrupt stops it between them; once it waits on other threads or on its consumer,
// Fetch refills the result (and blocks until a chunk is there, the InterruptWatcher ending the wait if need be).
unique_ptr<DataChunk> NextChunk(QueryResultStream<> &stream, ClientContext &caller, Connection &con) {
	for (;;) {
		CheckCaller(caller, con);
		unique_ptr<DataChunk> chunk;
		auto state = stream.TryFetch(chunk);
		if (state == QueryResultState::NOT_READY) {
			stream.ExecuteTask();
			continue;
		}
		if (!chunk && !IsTerminal(state)) {
			chunk = stream.Fetch();
		}
		ThrowIfFailed(stream, caller, con);
		if (chunk) {
			chunk->Flatten();
		}
		return chunk;
	}
}

unique_ptr<QueryResultStream<>> Submit(Connection &con, const string &sql) {
	auto result = con.Submit(sql);
	if (result->HasError()) {
		result->ThrowError();
	}
	return make_uniq<QueryResultStream<>>(std::move(result));
}

// The one value of a one-row query.
Value QueryValue(Connection &con, ClientContext &caller, const string &sql) {
	auto stream = Submit(con, sql);
	auto chunk = NextChunk(*stream, caller, con);
	return chunk && chunk->size() > 0 ? chunk->GetValue(0, 0) : Value();
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//

// The vectors' size: the array's, or the first list's.
idx_t VectorSize(Connection &con, ClientContext &context, const StreamBindData &data, const LogicalType &type) {
	if (type.id() == LogicalTypeId::ARRAY) {
		return ArrayType::GetSize(type);
	}
	auto first = QueryValue(con, context, "SELECT len(v) FROM " + data.source + " LIMIT 1");
	if (first.IsNull()) {
		throw InvalidInputException("%s: the query returned no vectors", data.name);
	}
	auto d = first.GetValue<int64_t>();
	if (d < 1) {
		throw InvalidInputException("%s: the vectors are empty", data.name);
	}
	return NumericCast<idx_t>(d);
}

const LogicalType &VectorElement(const LogicalType &type) {
	static const LogicalType invalid = LogicalType::INVALID;
	if (type.id() == LogicalTypeId::ARRAY) {
		return ArrayType::GetChildType(type);
	}
	return type.id() == LogicalTypeId::LIST ? ListType::GetChildType(type) : invalid;
}

LogicalType SourceType(Connection &con, const StreamBindData &data, const string &query) {
	auto prepared = con.Prepare("SELECT * FROM (" + query + "\n)");
	if (prepared->HasError()) {
		throw BinderException("%s: %s", data.name, prepared->GetError());
	}
	auto &types = prepared->GetTypes();
	if (types.size() != 1 || !VectorElement(types[0]).IsNumeric()) {
		throw BinderException("%s: the query must return one column of vectors, lists or arrays of numbers",
		                      data.name);
	}
	return types[0];
}

// The seed, in emeans()'s range (src/functions/emeans_cluster.c), so different seeds never wrap onto one.
uint32_t Seed(int64_t seed, const StreamBindData &data) {
	if (seed < 0 || seed > static_cast<int64_t>(NumericLimits<uint32_t>::Maximum())) {
		throw BinderException("%s: seed must be in [0, 4294967295], got %lld", data.name, seed);
	}
	return static_cast<uint32_t>(seed);
}

// One named argument; NULL means its default.
void ReadNamed(const string &name, const Value &value, StreamBindData &data) {
	if (value.IsNull()) {
		return;
	}
	if (name == "nepochs") {
		data.nepochs = value.GetValue<double>();
	} else if (name == "seed") {
		data.seed = Seed(value.GetValue<int64_t>(), data);
	} else if (name == "batch_size") {
		data.batch_size = value.GetValue<int64_t>();
	}
}

void ReadArguments(TableFunctionBindInput &input, StreamBindData &data) {
	if (input.inputs[0].IsNull() || input.inputs[1].IsNull()) {
		throw BinderException("%s: query and k must not be NULL", data.name);
	}
	data.k = input.inputs[1].GetValue<int64_t>();
	for (auto &entry : input.named_parameters) {
		ReadNamed(entry.first.GetIdentifierName(), entry.second, data);
	}
	if (data.k < 1) {
		throw BinderException("%s: k must be at least 1", data.name);
	}
	if (data.batch_size < 0) {
		throw BinderException("%s: batch_size must not be negative (0: the batch law)", data.name);
	}
}

// The query is spliced into subqueries (the source, the passes), so it must be one SELECT on its own: a string that
// parses alone as a single SELECT has balanced parentheses, quotes and block comments, and cannot close the
// subquery around it or add a statement after it (a line comment at its end is closed by the newline the splices
// add). The nested connection's transaction is read-only besides.
void CheckSingleSelect(ClientContext &context, const string &query, const StreamBindData &data) {
	Parser parser(context);
	try {
		parser.ParseQuery(query);
	} catch (std::exception &e) {
		throw BinderException("%s: %s", data.name, ErrorData(e).RawMessage());
	}
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw BinderException("%s: the query must be a single SELECT statement", data.name);
	}
}

// The query without the semicolons and whitespace it may end with, so it can go in a subquery.
string Unterminated(string query) {
	while (!query.empty() && (query.back() == ';' || StringUtil::CharacterIsSpace(query.back()))) {
		query.pop_back();
	}
	return query;
}

LogicalType RunType() {
	child_list_t<LogicalType> members {{"n", LogicalType::BIGINT},
	                                   {"batch_size", LogicalType::BIGINT},
	                                   {"alpha0", LogicalType::DOUBLE},
	                                   {"n_splits", LogicalType::BIGINT},
	                                   {"obj", LogicalType::LIST(LogicalType::DOUBLE)}};
	return LogicalType::STRUCT(members);
}

unique_ptr<FunctionData> Bind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types,
                              vector<Identifier> &names, bool spherical) {
	auto data = make_uniq<StreamBindData>();
	data->spherical = spherical;
	data->name = spherical ? "emeans_stream_spherical" : "emeans_stream";
	ReadArguments(input, *data);
	auto query = Unterminated(input.inputs[0].GetValue<string>());
	CheckSingleSelect(context, query, *data);
	data->search_path = SearchPath(context);
	// The newline ends a line comment the query may end with.
	data->source = "(SELECT v FROM (" + query + "\n) AS emeans_query(v) WHERE v IS NOT NULL)";
	auto con = OpenConnection(context, data->search_path);
	InterruptWatcher watcher(context, *con); // VectorSize runs the query, which may sort or aggregate first
	data->d = VectorSize(*con, context, *data, SourceType(*con, *data, query));
	CClientContextWrapper wrapper(context);
	emeans_settings_read(reinterpret_cast<duckdb_client_context>(&wrapper), &data->settings);
	return_types = {LogicalType::INTEGER, LogicalType::ARRAY(LogicalType::FLOAT, data->d), LogicalType::BIGINT,
	                RunType()};
	names = {Identifier("cluster"), Identifier("centroid"), Identifier("n"), Identifier("run")};
	return std::move(data);
}

unique_ptr<FunctionData> BindStream(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<Identifier> &names) {
	return Bind(context, input, return_types, names, false);
}

unique_ptr<FunctionData> BindSpherical(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<Identifier> &names) {
	return Bind(context, input, return_types, names, true);
}

//===--------------------------------------------------------------------===//
// The fit
//===--------------------------------------------------------------------===//

struct StreamDeleter {
	void operator()(emeans_stream *s) const {
		emeans_stream_free(s);
	}
};
using StreamPtr = unique_ptr<emeans_stream, StreamDeleter>;

// The distinct vectors and how many rows hold each: a hash aggregate, which spills to temp_directory.
string Copies(const StreamBindData &data) {
	return "(SELECT v, count(*) AS emeans_copies FROM " + data.source + " GROUP BY v)";
}

// The vectors as FLOAT[d] (a list of another length fails the cast) in the order of pass `salt`: a full sort, which
// spills to temp_directory (a LIMIT would make it a top-N, which holds its rows in memory). Ordered by hash(v)
// alone, the copies of a repeated vector would sit together and fill batches; copy j < copies gets the key
// hash(v, j) instead. That only tells identical rows apart, so what the core sees stays deterministic. When no
// vector repeats, every row is copy 0 and the aggregate is skipped: the same keys, so the same result.
string PassQuery(const StreamBindData &data, bool repeated, int64_t salt) {
	auto rows = repeated ? "(SELECT v, unnest(range(emeans_copies)) AS emeans_copy FROM " + Copies(data) + ")"
	                     : "(SELECT v, 0::BIGINT AS emeans_copy FROM " + data.source + ")";
	return StringUtil::Format("SELECT v::FLOAT[%llu] FROM %s ORDER BY hash(v, emeans_copy, %llu, %lld)", data.d, rows,
	                          static_cast<uint64_t>(data.seed), salt);
}

// The start: k distinct vectors drawn without replacement with probability proportional to their rows, the
// reference's draw of k rows with repeated rows drawn once (Efraimidis-Spirakis keys ln(u) / copies, u from the
// hash). With no repeated vectors it is a uniform draw of k rows. A top-N of k rows.
string StartQuery(const StreamBindData &data, bool repeated) {
	auto rows = repeated ? Copies(data) : "(SELECT v, 1 AS emeans_copies FROM " + data.source + ")";
	return StringUtil::Format("SELECT v::FLOAT[%llu] FROM %s ORDER BY ln((hash(v, %llu, %lld)::DOUBLE + 0.5) / "
	                          "18446744073709551616.0::DOUBLE) / emeans_copies DESC LIMIT %lld",
	                          data.d, rows, static_cast<uint64_t>(data.seed), INIT_PASS, data.k);
}

// The float rows of a chunk of FLOAT[d], which the cast leaves flat; a NULL element fails.
const float *ChunkRows(const DataChunk &chunk, const StreamBindData &data) {
	auto &child = ArrayVector::GetChild(chunk.data[0]);
	if (!FlatVector::Validity(child).CheckAllValid(chunk.size() * data.d)) {
		throw InvalidInputException("%s: a vector contains NULL", data.name);
	}
	return FlatVector::GetData<float>(child);
}

[[noreturn]] void ThrowFit(int rc, const StreamBindData &data, const char *err) {
	if (rc == EMEANS_INTERRUPTED) {
		ThrowInterrupted(data.settings.query);
	}
	if (rc == EMEANS_BUDGET_EXCEEDED) {
		throw InvalidInputException("%s: %s; raise the limit with SET emeans_max_work (0 = no limit)", data.name, err);
	}
	throw InvalidInputException("%s: %s", data.name, err);
}

void Check(int rc, const StreamBindData &data, const char *err) {
	if (rc != EMEANS_OK) {
		ThrowFit(rc, data, err);
	}
}

// The start (StartQuery). With fewer than k distinct vectors the rest repeat them: there are no other rows to take.
vector<float> Start(Connection &con, ClientContext &context, const StreamBindData &data, bool repeated) {
	vector<float> init;
	auto stream = Submit(con, StartQuery(data, repeated));
	while (auto chunk = NextChunk(*stream, context, con)) {
		auto rows = ChunkRows(*chunk, data);
		init.insert(init.end(), rows, rows + chunk->size() * data.d);
	}
	// The start is rows of x, so a NaN in it is one in x, as the passes would find.
	for (auto v : init) {
		if (!std::isfinite(v)) {
			throw InvalidInputException("%s: x contains NaN or infinity", data.name);
		}
	}
	auto distinct = init.size();
	init.resize(NumericCast<idx_t>(data.k) * data.d);
	for (idx_t t = distinct; t < init.size(); t++) {
		init[t] = init[t % distinct];
	}
	return init;
}

void RunPass(Connection &con, ClientContext &context, const StreamBindData &data, bool repeated,
             emeans_stream *fit) {
	char err[EMEANS_ERR_LEN];
	auto stream = Submit(con, PassQuery(data, repeated, emeans_stream_pass(fit)));
	// A fractional pass stops reading once it has its rows; closing the stream ends the query.
	while (emeans_stream_wanted(fit) > 0) {
		auto chunk = NextChunk(*stream, context, con);
		if (!chunk) {
			break;
		}
		Check(emeans_stream_add(fit, ChunkRows(*chunk, data), NumericCast<int64_t>(chunk->size()), err), data, err);
	}
	Check(emeans_stream_end_pass(fit, err), data, err);
}

emeans_params Params(ClientContext &context, const StreamBindData &data, const float *init) {
	emeans_params p;
	emeans_default_params(&p);
	p.nepochs = data.nepochs;
	p.seed = data.seed;
	p.batch_size = data.batch_size;
	p.spherical = data.spherical;
	p.init = init;
	p.max_work = data.settings.max_work;
	p.threads = data.settings.threads;
	p.allocator = emeans_mem_allocator(reinterpret_cast<emeans_mem *>(&BufferManager::GetBufferManager(context)));
	emeans_query_attach(&p, &context);
	return p;
}

// The output buffers, sized by the recipe (which also checks k against n before anything is read).
void Prepare(StreamState &state, const StreamBindData &data, int64_t n) {
	if (n == 0) {
		throw InvalidInputException("%s: the query returned no vectors", data.name);
	}
	char err[EMEANS_ERR_LEN];
	emeans_recipe recipe;
	if (emeans_resolve_recipe(n, data.k, NumericCast<int64_t>(data.d), data.nepochs, EMEANS_LAW_AUTO, data.batch_size, 0, &recipe,
	                          err)) {
		ThrowFit(EMEANS_INVALID, data, err);
	}
	// emeans_max_work, before the start's scan: the core checks it again, with the same rows x k x d.
	auto work = emeans_total_rows(n, data.nepochs) * static_cast<double>(data.k) * static_cast<double>(data.d);
	if (data.settings.max_work > 0 && work > data.settings.max_work) {
		snprintf(err, EMEANS_ERR_LEN, "work limit exceeded: rows x k x d over all passes is %.4g, the limit %.4g", work,
		         data.settings.max_work);
		ThrowFit(EMEANS_BUDGET_EXCEEDED, data, err);
	}
	state.n = n;
	state.centroids.resize(NumericCast<idx_t>(data.k) * data.d);
	state.counts.resize(NumericCast<idx_t>(data.k));
	state.obj.resize(NumericCast<idx_t>(recipe.n_passes));
	state.out = emeans_output {state.centroids.data(), nullptr, state.obj.data(), 0, recipe};
}

void Fit(ClientContext &context, const StreamBindData &data, StreamState &state) {
	auto con = OpenConnection(context, data.search_path);
	InterruptWatcher watcher(context, *con);
	// n and the distinct vectors from a GROUP BY, which spills (count(DISTINCT v) does not).
	auto counts = Submit(*con, "SELECT coalesce(sum(c), 0)::BIGINT, count(*) FROM (SELECT count(*) AS c FROM " + data.source +
	                               " GROUP BY v)");
	auto chunk = NextChunk(*counts, context, *con);
	auto n = chunk->GetValue(0, 0).GetValue<int64_t>();
	auto repeated = chunk->GetValue(1, 0).GetValue<int64_t>() < n;
	counts.reset();
	Prepare(state, data, n);
	auto init = Start(*con, context, data, repeated);
	auto p = Params(context, data, init.data());
	char err[EMEANS_ERR_LEN];
	emeans_stream *raw = nullptr;
	Check(emeans_stream_begin(state.n, NumericCast<int64_t>(data.d), data.k, &p, &state.out, &raw, err), data, err);
	StreamPtr fit(raw);
	while (emeans_stream_wanted(fit.get()) >= 0) {
		RunPass(*con, context, data, repeated, fit.get());
	}
	Check(emeans_stream_finish(fit.get(), state.counts.data(), err), data, err);
	con->Commit();
}

//===--------------------------------------------------------------------===//
// Output
//===--------------------------------------------------------------------===//

Value RunValue(const StreamState &state) {
	vector<Value> obj;
	for (auto v : state.obj) {
		obj.push_back(Value::DOUBLE(v));
	}
	child_list_t<Value> members {{"n", Value::BIGINT(state.n)},
	                             {"batch_size", Value::BIGINT(state.out.recipe.batch_size)},
	                             {"alpha0", Value::DOUBLE(state.out.recipe.alpha0)},
	                             {"n_splits", Value::BIGINT(state.out.n_splits)},
	                             {"obj", Value::LIST(LogicalType::DOUBLE, std::move(obj))}};
	return Value::STRUCT(std::move(members));
}

void Emit(const StreamBindData &data, StreamState &state, DataChunk &output) {
	auto count = MinValue<idx_t>(NumericCast<idx_t>(data.k) - state.emitted, STANDARD_VECTOR_SIZE);
	auto cluster = FlatVector::GetDataMutable<int32_t>(output.data[0]);
	auto centroid = FlatVector::GetDataMutable<float>(ArrayVector::GetChildMutable(output.data[1]));
	auto n = FlatVector::GetDataMutable<int64_t>(output.data[2]);
	auto run = RunValue(state);
	for (idx_t i = 0; i < count; i++) {
		auto j = state.emitted + i;
		cluster[i] = NumericCast<int32_t>(j + 1);
		memcpy(centroid + i * data.d, state.centroids.data() + j * data.d, data.d * sizeof(float));
		n[i] = state.counts[j];
		output.data[3].SetValue(i, run);
	}
	state.emitted += count;
	output.SetChildCardinality(count);
}

unique_ptr<GlobalTableFunctionState> Init(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<StreamState>();
}

void Execute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &data = input.bind_data->Cast<StreamBindData>();
	auto &state = input.global_state->Cast<StreamState>();
	if (!state.fitted) {
		Fit(context, data, state);
		state.fitted = true;
	}
	Emit(data, state, output);
}

TableFunction StreamFunction(const char *name, table_function_bind_t bind) {
	TableFunction f(name,
	                FunctionSignature()
	                    .AddPositionalOnly("query", LogicalType::VARCHAR)
	                    .AddPositionalOnly("k", LogicalType::BIGINT)
	                    .AddKeywordOnly("nepochs", LogicalType::DOUBLE, Value::DOUBLE(5.0))
	                    .AddKeywordOnly("seed", LogicalType::BIGINT, Value::BIGINT(1234))
	                    .AddKeywordOnly("batch_size", LogicalType::BIGINT, Value::BIGINT(0)),
	                Execute, bind, Init);
	return f;
}

} // namespace

void RegisterStream(ExtensionLoader &loader) {
	loader.RegisterFunction(StreamFunction("emeans_stream", BindStream));
	loader.RegisterFunction(StreamFunction("emeans_stream_spherical", BindSpherical));
}

} // namespace duckdb
