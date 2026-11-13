// emeans_query.h: inside DuckDB a duckdb_client_context is a CClientContextWrapper around the ClientContext
// (src/main/capi/v1/duckdb-c.cpp), which outlives the wrapper the bind destroys.

#include "duckdb.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "emeans_query.hpp"

extern "C" {
#include "emeans_query.h"
}

namespace {

// One part of a batch as a task of the executor.
class PartTask : public duckdb::BaseExecutorTask {
public:
	PartTask(duckdb::TaskExecutor &executor, emeans_task_fn fn, void *arg, int part, int parts)
	    : BaseExecutorTask(executor), fn(fn), arg(arg), part(part), parts(parts) {
	}
	void ExecuteTask() override {
		fn(arg, part, parts);
	}
	std::string TaskType() const override {
		return "EmeansBatchPart";
	}

private:
	emeans_task_fn fn;
	void *arg;
	int part, parts;
};

} // namespace

extern "C" void *emeans_query_context(duckdb_client_context context) {
	return context ? &reinterpret_cast<duckdb::CClientContextWrapper *>(context)->context : nullptr;
}

// Interrupted, or past max_execution_time: BlockingInterruptCheck throws for either, and reads the clock on every call
// (IsInterrupted only sees the interrupt). Asked once per step, so the clock costs nothing that matters.
extern "C" int emeans_query_interrupted(void *query) {
	if (!query) {
		return 0;
	}
	try {
		static_cast<duckdb::ClientContext *>(query)->BlockingInterruptCheck();
		return 0;
	} catch (duckdb::InterruptException &) {
		return 1;
	}
}

namespace duckdb {

std::string InterruptMessage(void *query) {
	try {
		if (query) {
			static_cast<ClientContext *>(query)->BlockingInterruptCheck();
		}
	} catch (InterruptException &e) {
		return ErrorData(e).RawMessage();
	}
	return InterruptException::INTERRUPT_MESSAGE;
}

void ThrowInterrupted(void *query) {
	throw InterruptException(InterruptMessage(query));
}

} // namespace duckdb

// The parts do no allocation and cannot fail, so WorkOnTasks has no error to rethrow; if scheduling
// itself throws, the caller (a C function) must not see the exception, and the parts run here instead.
extern "C" void emeans_query_parallel(void *query, emeans_task_fn fn, void *arg, int parts) {
	if (!query) {
		emeans_parallel_for(fn, arg, parts);
		return;
	}
	try {
		duckdb::TaskExecutor executor(*static_cast<duckdb::ClientContext *>(query));
		for (int part = 0; part < parts; part++) {
			executor.ScheduleTask(duckdb::make_uniq<PartTask>(executor, fn, arg, part, parts));
		}
		executor.WorkOnTasks();
	} catch (...) {
		emeans_parallel_for(fn, arg, parts);
	}
}

extern "C" void emeans_query_attach(emeans_params *p, void *query) {
	if (query) {
		p->interrupted = emeans_query_interrupted;
		p->interrupt_ctx = query;
		p->parallel = emeans_query_parallel;
		p->parallel_ctx = query;
	}
}
