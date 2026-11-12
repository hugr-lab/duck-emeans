#include "parallel.h"

#if defined(_WIN32)
#include <windows.h>
typedef HANDLE thread_handle;
#elif !defined(__EMSCRIPTEN__)
#include <pthread.h>
typedef pthread_t thread_handle;
#else
typedef int thread_handle;
#endif

typedef struct {
	emeans_task_fn fn;
	void *arg;
	int part;
	int parts;
} task;

static void run_task(task *t) {
	t->fn(t->arg, t->part, t->parts);
}

#if defined(_WIN32)
static DWORD WINAPI thread_main(LPVOID p) {
	run_task((task *)p);
	return 0;
}

static int start_thread(thread_handle *h, task *t) {
	*h = CreateThread(NULL, 0, thread_main, t, 0, NULL);
	return *h != NULL;
}

static void join_thread(thread_handle h) {
	WaitForSingleObject(h, INFINITE);
	CloseHandle(h);
}
#elif !defined(__EMSCRIPTEN__)
static void *thread_main(void *p) {
	run_task((task *)p);
	return NULL;
}

static int start_thread(thread_handle *h, task *t) {
	return pthread_create(h, NULL, thread_main, t) == 0;
}

static void join_thread(thread_handle h) {
	pthread_join(h, NULL);
}
#else
static int start_thread(thread_handle *h, task *t) {
	(void)h;
	(void)t;
	return 0;
}

static void join_thread(thread_handle h) {
	(void)h;
}
#endif

static int clamp_parts(int parts) {
	return parts < 1 ? 1 : (parts > EMEANS_MAX_THREADS ? EMEANS_MAX_THREADS : parts);
}

void emeans_parallel_for(emeans_task_fn fn, void *arg, int parts) {
	task tasks[EMEANS_MAX_THREADS];
	thread_handle handles[EMEANS_MAX_THREADS];
	int started[EMEANS_MAX_THREADS];
	parts = clamp_parts(parts);
	for (int p = 0; p < parts; p++) {
		tasks[p] = (task) {fn, arg, p, parts};
		started[p] = p > 0 && start_thread(&handles[p], &tasks[p]);
	}
	run_task(&tasks[0]);
	for (int p = 1; p < parts; p++) {
		if (started[p]) {
			join_thread(handles[p]);
		} else {
			run_task(&tasks[p]);
		}
	}
}
