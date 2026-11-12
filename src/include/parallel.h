#pragma once

// A minimal fork-join: run a task on a few threads and wait for all of them. POSIX threads on Linux and
// macOS, Win32 threads on Windows; under Emscripten (the WASM build), or when a thread cannot be started,
// that part runs on the calling thread instead. No pool and no global state: threads live for one call,
// which is cheap next to the batches worth splitting.

#define EMEANS_MAX_THREADS 64

// fn(arg, part, parts) for every part in [0, parts), parts clamped to [1, EMEANS_MAX_THREADS]. Part 0 runs
// on the calling thread. Returns when every part has finished.
typedef void (*emeans_task_fn)(void *arg, int part, int parts);
void emeans_parallel_for(emeans_task_fn fn, void *arg, int parts);
