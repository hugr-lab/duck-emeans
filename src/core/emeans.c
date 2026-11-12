#include "emeans.h"

#include "assign.h"
#include "emeans_alloc.h"
#include "parallel.h"
#include "rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INIT_MASS 1.52587890625e-05 // 2^-16: the mass every cluster starts with
#define REVIVE_EVERY 5              // steps between dead-cluster checks
#define SPLIT_EPSILON 1e-4f         // relative offset between the two halves of a split
#define INIT_STREAM 1               // PCG stream of the initial-centroid draw
#define PASS_STREAM 2               // ... of the pass permutations
#define REVIVE_STREAM 3             // ... of the donor sampling
#define WORK_PER_THREAD 4e6         // rows x k x d below which another thread costs more than it saves

typedef struct {
	double key;
	int32_t index;
} donor_key;

// Every buffer of one call, carved from one allocation (so out-of-memory has a single path).
typedef struct {
	void *block;
	void *handle;         // the allocator's, for release
	double *mass;         // N_j, k
	double *sum;          // s_j of the current batch, k x d
	double *pass_sum;     // sums of the closing pass, k x d
	double *batch_dist;   // B
	int64_t *count;       // n_j of the current batch, k
	int64_t *pass_count;  // counts of the closing pass, k
	int64_t *perm;        // n
	double *min_dist;     // k-means++: each row's squared distance to the nearest chosen centroid, n
	donor_key *keys;      // k
	int32_t *batch_label; // B
	int32_t *touched;     // clusters with n_j > 0 in the current batch, k
	int32_t *targets;     // dead clusters at a revival, k
	float *unit;          // spherical: the centroid directions, k x d
	float *scratch;       // the kernel's scratch, one block per thread
	emeans_kernel kernel; // ct (d x k) and bias (k) of the assignment centroids
} workspace;

typedef struct {
	const float *x;
	int64_t n, d, k;
	uint32_t seed;
	int spherical;
	int threads;
	int64_t scratch_floats; // per thread
	const emeans_params *params;
	int64_t rows_done;      // rows assigned so far, for the epoch of a frame
	emeans_recipe recipe;
	emeans_output *out;
	workspace ws;
	emeans_rng revive_rng;
	int64_t n_touched;
	int64_t step;
	int stopped; // the caller's interrupted() said so
} run_state;

// Whether to stop: once the caller's callback has said so, without asking it again.
static int stopping(run_state *s) {
	const emeans_params *p = s->params;
	s->stopped = s->stopped || (p->interrupted && p->interrupted(p->interrupt_ctx));
	return s->stopped;
}

void emeans_default_params(emeans_params *p) {
	p->nepochs = 5.0;
	p->seed = 1234;
	p->law = EMEANS_LAW_AUTO;
	p->batch_size = 0;
	p->alpha0 = 0;
	p->unused_threshold = -1;
	p->init = NULL;
	p->kmeans_plus_plus = 0;
	p->max_work = 0;
	p->spherical = 0;
	p->threads = 1;
	p->constant_step = 0;
	p->on_frame = NULL;
	p->frame_ctx = NULL;
	p->frame_every = 0;
	p->allocator = (emeans_allocator) {NULL, NULL, NULL};
	p->interrupted = NULL;
	p->interrupt_ctx = NULL;
}

//===--------------------------------------------------------------------===//
// Checks and the workspace
//===--------------------------------------------------------------------===//

static int resolve(run_state *s, const emeans_params *p, char *err) {
	if (emeans_resolve_recipe(s->n, s->k, s->d, p->nepochs, p->law, p->batch_size, p->alpha0, &s->recipe, err)) {
		return EMEANS_INVALID;
	}
	if (p->unused_threshold >= 0) {
		s->recipe.unused_threshold = p->unused_threshold;
	}
	return EMEANS_OK;
}

static int all_finite(const float *v, size_t count) {
	for (size_t i = 0; i < count; i++) {
		if (!isfinite(v[i])) {
			return 0;
		}
	}
	return 1;
}

static int check_input(const run_state *s, const emeans_params *p, char *err) {
	if (s->k > INT32_MAX) {
		return emeans_error(err, "k must be below 2^31");
	}
	if (!all_finite(s->x, (size_t)(s->n * s->d))) {
		return emeans_error(err, "x contains NaN or infinity");
	}
	if (p->init && !all_finite(p->init, (size_t)(s->k * s->d))) {
		return emeans_error(err, "init contains NaN or infinity");
	}
	return EMEANS_OK;
}

static double total_rows(int64_t n, double nepochs) {
	double rows = 0;
	int64_t passes = emeans_n_passes(n, nepochs);
	for (int64_t pass = 0; pass < passes; pass++) {
		rows += (double)emeans_pass_rows(n, nepochs, pass);
	}
	return rows;
}

static int check_budget(const run_state *s, const emeans_params *p, char *err) {
	double rows = total_rows(s->n, p->nepochs) + (p->kmeans_plus_plus && !p->init ? (double)s->n : 0.0);
	double work = rows * (double)s->k * (double)s->d;
	if (p->max_work > 0 && work > p->max_work) {
		snprintf(err, EMEANS_ERR_LEN, "work limit exceeded: rows x k x d over all passes is %.4g, the limit %.4g",
		         work, p->max_work);
		return EMEANS_BUDGET_EXCEEDED;
	}
	return EMEANS_OK;
}

static int prepare(run_state *s, const emeans_params *p, char *err) {
	int rc = resolve(s, p, err);
	if (rc == EMEANS_OK) {
		rc = check_input(s, p, err);
	}
	if (rc == EMEANS_OK) {
		rc = check_budget(s, p, err);
	}
	return rc;
}

// Ordered by alignment: 8-byte arrays, the 16-byte keys, then the 4-byte arrays.
static double workspace_bytes(const run_state *s, int64_t b) {
	double n = (double)s->n, k = (double)s->k, kd = k * (double)s->d;
	double scratch = (double)s->threads * (double)s->scratch_floats;
	return 8.0 * (2.0 * kd + k + (double)b) + 8.0 * (2.0 * k + 2.0 * n) + (double)sizeof(donor_key) * k +
	       4.0 * ((double)b + 2.0 * k) + 4.0 * (2.0 * kd + scratch) + 4.0 * (kd + k);
}

static void *carve(char **cursor, int64_t count, size_t size) {
	void *p = *cursor;
	*cursor += (size_t)count * size;
	return p;
}

// The workspace's one allocation, zeroed: from params->allocator when it has one, else calloc.
static char *workspace_block(run_state *s, double bytes, char *err) {
	const emeans_allocator *a = &s->params->allocator;
	if (bytes >= (double)(SIZE_MAX / 2)) {
		emeans_error(err, "out of memory");
		return NULL;
	}
	if (!a->alloc) {
		char *p = emeans_calloc(1, (size_t)bytes);
		if (!p) {
			emeans_error(err, "out of memory");
		}
		return p;
	}
	char *p = a->alloc(a->ctx, (size_t)bytes, &s->ws.handle, err);
	return p ? memset(p, 0, (size_t)bytes) : NULL;
}

static void workspace_free(run_state *s) {
	const emeans_allocator *a = &s->params->allocator;
	if (a->alloc) {
		a->release(a->ctx, s->ws.handle);
	} else {
		free(s->ws.block);
	}
}

static int workspace_alloc(run_state *s, int64_t b, char *err) {
	workspace *ws = &s->ws;
	int64_t n = s->n, d = s->d, k = s->k;
	char *cursor = workspace_block(s, workspace_bytes(s, b), err);
	ws->block = cursor;
	if (!cursor) {
		return 1;
	}
	ws->mass = carve(&cursor, k, sizeof(double));
	ws->sum = carve(&cursor, k * d, sizeof(double));
	ws->pass_sum = carve(&cursor, k * d, sizeof(double));
	ws->batch_dist = carve(&cursor, b, sizeof(double));
	ws->count = carve(&cursor, k, sizeof(int64_t));
	ws->pass_count = carve(&cursor, k, sizeof(int64_t));
	ws->perm = carve(&cursor, n, sizeof(int64_t));
	ws->min_dist = carve(&cursor, n, sizeof(double));
	ws->keys = carve(&cursor, k, sizeof(donor_key));
	ws->batch_label = carve(&cursor, b, sizeof(int32_t));
	ws->touched = carve(&cursor, k, sizeof(int32_t));
	ws->targets = carve(&cursor, k, sizeof(int32_t));
	ws->unit = carve(&cursor, k * d, sizeof(float));
	ws->scratch = carve(&cursor, s->threads * s->scratch_floats, sizeof(float));
	float *ct = carve(&cursor, k * d, sizeof(float));
	float *bias = carve(&cursor, k, sizeof(float));
	emeans_kernel_init(&ws->kernel, ct, bias, k, d, s->spherical);
	return 0;
}

//===--------------------------------------------------------------------===//
// One batch: assign, accumulate, EMA update
//===--------------------------------------------------------------------===//

// The kernel's copy of centroid j: the centroid itself, or (spherical) its direction.
static void refresh(run_state *s, int64_t j) {
	const float *c = s->out->centroids + j * s->d;
	if (s->spherical) {
		float *u = s->ws.unit + j * s->d;
		emeans_normalize(c, u, s->d);
		c = u;
	}
	emeans_kernel_set(&s->ws.kernel, j, c);
}

typedef struct {
	run_state *s;
	const int64_t *rows;
	int64_t count;
} batch_task;

// Labels from the kernel for rows [lo, hi) of the batch; each distance recomputed exactly for the centroid
// chosen. Ranges are disjoint, and the kernel and centroids are only read, so parts can run concurrently.
static void assign_part(void *arg, int part, int parts) {
	const batch_task *t = arg;
	run_state *s = t->s;
	workspace *ws = &s->ws;
	int64_t lo = t->count * part / parts, hi = t->count * (part + 1) / parts;
	emeans_kernel_nearest(&ws->kernel, s->x, t->rows + lo, hi - lo, ws->batch_label + lo,
	                      ws->scratch + part * s->scratch_floats);
	for (int64_t i = lo; i < hi; i++) {
		const float *row = s->x + t->rows[i] * s->d;
		int64_t j = ws->batch_label[i];
		ws->batch_dist[i] = s->spherical ? 2.0 - 2.0 * emeans_dot(row, ws->unit + j * s->d, s->d)
		                                 : emeans_sq_dist(row, s->out->centroids + j * s->d, s->d);
	}
}

// As many threads as the batch has work for: at least WORK_PER_THREAD each, and at most s->threads.
static int batch_parts(const run_state *s, int64_t count) {
	double by_work = (double)count * (double)s->k * (double)s->d / WORK_PER_THREAD;
	return by_work < (double)s->threads ? (by_work < 1 ? 1 : (int)by_work) : s->threads;
}

static void assign_batch(run_state *s, const int64_t *rows, int64_t count) {
	batch_task t = {s, rows, count};
	int parts = batch_parts(s, count);
	if (parts > 1) {
		emeans_parallel_for(assign_part, &t, parts);
	} else {
		assign_part(&t, 0, 1);
	}
}

static void add_row(double *sum, const float *row, int64_t d) {
	for (int64_t t = 0; t < d; t++) {
		sum[t] += (double)row[t];
	}
}

// Counts and sums per cluster, the touched list, and the batch objective.
static double accumulate(run_state *s, const int64_t *rows, int64_t count) {
	workspace *ws = &s->ws;
	double obj = 0;
	for (int64_t i = 0; i < count; i++) {
		int32_t j = ws->batch_label[i];
		if (ws->count[j]++ == 0) {
			ws->touched[s->n_touched++] = j;
		}
		add_row(ws->sum + j * s->d, s->x + rows[i] * s->d, s->d);
		obj += ws->batch_dist[i];
	}
	return obj;
}

// In the closing pass: the labels, and the sums and counts the closing means are taken from.
static void record_closing(run_state *s, const int64_t *rows, int64_t count) {
	workspace *ws = &s->ws;
	for (int64_t i = 0; i < count; i++) {
		s->out->labels[rows[i]] = ws->batch_label[i];
	}
	for (int64_t t = 0; t < s->n_touched; t++) {
		int64_t j = ws->touched[t];
		ws->pass_count[j] += ws->count[j];
		for (int64_t u = 0; u < s->d; u++) {
			ws->pass_sum[j * s->d + u] += ws->sum[j * s->d + u];
		}
	}
}

// c <- (1 - w) c + w * mean, and the batch sum back to zero. With w = 1 (alpha = 1) c is the mean exactly.
static void move_centroid(float *c, double *sum, int64_t d, double w, double n) {
	for (int64_t t = 0; t < d; t++) {
		c[t] = (float)((1.0 - w) * (double)c[t] + w * (sum[t] / n));
		sum[t] = 0;
	}
}

// N_j <- (1 - alpha) N_j for every cluster; for those in the batch also N_j += alpha n_j, and the
// centroid moves by alpha n_j / N_j towards the batch mean.
static void update(run_state *s, double alpha) {
	workspace *ws = &s->ws;
	for (int64_t j = 0; j < s->k; j++) {
		ws->mass[j] *= 1.0 - alpha;
	}
	for (int64_t t = 0; t < s->n_touched; t++) {
		int64_t j = ws->touched[t];
		double nj = (double)ws->count[j];
		ws->mass[j] += alpha * nj;
		move_centroid(s->out->centroids + j * s->d, ws->sum + j * s->d, s->d, alpha * nj / ws->mass[j], nj);
		refresh(s, j);
		ws->count[j] = 0;
	}
	s->n_touched = 0;
}

//===--------------------------------------------------------------------===//
// Revival: a cluster whose mass fell below tau times the mean mass takes half of a live one
//===--------------------------------------------------------------------===//

static double total_mass(const double *mass, int64_t k) {
	double total = 0;
	for (int64_t j = 0; j < k; j++) {
		total += mass[j];
	}
	return total;
}

// The dead clusters, in index order.
static int64_t collect_targets(run_state *s, double thresh) {
	int64_t n = 0;
	for (int64_t j = 0; j < s->k; j++) {
		if (s->ws.mass[j] < thresh) {
			s->ws.targets[n++] = (int32_t)j;
		}
	}
	return n;
}

static int by_key_descending(const void *a, const void *b) {
	const donor_key *x = a;
	const donor_key *y = b;
	if (x->key != y->key) {
		return x->key > y->key ? -1 : 1;
	}
	return x->index - y->index;
}

// The live clusters, ordered by a Gumbel key log N_j - log(-log u): sorting by it draws donors without
// replacement with probability proportional to their mass.
static int64_t collect_donors(run_state *s, double thresh) {
	int64_t n = 0;
	for (int64_t j = 0; j < s->k; j++) {
		double mass = s->ws.mass[j];
		if (mass >= thresh && mass > 0) {
			double u = emeans_rng_open01(&s->revive_rng);
			s->ws.keys[n].key = log(mass) - log(-log(u));
			s->ws.keys[n++].index = (int32_t)j;
		}
	}
	qsort(s->ws.keys, (size_t)n, sizeof(donor_key), by_key_descending);
	return n;
}

// The donor keeps half its mass and moves by -eps (relative), the dead cluster gets the other half at
// +eps; the sign alternates by pair. If the two still coincide in coordinate 0, the child moves one ULP.
static void split(run_state *s, int64_t src, int64_t dst, int64_t pair) {
	double *mass = s->ws.mass;
	mass[src] /= 2;
	mass[dst] = mass[src];
	float sign = pair % 2 == 0 ? 1.0f : -1.0f;
	float *cs = s->out->centroids + src * s->d;
	float *ct = s->out->centroids + dst * s->d;
	for (int64_t t = 0; t < s->d; t++) {
		float v = cs[t];
		cs[t] = v * (1.0f - SPLIT_EPSILON * sign);
		ct[t] = v * (1.0f + SPLIT_EPSILON * sign);
	}
	if (cs[0] == ct[0]) {
		ct[0] = nextafterf(ct[0], INFINITY);
	}
	refresh(s, src);
	refresh(s, dst);
}

static void revive(run_state *s) {
	double tau = s->recipe.unused_threshold;
	if (!(tau > 0)) {
		return;
	}
	double thresh = tau * total_mass(s->ws.mass, s->k) / (double)s->k;
	int64_t n_targets = collect_targets(s, thresh);
	if (n_targets == 0) {
		return;
	}
	int64_t n_donors = collect_donors(s, thresh);
	int64_t pairs = n_targets < n_donors ? n_targets : n_donors;
	for (int64_t i = 0; i < pairs; i++) {
		split(s, s->ws.keys[i].index, s->ws.targets[i], i);
	}
	s->out->n_splits += pairs;
}

//===--------------------------------------------------------------------===//
// Passes
//===--------------------------------------------------------------------===//

static void emit_frame(run_state *s, int final) {
	const emeans_params *p = s->params;
	if (p->on_frame) {
		emeans_frame f = {(double)s->rows_done / (double)s->n, s->step, s->out->centroids, final};
		p->on_frame(p->frame_ctx, &f);
	}
}

static double step_alpha(const run_state *s) {
	return s->params->constant_step ? s->recipe.alpha0
	                                : emeans_scheduled_alpha(s->recipe.alpha0, s->step, s->recipe.total_steps);
}

static double run_batch(run_state *s, const int64_t *rows, int64_t count, int closing) {
	assign_batch(s, rows, count);
	double obj = accumulate(s, rows, count);
	if (closing) {
		record_closing(s, rows, count);
	}
	update(s, step_alpha(s));
	if ((s->step + 1) % REVIVE_EVERY == 0) {
		revive(s);
	}
	s->step++;
	s->rows_done += count;
	if (s->params->frame_every > 0 && s->step % s->params->frame_every == 0) {
		emit_frame(s, 0);
	}
	return obj;
}

static void run_pass(run_state *s, int64_t pass, int64_t rows, int closing) {
	emeans_rng rng;
	emeans_rng_seed(&rng, emeans_epoch_seed(s->seed, pass), PASS_STREAM);
	emeans_shuffle_prefix(&rng, s->ws.perm, s->n, rows);
	int64_t b = s->recipe.batch_size;
	double obj = 0;
	for (int64_t start = 0; start < rows && !stopping(s); start += b) {
		obj += run_batch(s, s->ws.perm + start, rows - start < b ? rows - start : b, closing);
	}
	s->out->obj[pass] = obj;
}

//===--------------------------------------------------------------------===//
// Initial centroids
//===--------------------------------------------------------------------===//

static void copy_row(run_state *s, int64_t j, int64_t row) {
	memcpy(s->out->centroids + j * s->d, s->x + row * s->d, (size_t)s->d * sizeof(float));
}

// Lowers each row's distance to the nearest chosen centroid with centroid j; returns the new total.
static double update_min_dist(run_state *s, int64_t j) {
	const float *c = s->out->centroids + j * s->d;
	double total = 0;
	for (int64_t i = 0; i < s->n; i++) {
		double d = emeans_sq_dist(s->x + i * s->d, c, s->d);
		s->ws.min_dist[i] = j == 0 || d < s->ws.min_dist[i] ? d : s->ws.min_dist[i];
		total += s->ws.min_dist[i];
	}
	return total;
}

// The row a draw lands on when rows are weighted by min_dist; the last row with weight if rounding runs
// past the end. A total of 0 (every row on a centroid already) gives a uniform draw.
static int64_t weighted_row(run_state *s, emeans_rng *rng, double total) {
	if (!(total > 0)) {
		return (int64_t)emeans_rng_below(rng, (uint64_t)s->n);
	}
	double target = emeans_rng_open01(rng) * total;
	int64_t last = 0;
	for (int64_t i = 0; i < s->n; i++) {
		last = s->ws.min_dist[i] > 0 ? i : last;
		target -= s->ws.min_dist[i];
		if (target <= 0 && s->ws.min_dist[i] > 0) {
			return i;
		}
	}
	return last;
}

// k-means++ (Arthur and Vassilvitskii 2007): a uniform first centroid, then each next one drawn with
// probability proportional to its squared distance from the nearest centroid chosen so far. One pass
// over the rows per centroid, like a pass of the loop.
static void init_plus_plus(run_state *s, emeans_rng *rng) {
	copy_row(s, 0, (int64_t)emeans_rng_below(rng, (uint64_t)s->n));
	double total = update_min_dist(s, 0);
	for (int64_t j = 1; j < s->k && !stopping(s); j++) {
		copy_row(s, j, weighted_row(s, rng, total));
		total = update_min_dist(s, j);
	}
}

static void init_centroids(run_state *s, const emeans_params *p) {
	if (p->init) {
		memcpy(s->out->centroids, p->init, (size_t)(s->k * s->d) * sizeof(float));
		return;
	}
	emeans_rng rng;
	emeans_rng_seed(&rng, s->seed, INIT_STREAM);
	if (p->kmeans_plus_plus) {
		init_plus_plus(s, &rng);
		return;
	}
	emeans_shuffle_prefix(&rng, s->ws.perm, s->n, s->k);
	for (int64_t j = 0; j < s->k; j++) {
		copy_row(s, j, s->ws.perm[j]);
	}
}

static void init_state(run_state *s) {
	s->ws.kernel.inner_product = s->spherical;
	for (int64_t j = 0; j < s->k; j++) {
		s->ws.mass[j] = INIT_MASS;
		if (s->spherical) {
			emeans_normalize(s->out->centroids + j * s->d, s->out->centroids + j * s->d, s->d);
		}
		refresh(s, j);
	}
	for (int64_t i = 0; i < s->n; i++) {
		s->out->labels[i] = -1;
	}
	s->out->n_splits = 0;
	emeans_rng_seed(&s->revive_rng, s->seed, REVIVE_STREAM);
}

// Every centroid that was assigned rows in the closing pass becomes their exact mean; spherical centroids,
// assigned or not, then become directions.
static void close_centroids(run_state *s) {
	workspace *ws = &s->ws;
	for (int64_t j = 0; j < s->k; j++) {
		float *c = s->out->centroids + j * s->d;
		for (int64_t t = 0; ws->pass_count[j] > 0 && t < s->d; t++) {
			c[t] = (float)(ws->pass_sum[j * s->d + t] / (double)ws->pass_count[j]);
		}
		if (s->spherical) {
			emeans_normalize(c, c, s->d);
		}
	}
}

static void run(run_state *s, const emeans_params *p) {
	init_centroids(s, p);
	init_state(s);
	emit_frame(s, 0);
	int64_t passes = s->recipe.n_passes;
	for (int64_t pass = 0; pass < passes && !stopping(s); pass++) {
		run_pass(s, pass, emeans_pass_rows(s->n, p->nepochs, pass), pass == passes - 1);
	}
	if (!s->stopped) {
		close_centroids(s);
		emit_frame(s, 1);
	}
}

int emeans_fit(const float *x, int64_t n, int64_t d, int64_t k, const emeans_params *p, emeans_output *out,
               char *err) {
	if (!x || !p || !out) {
		return emeans_error(err, "x, params and output must not be NULL");
	}
	run_state s = {.x = x,
	               .n = n,
	               .d = d,
	               .k = k,
	               .seed = p->seed,
	               .spherical = p->spherical != 0,
	               .threads = p->threads < 1 ? 1 : (p->threads > EMEANS_MAX_THREADS ? EMEANS_MAX_THREADS : p->threads),
	               .scratch_floats = emeans_kernel_scratch(d),
	               .params = p,
	               .out = out};
	int rc = prepare(&s, p, err);
	if (rc != EMEANS_OK) {
		return rc;
	}
	if (workspace_alloc(&s, s.recipe.batch_size, err)) {
		return EMEANS_OOM;
	}
	run(&s, p);
	workspace_free(&s);
	out->recipe = s.recipe;
	if (s.stopped) {
		emeans_error(err, "interrupted");
		return EMEANS_INTERRUPTED;
	}
	return EMEANS_OK;
}
