# Algorithm

The reference is `faiss/contrib/e_means/base.py` and `e_means.py` at the pinned commit, not the paper:
the shipped coefficients differ from the paper's Table 2, and the code is what users of FAISS get.

## Recipe (`src/core/recipe.c`)

Inputs: n rows, dimension d, k clusters, T passes (fractional allowed).

Batch law, equation 5: B = c · k^0.134749 · n^0.876473 · T^0.813344, with c = 0.035134 for the
throughput law and 0.00439175 for the quality law. `auto` picks quality for d ≤ 16, throughput above.
B is rounded to the nearest power of two in the geometric sense (value / low < high / value picks low),
then capped: the throughput cap is round_pow2(n / 2), the quality cap n // 2; finally 1 ≤ B ≤ n.

Step law, equation 6: α0 = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057, clamped to
[1e-4, 1]. A given B replaces the batch law (capped at n); a given α0 replaces the step law.

Pass schedule: [n] × (⌊T⌋ − 1), then round((T − ⌊T⌋) · n) rows if non-zero, then a full closing pass.
Below one pass, only the rounded prefix. Rounding is ties-to-even, like Python's `round()`.

Step schedule: α(step) = α0 · (0.1 + 0.9 · ½(1 + cos(π · min(step, total) / max(1, total)))), with
total = Σ_passes ⌈rows / B⌉.

Permutation seed of pass e: (seed · 1000003 + e) mod 2^32. Revival threshold τ = 0.1 for d > 2, 0
(disabled) otherwise.

## Loop (`src/core/emeans.c`)

State: centroid c_j and mass N_j, initially N_j = 2^-16 and c_j the first k rows of a random
permutation (or the given `init`).

Per batch of B rows from the pass's permutation:

1. a_i = argmin_j ‖x_i − c_j‖²; accumulate n_j, s_j = Σ x_i, and the objective Σ ‖x_i − c_{a_i}‖² in
   double.
2. N_j ← (1 − α) N_j for every j.
3. For n_j > 0: N_j += α n_j, then c_j += (α n_j / N_j)(s_j / n_j − c_j). With α = 1 this is
   N_j = n_j and c_j = s_j / n_j, one Lloyd step.
4. Every 5 steps, if τ > 0: thresh = τ ΣN_j / k; unused_j = N_j < thresh; donors are the others with
   N_j > 0; nsplit = min(#unused, #donors). Donors are the nsplit largest keys log N_j − log(−log u_j),
   u ~ U(0, 1]; targets the first nsplit unused in index order. For each pair, halve N_s, copy it to
   N_t, and set c_s = c_s (1 − 1e-4 · sign), c_t = c_s (1 + 1e-4 · sign) with sign alternating +1/−1
   by pair index; if the two coincide in coordinate 0, nudge the child by one ULP.
5. In the last pass, accumulate pass_counts_j and pass_sums_j.

Closing: c_j = pass_sums_j / pass_counts_j where pass_counts_j > 0. Labels are the last pass's
assignments, so a label is not necessarily the nearest final centroid; `emeans_assign` gives that.

## Where this differs from the reference

- Random numbers come from PCG32, not numpy's MT19937, so permutations, initial centroids and donor
  draws differ from `NumpyEmeans` with the same seed. Results agree in distribution, not in bits. The
  initial centroids, the pass permutations and the donor draws use separate streams of one seed; pass e
  reseeds from (seed · 1000003 + e) mod 2^32 as the reference does.
- u for the Gumbel keys is drawn from the open interval (0, 1), so no key is infinite; the reference
  draws from (0, 1] and clamps.
- Distances are exact squared differences accumulated in double. The reference (FAISS) computes
  ‖x‖² + ‖c‖² − 2 x·c in float32, which can differ in the last bits and, rarely, in a tie.
- An input with NaN or infinity is rejected; the reference does not check.
- `max_points_per_centroid` (training on a subsample) is not implemented; sample in SQL instead.
- Spherical mode is not implemented yet (phase 5).

## Spherical (planned)

Assign with ĉ_j = c_j / ‖c_j‖ (zero stays zero) and argmax of the dot product; objective
Σ(2 − 2·sim); the closing mean is normalised.
