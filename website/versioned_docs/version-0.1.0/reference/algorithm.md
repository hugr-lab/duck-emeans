---
title: Algorithm
sidebar_position: 2
---

# Algorithm

The reference is `faiss/contrib/e_means/base.py` and `e_means.py` at the pinned commit, not the paper: the
shipped coefficients differ from the paper's Table 2, and the code is what users of FAISS get. The
extension reimplements it in C11; no code is copied.

## Recipe

Inputs: n rows, dimension d, k clusters, T passes (fractional allowed).

**Batch law** (equation 5): B = c · k^0.134749 · n^0.876473 · T^0.813344, with c = 0.035134 for the
throughput law and 0.00439175 for the quality law. `auto` picks quality for d ≤ 16, throughput above. B is
rounded to the nearest power of two in the geometric sense, then capped: the throughput cap is
round_pow2(n / 2), the quality cap n // 2; finally 1 ≤ B ≤ n.

**Step law** (equation 6): α0 = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057, clamped to
[1e-4, 1].

**Pass schedule**: [n] × (⌊T⌋ − 1), then round((T − ⌊T⌋) · n) rows if non-zero, then a full closing pass.
Below one pass, only the rounded prefix. Rounding is ties-to-even, like Python's `round()`.

**Step schedule**: α(step) = α0 · (0.1 + 0.9 · ½(1 + cos(π · min(step, total) / max(1, total)))), with
total = Σ over the passes of ⌈rows / B⌉.

**Seeds**: pass e reseeds from (seed · 1000003 + e) mod 2^32. The revival threshold is τ = 0.1 for d > 2,
and 0 (disabled) otherwise.

## Loop

State: centroid c_j and mass N_j, initially N_j = 2^-16 and c_j the first k rows of a random permutation.

Per batch of B rows from the pass's permutation:

1. a_i = argmin_j ‖x_i − c_j‖²; accumulate n_j, s_j = Σ x_i, and the objective Σ ‖x_i − c_(a_i)‖² in double.
2. N_j ← (1 − α) N_j for every j.
3. For n_j > 0: N_j += α n_j, then c_j += (α n_j / N_j)(s_j / n_j − c_j). With α = 1 this is N_j = n_j and
   c_j = s_j / n_j, one Lloyd step.
4. Every 5 steps, if τ > 0: thresh = τ ΣN_j / k; clusters with N_j below it are dead, the others with
   N_j > 0 donors. Donors are drawn by the largest keys log N_j − log(−log u_j); each pair halves the
   donor's mass, gives the other half to the dead cluster, and places the two at ±1e-4 relative offsets.
5. In the last pass, accumulate the closing counts and sums.

**Closing**: c_j = pass_sums_j / pass_counts_j where pass_counts_j > 0. Labels are the last pass's
assignments, so a label is not necessarily the nearest final centroid; `emeans_assign` gives that.

## Spherical

The initial centroids are normalised. Each row is assigned to the largest x·ĉ_j, where ĉ_j = c_j / ‖c_j‖,
and the objective is Σ(2 − 2 x·ĉ). The EMA updates and the splits move the raw centroids, as in the
Euclidean variant. The closing means are normalised. Like the reference, the rows are not normalised; for
unit rows 2 − 2 x·ĉ is ‖x − ĉ‖².

## Where this differs from the reference

- Random numbers come from PCG32, not numpy's MT19937, so permutations, initial centroids and donor draws
  differ from `NumpyEmeans` with the same seed. Results agree in distribution, not in bits.
- The centroid is chosen, as in FAISS, by ‖c‖² − 2 x·c in float32, which can pick a different centroid
  than exact arithmetic for near-ties; the distance to the chosen centroid, and so the objective, is then
  recomputed exactly in double.
- An input with NaN or infinity is rejected; the reference does not check.
- `max_points_per_centroid` (training on a subsample) is not implemented; sample in SQL instead.
- `emeans_stream` orders its passes by a hash of the vectors instead of a PCG32 permutation and draws its
  start from the data by hash; [Streaming](../functions/stream.md#how-a-run-goes) has the details.
