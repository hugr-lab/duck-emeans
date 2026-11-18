---
title: e-means explained
sidebar_position: 3
---

# e-means, explained

The formulas behind the extension, one at a time, with what each term does. A
[marimo](https://marimo.io) notebook goes through the same steps and runs every formula against the
extension: [**the notebook, rendered**](https://hugr-lab.github.io/duck-emeans/notebook/) (static), or run it
yourself with sliders for $n$, $k$, $d$ and $T$:

```sh
git clone https://github.com/hugr-lab/duck-emeans && cd duck-emeans
uvx marimo edit --sandbox notebooks/emeans_explained.py
```

It installs the extension from the community repository, or loads a build named by `$EMEANS_EXTENSION`.

e-means ([Bessa et al., arXiv 2611.09501](https://arxiv.org/abs/2611.09501)) is mini-batch k-means in which
every centroid follows an exponential moving average of its batch means, the batch size $B$ and initial
step $\alpha_0$ come from fitted scaling laws, dead clusters are revived, and a closing pass sets every
centroid to the exact mean of its points. The reference is the code in `faiss/contrib/e_means`, whose
coefficients differ from the paper's Table 2.

## 1. The batch law

For $n$ points of dimension $d$, $k$ clusters and $T$ epochs (fractional values allowed):

$$
B \;=\; c \cdot k^{0.134749} \cdot n^{0.876473} \cdot T^{0.813344}
$$

- $n^{0.876}$: the batch grows almost in proportion to the data, so the steps per pass, $n / B \propto n^{0.12}$,
  grow only slowly.
- $T^{0.813}$: more epochs buy bigger, steadier batches rather than more steps.
- $k^{0.135}$: more clusters need slightly bigger batches, so each still sees enough points.
- $c = 0.035134$ for the **throughput** law, $c = 0.00439175$ for the **quality** law (8× smaller batches,
  more steps). `auto` picks quality for $d \le 16$, where a step is cheap.

$B$ is rounded to the nearest power of two in the geometric sense, capped at about $n/2$, and kept in
$[1, n]$. [`emeans_recipe`](./functions/recipe-and-trace.md) shows the result for any shape.

## 2. The step law

$$
\alpha_0 \;=\; 0.8244 \cdot k^{-0.4617} \cdot B^{0.9003} \cdot n^{-0.4498} \cdot T^{-0.1057},
\qquad \alpha_0 \in [10^{-4},\, 1]
$$

- $B^{0.90}\,n^{-0.45}$: a big batch estimates a cluster's mean well, so the centroid may move far towards it.
- $k^{-0.46}$: more clusters mean fewer points per cluster per batch, noisier means, smaller steps.
- $T^{-0.11}$: a longer run starts slightly more cautiously.

## 3. The cosine schedule

$$
\alpha(s) \;=\; \alpha_0 \left(0.1 + 0.9 \cdot \tfrac{1}{2}\Bigl(1 + \cos \frac{\pi s}{S}\Bigr)\right),
\qquad S = \sum_{\text{passes}} \Bigl\lceil \frac{\text{rows}}{B} \Bigr\rceil
$$

The step falls from $\alpha_0$ to $\alpha_0 / 10$ over the $S$ steps of the run: early steps move the
centroids fast, late ones only polish them.

## 4. The EMA update

Each cluster $j$ has a centroid $c_j$ and a mass $N_j$, a decayed count of its points (initially $2^{-16}$).
For a batch with step $\alpha$, where $n_j$ of its points are nearest to $c_j$ and $\bar x_j$ is their mean:

$$
\begin{aligned}
N_j &\leftarrow (1 - \alpha)\, N_j + \alpha\, n_j \\
c_j &\leftarrow c_j + \frac{\alpha\, n_j}{N_j}\,\bigl(\bar x_j - c_j\bigr)
\end{aligned}
$$

- Every mass decays by $(1 - \alpha)$: older batches count less and less.
- $\alpha n_j / N_j$ is the new points' share of the cluster's mass. A cluster that has seen many points moves
  a little; a new or starved one jumps to its batch mean.
- With $\alpha = 1$, $N_j = n_j$ and $c_j = \bar x_j$: one step of Lloyd's k-means.

## 5. Revival

Every 5 steps (for $d > 2$) a cluster is dead if

$$
N_j \;<\; \tau \cdot \frac{1}{k} \sum_l N_l, \qquad \tau = 0.1 .
$$

Each dead cluster takes half of a live one. Donors are drawn in proportion to their mass, by the largest
Gumbel keys $\log N_j - \log(-\log u_j)$ with $u_j$ uniform. The two halves start $10^{-4}$ apart on either
side of the donor's centroid, so the next batches pull them apart.

## 6. The closing pass

The last pass also adds up each cluster's points $P_j$, and each centroid becomes their exact mean:

$$
c_j \;=\; \frac{1}{|P_j|} \sum_{i \in P_j} x_i
$$

The labels the extension returns are these assignments.

## 7. Spherical

`emeans_spherical` assigns each point to the direction $\hat c_j = c_j / \lVert c_j \rVert$ with the largest
inner product and minimises

$$
\sum_i \bigl(2 - 2\, x_i \cdot \hat c_{a_i}\bigr),
$$

which equals $\sum_i \lVert x_i - \hat c_{a_i} \rVert^2$ for unit vectors. The updates and splits are as above.

The [algorithm reference](./reference/algorithm.md) has the exact rules: rounding, caps, seeds, and where the
extension differs from the reference.
