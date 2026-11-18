# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb==2.0.0",
#     "marimo==0.25.2",
#     "matplotlib==3.10.7",
#     "numpy==2.3.4",
# ]
# ///
"""e-means explained: the formulas of the emeans DuckDB extension, annotated and run.

Run it with `uvx marimo edit --sandbox notebooks/emeans_explained.py` (interactive), or `make notebook`
(static HTML, against the build; any failing cell, such as the formulas disagreeing with emeans_recipe,
fails it).
It loads the extension from $EMEANS_EXTENSION (a built emeans.duckdb_extension) if set, else installs it
from the community repository.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="e-means explained")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # e-means, explained

        **e-means** ([Bessa et al., arXiv 2611.09501](https://arxiv.org/abs/2611.09501)) is mini-batch
        k-means with three changes:

        1. every centroid follows an **exponential moving average** of the means of its batches;
        2. the **batch size** $B$ and the **initial step** $\alpha_0$ come from scaling laws fitted to many
           runs, so there is nothing to tune;
        3. **dead clusters are revived** by splitting live ones, and a **closing pass** sets every centroid to
           the exact mean of its points.

        This notebook takes the formulas one at a time, as the
        [emeans DuckDB extension](https://hugr-lab.github.io/duck-emeans/) implements them, and runs each
        against the extension itself. The reference is the code in `faiss/contrib/e_means`, not the paper:
        the shipped coefficients differ from the paper's Table 2.
        """
    )
    return


@app.cell
def _():
    import math
    import os

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return duckdb, math, mo, np, os, plt


@app.cell
def _(duckdb, os):
    # A built extension ($EMEANS_EXTENSION) or the community one.
    extension = os.environ.get("EMEANS_EXTENSION")
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"} if extension else {})
    if extension:
        con.execute(f"LOAD '{extension}'")
    else:
        con.execute("INSTALL emeans FROM community")
        con.execute("LOAD emeans")
    return (con,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. The batch law

        For $n$ points of dimension $d$, $k$ clusters and $T$ passes over the data (epochs; fractional
        values allowed), the batch size is

        $$
        B \;=\; c \cdot k^{0.134749} \cdot n^{0.876473} \cdot T^{0.813344}
        $$

        - $n^{0.876}$: the batch grows almost in proportion to the data, so the number of steps per pass,
          $n / B \propto n^{0.12}$, grows only slowly.
        - $T^{0.813}$: more epochs buy bigger, steadier batches rather than more steps.
        - $k^{0.135}$: more clusters need slightly bigger batches, so each cluster still sees enough points.
        - $c$ picks the law: $c = 0.035134$ for **throughput** and $c = 0.00439175$ for **quality**, whose
          batches are 8× smaller, so it makes more steps. `auto` uses quality for $d \le 16$, where a step is
          cheap, and throughput above.

        $B$ is then rounded to the nearest power of two in the geometric sense (between $2^m$ and $2^{m+1}$,
        the one whose ratio to $B$ is smaller), capped at about $n / 2$, and kept in $[1, n]$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(1_000, 10_000_000, value=1_000_000, step=1_000, label="n (points)", show_value=True)
    k_slider = mo.ui.slider(2, 4096, value=256, label="k (clusters)", show_value=True)
    d_slider = mo.ui.slider(2, 1536, value=128, label="d (dimension)", show_value=True)
    t_slider = mo.ui.slider(0.5, 25.0, value=2.0, step=0.5, label="T (epochs)", show_value=True)
    mo.vstack([n_slider, k_slider, d_slider, t_slider])
    return d_slider, k_slider, n_slider, t_slider


@app.cell
def _(math):
    def round_pow2(value):
        """The power of two nearest to value in the geometric sense; 1 for value <= 1."""
        if value <= 1:
            return 1
        low = 2 ** math.floor(math.log2(value))
        return low if value / low < (2 * low) / value else 2 * low

    def batch_law(n, k, d, T):
        quality = d <= 16
        c = 0.00439175 if quality else 0.035134
        raw = c * k**0.134749 * n**0.876473 * T**0.813344
        b = round_pow2(raw)
        b = min(b, n // 2) if quality else min(b, round_pow2(n / 2))
        return max(1, min(b, n)), raw, "quality" if quality else "throughput"

    def step_law(n, k, B, T):
        alpha0 = 0.8244 * k**-0.4617 * B**0.9003 * n**-0.4498 * T**-0.1057
        return min(1.0, max(1e-4, alpha0)), alpha0

    return batch_law, round_pow2, step_law


@app.cell
def _(batch_law, con, d_slider, k_slider, mo, n_slider, step_law, t_slider):
    n, k, d, T = n_slider.value, k_slider.value, d_slider.value, t_slider.value
    B, B_raw, law = batch_law(n, k, d, T)
    alpha0, alpha0_raw = step_law(n, k, B, T)
    sql = con.execute("SELECT emeans_recipe(?, ?, ?, ?)", [n, k, d, T]).fetchone()[0]
    agree = sql["batch_law"] == law and sql["batch_size"] == B and abs(sql["alpha0"] - alpha0) <= 1e-12 * alpha0
    # A disagreement is a bug in the formulas above or in the extension: stop here (and fail the export in CI).
    assert agree, f"the formulas give B = {B}, alpha0 = {alpha0}; emeans_recipe gives {sql}"
    mo.md(
        f"""
        | | the formulas, in Python | `emeans_recipe({n}, {k}, {d}, {T})` |
        |---|---:|---:|
        | law | {law} | {sql["batch_law"]} |
        | $B$ before rounding | {B_raw:,.1f} | |
        | $B$ | {B:,} | {sql["batch_size"]:,} |
        | $\\alpha_0$ before clamping | {alpha0_raw:.6g} | |
        | $\\alpha_0$ | {alpha0:.6g} | {sql["alpha0"]:.6g} |
        | steps in all | | {sql["total_steps"]:,} over {sql["n_passes"]} passes |

        ✅ The formulas and the extension agree.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. The step law

        The first step is

        $$
        \alpha_0 \;=\; 0.8244 \cdot k^{-0.4617} \cdot B^{0.9003} \cdot n^{-0.4498} \cdot T^{-0.1057},
        \qquad \text{clamped to } [10^{-4},\, 1].
        $$

        - $B^{0.90}\, n^{-0.45}$: a big batch is a good estimate of a cluster's mean, so the centroid may move
          far towards it; a step that is small relative to all the data moves it less.
        - $k^{-0.46}$: with more clusters each one gets fewer points per batch, so its mean is noisier and
          the step is smaller.
        - $T^{-0.11}$: a longer run can afford a slightly more cautious start.

        At $\alpha_0 = 1$ (often reached for large batches) the first step jumps straight to the batch mean.

        ## 3. The cosine schedule

        The step decays over the run, from $\alpha_0$ down to $\alpha_0 / 10$, along half a cosine:

        $$
        \alpha(s) \;=\; \alpha_0 \left( 0.1 + 0.9 \cdot \tfrac{1}{2}\Bigl(1 + \cos\frac{\pi\, s}{S}\Bigr) \right),
        \qquad S = \sum_{\text{passes}} \Bigl\lceil \frac{\text{rows}}{B} \Bigr\rceil .
        $$

        Early steps move the centroids fast; late steps only polish them, so the noise of single batches
        averages out instead of throwing the centroids around at the end.
        """
    )
    return


@app.cell
def _(np, plt):
    S = 100
    steps = np.arange(S + 1)
    schedule = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * steps / S))
    fig_schedule, ax_schedule = plt.subplots(figsize=(6, 2.6), layout="constrained")
    ax_schedule.plot(steps, schedule, color="#1c7d78")
    ax_schedule.axhline(0.1, color="grey", lw=0.8, ls="--")
    ax_schedule.set(xlabel="step s (of S = 100)", ylabel=r"$\alpha(s) / \alpha_0$", ylim=(0, 1.05))
    ax_schedule.set_title("The cosine schedule: from α0 down to α0 / 10")
    fig_schedule
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. The EMA update

        Each cluster $j$ has a centroid $c_j$ and a **mass** $N_j$, a decayed count of the points it has
        been given (it starts at $2^{-16}$, nearly nothing). For a batch with step $\alpha$, let $n_j$ be the
        batch's points assigned to $j$ (its nearest centroid) and $\bar x_j$ their mean. Then

        $$
        \begin{aligned}
        N_j &\leftarrow (1 - \alpha)\, N_j \;+\; \alpha\, n_j \\[2pt]
        c_j &\leftarrow c_j \;+\; \frac{\alpha\, n_j}{N_j}\,\bigl(\bar x_j - c_j\bigr)
        \end{aligned}
        $$

        - Every mass decays by $(1 - \alpha)$, so old batches count less and less: an exponential moving
          average.
        - $\dfrac{\alpha\, n_j}{N_j}$ is the share of the new points in the cluster's mass. A cluster that has
          seen many points moves a little; a new or starved one jumps to its batch mean.
        - **With $\alpha = 1$** the mass becomes $N_j = n_j$ and the weight is 1, so $c_j = \bar x_j$: one
          step of Lloyd's k-means. e-means is Lloyd with a learning rate and mini-batches.

        The objective of a pass is $\sum_i \lVert x_i - c_{a_i} \rVert^2$, where $a_i$ is point $i$'s cluster, measured as the
        centroids move.

        ## 5. Revival

        Every 5 steps (for $d > 2$), a cluster is **dead** if its mass fell below a tenth of the average:

        $$
        N_j \;<\; \tau\, \frac{1}{k} \sum_{l} N_l, \qquad \tau = 0.1 .
        $$

        Each dead cluster takes half of a live one: donors are drawn in proportion to their mass, by the
        largest Gumbel keys $\log N_j - \log(-\log u_j)$ with $u_j$ uniform. The donor keeps half its mass,
        the dead cluster gets the other half, and the two centroids are placed $10^{-4}$ apart on either side
        of the donor's, so the next batches pull them to different sides of what was one cluster.

        ## 6. The closing pass

        The last pass also adds up, per cluster, the points assigned to it. At the end each centroid becomes
        their exact mean,

        $$
        c_j \;=\; \frac{1}{|P_j|} \sum_{i \in P_j} x_i ,
        $$

        where $P_j$ is the set of points assigned to $j$ in the last pass. That removes the last trace of the
        moving average. The labels the extension returns are those assignments.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Lloyd and e-means on the same data

        `emeans_trace` records a run frame by frame: the centroids and the mean squared distance of the
        training and the held-out points to their nearest centroid. Below, 8 Gaussian blobs in the plane,
        4,000 training and 1,000 test points, $k = 8$, both from the same start (`seed`).
        """
    )
    return


@app.cell
def _(np):
    rng = np.random.default_rng(7)
    centres = rng.uniform(-10, 10, size=(8, 2))
    points = np.vstack([centres[i % 8] + rng.normal(0, 1.2, size=2) for i in range(5000)])
    train, test = points[:4000], points[4000:]
    return centres, test, train


@app.cell
def _(con, test, train):
    def trace(method, nepochs):
        row = con.execute(
            "SELECT emeans_trace(?::DOUBLE[][], ?::DOUBLE[][], 8, ?, 1, ?, 4, NULL, NULL)",
            [train.tolist(), test.tolist(), nepochs, method],
        ).fetchone()[0]
        return row

    lloyd = trace("lloyd", 10.0)
    emeans = trace("emeans", 2.0)
    return emeans, lloyd


@app.cell
def _(emeans, lloyd, mo):
    mo.md(
        f"""
        e-means used $B$ = {emeans["batch_size"]} and $\\alpha_0$ = {emeans["alpha0"]:.3g}; Lloyd uses
        $B = n$ = {lloyd["batch_size"]} and $\\alpha = 1$.

        | | epochs | final train MSE | final test MSE |
        |---|---:|---:|---:|
        | Lloyd | {lloyd["frames"][-1]["epoch"]:.0f} | {lloyd["frames"][-1]["train_mse"]:.4f} | {lloyd["frames"][-1]["test_mse"]:.4f} |
        | e-means | {emeans["frames"][-1]["epoch"]:.0f} | {emeans["frames"][-1]["train_mse"]:.4f} | {emeans["frames"][-1]["test_mse"]:.4f} |

        After {emeans["frames"][-1]["epoch"]:.0f} epochs e-means's held-out error is
        {100 * (emeans["frames"][-1]["test_mse"] / lloyd["frames"][-1]["test_mse"] - 1):+.2f}% from Lloyd's after
        {lloyd["frames"][-1]["epoch"]:.0f}: about the same clustering, from a fifth of the passes over the data.
        """
    )
    return


@app.cell
def _(centres, emeans, lloyd, np, plt, train):
    fig_runs, (ax_mse, ax_plane) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for _run, _name, _colour in ((lloyd, "Lloyd", "#888888"), (emeans, "e-means", "#1c7d78")):
        _epochs = [f["epoch"] for f in _run["frames"]]
        ax_mse.plot(_epochs, [f["test_mse"] for f in _run["frames"]], marker=".", color=_colour, label=_name)
    ax_mse.set(xlabel="epoch", ylabel="held-out MSE", yscale="log")
    ax_mse.set_title("Held-out error over epochs")
    ax_mse.legend()

    ax_plane.scatter(train[:, 0], train[:, 1], s=2, color="#cccccc")
    _path = np.array([f["centroids"] for f in emeans["frames"]])  # frames x k x 2
    for _j in range(_path.shape[1]):
        ax_plane.plot(_path[:, _j, 0], _path[:, _j, 1], color="#1c7d78", lw=1)
    ax_plane.scatter(_path[-1, :, 0], _path[-1, :, 1], color="#1c7d78", s=40, label="e-means, final")
    ax_plane.scatter(_path[0, :, 0], _path[0, :, 1], color="black", marker="x", s=30, label="start")
    ax_plane.scatter(centres[:, 0], centres[:, 1], facecolor="none", edgecolor="red", s=90, label="true centres")
    ax_plane.set_title("e-means centroids, frame by frame")
    ax_plane.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False)
    fig_runs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. The spherical variant

        For embeddings compared by cosine similarity, `emeans_spherical` assigns each point to the centroid
        **direction** $\hat c_j = c_j / \lVert c_j \rVert$ with the largest inner product, and the objective is

        $$
        \sum_i \bigl(2 - 2\, x_i \cdot \hat c_{a_i}\bigr),
        $$

        which for unit vectors $x_i$ equals $\sum_i \lVert x_i - \hat c_{a_i} \rVert^2$. The EMA updates and
        the splits move the raw centroids $c_j$ exactly as above; only the assignment and the closing
        normalisation change.

        ---

        More: the [documentation](https://hugr-lab.github.io/duck-emeans/), the
        [algorithm reference](https://hugr-lab.github.io/duck-emeans/reference/algorithm/), and the
        [source of this notebook](https://github.com/hugr-lab/duck-emeans/blob/main/notebooks/emeans_explained.py).
        """
    )
    return


if __name__ == "__main__":
    app.run()
