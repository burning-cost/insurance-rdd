"""RDD plots for insurance threshold analysis.

All plots return matplotlib Figure objects so they can be saved, shown,
or embedded in reports without forcing a display.

RDPlot:
    Binned scatter with polynomial fit on each side — the standard RD figure.
    Shows the data honestly (binned means) alongside the fitted discontinuity.

DensityPlot:
    Histogram/density at the cutoff, highlighting any manipulation spike.

BandwidthSensitivityPlot:
    tau as a function of bandwidth multiplier — key sensitivity check.

CovariateBalancePlot:
    Forest plot of covariate balance test results.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")


def _bin_data(
    x: np.ndarray,
    y: np.ndarray,
    cutoff: float,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create binned means for RD scatter plot.

    Returns arrays (bin_centers, bin_means, bin_se).
    """
    left_mask = x < cutoff
    right_mask = x >= cutoff

    bin_centers = []
    bin_means = []
    bin_ses = []

    for mask in [left_mask, right_mask]:
        x_side = x[mask]
        y_side = y[mask]
        w_side = weights[mask] if weights is not None else np.ones(mask.sum())

        if mask.sum() < 2:
            continue

        bins = np.linspace(x_side.min(), x_side.max(), n_bins + 1)
        for i in range(len(bins) - 1):
            in_bin = (x_side >= bins[i]) & (x_side < bins[i + 1])
            if in_bin.sum() == 0:
                continue
            w_bin = w_side[in_bin]
            y_bin = y_side[in_bin]
            wsum = w_bin.sum()
            if wsum == 0:
                continue
            wmean = np.sum(w_bin * y_bin) / wsum
            # Weighted SE.
            wvar = np.sum(w_bin * (y_bin - wmean) ** 2) / wsum
            wse = np.sqrt(wvar / in_bin.sum())
            bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
            bin_means.append(wmean)
            bin_ses.append(wse)

    return np.array(bin_centers), np.array(bin_means), np.array(bin_ses)


def _poly_fit(
    x: np.ndarray,
    y: np.ndarray,
    cutoff: float,
    p: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit local polynomial on each side and return smooth curve.

    Returns (x_curve, y_curve) arrays for plotting.
    """
    left = x < cutoff
    right = x >= cutoff
    x_left = x[left]
    y_left = y[left]
    x_right = x[right]
    y_right = y[right]

    w_left = weights[left] if weights is not None else None
    w_right = weights[right] if weights is not None else None

    x_curves = []
    y_curves = []

    for x_side, y_side, w_side in [
        (x_left, y_left, w_left),
        (x_right, y_right, w_right),
    ]:
        if len(x_side) < p + 2:
            continue
        coef = np.polyfit(x_side - cutoff, y_side, p, w=w_side)
        x_fit = np.linspace(x_side.min(), x_side.max(), 200)
        y_fit = np.polyval(coef, x_fit - cutoff)
        x_curves.append(x_fit)
        y_curves.append(y_fit)

    if not x_curves:
        return np.array([]), np.array([])

    return np.concatenate(x_curves), np.concatenate(y_curves)


class RDPlot:
    """Standard RD figure: binned scatter with polynomial fit.

    Parameters
    ----------
    outcome:
        Column name of the outcome.
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame.
    exposure:
        Column name of exposure (used for rate computation and weighting).
    outcome_type:
        'poisson', 'gamma', or 'gaussian'. Affects y-axis label.
    n_bins:
        Number of bins on each side (default 30).
    p:
        Polynomial order for the smooth fit.
    title:
        Plot title. Auto-generated if None.
    xlabel:
        x-axis label. Auto-generated if None.
    ylabel:
        y-axis label. Auto-generated if None.
    figsize:
        Figure size tuple.
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        exposure: str | None = None,
        outcome_type: str = "poisson",
        n_bins: int = 30,
        p: int = 1,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.exposure = exposure
        self.outcome_type = outcome_type
        self.n_bins = n_bins
        self.p = p
        self.title = title
        self.xlabel = xlabel or running_var
        self.ylabel = ylabel
        self.figsize = figsize

    def plot(
        self,
        tau: float | None = None,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
        ax: Any | None = None,
    ) -> "Figure":
        """Generate the RD scatter plot.

        Parameters
        ----------
        tau:
            Point estimate (for annotation). Optional.
        ci_lower:
            CI lower bound (for annotation).
        ci_upper:
            CI upper bound.
        ax:
            Existing matplotlib axes to plot into. Creates new figure if None.

        Returns
        -------
        matplotlib Figure.
        """
        _require_matplotlib()

        df = self.data.dropna(subset=[self.outcome, self.running_var])

        x = df[self.running_var].to_numpy(dtype=float)
        y_raw = df[self.outcome].to_numpy(dtype=float)

        weights = None
        if self.exposure and self.exposure in df.columns:
            exp_arr = df[self.exposure].to_numpy(dtype=float)
            if self.outcome_type == "poisson":
                y_plot = y_raw / np.maximum(exp_arr, 1e-10)  # rate
            else:
                y_plot = y_raw
            weights = exp_arr
        else:
            y_plot = y_raw

        bin_x, bin_y, bin_se = _bin_data(x, y_plot, self.cutoff, self.n_bins, weights)
        x_curve, y_curve = _poly_fit(x, y_plot, self.cutoff, self.p, weights)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        # Separate left and right for colouring.
        left_bins = bin_x < self.cutoff
        right_bins = bin_x >= self.cutoff

        ax.errorbar(
            bin_x[left_bins], bin_y[left_bins], yerr=bin_se[left_bins],
            fmt="o", color="#2166AC", alpha=0.7, markersize=4, linewidth=0.8,
            label="Below cutoff", capsize=2,
        )
        ax.errorbar(
            bin_x[right_bins], bin_y[right_bins], yerr=bin_se[right_bins],
            fmt="o", color="#D6604D", alpha=0.7, markersize=4, linewidth=0.8,
            label="Above cutoff", capsize=2,
        )

        if len(x_curve) > 0:
            left_curve = x_curve < self.cutoff
            right_curve = x_curve >= self.cutoff
            ax.plot(x_curve[left_curve], y_curve[left_curve], "-", color="#2166AC", linewidth=1.5)
            ax.plot(x_curve[right_curve], y_curve[right_curve], "-", color="#D6604D", linewidth=1.5)

        ax.axvline(self.cutoff, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="Cutoff")

        # Annotate tau if provided.
        if tau is not None:
            rr = np.exp(tau)
            ci_str = ""
            if ci_lower is not None and ci_upper is not None:
                ci_str = f"\n95% CI: [{np.exp(ci_lower):.3f}, {np.exp(ci_upper):.3f}]"
            annotation = f"Rate ratio: {rr:.3f}{ci_str}"
            ax.text(
                0.97, 0.95, annotation,
                transform=ax.transAxes,
                va="top", ha="right",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ylabel = self.ylabel
        if ylabel is None:
            if self.outcome_type == "poisson":
                ylabel = f"Claim frequency (claims / exposure-year)"
            elif self.outcome_type == "gamma":
                ylabel = f"Average claim amount"
            else:
                ylabel = self.outcome

        ax.set_xlabel(self.xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            self.title or f"RD Plot: {self.outcome} at {self.running_var} = {self.cutoff}",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        return fig


class BandwidthSensitivityPlot:
    """Plot tau as a function of bandwidth multiplier.

    Parameters
    ----------
    sensitivity_df:
        DataFrame from InsuranceRD.bandwidth_sensitivity() with columns:
        h_factor, tau, ci_lower, ci_upper.
    cutoff:
        The cutoff value (for title).
    outcome_type:
        For y-axis labelling.
    figsize:
        Figure size.
    """

    def __init__(
        self,
        sensitivity_df: pd.DataFrame,
        cutoff: float | None = None,
        outcome_type: str = "poisson",
        figsize: tuple[float, float] = (9, 5),
    ) -> None:
        self.df = sensitivity_df
        self.cutoff = cutoff
        self.outcome_type = outcome_type
        self.figsize = figsize

    def plot(self, ax: Any | None = None) -> "Figure":
        """Generate the bandwidth sensitivity plot."""
        _require_matplotlib()

        df = self.df.dropna(subset=["tau"])

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        ax.plot(df["h_factor"], df["tau"], "b-o", markersize=4, linewidth=1.5, label="tau (log scale)")
        ax.fill_between(df["h_factor"], df["ci_lower"], df["ci_upper"], alpha=0.2, color="blue", label="95% CI")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Optimal h*")

        ax.set_xlabel("Bandwidth multiplier (h / h*)", fontsize=11)
        ylabel = "tau (log rate ratio)" if self.outcome_type in ("poisson", "gamma") else "tau"
        ax.set_ylabel(ylabel, fontsize=11)
        title = "Bandwidth Sensitivity"
        if self.cutoff is not None:
            title += f" (cutoff = {self.cutoff})"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        return fig


class DensityPlot:
    """Density/histogram plot at the cutoff, for visual McCrary check.

    Parameters
    ----------
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame.
    n_bins:
        Number of histogram bins.
    figsize:
        Figure size.
    """

    def __init__(
        self,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        n_bins: int = 50,
        figsize: tuple[float, float] = (9, 5),
    ) -> None:
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.n_bins = n_bins
        self.figsize = figsize

    def plot(self, ax: Any | None = None) -> "Figure":
        """Generate density plot."""
        _require_matplotlib()

        df = self.data.dropna(subset=[self.running_var])
        x = df[self.running_var].to_numpy(dtype=float)

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        left = x[x < self.cutoff]
        right = x[x >= self.cutoff]

        bins = np.linspace(x.min(), x.max(), self.n_bins + 1)

        ax.hist(left, bins=bins[bins < self.cutoff], color="#2166AC", alpha=0.6, label="Below cutoff")
        ax.hist(right, bins=bins[bins >= self.cutoff], color="#D6604D", alpha=0.6, label="Above cutoff")
        ax.axvline(self.cutoff, color="black", linestyle="--", linewidth=1.2, label="Cutoff")

        ax.set_xlabel(self.running_var, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"Running Variable Density: {self.running_var} at {self.cutoff}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        fig.tight_layout()
        return fig


class CovariateBalancePlot:
    """Forest plot of covariate balance test results.

    Parameters
    ----------
    balance_df:
        DataFrame from CovariateBalance with columns: covariate, tau,
        ci_lower, ci_upper, p_value.
    alpha:
        Significance level for highlighting.
    figsize:
        Figure size (height auto-scales with number of covariates).
    """

    def __init__(
        self,
        balance_df: pd.DataFrame,
        alpha: float = 0.05,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self.df = balance_df
        self.alpha = alpha
        self.figsize = figsize

    def plot(self, ax: Any | None = None) -> "Figure":
        """Generate the forest plot."""
        _require_matplotlib()

        df = self.df.dropna(subset=["tau"])
        n = len(df)

        if n == 0:
            fig, ax = plt.subplots(figsize=self.figsize or (8, 3))
            ax.text(0.5, 0.5, "No covariates to plot.", ha="center", va="center")
            return fig

        figsize = self.figsize or (9, max(3, n * 0.5 + 1.5))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        y_pos = np.arange(n)
        colors = ["#D6604D" if p < self.alpha else "#2166AC" for p in df["p_value"]]

        ax.barh(y_pos, df["tau"], xerr=[
            df["tau"] - df["ci_lower"],
            df["ci_upper"] - df["tau"],
        ], color=colors, alpha=0.7, height=0.5, capsize=4)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["covariate"], fontsize=9)
        ax.set_xlabel("Estimated tau (RD on covariate)", fontsize=11)
        ax.set_title("Covariate Balance at Cutoff", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=":", axis="x")

        red_patch = mpatches.Patch(color="#D6604D", alpha=0.7, label=f"Significant (p < {self.alpha})")
        blue_patch = mpatches.Patch(color="#2166AC", alpha=0.7, label="Not significant")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9)

        fig.tight_layout()
        return fig


class MultiCutoffPlot:
    """Forest plot of per-cutoff effects from MultiCutoffRD.

    Parameters
    ----------
    mc_result:
        MultiCutoffRDResult object.
    outcome_type:
        For axis labelling.
    figsize:
        Figure size.
    """

    def __init__(
        self,
        mc_result: Any,
        outcome_type: str = "poisson",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self.mc_result = mc_result
        self.outcome_type = outcome_type
        self.figsize = figsize

    def plot(self, ax: Any | None = None) -> "Figure":
        """Generate the multi-cutoff forest plot."""
        _require_matplotlib()

        df = self.mc_result.cutoff_effects_df()
        n = len(df)
        figsize = self.figsize or (9, max(3, n * 0.6 + 2))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        y_pos = np.arange(n)
        colors = ["#D6604D" if p < 0.05 else "#2166AC" for p in df["p_value"]]

        ax.barh(y_pos, df["tau"], xerr=[
            df["tau"] - df["ci_lower"],
            df["ci_upper"] - df["tau"],
        ], color=colors, alpha=0.7, height=0.5, capsize=4)

        # Pooled estimate as a vertical line.
        pooled = self.mc_result.pooled_tau
        ax.axvline(pooled, color="green", linestyle="-", linewidth=1.5, label=f"Pooled: {pooled:.3f}")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Cutoff = {c:.1f}" for c in df["cutoff"]], fontsize=9)

        xlabel = "tau (log rate ratio)" if self.outcome_type in ("poisson", "gamma") else "tau"
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title("Multi-Cutoff RDD Effects", fontsize=12)

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":", axis="x")
        fig.tight_layout()
        return fig
