"""Tests for plot classes. Run without display (Agg backend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests.

from insurance_rdd.plots import (
    RDPlot,
    DensityPlot,
    BandwidthSensitivityPlot,
    CovariateBalancePlot,
    MultiCutoffPlot,
)

from conftest import make_sharp_rdd_data, make_multicutoff_data


@pytest.fixture
def sensitivity_df():
    """Minimal bandwidth sensitivity DataFrame."""
    return pd.DataFrame({
        "h_factor": [0.5, 0.75, 1.0, 1.5, 2.0],
        "tau": [-0.1, -0.15, -0.35, -0.38, -0.40],
        "ci_lower": [-0.3, -0.35, -0.55, -0.58, -0.60],
        "ci_upper": [0.1, 0.05, -0.15, -0.18, -0.20],
        "p_value": [0.3, 0.1, 0.01, 0.008, 0.005],
    })


@pytest.fixture
def balance_df():
    return pd.DataFrame({
        "covariate": ["age", "gender", "region"],
        "tau": [0.05, -0.02, 0.01],
        "ci_lower": [-0.1, -0.15, -0.08],
        "ci_upper": [0.2, 0.11, 0.10],
        "p_value": [0.4, 0.8, 0.9],
        "significant": [False, False, False],
    })


class TestRDPlot:
    def test_returns_figure(self, gaussian_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_with_tau_annotation(self, gaussian_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df
        )
        fig = p.plot(tau=-0.35, ci_lower=-0.5, ci_upper=-0.2)
        assert isinstance(fig, Figure)

    def test_poisson_outcome_type(self, poisson_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y",
            running_var="x",
            cutoff=0.0,
            data=poisson_df,
            exposure="exposure",
            outcome_type="poisson",
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_gamma_outcome_type(self, gamma_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y", running_var="x", cutoff=0.0, data=gamma_df,
            outcome_type="gamma"
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_custom_title(self, gaussian_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            title="My Custom Title"
        )
        fig = p.plot()
        ax = fig.axes[0]
        assert "My Custom Title" in ax.get_title()

    def test_different_n_bins(self, gaussian_df):
        from matplotlib.figure import Figure
        for n in [5, 20, 50]:
            p = RDPlot(
                outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
                n_bins=n
            )
            fig = p.plot()
            assert isinstance(fig, Figure)

    def test_polynomial_order_2(self, gaussian_df):
        from matplotlib.figure import Figure
        p = RDPlot(
            outcome="y", running_var="x", cutoff=0.0, data=gaussian_df,
            p=2
        )
        fig = p.plot()
        assert isinstance(fig, Figure)


class TestDensityPlot:
    def test_returns_figure(self, gaussian_df):
        from matplotlib.figure import Figure
        p = DensityPlot(
            running_var="x", cutoff=0.0, data=gaussian_df
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_custom_n_bins(self, gaussian_df):
        from matplotlib.figure import Figure
        p = DensityPlot(
            running_var="x", cutoff=0.0, data=gaussian_df, n_bins=20
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_into_existing_axes(self, gaussian_df):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p = DensityPlot(running_var="x", cutoff=0.0, data=gaussian_df)
        returned_fig = p.plot(ax=ax)
        assert returned_fig is fig


class TestBandwidthSensitivityPlot:
    def test_returns_figure(self, sensitivity_df):
        from matplotlib.figure import Figure
        p = BandwidthSensitivityPlot(sensitivity_df=sensitivity_df, cutoff=0.0)
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_with_poisson_outcome(self, sensitivity_df):
        from matplotlib.figure import Figure
        p = BandwidthSensitivityPlot(
            sensitivity_df=sensitivity_df,
            outcome_type="poisson",
        )
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_empty_df_does_not_crash(self):
        from matplotlib.figure import Figure
        empty_df = pd.DataFrame({
            "h_factor": [], "tau": [], "ci_lower": [], "ci_upper": [], "p_value": []
        })
        p = BandwidthSensitivityPlot(sensitivity_df=empty_df)
        fig = p.plot()
        assert isinstance(fig, Figure)


class TestCovariateBalancePlot:
    def test_returns_figure(self, balance_df):
        from matplotlib.figure import Figure
        p = CovariateBalancePlot(balance_df=balance_df)
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_empty_df_does_not_crash(self):
        from matplotlib.figure import Figure
        empty_df = pd.DataFrame({
            "covariate": [], "tau": [], "ci_lower": [], "ci_upper": [], "p_value": []
        })
        p = CovariateBalancePlot(balance_df=empty_df)
        fig = p.plot()
        assert isinstance(fig, Figure)

    def test_significant_rows_differ_in_color(self, balance_df):
        """Just verify it doesn't crash with mixed significance."""
        from matplotlib.figure import Figure
        df = balance_df.copy()
        df.loc[0, "p_value"] = 0.01
        df.loc[0, "significant"] = True
        p = CovariateBalancePlot(balance_df=df)
        fig = p.plot()
        assert isinstance(fig, Figure)


class TestMultiCutoffPlot:
    def test_returns_figure(self, multicutoff_df):
        from matplotlib.figure import Figure
        from insurance_rdd import MultiCutoffRD

        mc = MultiCutoffRD(
            outcome="claim_count",
            running_var="x",
            cutoffs=[1.0, 2.0, 3.0],
            data=multicutoff_df,
        )
        result = mc.fit()
        p = MultiCutoffPlot(mc_result=result)
        fig = p.plot()
        assert isinstance(fig, Figure)
