"""Validity tests for RDD assumptions.

Every published RDD paper must report these. The validity suite here
covers all standard checks in one place.

DensityTest:
    McCrary manipulation test via rddensity (Cattaneo, Jansson, Ma 2018).
    A significant density jump at the cutoff suggests sorting/manipulation.
    For NCD at maximum step, failure is expected and documented.

CovariateBalance:
    Tests that predetermined covariates (those that cannot be affected by
    treatment) are smooth at the cutoff. Run InsuranceRD with each
    covariate as the outcome; significant tau indicates a balance failure.

PlaceboTest:
    Tests for spurious effects at false cutoffs away from the true cutoff.
    If the method produces significant effects at placebo cutoffs, the true
    effect estimate is suspect.

DonutRDD:
    Re-estimates the main effect after excluding observations within delta
    of the cutoff. Robustness to heaping — if the estimate is driven by
    the heaped observations right at the cutoff, it will collapse when
    excluded.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._rdrobust import import_rddensity, import_rdrobust


@dataclass
class DensityTestResult:
    """Result from the McCrary/rddensity manipulation test.

    Attributes
    ----------
    test_stat:
        Test statistic (z or t).
    p_value:
        p-value for H0: density is continuous at cutoff.
    significant:
        True if p_value < alpha (evidence of manipulation).
    alpha:
        Significance level used.
    bandwidth_left:
        Bandwidth used on the left side.
    bandwidth_right:
        Bandwidth used on the right side.
    interpretation:
        Human-readable string. Always read this — for NCD at max, a
        failure is documented and expected, not a surprise.
    """

    test_stat: float
    p_value: float
    significant: bool
    alpha: float
    bandwidth_left: float
    bandwidth_right: float
    interpretation: str
    _raw: Any = field(default=None, repr=False)

    def summary(self) -> str:
        sig_str = "SIGNIFICANT (manipulation concern)" if self.significant else "not significant"
        return (
            f"Density Test (rddensity / Cattaneo-Jansson-Ma 2018)\n"
            f"  Test statistic : {self.test_stat:.4f}\n"
            f"  p-value        : {self.p_value:.4f}\n"
            f"  Result         : {sig_str} at alpha={self.alpha}\n"
            f"  Bandwidth (L/R): {self.bandwidth_left:.4f} / {self.bandwidth_right:.4f}\n"
            f"\n  {self.interpretation}"
        )


class DensityTest:
    """McCrary manipulation test via rddensity.

    Tests whether the density of the running variable is continuous at
    the cutoff. A significant discontinuity in density is evidence that
    agents are manipulating the running variable to be above or below
    the cutoff.

    Parameters
    ----------
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame.
    alpha:
        Significance level (default 0.05).
    manipulation_expected:
        Set to True for thresholds where manipulation is known (e.g. NCD
        max step). Affects the interpretation string only — does not change
        the test result.
    """

    def __init__(
        self,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        alpha: float = 0.05,
        manipulation_expected: bool = False,
    ) -> None:
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.alpha = alpha
        self.manipulation_expected = manipulation_expected
        self._result: DensityTestResult | None = None

    def fit(self) -> DensityTestResult:
        """Run the density test."""
        rddensity = import_rddensity()

        df = self.data.dropna(subset=[self.running_var])
        x = df[self.running_var].to_numpy(dtype=float)

        rdd = rddensity.rddensity(X=x, c=self.cutoff)

        # Extract statistics from rddensity result.
        try:
            # rddensity stores results in .hat, .se, .test attributes
            T_stat = float(np.asarray(rdd.test["t_jk"]).ravel()[0])
            p_val = float(np.asarray(rdd.test["p_jk"]).ravel()[0])
            bw_left = float(np.asarray(rdd.h).ravel()[0])
            bw_right = float(np.asarray(rdd.h).ravel()[1]) if len(np.asarray(rdd.h).ravel()) > 1 else bw_left
        except (AttributeError, KeyError, IndexError):
            # Fallback if API differs between versions.
            try:
                T_stat = float(np.asarray(rdd.T).ravel()[0])
                p_val = float(np.asarray(rdd.pv).ravel()[0])
                bw_left = float(np.asarray(rdd.h).ravel()[0])
                bw_right = bw_left
            except Exception:
                T_stat = np.nan
                p_val = np.nan
                bw_left = np.nan
                bw_right = np.nan
                warnings.warn(
                    "Could not extract rddensity test statistics. "
                    "Inspect the _raw attribute directly.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        significant = bool(p_val < self.alpha) if not np.isnan(p_val) else False

        if self.manipulation_expected:
            if significant:
                interp = (
                    "Density spike detected. NOTE: manipulation at this threshold is "
                    "expected and well-documented in the literature (e.g. NCD claim "
                    "withholding at Artis et al. 2002). The RDD identifies the LATE "
                    "for non-manipulators. Interpret treatment effect as a lower bound "
                    "on the true causal effect. Report this failure prominently."
                )
            else:
                interp = (
                    "No significant density spike, despite manipulation being expected. "
                    "Either the sample is too small to detect it, or manipulation is "
                    "less prevalent than expected in this dataset."
                )
        else:
            if significant:
                interp = (
                    "DENSITY SPIKE DETECTED. This is evidence that agents are "
                    "manipulating the running variable near the cutoff. The continuity "
                    "assumption for RDD identification may be violated. Investigate "
                    "whether self-selection into the treated group is occurring. "
                    "Consider using a donut RDD to exclude the heaped region."
                )
            else:
                interp = (
                    "No significant density manipulation detected. The running variable "
                    "appears to be continuous at the cutoff. This supports the "
                    "identification assumption for RDD."
                )

        self._result = DensityTestResult(
            test_stat=T_stat,
            p_value=p_val,
            significant=significant,
            alpha=self.alpha,
            bandwidth_left=bw_left,
            bandwidth_right=bw_right,
            interpretation=interp,
            _raw=rdd,
        )
        return self._result


@dataclass
class CovariateBalanceResult:
    """Result from covariate balance testing.

    Attributes
    ----------
    results_df:
        DataFrame with one row per covariate: tau, se, ci_lower, ci_upper,
        p_value, significant.
    n_covariates:
        Number of covariates tested.
    n_significant:
        Number of covariates with significant RD effect (balance failures).
    balanced:
        True if no covariate shows significant imbalance.
    """

    results_df: pd.DataFrame
    n_covariates: int
    n_significant: int
    balanced: bool

    def summary(self) -> str:
        lines = [
            f"Covariate Balance Test ({self.n_covariates} covariates)",
            f"  Balance failures: {self.n_significant}/{self.n_covariates}",
            f"  Overall: {'BALANCED' if self.balanced else 'IMBALANCE DETECTED'}",
            "",
            self.results_df.to_string(index=False),
        ]
        return "\n".join(lines)


class CovariateBalance:
    """Test whether predetermined covariates are smooth at the cutoff.

    Runs InsuranceRD with each covariate as the outcome. A significant
    treatment effect on a covariate means that covariate changes at the
    cutoff — which should not happen for variables determined before the
    treatment. Balance failures cast doubt on the continuity assumption.

    Parameters
    ----------
    covariates:
        List of covariate column names to test.
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame.
    kernel:
        Kernel for local polynomial.
    bwselect:
        Bandwidth selector.
    alpha:
        Significance level (default 0.05).
    """

    def __init__(
        self,
        covariates: list[str],
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        alpha: float = 0.05,
    ) -> None:
        self.covariates = covariates
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.kernel = kernel
        self.bwselect = bwselect
        self.alpha = alpha
        self._result: CovariateBalanceResult | None = None

    def fit(self) -> CovariateBalanceResult:
        """Run balance tests on all covariates."""
        from .core import InsuranceRD

        rows = []
        for cov in self.covariates:
            if cov not in self.data.columns:
                warnings.warn(f"Covariate '{cov}' not found in data. Skipping.", RuntimeWarning)
                continue

            # Covariates must be numeric for rdrobust.
            cov_data = self.data[[cov, self.running_var]].dropna()
            if not pd.api.types.is_numeric_dtype(cov_data[cov]):
                warnings.warn(
                    f"Covariate '{cov}' is not numeric. Skipping.", RuntimeWarning
                )
                continue

            try:
                rd = InsuranceRD(
                    outcome=cov,
                    running_var=self.running_var,
                    cutoff=self.cutoff,
                    data=cov_data,
                    outcome_type="gaussian",
                    kernel=self.kernel,
                    bwselect=self.bwselect,
                )
                res = rd.fit()
                rows.append(
                    {
                        "covariate": cov,
                        "tau": res.tau_bc,
                        "se": res.se,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                        "p_value": res.p_value,
                        "significant": bool(res.p_value < self.alpha),
                        "n_left": res.n_left_eff,
                        "n_right": res.n_right_eff,
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Balance test failed for '{cov}': {e}", RuntimeWarning
                )
                rows.append(
                    {
                        "covariate": cov,
                        "tau": np.nan,
                        "se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_left": 0,
                        "n_right": 0,
                    }
                )

        df = pd.DataFrame(rows)
        n_sig = int(df["significant"].sum()) if len(df) > 0 else 0

        self._result = CovariateBalanceResult(
            results_df=df,
            n_covariates=len(rows),
            n_significant=n_sig,
            balanced=(n_sig == 0),
        )
        return self._result


@dataclass
class PlaceboTestResult:
    """Result from placebo cutoff testing.

    Attributes
    ----------
    results_df:
        DataFrame with one row per placebo cutoff: placebo_cutoff, tau,
        se, ci_lower, ci_upper, p_value, significant.
    true_tau:
        True effect estimate (at the real cutoff).
    n_significant:
        Number of placebo cutoffs with significant false effects.
    """

    results_df: pd.DataFrame
    true_tau: float
    n_significant: int

    def summary(self) -> str:
        lines = [
            f"Placebo Test ({len(self.results_df)} placebo cutoffs)",
            f"  True tau: {self.true_tau:.4f}",
            f"  Significant placebo effects: {self.n_significant}/{len(self.results_df)}",
            "",
            self.results_df.to_string(index=False),
        ]
        if self.n_significant > 0:
            lines.append(
                "\n  WARNING: Significant effects at placebo cutoffs suggest the "
                "running variable may correlate with treatment through channels "
                "other than the cutoff. Inspect the placebo table carefully."
            )
        return "\n".join(lines)


class PlaceboTest:
    """Run RDD at false cutoffs to test for spurious effects.

    If the RDD method is identifying a real discontinuity at the cutoff,
    it should not find significant effects at nearby placebo cutoffs. Finding
    them suggests the effect may be due to a pre-existing trend rather than
    a genuine jump at the true cutoff.

    Parameters
    ----------
    outcome:
        Column name of the main outcome.
    running_var:
        Column name of the running variable.
    cutoff:
        The true cutoff value.
    data:
        DataFrame.
    placebo_cutoffs:
        List of false cutoff values to test. Should be away from the true
        cutoff but within the data support.
    true_tau:
        The true effect estimate (for comparison in the summary).
    outcome_type:
        Outcome family.
    exposure:
        Column name for exposure variable.
    kernel:
        Kernel type.
    bwselect:
        Bandwidth selector.
    alpha:
        Significance level.
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        placebo_cutoffs: list[float],
        true_tau: float = 0.0,
        outcome_type: str = "gaussian",
        exposure: str | None = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        alpha: float = 0.05,
    ) -> None:
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.placebo_cutoffs = placebo_cutoffs
        self.true_tau = true_tau
        self.outcome_type = outcome_type
        self.exposure = exposure
        self.kernel = kernel
        self.bwselect = bwselect
        self.alpha = alpha
        self._result: PlaceboTestResult | None = None

    def fit(self) -> PlaceboTestResult:
        """Run placebo tests at all false cutoffs."""
        from .core import InsuranceRD

        rows = []
        for c_placebo in self.placebo_cutoffs:
            # Restrict data to one side of the true cutoff to avoid contamination.
            # Use only data strictly below or above the true cutoff depending on
            # which side the placebo falls.
            if c_placebo < self.cutoff:
                df_placebo = self.data[self.data[self.running_var] < self.cutoff].copy()
            else:
                df_placebo = self.data[self.data[self.running_var] >= self.cutoff].copy()

            if len(df_placebo) < 20:
                warnings.warn(
                    f"Insufficient data for placebo at cutoff {c_placebo}. Skipping.",
                    RuntimeWarning,
                )
                continue

            try:
                rd = InsuranceRD(
                    outcome=self.outcome,
                    running_var=self.running_var,
                    cutoff=c_placebo,
                    data=df_placebo,
                    outcome_type=self.outcome_type,
                    exposure=self.exposure,
                    kernel=self.kernel,
                    bwselect=self.bwselect,
                )
                res = rd.fit()
                rows.append(
                    {
                        "placebo_cutoff": c_placebo,
                        "tau": res.tau_bc,
                        "se": res.se,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                        "p_value": res.p_value,
                        "significant": bool(res.p_value < self.alpha),
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Placebo test failed at cutoff {c_placebo}: {e}", RuntimeWarning
                )

        df = pd.DataFrame(rows)
        n_sig = int(df["significant"].sum()) if len(df) > 0 else 0

        self._result = PlaceboTestResult(
            results_df=df,
            true_tau=self.true_tau,
            n_significant=n_sig,
        )
        return self._result


@dataclass
class DonutRDDResult:
    """Result from Donut RDD sensitivity analysis.

    Attributes
    ----------
    results_df:
        DataFrame with one row per donut radius tested: donut_radius, tau,
        se, ci_lower, ci_upper, p_value, n_left, n_right.
    original_tau:
        Estimate with no donut (donut_radius=0).
    """

    results_df: pd.DataFrame
    original_tau: float

    def summary(self) -> str:
        lines = [
            f"Donut RDD Sensitivity (original tau: {self.original_tau:.4f})",
            "",
            self.results_df.to_string(index=False),
        ]
        return "\n".join(lines)


class DonutRDD:
    """Re-estimate RDD effect across a range of donut exclusion radii.

    If the estimate is sensitive to small donut sizes, it suggests the
    effect is driven by observations immediately adjacent to the cutoff
    — potentially due to heaping or rounding artefacts.

    Parameters
    ----------
    outcome:
        Outcome column name.
    running_var:
        Running variable column name.
    cutoff:
        Cutoff value.
    data:
        DataFrame.
    donut_radii:
        List of donut radius values to test.
    outcome_type:
        Outcome family.
    exposure:
        Exposure column name.
    kernel:
        Kernel type.
    bwselect:
        Bandwidth selector.
    original_tau:
        The estimate with no donut (for comparison).
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        donut_radii: list[float],
        outcome_type: str = "gaussian",
        exposure: str | None = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        original_tau: float = 0.0,
    ) -> None:
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.donut_radii = donut_radii
        self.outcome_type = outcome_type
        self.exposure = exposure
        self.kernel = kernel
        self.bwselect = bwselect
        self.original_tau = original_tau
        self._result: DonutRDDResult | None = None

    def fit(self) -> DonutRDDResult:
        """Run Donut RDD at each exclusion radius."""
        from .core import InsuranceRD

        rows = []
        for radius in self.donut_radii:
            try:
                rd = InsuranceRD(
                    outcome=self.outcome,
                    running_var=self.running_var,
                    cutoff=self.cutoff,
                    data=self.data,
                    outcome_type=self.outcome_type,
                    exposure=self.exposure,
                    kernel=self.kernel,
                    bwselect=self.bwselect,
                    donut_radius=radius,
                )
                res = rd.fit()
                rows.append(
                    {
                        "donut_radius": radius,
                        "tau": res.tau_bc,
                        "se": res.se,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                        "p_value": res.p_value,
                        "n_left": res.n_left_eff,
                        "n_right": res.n_right_eff,
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Donut RDD failed at radius {radius}: {e}", RuntimeWarning
                )

        df = pd.DataFrame(rows)
        self._result = DonutRDDResult(results_df=df, original_tau=self.original_tau)
        return self._result
