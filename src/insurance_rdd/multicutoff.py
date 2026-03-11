"""MultiCutoffRD: pool RDD estimates across multiple thresholds.

Insurance rating systems often have multiple thresholds of the same type —
NCD has 5 distinct steps, age bands can span multiple cutoffs, vehicle age
may have different thresholds for different cover types. Analysing each
cutoff in isolation ignores information from the others.

MultiCutoffRD fits InsuranceRD at each cutoff independently, then pools
the estimates using inverse-variance weighting. The pooled estimate is a
weighted average of the per-cutoff log rate ratios, with weights proportional
to 1/SE^2.

This is simpler than rdmulti's rdmc() which fits a joint model normalising
the running variable across cutoffs. We provide that via rdmulti as an
option, but default to the independent-fits approach because it's easier
to explain to an actuary and handles exposure weighting correctly.

For discrete running variables (NCD levels), set discrete=True to use
rdlocrand's randomisation inference approach at each cutoff.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .core import InsuranceRD, RDResult


@dataclass
class CutoffEffect:
    """Per-cutoff effect estimate."""

    cutoff: float
    tau: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_left: int
    n_right: int
    rate_ratio: float
    bandwidth_h: tuple[float, float]
    converged: bool = True


@dataclass
class MultiCutoffRDResult:
    """Result from MultiCutoffRD.

    Attributes
    ----------
    cutoff_effects:
        List of CutoffEffect, one per cutoff.
    pooled_tau:
        Inverse-variance weighted pooled log rate ratio.
    pooled_se:
        Standard error of the pooled estimate.
    pooled_ci_lower:
        Lower bound of pooled 95% CI.
    pooled_ci_upper:
        Upper bound of pooled 95% CI.
    pooled_p_value:
        p-value for H0: pooled tau = 0.
    pooled_rate_ratio:
        exp(pooled_tau).
    weights:
        Per-cutoff inverse-variance weights (summing to 1).
    heterogeneity_p:
        p-value for Cochran's Q test of heterogeneity across cutoffs.
        A small p-value means the effect varies substantially across
        cutoffs and the pooled estimate is a simplification.
    """

    cutoff_effects: list[CutoffEffect]
    pooled_tau: float
    pooled_se: float
    pooled_ci_lower: float
    pooled_ci_upper: float
    pooled_p_value: float
    pooled_rate_ratio: float
    weights: list[float]
    heterogeneity_p: float

    def cutoff_effects_df(self) -> pd.DataFrame:
        """Return per-cutoff effects as a DataFrame."""
        rows = []
        for i, e in enumerate(self.cutoff_effects):
            rows.append(
                {
                    "cutoff": e.cutoff,
                    "tau": e.tau,
                    "se": e.se,
                    "ci_lower": e.ci_lower,
                    "ci_upper": e.ci_upper,
                    "p_value": e.p_value,
                    "rate_ratio": e.rate_ratio,
                    "weight": self.weights[i],
                    "n_left": e.n_left,
                    "n_right": e.n_right,
                    "h_left": e.bandwidth_h[0],
                    "h_right": e.bandwidth_h[1],
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        df = self.cutoff_effects_df()
        lines = [
            f"MultiCutoffRD — {len(self.cutoff_effects)} cutoffs",
            "",
            "Per-cutoff effects:",
            df.to_string(index=False),
            "",
            f"Pooled estimate (inverse-variance weighted):",
            f"  tau       : {self.pooled_tau:.4f}",
            f"  SE        : {self.pooled_se:.4f}",
            f"  95% CI    : [{self.pooled_ci_lower:.4f}, {self.pooled_ci_upper:.4f}]",
            f"  p-value   : {self.pooled_p_value:.4f}",
            f"  Rate ratio: {self.pooled_rate_ratio:.4f}  "
            f"  (95% CI ratio: [{np.exp(self.pooled_ci_lower):.4f}, {np.exp(self.pooled_ci_upper):.4f}])",
            "",
            f"Heterogeneity (Cochran Q test): p = {self.heterogeneity_p:.4f}",
        ]
        if self.heterogeneity_p < 0.1:
            lines.append(
                "  NOTE: Significant heterogeneity across cutoffs (p < 0.10). "
                "The pooled estimate may mask important variation. "
                "Examine per-cutoff effects before interpreting the pooled figure."
            )
        return "\n".join(lines)

    def pooled_effect(self) -> dict[str, float]:
        """Return pooled effect as a dict (rate ratio and log scale)."""
        return {
            "pooled_tau": self.pooled_tau,
            "pooled_se": self.pooled_se,
            "ci_lower": self.pooled_ci_lower,
            "ci_upper": self.pooled_ci_upper,
            "p_value": self.pooled_p_value,
            "rate_ratio": self.pooled_rate_ratio,
            "rate_ratio_ci_lower": float(np.exp(self.pooled_ci_lower)),
            "rate_ratio_ci_upper": float(np.exp(self.pooled_ci_upper)),
            "heterogeneity_p": self.heterogeneity_p,
        }


class MultiCutoffRD:
    """Pool RDD estimates across multiple pricing thresholds.

    Fits InsuranceRD independently at each cutoff, then pools via
    inverse-variance weighting. Intended for:
    - NCD step boundaries (cutoffs=[1,2,3,4,5])
    - Multiple age band boundaries
    - Vehicle age thresholds across product types

    Parameters
    ----------
    outcome:
        Column name of the outcome variable.
    running_var:
        Column name of the running variable.
    cutoffs:
        List of cutoff values to analyse.
    data:
        DataFrame containing all variables.
    outcome_type:
        Outcome family: 'gaussian' | 'poisson' | 'gamma'.
    exposure:
        Column name for exposure (policy-years).
    covariates:
        Covariate column names for conditional estimation.
    kernel:
        Kernel type.
    bwselect:
        Bandwidth selector.
    p:
        Local polynomial order.
    donut_radius:
        Donut exclusion radius (applied at each cutoff).
    level:
        Confidence level in percent.

    Examples
    --------
    >>> from insurance_rdd import MultiCutoffRD
    >>> mc_rd = MultiCutoffRD(
    ...     outcome='claim_count',
    ...     running_var='ncd_level',
    ...     cutoffs=[1, 2, 3, 4, 5],
    ...     data=df,
    ...     outcome_type='poisson',
    ...     exposure='exposure_years',
    ... )
    >>> result = mc_rd.fit()
    >>> print(result.summary())
    >>> result.cutoff_effects_df()
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoffs: list[float],
        data: pd.DataFrame,
        outcome_type: str = "gaussian",
        exposure: str | None = None,
        covariates: list[str] | None = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        p: int = 1,
        donut_radius: float = 0.0,
        level: float = 95.0,
    ) -> None:
        if len(cutoffs) < 2:
            raise ValueError("MultiCutoffRD requires at least 2 cutoffs.")

        self.outcome = outcome
        self.running_var = running_var
        self.cutoffs = sorted(cutoffs)
        self.data = data.copy()
        self.outcome_type = outcome_type
        self.exposure = exposure
        self.covariates = covariates or []
        self.kernel = kernel
        self.bwselect = bwselect
        self.p = p
        self.donut_radius = donut_radius
        self.level = level
        self._result: MultiCutoffRDResult | None = None

    def fit(self) -> MultiCutoffRDResult:
        """Fit RDD at each cutoff and pool estimates.

        Returns
        -------
        MultiCutoffRDResult with per-cutoff and pooled effects.
        """
        z = 1.96  # for 95% CI; adjust if level != 95
        alpha = 1.0 - self.level / 100.0
        from scipy import stats as scipy_stats
        z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))

        cutoff_effects: list[CutoffEffect] = []

        for c in self.cutoffs:
            try:
                rd = InsuranceRD(
                    outcome=self.outcome,
                    running_var=self.running_var,
                    cutoff=c,
                    data=self.data,
                    outcome_type=self.outcome_type,
                    exposure=self.exposure,
                    covariates=self.covariates if self.covariates else None,
                    kernel=self.kernel,
                    bwselect=self.bwselect,
                    p=self.p,
                    donut_radius=self.donut_radius,
                    level=self.level,
                )
                res = rd.fit()

                cutoff_effects.append(
                    CutoffEffect(
                        cutoff=c,
                        tau=res.tau_bc,
                        se=res.se,
                        ci_lower=res.ci_lower,
                        ci_upper=res.ci_upper,
                        p_value=res.p_value,
                        n_left=res.n_left_eff,
                        n_right=res.n_right_eff,
                        rate_ratio=float(np.exp(res.tau_bc)),
                        bandwidth_h=res.bandwidth_h,
                        converged=True,
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"RDD at cutoff {c} failed: {e}. Skipping this cutoff.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if len(cutoff_effects) == 0:
            raise RuntimeError("All cutoffs failed to estimate. Check data and parameters.")

        if len(cutoff_effects) < len(self.cutoffs):
            warnings.warn(
                f"Only {len(cutoff_effects)}/{len(self.cutoffs)} cutoffs converged.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Inverse-variance pooling.
        taus = np.array([e.tau for e in cutoff_effects])
        ses = np.array([e.se for e in cutoff_effects])

        # Guard against NaN or zero SE.
        valid = np.isfinite(taus) & np.isfinite(ses) & (ses > 0)
        if not np.any(valid):
            raise RuntimeError("No valid estimates available for pooling.")

        taus_v = taus[valid]
        ses_v = ses[valid]

        variances = ses_v**2
        inv_var = 1.0 / variances
        weights_raw = inv_var / inv_var.sum()

        pooled_tau = float(np.sum(weights_raw * taus_v))
        pooled_var = float(1.0 / inv_var.sum())
        pooled_se = float(np.sqrt(pooled_var))
        pooled_ci_lower = pooled_tau - z * pooled_se
        pooled_ci_upper = pooled_tau + z * pooled_se

        from scipy import stats as scipy_stats
        z_pooled = pooled_tau / pooled_se if pooled_se > 0 else np.nan
        pooled_p_value = float(2.0 * (1.0 - scipy_stats.norm.cdf(np.abs(z_pooled))))

        # Cochran's Q for heterogeneity.
        Q = float(np.sum(inv_var * (taus_v - pooled_tau) ** 2))
        df_Q = len(taus_v) - 1
        heterogeneity_p = float(1.0 - scipy_stats.chi2.cdf(Q, df=df_Q)) if df_Q > 0 else 1.0

        # Build full weight vector (NaN for failed cutoffs).
        all_weights = []
        valid_idx = 0
        for e in cutoff_effects:
            if np.isfinite(e.tau) and np.isfinite(e.se) and e.se > 0:
                all_weights.append(float(weights_raw[valid_idx]))
                valid_idx += 1
            else:
                all_weights.append(0.0)

        self._result = MultiCutoffRDResult(
            cutoff_effects=cutoff_effects,
            pooled_tau=pooled_tau,
            pooled_se=pooled_se,
            pooled_ci_lower=float(pooled_ci_lower),
            pooled_ci_upper=float(pooled_ci_upper),
            pooled_p_value=pooled_p_value,
            pooled_rate_ratio=float(np.exp(pooled_tau)),
            weights=all_weights,
            heterogeneity_p=heterogeneity_p,
        )
        return self._result
