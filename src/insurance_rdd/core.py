"""InsuranceRD: main RDD estimator for insurance pricing thresholds.

This wraps rdrobust with insurance-specific defaults:
- Exposure weighting (policy-years as weights or Poisson offset)
- Donut RDD support (exclude heaped observations near cutoff)
- Rate ratio output (exp(tau) — directly comparable to GLM relativities)
- Fuzzy RDD support (when treatment probability jumps but not fully)
- Preset-driven defaults for known UK thresholds

The core estimation is rdrobust's CCT bias-corrected local polynomial regression.
We do not reimplement the CCT methodology — rdrobust does that correctly.
Our value is the insurance wrapper: exposure handling, outcome family routing,
domain defaults, and regulatory output format.

For Poisson and Gamma outcomes with exposure offsets, use PoissonRD or GammaRD
from outcomes.py — those implement local GLM regression rather than weighted OLS,
which is the methodologically correct approach for count data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._rdrobust import import_rdrobust
from .presets import ThresholdPreset


@dataclass
class RDResult:
    """Result from fitting an InsuranceRD model.

    Attributes
    ----------
    tau:
        Point estimate of the treatment effect (jump at cutoff).
        For Poisson/Gamma outcomes this is on the log scale; use
        rate_ratio() to exponentiate.
    tau_bc:
        Bias-corrected point estimate (rdrobust robust estimator).
    se:
        Standard error of tau_bc.
    ci_lower:
        Lower bound of 95% confidence interval (bias-corrected robust).
    ci_upper:
        Upper bound of 95% confidence interval.
    p_value:
        p-value for H0: tau = 0 (bias-corrected robust test).
    bandwidth_h:
        Main bandwidth used (left and right if asymmetric).
    bandwidth_b:
        Bias correction bandwidth.
    n_left:
        Observations used in estimation, left of cutoff.
    n_right:
        Observations used in estimation, right of cutoff.
    n_left_eff:
        Effective observations within main bandwidth, left.
    n_right_eff:
        Effective observations within main bandwidth, right.
    outcome_type:
        Outcome family used ('gaussian', 'poisson', 'gamma').
    cutoff:
        The cutoff value.
    running_var_name:
        Name of the running variable column.
    outcome_name:
        Name of the outcome column.
    _rdrobust_result:
        Raw rdrobust result object for advanced use.
    """

    tau: float
    tau_bc: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    bandwidth_h: tuple[float, float]
    bandwidth_b: tuple[float, float]
    n_left: int
    n_right: int
    n_left_eff: int
    n_right_eff: int
    outcome_type: str
    cutoff: float
    running_var_name: str
    outcome_name: str
    _rdrobust_result: Any = field(default=None, repr=False)
    _estimator: Any = field(default=None, repr=False)

    def rate_ratio(self) -> dict[str, float]:
        """Exponentiate tau to get a multiplicative rate ratio.

        For log-link outcomes (Poisson frequency, Gamma severity), the
        treatment effect tau is on the log scale. exp(tau) is the causal
        rate ratio at the cutoff — directly comparable to a GLM pricing
        relativity.

        Returns
        -------
        dict with keys: rate_ratio, ci_lower, ci_upper, tau, ci_tau_lower,
        ci_tau_upper.

        Notes
        -----
        The CI is obtained by exponentiating the log-scale CI endpoints,
        which is correct for the monotone exp() transformation.
        """
        if self.outcome_type == "gaussian":
            warnings.warn(
                "rate_ratio() is designed for log-link outcomes (poisson, gamma). "
                "For gaussian outcomes, tau is already on the natural scale. "
                "Returning exp(tau) anyway — interpret with care.",
                UserWarning,
                stacklevel=2,
            )
        return {
            "rate_ratio": float(np.exp(self.tau_bc)),
            "ci_lower": float(np.exp(self.ci_lower)),
            "ci_upper": float(np.exp(self.ci_upper)),
            "tau": self.tau_bc,
            "ci_tau_lower": self.ci_lower,
            "ci_tau_upper": self.ci_upper,
        }

    def summary(self) -> str:
        """Formatted results table."""
        h_l, h_r = self.bandwidth_h
        bw_str = f"{h_l:.3f}" if h_l == h_r else f"{h_l:.3f} / {h_r:.3f}"
        n_eff = self.n_left_eff + self.n_right_eff

        lines = [
            f"InsuranceRD — {self.outcome_name} ~ RD at {self.running_var_name} = {self.cutoff}",
            f"Outcome type : {self.outcome_type}",
            f"Bandwidth (h): {bw_str}  [left / right]",
            f"N in bandwidth: {n_eff} ({self.n_left_eff} left, {self.n_right_eff} right)",
            f"N total: {self.n_left + self.n_right} ({self.n_left} left, {self.n_right} right)",
            "",
            f"{'':30} {'Estimate':>12} {'Std Err':>10} {'95% CI':>22} {'p-value':>10}",
            "-" * 88,
            f"  {'tau (robust BC)':28} {self.tau_bc:>12.4f} {self.se:>10.4f} "
            f"[{self.ci_lower:>8.4f}, {self.ci_upper:>8.4f}] {self.p_value:>10.4f}",
        ]

        if self.outcome_type in ("poisson", "gamma"):
            rr = np.exp(self.tau_bc)
            rr_lo = np.exp(self.ci_lower)
            rr_hi = np.exp(self.ci_upper)
            lines += [
                "",
                f"  Rate ratio (exp(tau)): {rr:.4f}  (95% CI: {rr_lo:.4f}, {rr_hi:.4f})",
                "  Interpretation: crossing the cutoff multiplies the claims rate by this factor.",
            ]

        return "\n".join(lines)

    def regulatory_report(
        self,
        tariff_relativity: float | None = None,
        threshold_name: str = "threshold",
    ) -> str:
        """Generate FCA Consumer Duty formatted interpretation.

        Parameters
        ----------
        tariff_relativity:
            The multiplicative rating factor your tariff applies at this
            threshold (e.g. 0.70 means a 30% premium reduction above the cutoff).
            If provided, the report compares the empirical rate ratio to this
            figure and flags any discrepancy.
        threshold_name:
            Human-readable name for the threshold (e.g. 'age 25 driver boundary').

        Returns
        -------
        Markdown-formatted string suitable for inclusion in an FCA Consumer
        Duty pricing justification document.
        """
        from .report import _format_regulatory_report

        return _format_regulatory_report(
            result=self,
            tariff_relativity=tariff_relativity,
            threshold_name=threshold_name,
        )


class InsuranceRD:
    """Regression Discontinuity estimator for insurance pricing thresholds.

    Wraps rdrobust with insurance-specific defaults: exposure weighting,
    donut RDD, preset-driven configuration, and rate-ratio output.

    For count outcomes (claim frequency), the exposure offset should be
    passed via the ``exposure`` parameter. The outcome is then treated as a
    weighted regression with weights proportional to exposure_years, and
    the result is on the log rate scale if outcome_type is 'poisson'.

    For a fully correct Poisson local GLM (with log-offset), use PoissonRD
    from outcomes.py. InsuranceRD with outcome_type='poisson' uses rdrobust's
    weighted least squares as an approximation — acceptable for large samples
    but use PoissonRD when you want proper Poisson log-likelihood fitting.

    Parameters
    ----------
    outcome:
        Column name of the outcome variable in ``data``.
    running_var:
        Column name of the running variable (e.g. driver age in months).
    cutoff:
        The threshold value of the running variable.
    data:
        DataFrame containing all variables.
    outcome_type:
        Distribution family: 'gaussian' | 'poisson' | 'gamma'.
        Poisson and gamma route through rate-ratio output.
    exposure:
        Column name of exposure variable (policy-years). Used as offset
        for Poisson (outcome divided by exposure as rate) and as weights.
    fuzzy:
        Column name of the actual treatment variable for fuzzy RDD.
        If provided, estimates the LATE via 2SLS using the cutoff indicator
        as instrument.
    covariates:
        List of covariate column names for conditional RDD.
    kernel:
        Kernel type for rdrobust: 'triangular' | 'epanechnikov' | 'uniform'.
    bwselect:
        Bandwidth selector: 'mserd' | 'msetwo' | 'cerrd' | 'certwo' | ...
    p:
        Local polynomial order (1 = local linear, default and recommended).
    q:
        Bias correction polynomial order (default p+1).
    vce:
        Variance estimator: 'nn' | 'hc1' | 'hc2' | 'hc3' | 'cluster'.
    cluster:
        Column name for cluster variable (when vce='cluster').
    level:
        Confidence level in percent (default 95).
    donut_radius:
        Exclude observations within this distance of the cutoff.
        Useful when running variables are reported at integer precision
        (age in years, vehicle year) creating heaping.
    preset:
        ThresholdPreset instance to load default configuration.
        Any explicitly passed parameters override preset values.
    h:
        Fixed bandwidth (left, right). If provided, overrides data-driven
        bandwidth selection.

    Examples
    --------
    >>> from insurance_rdd import InsuranceRD, presets
    >>> rd = InsuranceRD(
    ...     outcome='claim_count',
    ...     running_var='driver_age_months',
    ...     cutoff=300,
    ...     data=df,
    ...     outcome_type='poisson',
    ...     exposure='exposure_years',
    ...     preset=presets.AGE_25,
    ... )
    >>> result = rd.fit()
    >>> print(result.summary())
    >>> print(result.rate_ratio())
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        outcome_type: str = "gaussian",
        exposure: str | None = None,
        fuzzy: str | None = None,
        covariates: list[str] | None = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        p: int = 1,
        q: int | None = None,
        vce: str = "nn",
        cluster: str | None = None,
        level: float = 95.0,
        donut_radius: float = 0.0,
        preset: ThresholdPreset | None = None,
        h: float | tuple[float, float] | None = None,
    ) -> None:
        # Apply preset defaults first, then override with explicit parameters.
        if preset is not None:
            if cutoff is None:
                cutoff = preset.cutoff
            if kernel == "triangular" and preset.kernel != "triangular":
                kernel = preset.kernel
            if bwselect == "mserd" and preset.bwselect != "mserd":
                bwselect = preset.bwselect
            if donut_radius == 0.0:
                donut_radius = preset.donut_radius

        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.outcome_type = outcome_type.lower()
        self.exposure = exposure
        self.fuzzy = fuzzy
        self.covariates = covariates or []
        self.kernel = kernel
        self.bwselect = bwselect
        self.p = p
        self.q = q
        self.vce = vce
        self.cluster = cluster
        self.level = level
        self.donut_radius = donut_radius
        self.preset = preset
        self.h = h

        self._result: RDResult | None = None

        if self.outcome_type not in ("gaussian", "poisson", "gamma", "tweedie"):
            raise ValueError(
                f"outcome_type must be 'gaussian', 'poisson', 'gamma', or 'tweedie'. "
                f"Got '{self.outcome_type}'."
            )

    def _prepare_data(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
        pd.DataFrame | None,
        NDArray[np.float64] | None,
    ]:
        """Extract and validate arrays from the DataFrame.

        Returns
        -------
        y, x, weights, fuzzy_treatment, covs_df, cluster_arr
        """
        df = self.data.copy()

        # Validate required columns.
        required = [self.outcome, self.running_var]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Donut exclusion: remove observations near cutoff.
        if self.donut_radius > 0:
            dist = np.abs(df[self.running_var] - self.cutoff)
            df = df[dist >= self.donut_radius].copy()
            if len(df) == 0:
                raise ValueError(
                    f"No observations remain after donut exclusion of radius {self.donut_radius}."
                )

        # Drop rows with missing values in key columns.
        cols_to_check = [self.outcome, self.running_var]
        if self.exposure:
            cols_to_check.append(self.exposure)
        if self.fuzzy:
            cols_to_check.append(self.fuzzy)
        cols_to_check += self.covariates
        df = df.dropna(subset=cols_to_check)

        x = df[self.running_var].to_numpy(dtype=float)
        y_raw = df[self.outcome].to_numpy(dtype=float)

        # Exposure handling.
        weights: NDArray[np.float64] | None = None
        if self.exposure is not None:
            exposure_arr = df[self.exposure].to_numpy(dtype=float)
            if np.any(exposure_arr <= 0):
                raise ValueError("All exposure values must be positive.")

            if self.outcome_type == "poisson":
                # For Poisson: convert count to rate, use exposure as weight.
                # This is the weighted OLS approximation; for exact Poisson GLM use PoissonRD.
                y_raw = y_raw / exposure_arr
                weights = exposure_arr
            else:
                weights = exposure_arr

        y = y_raw

        # Fuzzy treatment.
        fuzzy_arr: NDArray[np.float64] | None = None
        if self.fuzzy is not None:
            fuzzy_arr = df[self.fuzzy].to_numpy(dtype=float)

        # Covariates.
        covs_df: pd.DataFrame | None = None
        if self.covariates:
            missing_covs = [c for c in self.covariates if c not in df.columns]
            if missing_covs:
                raise ValueError(f"Covariate columns not found: {missing_covs}")
            covs_df = df[self.covariates].copy()

        # Cluster variable.
        cluster_arr: NDArray[np.float64] | None = None
        if self.cluster is not None:
            if self.cluster not in df.columns:
                raise ValueError(f"Cluster column '{self.cluster}' not found in data.")
            cluster_arr = df[self.cluster].to_numpy()

        return y, x, weights, fuzzy_arr, covs_df, cluster_arr

    def fit(self) -> RDResult:
        """Fit the RDD estimator.

        Returns
        -------
        RDResult with tau, CI, p-value, bandwidths, and effective sample sizes.
        """
        rdrobust = import_rdrobust()

        y, x, weights, fuzzy_arr, covs_df, cluster_arr = self._prepare_data()

        # Build kwargs for rdrobust.
        kwargs: dict[str, Any] = {
            "y": y,
            "x": x,
            "c": self.cutoff,
            "p": self.p,
            "kernel": self.kernel,
            "bwselect": self.bwselect,
            "vce": self.vce,
            "level": self.level,
            "all": True,
        }

        if self.q is not None:
            kwargs["q"] = self.q

        if weights is not None:
            kwargs["weights"] = weights

        if fuzzy_arr is not None:
            kwargs["fuzzy"] = fuzzy_arr

        if covs_df is not None:
            kwargs["covs"] = covs_df

        if cluster_arr is not None:
            kwargs["cluster"] = cluster_arr
            kwargs["vce"] = "cluster"

        if self.h is not None:
            if isinstance(self.h, (int, float)):
                kwargs["h"] = [float(self.h), float(self.h)]
            else:
                kwargs["h"] = list(self.h)

        rdr = rdrobust.rdrobust(**kwargs)

        # Extract results — rdrobust returns arrays; index depends on version.
        # We use the bias-corrected robust ('rbc') estimator throughout.
        # rdrobust returns coef[0]=conventional, coef[1]=BC, coef[2]=robust BC.
        coef = np.asarray(rdr.coef).ravel()
        se = np.asarray(rdr.se).ravel()
        ci = np.asarray(rdr.ci)
        pv = np.asarray(rdr.pv).ravel()
        bws = np.asarray(rdr.bws)

        # Conventional = index 0, BC = index 1, Robust BC = index 2.
        idx_rbc = min(2, len(coef) - 1)
        idx_bc = min(1, len(coef) - 1)

        tau_conv = float(coef[0])
        tau_bc = float(coef[idx_rbc])
        se_rbc = float(se[idx_rbc])
        p_val = float(pv[idx_rbc])

        ci_row = ci[idx_rbc] if ci.ndim > 1 else ci
        ci_lower = float(ci_row[0])
        ci_upper = float(ci_row[1])

        # Bandwidths: bws[0] = [h_left, h_right], bws[1] = [b_left, b_right].
        bws_flat = bws.ravel()
        h_left = float(bws_flat[0])
        h_right = float(bws_flat[1]) if len(bws_flat) > 1 else h_left
        b_left = float(bws_flat[2]) if len(bws_flat) > 2 else h_left
        b_right = float(bws_flat[3]) if len(bws_flat) > 3 else h_right

        # Sample sizes.
        N = np.asarray(rdr.N).ravel()
        n_left = int(N[0]) if len(N) > 0 else 0
        n_right = int(N[1]) if len(N) > 1 else 0

        N_h = np.asarray(rdr.N_h).ravel()
        n_left_eff = int(N_h[0]) if len(N_h) > 0 else 0
        n_right_eff = int(N_h[1]) if len(N_h) > 1 else 0

        self._result = RDResult(
            tau=tau_conv,
            tau_bc=tau_bc,
            se=se_rbc,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_val,
            bandwidth_h=(h_left, h_right),
            bandwidth_b=(b_left, b_right),
            n_left=n_left,
            n_right=n_right,
            n_left_eff=n_left_eff,
            n_right_eff=n_right_eff,
            outcome_type=self.outcome_type,
            cutoff=self.cutoff,
            running_var_name=self.running_var,
            outcome_name=self.outcome,
            _rdrobust_result=rdr,
            _estimator=self,
        )
        return self._result

    def bandwidth_sensitivity(
        self,
        h_factors: list[float] | None = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Estimate tau across a range of bandwidths.

        Tests stability of the point estimate as bandwidth varies between
        0.5h* and 2h*, where h* is the data-driven optimal bandwidth.
        A stable tau across this range is reassuring; a cliff edge at a
        particular h is a warning sign.

        Parameters
        ----------
        h_factors:
            Multipliers of the optimal bandwidth. Default: 0.5 to 2.0
            in n_points steps.
        n_points:
            Number of grid points if h_factors is None.

        Returns
        -------
        pd.DataFrame with columns: h_factor, h, tau, ci_lower, ci_upper,
        se, p_value, n_eff_left, n_eff_right.
        """
        if self._result is None:
            self.fit()
        assert self._result is not None

        h_opt_l, h_opt_r = self._result.bandwidth_h

        if h_factors is None:
            h_factors = np.linspace(0.5, 2.0, n_points).tolist()

        rows = []
        for factor in h_factors:
            h_l = h_opt_l * factor
            h_r = h_opt_r * factor

            try:
                # Refit with fixed bandwidth.
                orig_h = self.h
                self.h = (h_l, h_r)
                res = self.fit()
                self.h = orig_h

                rows.append(
                    {
                        "h_factor": factor,
                        "h_left": h_l,
                        "h_right": h_r,
                        "tau": res.tau_bc,
                        "ci_lower": res.ci_lower,
                        "ci_upper": res.ci_upper,
                        "se": res.se,
                        "p_value": res.p_value,
                        "n_eff_left": res.n_left_eff,
                        "n_eff_right": res.n_right_eff,
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Failed at h_factor={factor:.2f}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.h = orig_h

        # Restore optimal bandwidth as default.
        self.h = None
        return pd.DataFrame(rows)
