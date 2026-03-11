"""Local GLM regression RDD for non-Gaussian insurance outcomes.

InsuranceRD with outcome_type='poisson' uses rdrobust's weighted OLS as an
approximation. For methodologically correct Poisson and Gamma outcomes, use
PoissonRD and GammaRD here.

These implement local polynomial GLM regression:
  log E[Y | X] = log(t) + beta_0 + beta_1*(X-c) + ... [X < c]
  log E[Y | X] = log(t) + gamma_0 + gamma_1*(X-c) + ... [X >= c]

The treatment effect tau = gamma_0 - beta_0 is the log rate ratio at the
cutoff. exp(tau) is the causal rate ratio — directly comparable to a GLM
pricing relativity.

Bootstrap CIs are used throughout (v0.1.0). Analytical bias-corrected inference
for non-Gaussian local regression requires the full CCT derivation adapted to
the GLM score — this is non-trivial and is deferred to v0.2.

Design notes:
- Zero-inflated outcomes: for claim count data where most policies have 0 claims,
  Poisson is still appropriate (it handles zeros; ZIP is only needed when zeros
  exceed the Poisson prediction). We document this but default to Poisson.
- Gamma on zero-inflated severity: GammaRD operates on S_i | S_i > 0 by
  default (pass severity_only=True, which is the default). This is correct —
  you can't fit Gamma to zero claims.
- The local polynomial is fitted via scipy.minimize on the Poisson/Gamma
  log-likelihood with a kernel-weighted objective.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats


@dataclass
class LocalGLMResult:
    """Result from PoissonRD or GammaRD.

    Attributes
    ----------
    tau:
        Log rate ratio (treatment effect on log scale).
    se_boot:
        Bootstrap standard error.
    ci_lower:
        Lower bound of bootstrap CI.
    ci_upper:
        Upper bound of bootstrap CI.
    p_value:
        Two-sided p-value for H0: tau = 0.
    bandwidth:
        Bandwidth used.
    n_left:
        Effective observations left of cutoff.
    n_right:
        Effective observations right of cutoff.
    outcome_type:
        'poisson' or 'gamma'.
    n_boot:
        Number of bootstrap replications used.
    converged:
        Whether the main fit converged.
    coef_left:
        Full polynomial coefficient vector for the left side.
    coef_right:
        Full polynomial coefficient vector for the right side.
    """

    tau: float
    se_boot: float
    ci_lower: float
    ci_upper: float
    p_value: float
    bandwidth: float
    n_left: int
    n_right: int
    outcome_type: str
    n_boot: int
    converged: bool
    coef_left: NDArray[np.float64]
    coef_right: NDArray[np.float64]
    _estimator: Any = field(default=None, repr=False)

    def rate_ratio(self) -> dict[str, float]:
        """Exponentiate tau to get multiplicative rate ratio with CI."""
        return {
            "rate_ratio": float(np.exp(self.tau)),
            "ci_lower": float(np.exp(self.ci_lower)),
            "ci_upper": float(np.exp(self.ci_upper)),
            "tau": self.tau,
            "ci_tau_lower": self.ci_lower,
            "ci_tau_upper": self.ci_upper,
        }

    def summary(self) -> str:
        rr = np.exp(self.tau)
        rr_lo = np.exp(self.ci_lower)
        rr_hi = np.exp(self.ci_upper)
        lines = [
            f"{self.outcome_type.title()} Local GLM RDD",
            f"Bandwidth: {self.bandwidth:.4f}",
            f"N: {self.n_left + self.n_right} ({self.n_left} left, {self.n_right} right)",
            f"Bootstrap replications: {self.n_boot}",
            f"Converged: {self.converged}",
            "",
            f"  tau (log scale):  {self.tau:.4f}",
            f"  Bootstrap SE:     {self.se_boot:.4f}",
            f"  95% CI (log):     [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  p-value:          {self.p_value:.4f}",
            "",
            f"  Rate ratio:       {rr:.4f}",
            f"  95% CI (ratio):   [{rr_lo:.4f}, {rr_hi:.4f}]",
        ]
        return "\n".join(lines)


def _triangular_weights(x: NDArray, cutoff: float, h: float) -> NDArray[np.float64]:
    """Triangular kernel weights centred at cutoff with bandwidth h."""
    u = np.abs(x - cutoff) / h
    w = np.maximum(1.0 - u, 0.0)
    return w


def _epanechnikov_weights(x: NDArray, cutoff: float, h: float) -> NDArray[np.float64]:
    """Epanechnikov kernel weights."""
    u = np.abs(x - cutoff) / h
    w = np.where(u <= 1.0, 0.75 * (1.0 - u**2), 0.0)
    return w


def _uniform_weights(x: NDArray, cutoff: float, h: float) -> NDArray[np.float64]:
    """Uniform kernel weights."""
    return np.where(np.abs(x - cutoff) <= h, 1.0, 0.0)


def _kernel_weights(
    x: NDArray, cutoff: float, h: float, kernel: str
) -> NDArray[np.float64]:
    """Dispatch to kernel weight function."""
    if kernel == "triangular":
        return _triangular_weights(x, cutoff, h)
    elif kernel == "epanechnikov":
        return _epanechnikov_weights(x, cutoff, h)
    elif kernel == "uniform":
        return _uniform_weights(x, cutoff, h)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Use 'triangular', 'epanechnikov', or 'uniform'.")


def _build_design_matrix(
    x: NDArray, cutoff: float, p: int
) -> NDArray[np.float64]:
    """Build local polynomial design matrix [1, (x-c), (x-c)^2, ..., (x-c)^p]."""
    xc = x - cutoff
    cols = [np.ones_like(xc)]
    for k in range(1, p + 1):
        cols.append(xc**k)
    return np.column_stack(cols)


def _poisson_nll(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_offset: NDArray,
    kw: NDArray,
) -> float:
    """Kernel-weighted negative Poisson log-likelihood.

    Objective: - sum_i kw_i * [y_i*(eta_i) - exp(eta_i)]
    where eta_i = log(offset_i) + X_i @ beta
    """
    eta = log_offset + X @ beta
    # Clip to prevent exp overflow
    eta = np.clip(eta, -500, 500)
    mu = np.exp(eta)
    ll = y * eta - mu  # Poisson log-likelihood (drop log(y!) constant)
    return -float(np.sum(kw * ll))


def _poisson_grad(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    log_offset: NDArray,
    kw: NDArray,
) -> NDArray[np.float64]:
    """Gradient of kernel-weighted Poisson negative log-likelihood."""
    eta = log_offset + X @ beta
    eta = np.clip(eta, -500, 500)
    mu = np.exp(eta)
    residuals = y - mu  # score = y - mu
    grad = -X.T @ (kw * residuals)
    return grad


def _gamma_nll(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    kw: NDArray,
    dispersion: float = 1.0,
) -> float:
    """Kernel-weighted negative Gamma log-likelihood (log-link, unit dispersion).

    log E[Y|X] = X @ beta  =>  mu = exp(X @ beta)
    Log-likelihood (shape=1/dispersion, rate=1/(mu*dispersion)):
      ll_i = -log(mu_i) - y_i/mu_i  (up to constants not depending on beta)
    """
    eta = X @ beta
    eta = np.clip(eta, -500, 500)
    mu = np.exp(eta)
    # Gamma log-likelihood core terms (proportional to)
    ll = -eta - y / mu
    return -float(np.sum(kw * ll))


def _gamma_grad(
    beta: NDArray,
    X: NDArray,
    y: NDArray,
    kw: NDArray,
    dispersion: float = 1.0,
) -> NDArray[np.float64]:
    """Gradient of kernel-weighted Gamma negative log-likelihood."""
    eta = X @ beta
    eta = np.clip(eta, -500, 500)
    mu = np.exp(eta)
    # d/dbeta [-eta - y/mu] = [-1 + y/mu] * d(eta)/d(beta) = [-1 + y/mu] * X
    # score = y/mu - 1
    residuals = y / mu - 1.0
    grad = -X.T @ (kw * residuals)
    return grad


def _fit_one_side(
    x_side: NDArray,
    y_side: NDArray,
    exposure_side: NDArray | None,
    kw_side: NDArray,
    cutoff: float,
    p: int,
    outcome_type: str,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """Fit local polynomial GLM on one side of the cutoff.

    Returns
    -------
    (coef, converged): polynomial coefficients and convergence flag.
    """
    X = _build_design_matrix(x_side, cutoff, p)

    if x0 is None:
        x0 = np.zeros(p + 1)

    if outcome_type == "poisson":
        log_offset = np.log(exposure_side) if exposure_side is not None else np.zeros(len(y_side))
        res = optimize.minimize(
            _poisson_nll,
            x0,
            args=(X, y_side, log_offset, kw_side),
            jac=_poisson_grad,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
    elif outcome_type == "gamma":
        # Filter positives only for gamma (severity conditioning on claim)
        pos = y_side > 0
        if pos.sum() < p + 2:
            return np.full(p + 1, np.nan), False
        X_pos = X[pos]
        y_pos = y_side[pos]
        kw_pos = kw_side[pos]
        res = optimize.minimize(
            _gamma_nll,
            x0,
            args=(X_pos, y_pos, kw_pos),
            jac=_gamma_grad,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
        )
    else:
        raise ValueError(f"outcome_type must be 'poisson' or 'gamma', got '{outcome_type}'.")

    return np.asarray(res.x), res.success


def _auto_bandwidth(
    x: NDArray, cutoff: float, outcome_type: str
) -> float:
    """Simple data-driven bandwidth: Silverman's rule adapted for RDD.

    For a proper MSE-optimal bandwidth use InsuranceRD (wraps rdbwselect).
    This is a fallback when h is not provided to PoissonRD/GammaRD.
    """
    x_left = x[x < cutoff]
    x_right = x[x >= cutoff]

    n_l = len(x_left)
    n_r = len(x_right)

    if n_l < 5 or n_r < 5:
        return float(np.std(x) * len(x) ** (-0.2))

    # Use IQR-based bandwidth on each side, take the minimum.
    def silverman(v: NDArray) -> float:
        n = len(v)
        s = np.std(v, ddof=1)
        iqr_bw = np.percentile(v, 75) - np.percentile(v, 25)
        bw = 0.9 * min(s, iqr_bw / 1.34) * n ** (-0.2)
        return float(bw) if bw > 0 else float(s)

    return min(silverman(x_left), silverman(x_right))


class PoissonRD:
    """Local Poisson regression RDD with exposure offset.

    This is methodologically correct for claim frequency data:
    - Fits Poisson log-likelihood separately on each side of the cutoff
    - Applies kernel weights (triangular by default)
    - Includes log(exposure) as offset term
    - Bootstraps confidence intervals

    The treatment effect tau is the log rate ratio at the cutoff.
    exp(tau) is the causal frequency rate ratio.

    Parameters
    ----------
    outcome:
        Column name of the claim count outcome.
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame with all variables.
    exposure:
        Column name of exposure (policy-years). If None, no offset is used
        (i.e. exposure is assumed equal for all observations).
    kernel:
        Kernel type: 'triangular' | 'epanechnikov' | 'uniform'.
    p:
        Local polynomial order (1 recommended).
    h:
        Bandwidth. If None, estimated from data via simple rule-of-thumb
        (use InsuranceRD for CCT-optimal bandwidth).
    donut_radius:
        Exclude observations within this distance of the cutoff.
    n_boot:
        Number of bootstrap replications for CI.
    confidence:
        Confidence level (default 0.95).
    rng:
        Random number generator for reproducibility.

    Examples
    --------
    >>> from insurance_rdd.outcomes import PoissonRD
    >>> rd = PoissonRD(
    ...     outcome='claim_count',
    ...     running_var='driver_age_months',
    ...     cutoff=300,
    ...     data=df,
    ...     exposure='exposure_years',
    ... )
    >>> result = rd.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        exposure: str | None = None,
        kernel: str = "triangular",
        p: int = 1,
        h: float | None = None,
        donut_radius: float = 0.0,
        n_boot: int = 500,
        confidence: float = 0.95,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.exposure = exposure
        self.kernel = kernel
        self.p = p
        self.h = h
        self.donut_radius = donut_radius
        self.n_boot = n_boot
        self.confidence = confidence
        self.rng = rng or np.random.default_rng()
        self._result: LocalGLMResult | None = None

    def fit(self) -> LocalGLMResult:
        """Fit Poisson RDD and bootstrap confidence intervals."""
        df = self.data.copy()

        # Donut exclusion.
        if self.donut_radius > 0:
            dist = np.abs(df[self.running_var] - self.cutoff)
            df = df[dist >= self.donut_radius].copy()

        # Drop missing.
        cols = [self.outcome, self.running_var]
        if self.exposure:
            cols.append(self.exposure)
        df = df.dropna(subset=cols)

        x = df[self.running_var].to_numpy(dtype=float)
        y = df[self.outcome].to_numpy(dtype=float)
        exposure_arr = (
            df[self.exposure].to_numpy(dtype=float)
            if self.exposure
            else None
        )

        h = self.h if self.h is not None else _auto_bandwidth(x, self.cutoff, "poisson")

        tau, coef_left, coef_right, converged = self._fit_tau(
            x, y, exposure_arr, h
        )

        # Within-bandwidth samples for reporting.
        in_bw = np.abs(x - self.cutoff) <= h
        left_mask = in_bw & (x < self.cutoff)
        right_mask = in_bw & (x >= self.cutoff)

        # Bootstrap.
        boot_taus = self._bootstrap(x, y, exposure_arr, h)
        alpha = 1.0 - self.confidence
        ci_lower = float(np.nanpercentile(boot_taus, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(boot_taus, 100 * (1.0 - alpha / 2)))
        se_boot = float(np.nanstd(boot_taus, ddof=1))

        # p-value: proportion of bootstrap draws on opposite side of zero.
        shifted = boot_taus - np.nanmean(boot_taus)
        p_value = float(2.0 * np.mean(np.abs(shifted) >= np.abs(tau)))
        p_value = max(p_value, 2.0 / self.n_boot)

        self._result = LocalGLMResult(
            tau=tau,
            se_boot=se_boot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            bandwidth=h,
            n_left=int(np.sum(left_mask)),
            n_right=int(np.sum(right_mask)),
            outcome_type="poisson",
            n_boot=self.n_boot,
            converged=converged,
            coef_left=coef_left,
            coef_right=coef_right,
            _estimator=self,
        )
        return self._result

    def _fit_tau(
        self,
        x: NDArray,
        y: NDArray,
        exposure: NDArray | None,
        h: float,
    ) -> tuple[float, NDArray, NDArray, bool]:
        """Core fitting routine: returns (tau, coef_left, coef_right, converged)."""
        kw = _kernel_weights(x, self.cutoff, h, self.kernel)

        left = x < self.cutoff
        right = x >= self.cutoff

        # Need at least p+2 observations on each side with positive weight.
        in_bw_left = left & (kw > 0)
        in_bw_right = right & (kw > 0)

        if in_bw_left.sum() < self.p + 2 or in_bw_right.sum() < self.p + 2:
            return np.nan, np.full(self.p + 1, np.nan), np.full(self.p + 1, np.nan), False

        exp_l = exposure[in_bw_left] if exposure is not None else None
        exp_r = exposure[in_bw_right] if exposure is not None else None

        coef_l, conv_l = _fit_one_side(
            x[in_bw_left], y[in_bw_left], exp_l, kw[in_bw_left],
            self.cutoff, self.p, "poisson"
        )
        coef_r, conv_r = _fit_one_side(
            x[in_bw_right], y[in_bw_right], exp_r, kw[in_bw_right],
            self.cutoff, self.p, "poisson"
        )

        if np.any(np.isnan(coef_l)) or np.any(np.isnan(coef_r)):
            return np.nan, coef_l, coef_r, False

        # tau = intercept_right - intercept_left (log rate ratio at cutoff)
        tau = float(coef_r[0] - coef_l[0])
        return tau, coef_l, coef_r, conv_l and conv_r

    def _bootstrap(
        self,
        x: NDArray,
        y: NDArray,
        exposure: NDArray | None,
        h: float,
    ) -> NDArray[np.float64]:
        """Nonparametric bootstrap of tau."""
        n = len(x)
        boot_taus = np.full(self.n_boot, np.nan)

        for b in range(self.n_boot):
            idx = self.rng.integers(0, n, size=n)
            x_b = x[idx]
            y_b = y[idx]
            exp_b = exposure[idx] if exposure is not None else None

            tau_b, _, _, _ = self._fit_tau(x_b, y_b, exp_b, h)
            boot_taus[b] = tau_b

        return boot_taus


class GammaRD:
    """Local Gamma regression RDD for claim severity.

    Estimates the causal effect of a threshold on average claim severity,
    conditioning on positive claims (S > 0). The treatment effect is a log
    severity ratio — exp(tau) tells you how average claim cost changes at
    the cutoff.

    Parameters
    ----------
    outcome:
        Column name of the claim severity (cost) variable.
    running_var:
        Column name of the running variable.
    cutoff:
        Cutoff value.
    data:
        DataFrame with all variables.
    severity_only:
        If True (default), fit only on rows where outcome > 0. This is
        correct for severity analysis: Gamma cannot handle zero amounts.
        If False, zeros are included and treated as near-zero (adds 1e-8).
    kernel:
        Kernel type: 'triangular' | 'epanechnikov' | 'uniform'.
    p:
        Local polynomial order (1 recommended).
    h:
        Bandwidth. If None, estimated from data.
    donut_radius:
        Exclude observations within this distance of the cutoff.
    n_boot:
        Number of bootstrap replications.
    confidence:
        Confidence level.
    rng:
        Random number generator.

    Examples
    --------
    >>> from insurance_rdd.outcomes import GammaRD
    >>> rd = GammaRD(
    ...     outcome='claim_amount',
    ...     running_var='driver_age_months',
    ...     cutoff=300,
    ...     data=df[df['claim_amount'] > 0],
    ... )
    >>> result = rd.fit()
    >>> print(result.rate_ratio())
    """

    def __init__(
        self,
        outcome: str,
        running_var: str,
        cutoff: float,
        data: pd.DataFrame,
        severity_only: bool = True,
        kernel: str = "triangular",
        p: int = 1,
        h: float | None = None,
        donut_radius: float = 0.0,
        n_boot: int = 500,
        confidence: float = 0.95,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.data = data.copy()
        self.severity_only = severity_only
        self.kernel = kernel
        self.p = p
        self.h = h
        self.donut_radius = donut_radius
        self.n_boot = n_boot
        self.confidence = confidence
        self.rng = rng or np.random.default_rng()
        self._result: LocalGLMResult | None = None

    def fit(self) -> LocalGLMResult:
        """Fit Gamma RDD and bootstrap confidence intervals."""
        df = self.data.copy()

        if self.donut_radius > 0:
            dist = np.abs(df[self.running_var] - self.cutoff)
            df = df[dist >= self.donut_radius].copy()

        df = df.dropna(subset=[self.outcome, self.running_var])

        if self.severity_only:
            df = df[df[self.outcome] > 0].copy()
        else:
            # Avoid exact zeros in Gamma
            df[self.outcome] = df[self.outcome].clip(lower=1e-8)

        if len(df) < (self.p + 2) * 2:
            raise ValueError(
                f"Insufficient data for GammaRD: {len(df)} observations after filtering."
            )

        x = df[self.running_var].to_numpy(dtype=float)
        y = df[self.outcome].to_numpy(dtype=float)

        h = self.h if self.h is not None else _auto_bandwidth(x, self.cutoff, "gamma")

        tau, coef_left, coef_right, converged = self._fit_tau(x, y, h)

        in_bw = np.abs(x - self.cutoff) <= h
        n_left = int(np.sum(in_bw & (x < self.cutoff)))
        n_right = int(np.sum(in_bw & (x >= self.cutoff)))

        boot_taus = self._bootstrap(x, y, h)
        alpha = 1.0 - self.confidence
        ci_lower = float(np.nanpercentile(boot_taus, 100 * alpha / 2))
        ci_upper = float(np.nanpercentile(boot_taus, 100 * (1.0 - alpha / 2)))
        se_boot = float(np.nanstd(boot_taus, ddof=1))

        shifted = boot_taus - np.nanmean(boot_taus)
        p_value = float(2.0 * np.mean(np.abs(shifted) >= np.abs(tau)))
        p_value = max(p_value, 2.0 / self.n_boot)

        self._result = LocalGLMResult(
            tau=tau,
            se_boot=se_boot,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            bandwidth=h,
            n_left=n_left,
            n_right=n_right,
            outcome_type="gamma",
            n_boot=self.n_boot,
            converged=converged,
            coef_left=coef_left,
            coef_right=coef_right,
            _estimator=self,
        )
        return self._result

    def _fit_tau(
        self,
        x: NDArray,
        y: NDArray,
        h: float,
    ) -> tuple[float, NDArray, NDArray, bool]:
        kw = _kernel_weights(x, self.cutoff, h, self.kernel)
        left = (x < self.cutoff) & (kw > 0)
        right = (x >= self.cutoff) & (kw > 0)

        if left.sum() < self.p + 2 or right.sum() < self.p + 2:
            return np.nan, np.full(self.p + 1, np.nan), np.full(self.p + 1, np.nan), False

        coef_l, conv_l = _fit_one_side(
            x[left], y[left], None, kw[left], self.cutoff, self.p, "gamma"
        )
        coef_r, conv_r = _fit_one_side(
            x[right], y[right], None, kw[right], self.cutoff, self.p, "gamma"
        )

        if np.any(np.isnan(coef_l)) or np.any(np.isnan(coef_r)):
            return np.nan, coef_l, coef_r, False

        tau = float(coef_r[0] - coef_l[0])
        return tau, coef_l, coef_r, conv_l and conv_r

    def _bootstrap(self, x: NDArray, y: NDArray, h: float) -> NDArray[np.float64]:
        n = len(x)
        boot_taus = np.full(self.n_boot, np.nan)

        for b in range(self.n_boot):
            idx = self.rng.integers(0, n, size=n)
            tau_b, _, _, _ = self._fit_tau(x[idx], y[idx], h)
            boot_taus[b] = tau_b

        return boot_taus
