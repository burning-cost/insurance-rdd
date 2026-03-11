"""Shared fixtures for insurance-rdd tests.

All synthetic data is generated with known data-generating processes so tests
can verify that estimates recover the true treatment effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


RNG = np.random.default_rng(42)


def make_sharp_rdd_data(
    n: int = 2000,
    cutoff: float = 0.0,
    true_tau: float = 0.3,
    outcome_type: str = "gaussian",
    noise: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic sharp RDD data with a known treatment effect.

    Parameters
    ----------
    n:
        Number of observations.
    cutoff:
        Running variable cutoff.
    true_tau:
        True log treatment effect (rate ratio on log scale for Poisson/Gamma,
        additive for Gaussian).
    outcome_type:
        'gaussian', 'poisson', or 'gamma'.
    noise:
        Noise level.
    seed:
        Random seed.

    Returns
    -------
    DataFrame with columns: x (running variable), y (outcome), exposure,
    treatment (0/1), covariate_1, covariate_2.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-3.0, 3.0, size=n)
    treatment = (x >= cutoff).astype(float)

    # Smooth baseline: log-linear in x.
    baseline_log = 0.5 + 0.2 * x - 0.05 * x**2

    # Exposure: variable policy-years.
    exposure = rng.uniform(0.1, 1.0, size=n)

    if outcome_type == "gaussian":
        y = baseline_log + true_tau * treatment + noise * rng.standard_normal(n)
    elif outcome_type == "poisson":
        log_rate = baseline_log + true_tau * treatment
        mu = np.exp(log_rate) * exposure
        y = rng.poisson(mu).astype(float)
    elif outcome_type == "gamma":
        log_mean = baseline_log + true_tau * treatment
        mu = np.exp(log_mean)
        shape = 3.0
        y = rng.gamma(shape, mu / shape, size=n)
    else:
        raise ValueError(f"Unknown outcome_type '{outcome_type}'")

    # Predetermined covariates (should be smooth at cutoff).
    covariate_1 = 2.0 + 0.1 * x + 0.1 * rng.standard_normal(n)
    covariate_2 = rng.binomial(1, 0.4, size=n).astype(float)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "exposure": exposure,
            "treatment": treatment,
            "covariate_1": covariate_1,
            "covariate_2": covariate_2,
        }
    )


def make_age_rdd_data(
    n: int = 3000,
    cutoff_months: float = 300.0,
    true_tau: float = -0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate motor insurance data with an age-25 discontinuity.

    Driver age is in months (range 216-420 = ages 18-35).
    True treatment effect: log rate ratio of -0.35 (rate ratio 0.70) at age 25.
    """
    rng = np.random.default_rng(seed)

    # Driver ages: uniform between 18 and 35 years (216-420 months).
    age_months = rng.uniform(216, 420, size=n)
    above_25 = (age_months >= cutoff_months).astype(float)

    exposure = rng.uniform(0.1, 1.0, size=n)

    # Log claim rate: higher for young drivers (below 25 = above the threshold in log space).
    log_rate = -1.5 + 0.005 * (age_months - cutoff_months) + true_tau * above_25
    mu = np.exp(log_rate) * exposure
    claim_count = rng.poisson(mu).astype(float)

    # Severity (gamma conditional on claim).
    claim_amount = np.where(
        claim_count > 0,
        rng.gamma(3.0, 2000.0, size=n),
        0.0,
    )

    # Vehicle group (predetermined covariate — should be smooth).
    vehicle_group = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # Region (predetermined).
    region = rng.choice(["London", "SE", "Midlands", "North", "Scotland"], size=n)

    return pd.DataFrame(
        {
            "driver_age_months": age_months,
            "claim_count": claim_count,
            "claim_amount": claim_amount,
            "exposure_years": exposure,
            "above_25": above_25,
            "vehicle_group": vehicle_group.astype(float),
            "region": region,
        }
    )


def make_ncd_data(
    n: int = 5000,
    cutoffs: list[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate NCD-level data with effects at each step boundary."""
    if cutoffs is None:
        cutoffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    rng = np.random.default_rng(seed)

    ncd_level = rng.integers(0, 6, size=n).astype(float)
    exposure = rng.uniform(0.1, 1.0, size=n)

    # Log claim rate decreases with NCD (lower risk at higher NCD).
    log_rate = -1.0 - 0.15 * ncd_level
    mu = np.exp(log_rate) * exposure
    claim_count = rng.poisson(mu).astype(float)

    return pd.DataFrame(
        {
            "ncd_level": ncd_level,
            "claim_count": claim_count,
            "exposure_years": exposure,
        }
    )


def make_multicutoff_data(
    n: int = 6000,
    cutoffs: list[float] = None,
    true_taus: list[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data with effects at multiple running variable cutoffs."""
    if cutoffs is None:
        cutoffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    if true_taus is None:
        true_taus = [-0.15, -0.12, -0.10, -0.08, -0.08]

    rng = np.random.default_rng(seed)

    x = rng.uniform(-0.4, 5.9, size=n)  # NCD-like continuous variable
    exposure = rng.uniform(0.1, 1.0, size=n)

    # Cumulative effect at each cutoff.
    log_rate = -1.5 + 0.0 * x  # flat baseline
    for c, tau in zip(cutoffs, true_taus):
        log_rate += tau * (x >= c).astype(float)

    mu = np.exp(log_rate) * exposure
    claim_count = rng.poisson(mu).astype(float)

    return pd.DataFrame(
        {
            "x": x,
            "claim_count": claim_count,
            "exposure_years": exposure,
        }
    )


@pytest.fixture
def gaussian_df() -> pd.DataFrame:
    return make_sharp_rdd_data(n=2000, outcome_type="gaussian", true_tau=0.4)


@pytest.fixture
def poisson_df() -> pd.DataFrame:
    return make_sharp_rdd_data(n=3000, outcome_type="poisson", true_tau=0.3)


@pytest.fixture
def gamma_df() -> pd.DataFrame:
    return make_sharp_rdd_data(n=3000, outcome_type="gamma", true_tau=0.25)


@pytest.fixture
def age_df() -> pd.DataFrame:
    return make_age_rdd_data(n=3000)


@pytest.fixture
def ncd_df() -> pd.DataFrame:
    return make_ncd_data(n=5000)


@pytest.fixture
def multicutoff_df() -> pd.DataFrame:
    return make_multicutoff_data(n=6000)
