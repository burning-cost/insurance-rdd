"""insurance-rdd: Regression Discontinuity Design for insurance pricing thresholds.

UK insurers embed dozens of rating thresholds — age 25 young driver cliffs,
NCD step boundaries, vehicle age cutoffs, territory boundary lines. These
generate discontinuous premiums assumed to reflect discontinuous underlying
risk. That assumption is almost never formally tested.

Regression Discontinuity Design (RDD) is the credible method for this test.
It exploits the discontinuity in treatment assignment at the threshold to
identify a causal effect locally — without randomisation. The key question
for each threshold: does crossing it cause a corresponding jump in claims
outcomes, or is the premium jump larger than the risk warrants?

This library fills the Python gap: rdrobust handles Gaussian outcomes; we add
exposure weighting, Poisson/Gamma local GLM, geographic territory boundary RDD,
multi-cutoff pooling, and FCA Consumer Duty formatted output.

Quick start
-----------
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

Key classes
-----------
InsuranceRD
    Main estimator. Wraps rdrobust with insurance defaults.
    Handles Gaussian, Poisson (as weighted regression), and Gamma outcomes.

PoissonRD, GammaRD
    Local GLM regression RDD with correct Poisson/Gamma log-likelihood.
    Exposure offset support. Bootstrap CIs.

MultiCutoffRD
    Pool RDD estimates across multiple thresholds via inverse-variance weighting.
    Designed for NCD level transitions and other multi-step rating factors.

GeographicRD
    Spatial RDD at postcode/territory boundaries.
    Uses geopandas to compute signed distance to boundary, then runs standard RDD.
    The first Python implementation of geographic RDD.

DensityTest
    McCrary/rddensity manipulation test.

CovariateBalance
    Test that predetermined covariates are smooth at the cutoff.

PlaceboTest
    Test for spurious effects at false cutoffs.

ThresholdReport
    Combined report: RDD estimate + validity tests + FCA Consumer Duty framing.

presets
    ThresholdPreset objects for known UK thresholds:
    AGE_25, NCD_STEP, NCD_MAX, VEHICLE_AGE_10, VEHICLE_AGE_15, TENURE_3.
"""

from .core import InsuranceRD, RDResult
from .outcomes import PoissonRD, GammaRD, LocalGLMResult
from .multicutoff import MultiCutoffRD, MultiCutoffRDResult
from .validity import (
    DensityTest,
    DensityTestResult,
    CovariateBalance,
    CovariateBalanceResult,
    PlaceboTest,
    PlaceboTestResult,
    DonutRDD,
    DonutRDDResult,
)
from .plots import (
    RDPlot,
    DensityPlot,
    BandwidthSensitivityPlot,
    CovariateBalancePlot,
    MultiCutoffPlot,
)
from .report import ThresholdReport, ThresholdReportData
from .geographic import GeographicRD, GeographicRDResult
from . import presets
from .presets import ThresholdPreset, PRESETS, get_preset

__version__ = "0.1.0"

__all__ = [
    # Core estimator
    "InsuranceRD",
    "RDResult",
    # Local GLM outcomes
    "PoissonRD",
    "GammaRD",
    "LocalGLMResult",
    # Multi-cutoff
    "MultiCutoffRD",
    "MultiCutoffRDResult",
    # Validity tests
    "DensityTest",
    "DensityTestResult",
    "CovariateBalance",
    "CovariateBalanceResult",
    "PlaceboTest",
    "PlaceboTestResult",
    "DonutRDD",
    "DonutRDDResult",
    # Plots
    "RDPlot",
    "DensityPlot",
    "BandwidthSensitivityPlot",
    "CovariateBalancePlot",
    "MultiCutoffPlot",
    # Report
    "ThresholdReport",
    "ThresholdReportData",
    # Geographic
    "GeographicRD",
    "GeographicRDResult",
    # Presets
    "presets",
    "ThresholdPreset",
    "PRESETS",
    "get_preset",
    # Version
    "__version__",
]
