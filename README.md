# insurance-rdd

**Regression Discontinuity Design for insurance pricing thresholds.**

UK insurers embed pricing thresholds — age 25 young driver cliffs, NCD step boundaries, vehicle age cutoffs, territory lines. These generate discontinuous premiums assumed to reflect discontinuous underlying risk. That assumption is almost never formally tested.

Regression Discontinuity Design is the only credible method for this test. It uses the discontinuity in treatment assignment at the threshold to identify a causal effect locally, without randomisation. The question for each threshold: does crossing it cause a corresponding jump in claims outcomes, or is the premium jump larger than the risk justifies?

This library fills the Python gap. `rdrobust` handles Gaussian outcomes. We add exposure weighting, Poisson/Gamma local GLM, geographic territory boundary RDD, multi-cutoff pooling, and FCA Consumer Duty output.

## The problem this solves

Standard RDD packages (rdrobust, rddensity) assume:
- Outcome is approximately Gaussian
- Observations have equal weight

Insurance data is neither. Claim counts are Poisson with variable exposure (0.1–1.0 policy-years). Running OLS on claims per policy without accounting for exposure differences produces biased estimates and wrong standard errors. There is no existing Python package that handles this correctly.

Additionally, no Python implementation of geographic RDD exists. SpatialRDD is R-only. If you want to test whether a territory boundary discontinuity in claims matches the boundary in premiums, you previously had no Python tool.

## Installation

```bash
pip install insurance-rdd
```

For geographic RDD:
```bash
pip install insurance-rdd[geo]
```

## Quick start

### Age 25 threshold (Poisson, exposure offset)

```python
from insurance_rdd import InsuranceRD, presets

rd = InsuranceRD(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,                     # 25 * 12 months
    data=df,
    outcome_type='poisson',
    exposure='exposure_years',
    preset=presets.AGE_25,          # loads: donut=3mo, recommended bandwidth, FCA context
)
result = rd.fit()
print(result.summary())
print(result.rate_ratio())          # exp(tau): causal rate ratio at cutoff
print(result.regulatory_report(tariff_relativity=0.70))
```

Output:
```
InsuranceRD — claim_count ~ RD at driver_age_months = 300.0
Outcome type : poisson
Bandwidth (h): 28.451  [left / right]
N in bandwidth: 1842 (921 left, 921 right)

                                        Estimate    Std Err              95% CI    p-value
----------------------------------------------------------------------------------------
  tau (robust BC)                         -0.3214     0.0421   [-0.4039, -0.2389]     0.0000

  Rate ratio (exp(tau)): 0.7245  (95% CI: 0.6679, 0.7869)
  Interpretation: crossing the cutoff multiplies the claims rate by this factor.
```

### NCD transitions (multi-cutoff)

```python
from insurance_rdd import MultiCutoffRD

mc_rd = MultiCutoffRD(
    outcome='claim_count',
    running_var='ncd_level',
    cutoffs=[1, 2, 3, 4, 5],
    data=df,
    outcome_type='poisson',
    exposure='exposure_years',
)
result = mc_rd.fit()
print(result.summary())
result.cutoff_effects_df()          # per-cutoff table
result.pooled_effect()              # inverse-variance weighted pooled estimate
```

### Local Poisson GLM (correct likelihood, not weighted OLS)

```python
from insurance_rdd.outcomes import PoissonRD

rd = PoissonRD(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,
    data=df,
    exposure='exposure_years',
    n_boot=500,
)
result = rd.fit()
print(result.summary())
print(result.rate_ratio())
```

`PoissonRD` fits the Poisson log-likelihood directly rather than transforming to rates and running weighted OLS. The difference matters when exposure varies substantially across the bandwidth, or when claim counts are sparse.

### Claim severity (Gamma outcome)

```python
from insurance_rdd.outcomes import GammaRD

rd = GammaRD(
    outcome='claim_amount',
    running_var='driver_age_months',
    cutoff=300,
    data=df[df.claim_amount > 0],   # severity conditioning on claim
)
result = rd.fit()
print(result.rate_ratio())          # severity ratio at the cutoff
```

### Territory boundary (Geographic RDD)

```python
from insurance_rdd import GeographicRD

# If you have pre-computed signed distances:
geo_rd = GeographicRD(
    outcome='claim_count',
    treatment_col='territory_band',
    data=df,
    outcome_type='poisson',
    exposure='exposure_years',
    signed_distance_col='dist_to_boundary_m',  # negative = territory A
    border_segment_fes=False,
)

# With a boundary shapefile (requires geopandas):
geo_rd = GeographicRD(
    outcome='claim_count',
    treatment_col='territory_band',
    data=df,
    boundary_file='territory_AB.geojson',
    lat_col='lat',
    lon_col='lon',
    outcome_type='poisson',
    exposure='exposure_years',
    border_segment_fes=True,
    n_segments=10,
)
result = geo_rd.fit()
print(result.summary())
```

### Validity checks

```python
from insurance_rdd.validity import DensityTest, CovariateBalance, PlaceboTest

# McCrary manipulation test.
dt = DensityTest(running_var='driver_age_months', cutoff=300, data=df)
print(dt.fit().summary())

# Covariate balance (predetermined covariates should be smooth at cutoff).
cb = CovariateBalance(
    covariates=['vehicle_group', 'years_licensed'],
    running_var='driver_age_months',
    cutoff=300,
    data=df,
)
print(cb.fit().summary())

# Placebo test at false cutoffs.
pt = PlaceboTest(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,
    data=df,
    placebo_cutoffs=[264, 282, 318, 336],
    true_tau=result.tau_bc,
)
print(pt.fit().summary())
```

### Regulatory report

```python
from insurance_rdd.report import ThresholdReport, ThresholdReportData

report = ThresholdReport(
    ThresholdReportData(
        rd_result=result,
        density_result=density_result,
        balance_result=balance_result,
        tariff_relativity=0.70,     # your tariff factor at this threshold
        threshold_name='age 25 driver boundary',
    )
)
print(report.markdown())
report.save('age25_threshold_report.md')
report.save('age25_threshold_report.html')
```

The report compares your tariff relativity to the empirical causal rate ratio. If the tariff factor implies a larger premium reduction than the data supports, the report flags it as a potential FCA Consumer Duty concern and quantifies the discrepancy.

## Presets

```python
from insurance_rdd import presets

presets.AGE_25          # age 25 threshold, running variable in months
presets.AGE_25_YEARS    # age 25, running variable in years (discrete)
presets.NCD_STEP        # any NCD level transition
presets.NCD_MAX         # NCD 4→5 (manipulation expected, documented)
presets.VEHICLE_AGE_10  # vehicle age 10 years
presets.VEHICLE_AGE_15  # vehicle age 15 years
presets.TENURE_3        # policy tenure 3 years
presets.SUM_INSURED_100K  # sum insured £100k band
```

Each preset loads: cutoff value, recommended kernel, donut radius, bandwidth selector, FCA context, and methodological notes.

## Design choices

**Exposure weighting**: `InsuranceRD` passes `exposure` as weights to rdrobust and converts counts to rates. This is the weighted OLS approximation, which is correct when the mean-variance relationship is approximately proportional. For sparse count data or highly variable exposure, use `PoissonRD` which fits the Poisson log-likelihood directly.

**Bootstrap CIs for GLM outcomes**: `PoissonRD` and `GammaRD` use nonparametric bootstrap (500 replications by default). Analytical bias correction for non-Gaussian local regression requires adapting the CCT derivation to the score function of the GLM — non-trivial and deferred to v0.2. The bootstrap is honest and interpretation is straightforward.

**Donut RDD**: Many insurance running variables are reported at integer precision (driver age in years, vehicle year). This creates heaping at the integers nearest to the cutoff. The donut exclusion removes a neighbourhood around the cutoff before estimation. The default donut for `AGE_25` is 3 months.

**Geographic RDD**: We compute signed distance to the boundary line using geopandas and pyproj, reproject to the local UTM zone for accurate metre-scale distances, then run standard rdrobust with signed distance as the running variable. Border segment fixed effects are added by dividing the boundary into segments and one-hot encoding segment membership as covariates.

## FCA Consumer Duty context

FCA Consumer Duty (PS22/9, effective July 2023) requires insurers to demonstrate that pricing factors are risk-reflective. RDD provides causal evidence at each pricing threshold. The formal test is:

```
H0: log(tariff relativity at c) = tau_hat_poisson
```

If the tariff applies a 30% premium reduction at age 25 (relativity = 0.70, log = -0.357) and the empirical Poisson RDD estimates tau = -0.32 (rate ratio = 0.726), these are consistent. If the tariff applies a 50% reduction (relativity = 0.50, log = -0.693) for a 25% empirical claims reduction (tau = -0.29), that is evidence of over-pricing at the threshold.

## Methodology

Core estimation uses rdrobust's CCT bias-corrected robust inference (Calonico, Cattaneo, Titiunik 2014, *Econometrica* 82(6):2295-2326). We do not reimplement the CCT bandwidth selection or bias correction — rdrobust does this correctly and efficiently.

Our additions:
- Local Poisson/Gamma GLM via scipy.optimize (correct log-likelihood, not OLS on log Y)
- Exposure offset in Poisson log-likelihood
- Geographic distance computation and border segment fixed effects
- Multi-cutoff inverse-variance pooling
- Insurance domain presets and FCA regulatory output

## References

- Calonico, Cattaneo, Titiunik (2014) *Econometrica* 82(6):2295-2326 — CCT bias-corrected inference
- Lee & Lemieux (2010) *Journal of Economic Literature* 48(2):281-355 — comprehensive survey
- Keele & Titiunik (2015) *Political Analysis* 23(1):127-155 — geographic RDD
- Souza et al. (2025) *Health Economics* DOI:10.1002/hec.4898 — only published insurance RDD
- Cattaneo, Jansson, Ma (2018) — rddensity manipulation test
- Artis, Ayuso, Guillén (2002) — NCD claim withholding evidence

---

*Burning Cost — pricing tools that actuaries can actually use.*
