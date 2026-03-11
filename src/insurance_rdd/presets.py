"""Insurance threshold presets for common UK rating cutoffs.

These presets capture accumulated knowledge about the data characteristics
at each threshold — expected bandwidth ranges, heaping patterns, donut sizes,
and regulatory context. Using a preset is not mandatory, but it loads sensible
defaults so you don't have to remember that the age-25 boundary needs a donut
of 3 months because age is reported in integer years.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThresholdPreset:
    """Configuration defaults for a known insurance rating threshold.

    Attributes
    ----------
    name:
        Human-readable name for the threshold.
    description:
        What this threshold represents and why it exists.
    cutoff:
        The default cutoff value on the running variable scale.
    running_var_label:
        Description of the running variable (for plot axes).
    outcome_type:
        Default outcome family ('poisson', 'gamma', 'gaussian').
    bwselect:
        Recommended rdrobust bandwidth selector.
    kernel:
        Recommended kernel type.
    donut_radius:
        Size of exclusion zone around the cutoff to remove heaped observations.
        0 means no donut. Units are same as the running variable.
    discrete:
        Whether the running variable is discrete (use rdlocrand methods).
    placebo_cutoffs:
        Suggested placebo cutoff values for validity testing.
    fca_context:
        Description of FCA Consumer Duty relevance.
    notes:
        Additional methodological notes.
    """

    name: str
    description: str
    cutoff: float
    running_var_label: str
    outcome_type: str = "poisson"
    bwselect: str = "mserd"
    kernel: str = "triangular"
    donut_radius: float = 0.0
    discrete: bool = False
    placebo_cutoffs: list[float] = field(default_factory=list)
    fca_context: str = ""
    notes: str = ""


# Age 25 young driver threshold (running variable in months avoids integer-year heaping).
AGE_25 = ThresholdPreset(
    name="Age 25 (months)",
    description=(
        "Young driver premium cliff at age 25. UK motor insurers apply substantially "
        "higher rates below 25. ABI data shows under-25s pay approximately 3x the "
        "premium of 25-30 drivers. FCA scrutinised this boundary in GIPP market reviews."
    ),
    cutoff=300.0,  # 25 * 12 months
    running_var_label="Driver age (months)",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="triangular",
    donut_radius=3.0,  # 3 months — removes heaping from integer-year reporting
    discrete=False,
    placebo_cutoffs=[264.0, 282.0, 318.0, 336.0],  # 22, 23.5, 26.5, 28 years in months
    fca_context=(
        "FCA Consumer Duty (July 2023) requires demonstrable risk-reflective pricing. "
        "The age 25 boundary is high-visibility: if the premium cliff exceeds the "
        "causal claims risk discontinuity, this is a Consumer Duty exposure. "
        "FCA GIPP reviews (2021-2022) specifically examined young driver pricing."
    ),
    notes=(
        "Use driver_age_months not driver_age_years to avoid integer-year heaping. "
        "The donut of 3 months excludes ages [297, 303] months (24.75 to 25.25 years). "
        "McCrary density test should be clean — age is externally verified and "
        "policyholders cannot manipulate their date of birth."
    ),
)

# Age 25 for running variable in years (less preferred — heaping at integers).
AGE_25_YEARS = ThresholdPreset(
    name="Age 25 (years)",
    description="Young driver premium cliff at age 25, running variable in integer years.",
    cutoff=25.0,
    running_var_label="Driver age (years)",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="triangular",
    donut_radius=0.5,  # half a year either side
    discrete=True,  # integer years — treat as discrete
    placebo_cutoffs=[22.0, 23.0, 27.0, 28.0],
    fca_context=AGE_25.fca_context,
    notes=(
        "Prefer age in months if available — integer years creates heaping that "
        "distorts the density test. If only integer ages are available, set discrete=True "
        "and use rdlocrand for randomisation inference."
    ),
)

# NCD level boundaries (UK motor, 5-step scale).
NCD_STEP = ThresholdPreset(
    name="NCD Level Transition",
    description=(
        "No-Claims Discount step boundary. UK motor NCD: 0,1,2,3,4,5 steps "
        "corresponding to 0%,20%,30%,40%,50%,65% discounts (insurer-specific). "
        "Each step is a pricing threshold in its own right."
    ),
    cutoff=3.0,  # 3->4 transition as default (most common analysis point)
    running_var_label="NCD level",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="uniform",  # uniform preferred for discrete running variables
    donut_radius=0.0,
    discrete=True,
    placebo_cutoffs=[],
    fca_context=(
        "NCD pricing directly affects renewal pricing. FCA PS21/5 ENBP rules constrain "
        "how renewals can be priced relative to new business. RDD on NCD asks: does "
        "actual claims risk differ across NCD boundaries, or is NCD purely a mechanism "
        "for adverse selection and moral hazard?"
    ),
    notes=(
        "IMPORTANT: McCrary density test at NCD 4->5 WILL find a spike — NCD gaming "
        "(withholding small claims to preserve max NCD) is real and documented by "
        "Artis et al. (2002). The density failure does not invalidate the analysis but "
        "must be explicitly caveat. Use rdlocrand (rdrandinf) for randomisation inference. "
        "For all NCD cutoffs simultaneously, use MultiCutoffRD."
    ),
)

# NCD max (4->5) — the most gamed boundary.
NCD_MAX = ThresholdPreset(
    name="NCD Max Transition (4→5)",
    description=(
        "The 4→5 NCD step where manipulation (claim withholding) is most prevalent. "
        "The transition to maximum NCD (65% discount in UK) creates the strongest "
        "incentive to withhold claims just below the boundary."
    ),
    cutoff=5.0,
    running_var_label="NCD level",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="uniform",
    donut_radius=0.0,
    discrete=True,
    placebo_cutoffs=[2.0, 3.0],
    fca_context=NCD_STEP.fca_context,
    notes=(
        "Density test failure is expected here — manipulation is not incidental, "
        "it is the central phenomenon. The RDD still identifies the LATE for "
        "non-manipulators (compliers in the rdlocrand sense). Report the density "
        "failure prominently and interpret the treatment effect as a lower bound "
        "on the true causal effect for the population."
    ),
)

# Vehicle age 10-year threshold.
VEHICLE_AGE_10 = ThresholdPreset(
    name="Vehicle Age 10 Years",
    description=(
        "Vehicle age threshold at 10 years where some insurers reduce cover options "
        "or change rating factors. Running variable in months avoids year-boundary heaping."
    ),
    cutoff=120.0,  # 10 * 12 months
    running_var_label="Vehicle age (months)",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="triangular",
    donut_radius=2.0,  # 2 months to handle registration-year reporting
    discrete=False,
    placebo_cutoffs=[96.0, 108.0, 132.0, 144.0],  # 8, 9, 11, 12 years
    fca_context=(
        "Vehicle age pricing is risk-reflective if older vehicles have higher claims "
        "rates due to component reliability. However, older vehicles may also have "
        "lower replacement costs — frequency and severity effects can offset. "
        "RDD provides causal decomposition."
    ),
    notes=(
        "Vehicle age is externally verifiable from DVLA registration data — "
        "manipulation concern is low. Use months to avoid heaping at integer years."
    ),
)

# Vehicle age 15-year threshold (agreed value eligibility).
VEHICLE_AGE_15 = ThresholdPreset(
    name="Vehicle Age 15 Years",
    description="Vehicle age threshold at 15 years (agreed value cover eligibility in some markets).",
    cutoff=180.0,  # 15 * 12 months
    running_var_label="Vehicle age (months)",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="triangular",
    donut_radius=2.0,
    discrete=False,
    placebo_cutoffs=[156.0, 168.0, 192.0, 204.0],
    fca_context=VEHICLE_AGE_10.fca_context,
    notes=VEHICLE_AGE_10.notes,
)

# Policy tenure 3-year loyalty threshold.
TENURE_3 = ThresholdPreset(
    name="Policy Tenure 3 Years",
    description=(
        "Loyalty discount trigger at 3-year tenure. FCA PS21/5 ENBP rules require "
        "renewal pricing to be no worse than equivalent new business price for existing "
        "customers, but tenure-based discounts below this cap are still common."
    ),
    cutoff=36.0,  # 3 * 12 months
    running_var_label="Policy tenure (months)",
    outcome_type="poisson",
    bwselect="mserd",
    kernel="triangular",
    donut_radius=1.0,
    discrete=False,
    placebo_cutoffs=[24.0, 30.0, 42.0, 48.0],
    fca_context=(
        "FCA PS21/5 (ENBP, effective Jan 2022): insurers cannot charge renewing "
        "customers more than equivalent new business customers. RDD on tenure tests "
        "whether longer-tenure customers genuinely have different risk profiles "
        "(legitimate pricing basis) vs. pure inertia exploitation."
    ),
    notes=(
        "Moderate manipulation concern: policyholders who cancel at 3 years do not "
        "cross the threshold. Self-selection into continuation is possible. "
        "Run McCrary test and report."
    ),
)

# Sum insured £100k band boundary.
SUM_INSURED_100K = ThresholdPreset(
    name="Sum Insured £100k",
    description=(
        "Household sum insured rating band boundary at £100,000. The premium schedule "
        "may have a kink (slope change) rather than a level jump at this boundary. "
        "This is a Kink RDD design."
    ),
    cutoff=100_000.0,
    running_var_label="Sum insured (£)",
    outcome_type="gamma",  # severity analysis primary interest here
    bwselect="mserd",
    kernel="triangular",
    donut_radius=2_000.0,  # £2k either side — declared values cluster at round numbers
    discrete=False,
    placebo_cutoffs=[75_000.0, 90_000.0, 110_000.0, 125_000.0],
    fca_context=(
        "Sum insured bands require demonstrable actuarial basis — that exposure to loss "
        "increases continuously with declared value, with any kink at band boundaries "
        "reflecting a genuine change in loss rate."
    ),
    notes=(
        "This is a KINK RDD if the premium schedule is piecewise linear. The standard "
        "rdrobust sharp RD estimator measures the level discontinuity; for slope change "
        "you need Card et al. (2015) kink estimator (not yet in rdrobust Python). "
        "Declared sums insured cluster heavily at round numbers — donut is important."
    ),
)

# Dictionary of all available presets for programmatic access.
PRESETS: dict[str, ThresholdPreset] = {
    "age_25": AGE_25,
    "age_25_years": AGE_25_YEARS,
    "ncd_step": NCD_STEP,
    "ncd_max": NCD_MAX,
    "vehicle_age_10": VEHICLE_AGE_10,
    "vehicle_age_15": VEHICLE_AGE_15,
    "tenure_3": TENURE_3,
    "sum_insured_100k": SUM_INSURED_100K,
}


def get_preset(name: str) -> ThresholdPreset:
    """Look up a preset by name.

    Parameters
    ----------
    name:
        Preset key. Case-insensitive. One of: age_25, age_25_years, ncd_step,
        ncd_max, vehicle_age_10, vehicle_age_15, tenure_3, sum_insured_100k.

    Returns
    -------
    ThresholdPreset

    Raises
    ------
    KeyError
        If the preset name is not recognised.
    """
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available presets: {list(PRESETS.keys())}"
        )
    return PRESETS[key]
