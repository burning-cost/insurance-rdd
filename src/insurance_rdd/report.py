"""ThresholdReport: combined RDD results and FCA Consumer Duty interpretation.

This pulls together the RD estimate, validity tests, and regulatory framing
into a single document. The output is Markdown (or HTML via the html() method).

The FCA Consumer Duty framing is specific: we compare the empirical causal
rate ratio (from RDD) to the tariff relativity applied at the same threshold.
If these match, pricing is risk-reflective at the boundary. If the tariff
relativity is substantially larger than the empirical rate ratio, that is
evidence of over-pricing relative to risk — a Consumer Duty concern.

This is NOT a full actuarial sign-off. It is evidence to bring into a pricing
review, specifically addressing FCA PS22/9 (Consumer Duty) Section 8.8-8.12
which requires firms to demonstrate that pricing factors are risk-reflective.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _format_regulatory_report(
    result: Any,
    tariff_relativity: float | None = None,
    threshold_name: str = "threshold",
) -> str:
    """Internal helper to generate FCA-formatted regulatory report text.

    Called by RDResult.regulatory_report().
    """
    tau = result.tau_bc
    rr = np.exp(tau)
    rr_lo = np.exp(result.ci_lower)
    rr_hi = np.exp(result.ci_upper)
    p = result.p_value

    sig = p < 0.05
    direction = "reduction" if tau < 0 else "increase"
    direction_word = "decreases" if tau < 0 else "increases"

    lines = [
        f"## FCA Consumer Duty — Threshold Analysis: {threshold_name}",
        "",
        "### Evidence of Risk Discontinuity",
        "",
        f"Regression Discontinuity Design was applied to estimate the causal effect of "
        f"the {threshold_name} on claims outcomes.",
        "",
        f"**Result**: At the {threshold_name}, crossing the threshold causes a "
        f"{'statistically significant ' if sig else ''}causal {direction} in the claims "
        f"rate of **{abs(1.0 - rr) * 100:.1f}%** "
        f"(rate ratio: {rr:.3f}, 95% CI: {rr_lo:.3f}–{rr_hi:.3f}, p = {p:.3f}).",
        "",
    ]

    if not sig:
        lines += [
            f"The result is **not statistically significant** at the 5% level. This does not "
            f"prove there is no effect — it means the data do not provide sufficient evidence "
            f"to distinguish the estimated effect from noise. The 95% CI is "
            f"[{rr_lo:.3f}, {rr_hi:.3f}].",
            "",
        ]

    if tariff_relativity is not None:
        tr_log = np.log(tariff_relativity)
        diff_log = tau - tr_log

        lines += [
            "### Comparison to Tariff Relativity",
            "",
            f"The pricing model applies a tariff relativity of **{tariff_relativity:.3f}** "
            f"at this threshold (log scale: {tr_log:.4f}).",
            "",
        ]

        # Statistical test: does the log tariff relativity fall within the empirical CI?
        tariff_in_ci = result.ci_lower <= tr_log <= result.ci_upper
        tariff_ratio_vs_empirical = tariff_relativity / rr if rr != 0 else np.inf

        if tariff_in_ci:
            verdict = "CONSISTENT"
            colour_word = "consistent with"
            action = (
                "No action required on Consumer Duty grounds at this threshold — "
                "the pricing factor is calibrated in line with the empirical evidence."
            )
        elif tariff_relativity > rr * 1.1:
            verdict = "OVER-PRICED (potential Consumer Duty concern)"
            colour_word = "LARGER than"
            action = (
                "The tariff relativity exceeds the empirical causal rate ratio. "
                "If the threshold generates a premium REDUCTION above the cutoff "
                "that is smaller than warranted by risk, customers above the cutoff "
                "may be over-charged. Review and document the actuarial basis for "
                "the tariff relativity, or adjust to align with empirical evidence."
            )
        elif tariff_relativity < rr / 1.1:
            verdict = "UNDER-PRICED (potential under-reserving)"
            colour_word = "SMALLER than"
            action = (
                "The tariff relativity is smaller than the empirical causal rate ratio. "
                "Customers above/below the cutoff may be under-charged relative to "
                "their risk. While this is not a Consumer Duty concern (over-pricing), "
                "it is an under-reserving risk. Review pricing margins at this threshold."
            )
        else:
            verdict = "BROADLY CONSISTENT"
            colour_word = "broadly consistent with"
            action = (
                "Small discrepancy between tariff relativity and empirical rate ratio. "
                "Within normal estimation uncertainty. No immediate action required, "
                "but monitor at next pricing review."
            )

        lines += [
            f"**Assessment**: The tariff relativity ({tariff_relativity:.3f}) is {colour_word} "
            f"the empirical causal rate ratio ({rr:.3f}).",
            f"",
            f"**Verdict: {verdict}**",
            "",
            f"**Action**: {action}",
            "",
            "| | Log scale | Ratio scale |",
            "|---|---|---|",
            f"| Empirical rate ratio (RDD) | {tau:.4f} | {rr:.4f} |",
            f"| Tariff relativity | {tr_log:.4f} | {tariff_relativity:.4f} |",
            f"| Difference | {diff_log:.4f} | {tariff_ratio_vs_empirical:.4f} |",
            f"| 95% CI for empirical | [{result.ci_lower:.4f}, {result.ci_upper:.4f}] | [{rr_lo:.4f}, {rr_hi:.4f}] |",
            "",
        ]

    lines += [
        "### Methodological Notes",
        "",
        "- **Method**: Regression Discontinuity Design with CCT bias-corrected robust inference (rdrobust)",
        f"- **Estimand**: Local Average Treatment Effect (LATE) at the cutoff — not an average over the full portfolio",
        f"- **Bandwidth**: {result.bandwidth_h[0]:.3f} (left) / {result.bandwidth_h[1]:.3f} (right) [{result.bwselect if hasattr(result, 'bwselect') else 'data-driven'}]",
        f"- **Sample**: {result.n_left_eff} observations left, {result.n_right_eff} observations right of cutoff",
        "",
        "RDD identification relies on the continuity assumption: that absent the threshold, "
        "potential outcomes would not jump at the cutoff. This is credible for externally "
        "verifiable running variables (driver age, vehicle age). For self-reported or "
        "gameable variables (NCD level, mileage), the assumption requires explicit caveats.",
        "",
        "*Report generated by insurance-rdd v0.1.0 — Burning Cost*",
    ]

    return "\n".join(lines)


@dataclass
class ThresholdReportData:
    """Data bundle for building a ThresholdReport."""

    rd_result: Any  # RDResult
    density_result: Any = None  # DensityTestResult
    balance_result: Any = None  # CovariateBalanceResult
    placebo_result: Any = None  # PlaceboTestResult
    donut_result: Any = None  # DonutRDDResult
    bandwidth_sensitivity_df: pd.DataFrame | None = None
    tariff_relativity: float | None = None
    threshold_name: str = "threshold"
    additional_notes: str = ""


class ThresholdReport:
    """Full threshold analysis report combining RDD and validity tests.

    Generates a Markdown document suitable for an actuarial pricing review,
    incorporating:
    - Main RDD estimate with rate ratio
    - Density manipulation test
    - Covariate balance tests
    - Placebo test results
    - Donut RDD sensitivity
    - FCA Consumer Duty interpretation (when tariff_relativity is provided)

    Parameters
    ----------
    data:
        ThresholdReportData bundle with all analysis results.

    Examples
    --------
    >>> report = ThresholdReport(
    ...     ThresholdReportData(
    ...         rd_result=result,
    ...         density_result=density.fit(),
    ...         tariff_relativity=0.70,
    ...         threshold_name="age 25 driver boundary",
    ...     )
    ... )
    >>> print(report.markdown())
    """

    def __init__(self, data: ThresholdReportData) -> None:
        self.data = data

    def markdown(self) -> str:
        """Generate Markdown report."""
        rd = self.data.rd_result
        lines = [
            f"# Threshold Analysis Report: {self.data.threshold_name}",
            "",
            "---",
            "",
            "## 1. RDD Estimate",
            "",
            "```",
            rd.summary(),
            "```",
            "",
        ]

        if self.data.density_result is not None:
            lines += [
                "## 2. Density Manipulation Test",
                "",
                "```",
                self.data.density_result.summary(),
                "```",
                "",
            ]

        if self.data.balance_result is not None:
            lines += [
                "## 3. Covariate Balance",
                "",
                "```",
                self.data.balance_result.summary(),
                "```",
                "",
            ]

        if self.data.placebo_result is not None:
            lines += [
                "## 4. Placebo Test",
                "",
                "```",
                self.data.placebo_result.summary(),
                "```",
                "",
            ]

        if self.data.donut_result is not None:
            lines += [
                "## 5. Donut RDD Sensitivity",
                "",
                "```",
                self.data.donut_result.summary(),
                "```",
                "",
            ]

        if self.data.bandwidth_sensitivity_df is not None and len(self.data.bandwidth_sensitivity_df) > 0:
            df = self.data.bandwidth_sensitivity_df
            lines += [
                "## 6. Bandwidth Sensitivity",
                "",
                df[["h_factor", "tau", "ci_lower", "ci_upper", "p_value"]].to_markdown(
                    index=False, floatfmt=".4f"
                ),
                "",
            ]

        if self.data.tariff_relativity is not None:
            lines += [
                "## 7. FCA Consumer Duty Assessment",
                "",
                _format_regulatory_report(
                    result=rd,
                    tariff_relativity=self.data.tariff_relativity,
                    threshold_name=self.data.threshold_name,
                ),
                "",
            ]

        if self.data.additional_notes:
            lines += [
                "## Additional Notes",
                "",
                self.data.additional_notes,
                "",
            ]

        return "\n".join(lines)

    def html(self) -> str:
        """Generate HTML report by converting Markdown."""
        try:
            import markdown as md_lib
            return md_lib.markdown(self.markdown(), extensions=["tables"])
        except ImportError:
            # Simple HTML wrapping without Markdown processing.
            content = self.markdown()
            escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<html><body><pre>{escaped}</pre></body></html>"

    def save(self, path: str) -> None:
        """Save the report to a file.

        Parameters
        ----------
        path:
            File path. Extension determines format: .md for Markdown, .html for HTML.
        """
        if path.endswith(".html"):
            content = self.html()
        else:
            content = self.markdown()

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
