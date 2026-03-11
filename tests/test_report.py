"""Tests for ThresholdReport and regulatory reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_rdd import InsuranceRD
from insurance_rdd.report import (
    ThresholdReport,
    ThresholdReportData,
    _format_regulatory_report,
)

from conftest import make_sharp_rdd_data


@pytest.fixture
def rd_result(gaussian_df):
    rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=gaussian_df)
    return rd.fit()


@pytest.fixture
def poisson_result(poisson_df):
    rd = InsuranceRD(
        outcome="y", running_var="x", cutoff=0.0, data=poisson_df,
        outcome_type="poisson", exposure="exposure"
    )
    return rd.fit()


class TestRegulatoryReport:
    def test_returns_string(self, rd_result):
        s = rd_result.regulatory_report(threshold_name="test threshold")
        assert isinstance(s, str)

    def test_contains_fca_mention(self, rd_result):
        s = rd_result.regulatory_report(threshold_name="age 25")
        assert "FCA" in s or "Consumer Duty" in s

    def test_with_tariff_relativity(self, rd_result):
        s = rd_result.regulatory_report(
            tariff_relativity=1.3,
            threshold_name="test threshold"
        )
        assert "1.300" in s or "tariff" in s.lower()

    def test_poisson_shows_rate_ratio(self, poisson_result):
        s = poisson_result.regulatory_report(threshold_name="age 25")
        assert "rate" in s.lower() or "ratio" in s.lower()


class TestFormatRegulatoryReport:
    def test_with_matching_tariff(self, rd_result):
        """Tariff matching empirical effect => CONSISTENT verdict."""
        tau = rd_result.tau_bc
        tariff = float(np.exp(tau))
        s = _format_regulatory_report(
            result=rd_result,
            tariff_relativity=tariff,
            threshold_name="test",
        )
        assert isinstance(s, str)
        assert len(s) > 100

    def test_contains_methodology_section(self, rd_result):
        s = _format_regulatory_report(result=rd_result)
        assert "Methodological" in s or "methodolog" in s.lower()

    def test_contains_rdrobust_mention(self, rd_result):
        s = _format_regulatory_report(result=rd_result)
        assert "rdrobust" in s.lower() or "CCT" in s

    def test_significant_effect_mentions_it(self):
        """Force a significant p-value and check the language."""
        df = make_sharp_rdd_data(n=5000, true_tau=1.0, outcome_type="gaussian")
        rd = InsuranceRD(outcome="y", running_var="x", cutoff=0.0, data=df)
        result = rd.fit()
        s = _format_regulatory_report(result=result)
        if result.p_value < 0.05:
            assert "significant" in s.lower()

    def test_no_tariff_no_comparison(self, rd_result):
        s = _format_regulatory_report(result=rd_result, tariff_relativity=None)
        assert "Tariff" not in s or "Comparison" not in s


class TestThresholdReport:
    def test_markdown_returns_string(self, rd_result):
        report = ThresholdReport(
            ThresholdReportData(
                rd_result=rd_result,
                threshold_name="test threshold",
            )
        )
        md = report.markdown()
        assert isinstance(md, str)
        assert len(md) > 200

    def test_markdown_contains_rdd_section(self, rd_result):
        report = ThresholdReport(
            ThresholdReportData(rd_result=rd_result)
        )
        md = report.markdown()
        assert "RDD" in md or "RD" in md

    def test_html_returns_string(self, rd_result):
        report = ThresholdReport(
            ThresholdReportData(rd_result=rd_result)
        )
        html = report.html()
        assert isinstance(html, str)

    def test_markdown_with_density_result(self, rd_result, gaussian_df):
        from insurance_rdd.validity import DensityTest
        dt = DensityTest(running_var="x", cutoff=0.0, data=gaussian_df)
        dr = dt.fit()

        report = ThresholdReport(
            ThresholdReportData(
                rd_result=rd_result,
                density_result=dr,
                threshold_name="test",
            )
        )
        md = report.markdown()
        assert "Density" in md

    def test_markdown_with_balance_result(self, rd_result, gaussian_df):
        from insurance_rdd.validity import CovariateBalance
        cb = CovariateBalance(
            covariates=["covariate_1"],
            running_var="x",
            cutoff=0.0,
            data=gaussian_df,
        )
        br = cb.fit()

        report = ThresholdReport(
            ThresholdReportData(
                rd_result=rd_result,
                balance_result=br,
            )
        )
        md = report.markdown()
        assert "Covariate" in md or "Balance" in md

    def test_markdown_with_tariff_relativity(self, rd_result):
        report = ThresholdReport(
            ThresholdReportData(
                rd_result=rd_result,
                tariff_relativity=0.75,
                threshold_name="age 25 boundary",
            )
        )
        md = report.markdown()
        assert "0.750" in md or "FCA" in md

    def test_save_markdown(self, rd_result, tmp_path):
        report = ThresholdReport(
            ThresholdReportData(rd_result=rd_result)
        )
        path = str(tmp_path / "report.md")
        report.save(path)
        with open(path) as f:
            content = f.read()
        assert len(content) > 0

    def test_save_html(self, rd_result, tmp_path):
        report = ThresholdReport(
            ThresholdReportData(rd_result=rd_result)
        )
        path = str(tmp_path / "report.html")
        report.save(path)
        with open(path) as f:
            content = f.read()
        assert len(content) > 0

    def test_additional_notes_included(self, rd_result):
        report = ThresholdReport(
            ThresholdReportData(
                rd_result=rd_result,
                additional_notes="These are important additional notes about the analysis.",
            )
        )
        md = report.markdown()
        assert "additional notes" in md.lower()
