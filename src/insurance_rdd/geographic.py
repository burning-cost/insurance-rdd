"""GeographicRD: Spatial Regression Discontinuity at postcode/territory boundaries.

No Python spatial RDD library exists. SpatialRDD is R-only (Lehner 2020,
axlehner/SpatialRDD). This module implements the Keele-Titiunik (2015)
framework for geographic RDD in Python using geopandas and shapely.

The running variable is signed distance to the territory boundary:
  - Negative: inside territory A (the reference/control territory)
  - Positive: inside territory B (the treated territory)
  - Zero: exactly on the boundary

Treatment is territory assignment. The RD estimate gives the causal effect
of being in territory B vs A at the boundary — isolated from any confounders
that vary smoothly across the boundary.

Why geographic RDD is harder than standard RDD:
- Boundaries are 2D curves, not 1D thresholds
- Distance computation requires proper geodetic calculations
- Border segment fixed effects control for the fact that different segments
  of the boundary may have different characteristics
- Risk that residents sort across boundaries (though insurance is not
  usually a sorting motive — people don't move because of motor insurance
  territories)

The Keele-Titiunik (2015) paper (*Political Analysis* 23(1):127-155) is the
methodological foundation. We follow their approach of:
1. Compute signed distance to boundary for each observation
2. Use distance as the running variable in standard RDP
3. Optionally add border segment fixed effects

Requires geopandas and shapely. Install: pip install insurance-rdd[geo]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._rdrobust import check_geopandas


@dataclass
class GeographicRDResult:
    """Result from GeographicRD estimation.

    Attributes
    ----------
    tau:
        Log rate ratio (for Poisson/Gamma) or additive effect (Gaussian).
    tau_bc:
        Bias-corrected estimate.
    se:
        Standard error.
    ci_lower:
        95% CI lower bound.
    ci_upper:
        95% CI upper bound.
    p_value:
        p-value for H0: tau = 0.
    bandwidth_h:
        Main bandwidth (in metres, the unit of signed distance).
    n_left:
        Observations in territory A within bandwidth.
    n_right:
        Observations in territory B within bandwidth.
    n_left_eff:
        Effective observations, left.
    n_right_eff:
        Effective observations, right.
    outcome_type:
        Outcome family used.
    distances_computed:
        Whether distances were computed from coordinates (True) or
        provided directly (False).
    border_segment_fes:
        Whether border segment fixed effects were included.
    """

    tau: float
    tau_bc: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    bandwidth_h: tuple[float, float]
    n_left: int
    n_right: int
    n_left_eff: int
    n_right_eff: int
    outcome_type: str
    distances_computed: bool
    border_segment_fes: bool
    _rd_result: Any = field(default=None, repr=False)

    def rate_ratio(self) -> dict[str, float]:
        """Exponentiate tau for log-link outcomes."""
        return {
            "rate_ratio": float(np.exp(self.tau_bc)),
            "ci_lower": float(np.exp(self.ci_lower)),
            "ci_upper": float(np.exp(self.ci_upper)),
            "tau": self.tau_bc,
        }

    def summary(self) -> str:
        h_l, h_r = self.bandwidth_h
        lines = [
            "GeographicRD — Territory Boundary RDD",
            f"Bandwidth       : {h_l:.1f}m / {h_r:.1f}m",
            f"N in bandwidth  : {self.n_left_eff + self.n_right_eff} "
            f"({self.n_left_eff} territory A, {self.n_right_eff} territory B)",
            f"Border seg. FEs : {self.border_segment_fes}",
            "",
            f"  tau (robust BC): {self.tau_bc:.4f}",
            f"  SE             : {self.se:.4f}",
            f"  95% CI         : [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  p-value        : {self.p_value:.4f}",
        ]
        if self.outcome_type in ("poisson", "gamma"):
            rr = np.exp(self.tau_bc)
            lines += [
                "",
                f"  Rate ratio     : {rr:.4f}",
                f"  (95% CI ratio) : [{np.exp(self.ci_lower):.4f}, {np.exp(self.ci_upper):.4f}]",
            ]
        return "\n".join(lines)


class GeographicRD:
    """Spatial RDD at territory/postcode boundary.

    Computes signed distance from each observation to the boundary line,
    then runs standard RDD with distance as the running variable. The
    cutoff is distance = 0 (the boundary).

    This implements Keele & Titiunik (2015) Geographic RDD in Python —
    the first Python implementation of this method.

    Parameters
    ----------
    outcome:
        Column name of the outcome variable.
    treatment_col:
        Column name indicating which territory each observation is in
        (binary: 0 for territory A, 1 for territory B).
    data:
        DataFrame with outcome, treatment, and coordinates.
    boundary_file:
        Path to a geospatial file (GeoJSON, Shapefile, etc.) containing
        the territory boundary geometry. Must be loadable by geopandas.
        The boundary should be a (Multi)LineString or the border of a
        (Multi)Polygon.
    lat_col:
        Column name for latitude.
    lon_col:
        Column name for longitude.
    outcome_type:
        Outcome family: 'gaussian' | 'poisson' | 'gamma'.
    exposure:
        Column name for exposure variable.
    covariates:
        Covariate column names.
    kernel:
        Kernel type for rdrobust.
    bwselect:
        Bandwidth selector.
    border_segment_fes:
        If True, add border segment fixed effects. The boundary is divided
        into segments; each observation is assigned to its nearest segment,
        and segment indicators are added as covariates. This controls for
        spatial heterogeneity along the boundary. Recommended for long
        boundaries crossing diverse geographic areas.
    n_segments:
        Number of border segments if border_segment_fes=True. Default 10.
    distance_unit:
        Unit for computed distances: 'meters' (default) or 'km'.
    signed_distance_col:
        If provided, use this column directly as the signed distance
        running variable (bypasses coordinate-based computation).
        Negative = territory A, Positive = territory B.
    max_distance:
        Maximum distance (in distance_unit) to include. Observations
        further than this from the boundary are excluded.
    fuzzy:
        Column for actual treatment in fuzzy geographic RDD.
    level:
        Confidence level in percent.
    p:
        Local polynomial order.

    Examples
    --------
    >>> from insurance_rdd import GeographicRD
    >>> geo_rd = GeographicRD(
    ...     outcome='claim_count',
    ...     treatment_col='territory_band',
    ...     data=df,
    ...     boundary_file='territory_AB.geojson',
    ...     lat_col='lat',
    ...     lon_col='lon',
    ...     outcome_type='poisson',
    ...     exposure='exposure_years',
    ... )
    >>> result = geo_rd.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        outcome: str,
        treatment_col: str,
        data: pd.DataFrame,
        boundary_file: str | None = None,
        lat_col: str = "lat",
        lon_col: str = "lon",
        outcome_type: str = "poisson",
        exposure: str | None = None,
        covariates: list[str] | None = None,
        kernel: str = "triangular",
        bwselect: str = "mserd",
        border_segment_fes: bool = True,
        n_segments: int = 10,
        distance_unit: str = "meters",
        signed_distance_col: str | None = None,
        max_distance: float | None = None,
        fuzzy: str | None = None,
        level: float = 95.0,
        p: int = 1,
    ) -> None:
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.data = data.copy()
        self.boundary_file = boundary_file
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.outcome_type = outcome_type
        self.exposure = exposure
        self.covariates = covariates or []
        self.kernel = kernel
        self.bwselect = bwselect
        self.border_segment_fes = border_segment_fes
        self.n_segments = n_segments
        self.distance_unit = distance_unit
        self.signed_distance_col = signed_distance_col
        self.max_distance = max_distance
        self.fuzzy = fuzzy
        self.level = level
        self.p = p
        self._result: GeographicRDResult | None = None

    def _compute_signed_distances(self, df: pd.DataFrame) -> pd.Series:
        """Compute signed distance from each observation to the boundary.

        Returns a Series of signed distances (negative = territory A,
        positive = territory B).
        """
        check_geopandas()
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import nearest_points, unary_union

        # Load boundary.
        if self.boundary_file is None:
            raise ValueError(
                "boundary_file must be provided if signed_distance_col is not set."
            )
        boundary_gdf = gpd.read_file(self.boundary_file)

        # Convert all geometries to a single boundary line.
        all_geoms = boundary_gdf.geometry.tolist()
        boundary_union = unary_union(all_geoms)

        # Extract boundary line if polygon.
        from shapely.geometry import Polygon, MultiPolygon
        if isinstance(boundary_union, (Polygon, MultiPolygon)):
            boundary_line = boundary_union.boundary
        else:
            boundary_line = boundary_union

        # Reproject to metric CRS for distance computation.
        # Use UTM zone based on centroid of data.
        lat_mean = df[self.lat_col].mean()
        lon_mean = df[self.lon_col].mean()
        utm_zone = int((lon_mean + 180) / 6) + 1
        hemisphere = "north" if lat_mean >= 0 else "south"
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"

        # Create GeoDataFrame for points.
        points_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[self.lon_col], df[self.lat_col]),
            crs="EPSG:4326",
        )
        points_gdf = points_gdf.to_crs(utm_crs)

        # Also reproject boundary line.
        boundary_gdf_proj = boundary_gdf.to_crs(utm_crs)
        boundary_union_proj = unary_union(boundary_gdf_proj.geometry.tolist())
        if isinstance(boundary_union_proj, (Polygon, MultiPolygon)):
            boundary_line_proj = boundary_union_proj.boundary
        else:
            boundary_line_proj = boundary_union_proj

        # Compute unsigned distances.
        distances = points_gdf.geometry.distance(boundary_line_proj)

        # Apply sign based on treatment column.
        # treatment_col == 0 => territory A => negative distance
        # treatment_col == 1 => territory B => positive distance
        treatment = df[self.treatment_col].to_numpy()
        signed = np.where(treatment == 1, distances, -distances)

        if self.distance_unit == "km":
            signed = signed / 1000.0

        return pd.Series(signed.values, index=df.index, name="signed_distance_to_boundary")

    def _assign_border_segments(
        self, df: pd.DataFrame, signed_distances: pd.Series
    ) -> pd.DataFrame:
        """Assign each observation to its nearest border segment.

        Returns df with an added 'border_segment' column.
        """
        check_geopandas()
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union, substring

        if self.boundary_file is None:
            df["border_segment"] = 0
            return df

        boundary_gdf = gpd.read_file(self.boundary_file)
        all_geoms = boundary_gdf.geometry.tolist()
        boundary_union = unary_union(all_geoms)

        from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
        if isinstance(boundary_union, (Polygon, MultiPolygon)):
            boundary_line = boundary_union.boundary
        else:
            boundary_line = boundary_union

        # Extract a single LineString for segmentation.
        if isinstance(boundary_line, MultiLineString):
            # Use the longest component.
            geoms = list(boundary_line.geoms)
            boundary_line = max(geoms, key=lambda g: g.length)

        total_length = boundary_line.length
        segment_length = total_length / self.n_segments

        # Create segments.
        segments = []
        for i in range(self.n_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, total_length)
            try:
                seg = substring(boundary_line, start, end, normalized=False)
                segments.append(seg)
            except Exception:
                segments.append(boundary_line)

        # For each observation, find the nearest segment.
        points_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[self.lon_col], df[self.lat_col]),
            crs="EPSG:4326",
        )

        segment_ids = []
        for _, row in points_gdf.iterrows():
            pt = row.geometry
            dists = [pt.distance(seg) for seg in segments]
            segment_ids.append(int(np.argmin(dists)))

        df = df.copy()
        df["border_segment"] = segment_ids
        return df

    def fit(self) -> GeographicRDResult:
        """Fit the geographic RDD.

        Returns
        -------
        GeographicRDResult with tau, CI, and spatial diagnostics.
        """
        from .core import InsuranceRD

        df = self.data.copy()

        # Drop missing required columns.
        required = [self.outcome, self.treatment_col]
        if self.signed_distance_col is None:
            required += [self.lat_col, self.lon_col]
        df = df.dropna(subset=required)

        # Compute or use provided signed distance.
        distances_computed = False
        if self.signed_distance_col is not None:
            if self.signed_distance_col not in df.columns:
                raise ValueError(
                    f"signed_distance_col '{self.signed_distance_col}' not found in data."
                )
            df["_signed_dist"] = df[self.signed_distance_col]
        else:
            check_geopandas()
            signed_dist = self._compute_signed_distances(df)
            df["_signed_dist"] = signed_dist.values
            distances_computed = True

        # Restrict to max_distance if set.
        if self.max_distance is not None:
            df = df[np.abs(df["_signed_dist"]) <= self.max_distance].copy()
            if len(df) < 10:
                raise ValueError(
                    f"Only {len(df)} observations within max_distance={self.max_distance}."
                )

        # Build covariates including border segment FEs.
        covs = list(self.covariates)
        if self.border_segment_fes:
            if self.signed_distance_col is None and self.boundary_file is not None:
                df = self._assign_border_segments(df, df["_signed_dist"])
            elif "border_segment" not in df.columns:
                warnings.warn(
                    "border_segment_fes=True but boundary_file not provided and "
                    "no 'border_segment' column in data. FEs will not be added.",
                    RuntimeWarning,
                )
            if "border_segment" in df.columns:
                # One-hot encode border segments (drop first for identification).
                seg_dummies = pd.get_dummies(
                    df["border_segment"], prefix="seg", drop_first=True
                )
                df = pd.concat([df, seg_dummies], axis=1)
                covs += seg_dummies.columns.tolist()

        rd = InsuranceRD(
            outcome=self.outcome,
            running_var="_signed_dist",
            cutoff=0.0,
            data=df,
            outcome_type=self.outcome_type,
            exposure=self.exposure,
            covariates=covs if covs else None,
            kernel=self.kernel,
            bwselect=self.bwselect,
            p=self.p,
            level=self.level,
            fuzzy=self.fuzzy,
        )
        rd_res = rd.fit()

        self._result = GeographicRDResult(
            tau=rd_res.tau,
            tau_bc=rd_res.tau_bc,
            se=rd_res.se,
            ci_lower=rd_res.ci_lower,
            ci_upper=rd_res.ci_upper,
            p_value=rd_res.p_value,
            bandwidth_h=rd_res.bandwidth_h,
            n_left=rd_res.n_left,
            n_right=rd_res.n_right,
            n_left_eff=rd_res.n_left_eff,
            n_right_eff=rd_res.n_right_eff,
            outcome_type=self.outcome_type,
            distances_computed=distances_computed,
            border_segment_fes=self.border_segment_fes and "border_segment" in df.columns,
            _rd_result=rd_res,
        )
        return self._result
