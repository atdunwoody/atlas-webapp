from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import warnings

import pandas as pd
import geopandas as gpd
import pyogrio  # for fast layer listing & I/O


def _validate_inputs(in_gpkg: Path, csv_path: Path) -> None:
    """Basic existence checks."""
    if not in_gpkg.exists():
        raise FileNotFoundError(f"GeoPackage not found: {in_gpkg}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")


def _read_csv_dedup(csv_path: Path, key: str) -> pd.DataFrame:
    """
    Read CSV and ensure `key` exists; if duplicates are present, keep first and warn.
    Returns a DataFrame indexed by `key` for efficient merging.
    """
    df = pd.read_csv(csv_path)
    if key not in df.columns:
        raise KeyError(f"CSV is missing join key column '{key}'. Columns: {list(df.columns)}")

    # Track duplicates
    dup_mask = df.duplicated(subset=[key], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        warnings.warn(
            f"{dup_count} duplicate {key} values found in CSV; keeping first occurrence, "
            "dropping the rest."
        )
        df = df[~dup_mask].copy()

    # Set index to key for fast left-join via map/merge
    if df[key].isna().any():
        warnings.warn(f"CSV has {df[key].isna().sum()} rows with NaN in '{key}'; they cannot join.")
    return df.set_index(key)


def _scale_series_to_range(
    s: pd.Series, out_min: float, out_max: float
) -> pd.Series:
    """
    Min–max scale numeric series to [out_min, out_max], preserving NaNs.
    If the series has no finite values, returns it unchanged.
    If min == max (constant), fills finite entries with midpoint and warns.
    """
    s = s.astype("float64")  # ensure numeric (raise later if fails upstream)
    finite = s.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if finite.empty:
        warnings.warn("CurrTemp has no finite values; leaving as-is.")
        return s

    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        warnings.warn(
            f"CurrTemp is constant ({vmin}); setting scaled values to midpoint "
            f"{0.5 * (out_min + out_max)}."
        )
        out = s.copy()
        out.loc[s.notna()] = 0.5 * (out_min + out_max)
        return out

    scale = (out_max - out_min) / (vmax - vmin)
    out = out_min + (s - vmin) * scale
    # numerical safety: clip tiny overshoots
    return out.clip(lower=out_min, upper=out_max)


def process_geopackage(
    in_gpkg: str | Path,
    csv_path: str | Path,
    out_gpkg: Optional[str | Path] = None,
    *,
    scale_field: str = "CurrTemp",
    join_key: str = "BSR",
    out_range: tuple[float, float] = (0.0, 25.0),
) -> Path:
    """
    For every layer in the input GeoPackage:
      1) Min–max scale `scale_field` to `out_range` (per-layer).
      2) Left-join attributes from CSV on `join_key`.

    Parameters
    ----------
    in_gpkg : str | Path
        Input GeoPackage path.
    csv_path : str | Path
        CSV with attributes to join (must contain `join_key`).
    out_gpkg : str | Path, optional
        Output GeoPackage path. If None, creates sibling file with suffix `_scaled_joined.gpkg`.
    scale_field : str
        Name of the numeric field to scale (default 'CurrTemp').
    join_key : str
        Name of the join key present in both layer attribute tables and CSV (default 'BSR').
    out_range : tuple[float, float]
        Target range for scaling, default (0, 25).

    Returns
    -------
    Path
        Path to the written output GeoPackage.

    Notes
    -----
    - CRS is preserved; no reprojection performed.
    - Join is a left join (all geometries preserved). CSV duplicates on `join_key` are de-duplicated (first keeps).
    - If `scale_field` is constant per layer, scaled values are set to the midpoint of `out_range`.
    """
    in_gpkg = Path(in_gpkg)
    csv_path = Path(csv_path)
    _validate_inputs(in_gpkg, csv_path)

    if out_gpkg is None:
        out_gpkg = in_gpkg.with_name(in_gpkg.stem + "_scaled_joined.gpkg")
    out_gpkg = Path(out_gpkg)
    if out_gpkg.exists():
        out_gpkg.unlink()  # start fresh

    csv_df = _read_csv_dedup(csv_path, key=join_key)

    layers: Sequence[str] = [name for name, _geom in pyogrio.list_layers(str(in_gpkg))]
    if not layers:
        raise ValueError(f"No layers found in {in_gpkg}")

    write_mode = "w"
    for lyr in layers:
        gdf = gpd.read_file(in_gpkg, layer=lyr, engine="pyogrio")

        if gdf.empty:
            warnings.warn(f"Layer '{lyr}' is empty; writing through unchanged.")
        else:
            # Validate join key
            if join_key not in gdf.columns:
                raise KeyError(
                    f"Layer '{lyr}' is missing join key '{join_key}'. "
                    f"Columns: {list(gdf.columns)}"
                )

            # Scale field
            if scale_field not in gdf.columns:
                raise KeyError(
                    f"Layer '{lyr}' is missing scale field '{scale_field}'. "
                    f"Columns: {list(gdf.columns)}"
                )

            # ensure numeric
            try:
                gdf[scale_field] = pd.to_numeric(gdf[scale_field], errors="coerce")
            except Exception as e:
                raise TypeError(
                    f"Layer '{lyr}' field '{scale_field}' cannot be coerced to numeric."
                ) from e

            gdf[scale_field] = _scale_series_to_range(
                gdf[scale_field], out_min=float(out_range[0]), out_max=float(out_range[1])
            )

            # Left join attributes from CSV on join_key
            # Use merge to bring all CSV columns (avoid index becoming a column twice).
            attrs = csv_df.reset_index()
            # Warn if any layer keys are NaN
            if gdf[join_key].isna().any():
                warnings.warn(
                    f"Layer '{lyr}' has {int(gdf[join_key].isna().sum())} NaN '{join_key}' values; "
                    "those rows will not match CSV."
                )
            gdf = gdf.merge(attrs, on=join_key, how="left", validate="many_to_one")

        # Write out (preserve layer name). CRS preserved by GeoPandas.
        for lyr_name, _geom in pyogrio.list_layers(str(in_gpkg)):
            gdf.to_file(out_gpkg, layer=lyr_name, driver="GPKG", mode=write_mode, engine="pyogrio")

    return out_gpkg


if __name__ == "__main__":
    # --- Example execution using your provided paths ---
    IN_GPKG = r"data\inputs\base_bsr.gpkg"
    CSV = r"data\inputs\BSR_RCAT.csv"
    OUT = r"data\outputs\base_bsr_scaled_scores.gpkg"

    out_path = process_geopackage(IN_GPKG, CSV, OUT)
    print(f"Wrote: {out_path}")
