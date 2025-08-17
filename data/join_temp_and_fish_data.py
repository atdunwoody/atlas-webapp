
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings

import fiona
import geopandas as gpd
import pandas as pd

from pathlib import Path
from typing import Optional
import re
import numpy as np
import fiona
import geopandas as gpd
import pandas as pd

from pathlib import Path
from typing import Optional, Sequence
import fiona
import geopandas as gpd
import pandas as pd
import numpy as np


def join_temp_medians_to_fish(
    temp_points_path: str,
    fish_dist_gpkg: str,
    out_gpkg_path: str,
    *,
    temp_layer: Optional[str] = None,
    out_layer_suffix: str = "",          # e.g., "_with_temp_medians"
    buffer_meters: float = 50.0,
    value_fields: Sequence[str] = ("S30_2040D", "S32_2080D"),
) -> str:
    """
    For each layer in `fish_dist_gdb`:
      - Buffer temp points by `buffer_meters`
      - Reproject buffers to the fish layer CRS
      - Spatially join buffered temps to fish features (intersects)
      - Aggregate requested value fields by median per fish feature
      - Write an output layer to `out_gpkg_path` (one layer per input fish layer)

    Parameters
    ----------
    temp_points_path : str
        Vector dataset with temperature points (e.g., GPKG, SHP). Can be multi-layer (set `temp_layer`).
    fish_dist_gdb : str
        Path to fish distribution FileGDB (.gdb) with one or more layers (typically linework).
    out_gpkg_path : str
        Output GeoPackage path. Will contain one layer per fish layer.
    temp_layer : Optional[str], default None
        If `temp_points_path` has multiple layers, specify the layer name. If None and multi-layer,
        the first layer is used.
    out_layer_suffix : str, default ""
        Optional suffix appended to each fish layer name in the output GPKG.
    buffer_meters : float, default 50.0
        Buffer distance applied to temperature points BEFORE CRS matching to fish layers.
        (Assumes the temp dataset's CRS is already in meters.)
    value_fields : Sequence[str], default ("S30_2040D", "S32_2080D")
        Attribute fields from the temp points to aggregate via median.

    Returns
    -------
    str
        The output GeoPackage path.
    """
    # --- read temperature points (supports single- or multi-layer sources) ---
    if temp_layer is None:
        # If multi-layer, use the first layer by default
        try:
            _layers = fiona.listlayers(temp_points_path)
            if _layers:
                temp_gdf = gpd.read_file(temp_points_path, layer=_layers[0])
            else:
                temp_gdf = gpd.read_file(temp_points_path)
        except Exception:
            temp_gdf = gpd.read_file(temp_points_path)
    else:
        temp_gdf = gpd.read_file(temp_points_path, layer=temp_layer)

    if temp_gdf.crs is None:
        raise ValueError("Temperature points must have a valid CRS.")

    # Ensure required fields exist and are numeric
    for fld in value_fields:
        if fld not in temp_gdf.columns:
            raise KeyError(f"Field '{fld}' not found in temperature points.")
        temp_gdf[fld] = pd.to_numeric(temp_gdf[fld], errors="coerce")

    # Buffer BEFORE CRS matching (assumes units are meters)
    temp_buf = temp_gdf.copy()
    temp_buf["geometry"] = temp_buf.geometry.buffer(buffer_meters)

    # Keep only fields needed for join/aggregation
    temp_keep_cols = list(value_fields) + ["geometry"]
    temp_buf = temp_buf[temp_keep_cols]

    # --- iterate fish layers in the GeoPackage ---
    fish_layers = fiona.listlayers(fish_dist_gpkg)
    if not fish_layers:
        raise ValueError(f"No layers found in {fish_dist_gpkg}")

    out_path = Path(out_gpkg_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = "w"  # Overwrite file on first write, append thereafter

    for lyr in fish_layers:
        fish_gdf = gpd.read_file(fish_dist_gpkg, layer=lyr)
        if fish_gdf.empty:
            # Write empty layer to keep parity with inputs
            fish_gdf.to_file(out_path.as_posix(), layer=f"{lyr}{out_layer_suffix}", driver="GPKG", mode=write_mode)
            write_mode = "a"
            continue

        if fish_gdf.crs is None:
            raise ValueError(f"Fish layer '{lyr}' has no CRS.")

        # Reproject the buffered temps to fish CRS
        temp_buf_in_fish_crs = temp_buf.to_crs(fish_gdf.crs)

        # Stable index for per-feature aggregation
        fish_work = fish_gdf.reset_index(drop=False).rename(columns={"index": "fish_idx"})
        fish_work = fish_work.loc[~fish_work.geometry.isna(), ["fish_idx", "geometry"]]

        # Spatial join (fish features ⟂ buffered temp polygons)
        sj = gpd.sjoin(
            fish_work,                    # left
            temp_buf_in_fish_crs,        # right
            how="left",
            predicate="intersects",
        )

        # Median per fish feature for the requested fields
        med = (
            sj.groupby("fish_idx")[list(value_fields)]
            .median(numeric_only=True)
            .reset_index()
        )

        # Merge medians back onto full fish attributes (preserve all columns)
        fish_with_med = (
            fish_gdf.reset_index(drop=False).rename(columns={"index": "fish_idx"})
            .merge(med, on="fish_idx", how="left")
            .drop(columns=["fish_idx"])
        )

        # Write out one layer per fish layer
        gpd.GeoDataFrame(fish_with_med, geometry="geometry", crs=fish_gdf.crs).to_file(
            out_path.as_posix(),
            layer=f"{lyr}{out_layer_suffix}",
            driver="GPKG",
            mode=write_mode,
        )
        write_mode = "a"  # subsequent layers append

    return out_path.as_posix()





def _sanitize_layer_name(name: str) -> str:
    """Make a safe GPKG layer name."""
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    # GPKG layer name length is practically generous; keep it tidy
    return safe.strip("_")[:100] or "layer"


def _list_gpkg_layers(gpkg_path: str) -> List[str]:
    """List layers in a GeoPackage with informative errors."""
    if not Path(gpkg_path).exists():
        raise FileNotFoundError(f"Fish path not found: {gpkg_path}")
    try:
        layers = fiona.listlayers(gpkg_path)
    except Exception as ex:
        raise RuntimeError(f"Unable to list layers in {gpkg_path}: {ex}") from ex
    if not layers:
        raise ValueError(f"No layers found in {gpkg_path}")
    return layers


from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings

import geopandas as gpd
from pyproj import CRS

# ---------- small helpers ----------
def _list_gpkg_layers(path: str) -> List[str]:
    import fiona
    return list(fiona.listlayers(path))

def _sanitize_layer_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    return safe[:60] or "layer"

def _crs_is_meters(crs: CRS) -> bool:
    try:
        units = [ax.unit_name.lower() for ax in CRS.from_user_input(crs).axis_info]
        return any(u in ("metre", "meter", "metres", "meters") for u in units)
    except Exception:
        return False

def _auto_utm_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """
    Pick a UTM CRS (WGS84) using the dataset centroid in lon/lat.
    """
    if gdf.crs is None:
        raise ValueError("Cannot pick UTM: input has no CRS.")
    wgs84 = CRS.from_epsg(4326)
    xy = gdf.to_crs(wgs84).unary_union.centroid
    lon, lat = float(xy.x), float(xy.y)
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def _choose_metric_crs(bsr: gpd.GeoDataFrame) -> CRS:
    """
    If BSR CRS is projected in meters, use it; otherwise choose UTM.
    """
    bsr_crs = CRS.from_user_input(bsr.crs)
    if bsr_crs.is_projected and _crs_is_meters(bsr_crs):
        return bsr_crs
    return _auto_utm_crs(bsr)

def _make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make geometries valid (Shapely 2: shapely.make_valid; fallback: buffer(0)).
    """
    try:
        from shapely import make_valid  # Shapely >= 2
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.apply(make_valid)
        return gdf
    except Exception:
        gdf = gdf.copy()
        gdf.geometry = gdf.buffer(0)
        return gdf

# ---------- main function ----------
def join_fish_fields_to_bsr(
    fish_dist_path: str,
    bsr_path: str,
    *,
    out_path: Optional[str] = None,
    join_predicate: str = "intersects",
    value_fields: Tuple[str, str] = ("S30_2040D", "S32_2080D"),
) -> str:
    """
    For each layer in `fish_dist_path` (GPKG), spatially join the given `value_fields`
    onto the BSR features and write one output layer per fish layer into a GeoPackage.
    Also adds `species_stream_length` (meters): total length of fish lines within each BSR polygon.

    Assumptions
    -----------
    - BSR geometries are polygons (MultiPolygons ok) and are the *target* to enrich.
    - Median aggregation for duplicate overlaps.
    - Lengths are computed in meters in a projected CRS (BSR's if projected in meters; else auto-UTM).
    - Output preserves the original BSR CRS and attributes.

    Parameters
    ----------
    fish_dist_path : str
        Path to fish distribution GeoPackage containing multiple layers (line features expected).
    bsr_path : str
        Path to the BSR vector dataset (polygons).
    out_path : str, optional
        Output GeoPackage path. If None:
          - If `bsr_path` ends with .gpkg → write there.
          - Else → create `<bsr_stem>.gpkg` next to `bsr_path`.
    join_predicate : str
        Spatial join predicate (e.g., 'intersects', 'within', 'contains', 'crosses').
    value_fields : (str, str)
        Names of fish attributes to transfer (default: S30_2040D, S32_2080D).

    Returns
    -------
    str
        The output GeoPackage path.
    """
    # --- Validate / resolve output ---
    fish_dist_path = str(fish_dist_path)
    bsr_path = str(bsr_path)
    if out_path is None:
        bsr_p = Path(bsr_path)
        out_gpkg = bsr_path if bsr_p.suffix.lower() == ".gpkg" else str(bsr_p.with_suffix(".gpkg"))
    else:
        out_gpkg = str(out_path)

    # --- Read BSR once (preserve its CRS) ---
    if not Path(bsr_path).exists():
        raise FileNotFoundError(f"BSR path not found: {bsr_path}")
    try:
        bsr = gpd.read_file(bsr_path, engine="fiona")
    except Exception as ex:
        raise RuntimeError(f"Failed to read BSR at {bsr_path}: {ex}") from ex

    if bsr.empty:
        raise ValueError("BSR dataset is empty.")
    if bsr.crs is None:
        raise ValueError("BSR dataset has no CRS; cannot perform a safe spatial join.")

    # Stable row id for grouping/merging
    bsr = bsr.reset_index(drop=True).copy()
    bsr["_bsr_id_"] = bsr.index

    # Precompute metric CRS for length calc
    metric_crs = _choose_metric_crs(bsr)

    # --- Iterate fish layers ---
    layers = _list_gpkg_layers(fish_dist_path)

    for lyr in layers:
        try:
            fish = gpd.read_file(fish_dist_path, layer=lyr, engine="fiona")
        except Exception as ex:
            raise RuntimeError(f"Failed to read layer '{lyr}' in {fish_dist_path}: {ex}") from ex

        if fish.empty:
            warnings.warn(f"Layer '{lyr}' is empty; writing BSR with no added fields.")
            out = bsr.drop(columns=["_bsr_id_"]).copy()
            out_layer = _sanitize_layer_name(f"{lyr}")
            out.to_file(out_gpkg, layer=out_layer, driver="GPKG")
            continue

        if fish.crs is None:
            raise ValueError(f"Fish layer '{lyr}' has no CRS; cannot join safely.")

        # Require line features for length; filter if needed
        fish_lines = fish[fish.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        if fish_lines.empty:
            warnings.warn(f"Layer '{lyr}' has no line geometries; length will be 0.")

        # Check requested fields
        missing = [c for c in value_fields if c not in fish.columns]
        if missing:
            warnings.warn(f"Layer '{lyr}' missing fields {missing}; medians won't be added.")
            fish_use = fish[["geometry"]].copy()
            present_fields: List[str] = []
        else:
            fish_use = fish[list(value_fields) + ["geometry"]].copy()
            present_fields = list(value_fields)

        # Reproject fish to BSR CRS (for attribute join)
        if fish_use.crs != bsr.crs:
            fish_use = fish_use.to_crs(bsr.crs)

        # Spatial join → many-to-one; aggregate medians and match count
        try:
            hits = gpd.sjoin(
                bsr[["_bsr_id_", "geometry"]],
                fish_use,
                how="left",
                predicate=join_predicate,
            )
        except Exception as ex:
            raise RuntimeError(
                f"Spatial join failed for layer '{lyr}' with predicate '{join_predicate}': {ex}"
            ) from ex

        if present_fields:
            agg_parts: Dict[str, Tuple[str, str]] = {
                f"{fld}_median": (fld, "median") for fld in present_fields
            }
            count_field = present_fields[0]
            agg_parts["fish_matches"] = (count_field, "count")
            agg_df = (
                hits.drop(columns=["geometry"])
                .groupby("_bsr_id_", dropna=False)
                .agg(**agg_parts)
                .reset_index()
            )
        else:
            agg_df = (
                hits.drop(columns=["geometry"])
                .groupby("_bsr_id_", dropna=False)
                .size()
                .rename("fish_matches")
                .reset_index()
            )

        # --- Compute species_stream_length (meters) via overlay in a metric CRS ---
        try:
            bsr_len = bsr[["_bsr_id_", "geometry"]].to_crs(metric_crs)
            fish_len = fish_lines[["geometry"]].to_crs(metric_crs) if not fish_lines.empty else fish_use[["geometry"]].to_crs(metric_crs)
            # Ensure validity to avoid topology errors
            bsr_len = _make_valid_gdf(bsr_len)
            fish_len = _make_valid_gdf(fish_len)

            inter = gpd.overlay(fish_len, bsr_len, how="intersection")
            if inter.empty:
                length_df = bsr_len[["_bsr_id_"]].copy()
                length_df["species_stream_length"] = 0.0
            else:
                inter["seg_len_m"] = inter.geometry.length
                length_df = (
                    inter.groupby("_bsr_id_", dropna=False)["seg_len_m"]
                    .sum()
                    .rename("species_stream_length")
                    .reset_index()
                )
        except Exception as ex:
            raise RuntimeError(
                f"Length computation failed for layer '{lyr}' in metric CRS {metric_crs.to_string()}: {ex}"
            ) from ex

        # Merge back onto full BSR attributes
        out = bsr.merge(agg_df, on="_bsr_id_", how="left")
        out = out.merge(length_df, on="_bsr_id_", how="left")
        out["species_stream_length"] = out["species_stream_length"].fillna(0.0)
        out = out.drop(columns=["_bsr_id_"])

        # Write a layer per fish layer
        out_layer = _sanitize_layer_name(f"{lyr}")
        try:
            out.to_file(out_gpkg, layer=out_layer, driver="GPKG")
        except Exception as ex:
            raise RuntimeError(
                f"Failed to write layer '{out_layer}' to {out_gpkg}: {ex}"
            ) from ex

    return out_gpkg



if __name__ == "__main__":
    temp_points_gpkg = r"data\inputs\NorWeST_PredictedStreamTempPoints_MidColumbia_MWMT\NorWeST_PredictedStreamTempPoints_MidColumbia_MWMT.shp"
    fish_dist_gpkg = r"data\inputs\All_Fish_Dist.gpkg"
    fish_dist_path = r"data\outputs\All_Fish_Dist_with_temp.gpkg"

    #join_temp_medians_to_fish(temp_points_gpkg, fish_dist_gpkg, fish_dist_path)

    bsr_path = r"data\inputs\base_bsr.gpkg"
    fish_dist_path = r"data\outputs\All_Fish_Dist_with_temp.gpkg"
    bsr_temp_path = r"data\outputs\base_bsr_with_temp.gpkg"

    # If bsr_path is a .shp, outputs will be written to a sibling .gpkg with the same stem.
    out_gpkg_written = join_fish_fields_to_bsr(fish_dist_path, bsr_path, out_path = bsr_temp_path)
    print(f"Wrote joined layers to: {out_gpkg_written}")
