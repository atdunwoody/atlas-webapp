
from __future__ import annotations
from pathlib import Path
import warnings

import fiona
import geopandas as gpd
import pandas as pd
import re
import numpy as np
from pyproj import CRS



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


from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List
import warnings

import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Point
from shapely.ops import unary_union
try:
    # shapely>=2
    from shapely.validation import make_valid as _shp_make_valid
except Exception:
    _shp_make_valid = None

from pyproj import CRS

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _list_gpkg_layers(path: str) -> List[str]:
    return list(fiona.listlayers(path))

def _sanitize_layer_name(name: str, maxlen: int = 63) -> str:
    # keep alnum and underscore; replace others with underscore
    clean = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    return clean[:maxlen] if len(clean) > maxlen else clean

def _choose_metric_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """
    Return a metric CRS for accurate length/area.
    - If gdf.crs is projected w/ meters → keep it.
    - Else → pick a UTM zone EPSG based on centroid lon/lat.
    """
    if gdf.crs:
        try:
            crs = CRS.from_user_input(gdf.crs)
            if crs.is_projected and crs.axis_info and all(ax.unit_name.lower().startswith("metre") or ax.unit_name.lower().startswith("meter") for ax in crs.axis_info):
                return crs
        except Exception:
            pass

    # compute geographic centroid
    wgs84 = CRS.from_epsg(4326)
    geom = gdf.geometry
    if geom.is_empty.all():
        # fallback to WebMercator if empty (lengths will be 0 anyway)
        return CRS.from_epsg(3857)

    centroid = gpd.GeoSeries(geom).to_crs(4326).unary_union.centroid
    lon, lat = centroid.x, centroid.y

    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)

def _make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make geometries valid. Uses shapely.validation.make_valid if available,
    else falls back to buffer(0) trick.
    """
    out = gdf.copy()
    if _shp_make_valid is not None:
        out["geometry"] = out.geometry.map(lambda g: _shp_make_valid(g) if g is not None else None)
    else:
        # buffer(0) fallback
        out["geometry"] = out.geometry.buffer(0)
    return out


# ---------------------------------------------------------------------
# join_temp_medians_to_fish  (N-field support + robust writes)
# ---------------------------------------------------------------------
def join_temp_medians_to_fish(
    temp_points_path: str,
    fish_dist_gpkg: str,
    out_gpkg_path: str,
    *,
    temp_layer: Optional[str] = None,
    out_layer_suffix: str = "",          # e.g., "_with_temp_medians"
    buffer_meters: float = 50.0,
    value_fields: Sequence[str] = ("S1_93_11", "S30_2040D", "S32_2080D"),
) -> str:
    """
    For each layer in `fish_dist_gpkg`:
      - Buffer temp points by `buffer_meters`
      - Reproject buffers to the fish layer CRS
      - Spatially join buffered temps to fish features (intersects)
      - Aggregate requested value fields by median per fish feature (supports N>=1)
      - Write an output layer to `out_gpkg_path` (one layer per input fish layer)
    """
    # --- read temperature points (supports single- or multi-layer sources) ---
    if temp_layer is None:
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
            print(temp_gdf.columns)
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
            # write empty layer (schema preserved from fish_gdf)
            gpd.GeoDataFrame(fish_gdf, geometry="geometry", crs=fish_gdf.crs).to_file(
                out_path.as_posix(),
                layer=f"{lyr}{out_layer_suffix}",
                driver="GPKG",
                mode=write_mode,
                index=False,
            )
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

        # Median per fish feature for the requested fields (supports N fields)
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

        # --- clean before writing to avoid pyogrio index/schema issues ---
        fish_with_med = fish_with_med.reset_index(drop=True)
        fish_with_med = fish_with_med.drop(
            columns=[c for c in fish_with_med.columns if c in {"index_right"}],
            errors="ignore",
        )
        fish_with_med.columns = [str(c) for c in fish_with_med.columns]

        gpd.GeoDataFrame(fish_with_med, geometry="geometry", crs=fish_gdf.crs).to_file(
            out_path.as_posix(),
            layer=f"{lyr}{out_layer_suffix}",
            driver="GPKG",
            mode=write_mode,
            index=False,  # <- important
        )
        write_mode = "a"

    return out_path.as_posix()


# ---------------------------------------------------------------------
# join_fish_fields_to_bsr  (N-field support + robust writes)
# ---------------------------------------------------------------------
def join_fish_fields_to_bsr(
    fish_dist_path: str,
    bsr_path: str,
    *,
    out_path: Optional[str] = None,
    join_predicate: str = "intersects",
    value_fields: Sequence[str] = ("S1_93_11", "S30_2040D", "S32_2080D"),
) -> str:
    """
    For each layer in `fish_dist_path` (GPKG), spatially join any number of `value_fields`
    onto the BSR features and write one output layer per fish layer into a GeoPackage.

    Adds:
      - species_stream_length: total stream length inside each BSR (meters)
      - percent_above_18C: % of stream miles in each BSR with S1_93_11 > 18
      - percent_above_22C: % of stream miles in each BSR with S1_93_11 > 22

    Notes:
      - Lengths are computed in a metric CRS, converted to miles for the percentages.
      - If S1_93_11 is missing or entirely NaN, percentages are set to 0.0.
    """
    fish_dist_path = str(fish_dist_path)
    bsr_path = str(bsr_path)
    if out_path is None:
        bsr_p = Path(bsr_path)
        out_gpkg = bsr_path if bsr_p.suffix.lower() == ".gpkg" else str(bsr_p.with_suffix(".gpkg"))
    else:
        out_gpkg = str(out_path)

    # Read BSR once
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

    bsr = bsr.reset_index(drop=True).copy()
    bsr["_bsr_id_"] = bsr.index

    metric_crs = _choose_metric_crs(bsr)
    layers = _list_gpkg_layers(fish_dist_path)

    for i, lyr in enumerate(layers):
        try:
            fish = gpd.read_file(fish_dist_path, layer=lyr, engine="fiona")
        except Exception as ex:
            raise RuntimeError(f"Failed to read layer '{lyr}' in {fish_dist_path}: {ex}") from ex

        if fish.empty:
            warnings.warn(f"Layer '{lyr}' is empty; writing BSR with no added fields.")
            out = bsr.drop(columns=["_bsr_id_"]).copy()
            out = out.reset_index(drop=True)
            out["percent_above_18C"] = 0.0
            out["percent_above_22C"] = 0.0
            out_layer = _sanitize_layer_name(f"{lyr}")
            gpd.GeoDataFrame(out, geometry="geometry", crs=bsr.crs).to_file(
                out_gpkg, layer=out_layer, driver="GPKG", mode=("w" if i == 0 else "a"), index=False
            )
            continue

        if fish.crs is None:
            raise ValueError(f"Fish layer '{lyr}' has no CRS; cannot join safely.")

        # Keep only lines for length calculations
        fish_lines = fish[fish.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        if fish_lines.empty:
            warnings.warn(f"Layer '{lyr}' has no line geometries; lengths will be 0.")
            fish_lines = fish.copy()

        # Determine which value fields are present
        missing = [c for c in value_fields if c not in fish.columns]
        if missing:
            warnings.warn(f"Layer '{lyr}' missing fields {missing}; medians for those won't be added.")
        present_fields = [c for c in value_fields if c in fish.columns]

        # Spatial-join to bring medians of present value fields (same as before)
        fish_use_cols = (present_fields + ["geometry"]) if present_fields else ["geometry"]
        fish_use = fish[fish_use_cols].copy()
        if fish_use.crs != bsr.crs:
            fish_use = fish_use.to_crs(bsr.crs)

        hits = gpd.sjoin(
            bsr[["_bsr_id_", "geometry"]],
            fish_use,
            how="left",
            predicate=join_predicate,
        )

        match_counts = (
            hits.drop(columns=["geometry"])
            .groupby("_bsr_id_", dropna=False)
            .size()
            .rename("fish_matches")
            .reset_index()
        )

        if present_fields:
            med = (
                hits.drop(columns=["geometry"])
                .groupby("_bsr_id_", dropna=False)[present_fields]
                .median(numeric_only=True)
                .reset_index()
            )
            agg_df = match_counts.merge(med, on="_bsr_id_", how="left")
        else:
            agg_df = match_counts

        # ---------- Length & temperature-threshold percentages ----------
        # We need S1_93_11 for thresholding; coerce to numeric if present
        has_temp = "S1_93_11" in fish_lines.columns
        if has_temp:
            fish_lines = fish_lines.copy()
            fish_lines["S1_93_11"] = pd.to_numeric(fish_lines["S1_93_11"], errors="coerce")

        # Reproject to a metric CRS for length computation
        bsr_len = _make_valid_gdf(bsr[["_bsr_id_", "geometry"]].to_crs(metric_crs))
        fish_len_cols = ["geometry"] + (["S1_93_11"] if has_temp else [])
        fish_len = _make_valid_gdf(fish_lines[fish_len_cols].to_crs(metric_crs))

        # Intersect fish lines with BSR polygons
        inter = gpd.overlay(fish_len, bsr_len, how="intersection")
        if inter.empty:
            length_df = bsr_len[["_bsr_id_"]].copy()
            length_df["species_stream_length"] = 0.0
            length_df["total_len_miles"] = 0.0
            length_df["len_gt18_miles"] = 0.0
            length_df["len_gt22_miles"] = 0.0
        else:
            inter["seg_len_m"] = inter.geometry.length
            M_TO_MILES = 1.0 / 1609.344

            # Total in-BSR length (meters + miles)
            total_len = (
                inter.groupby("_bsr_id_", dropna=False)["seg_len_m"]
                .sum()
                .rename("species_stream_length")
                .reset_index()
            )
            total_len["total_len_miles"] = total_len["species_stream_length"] * M_TO_MILES

            # Conditional lengths (miles) if we have temperature
            if has_temp:
                inter_temp = inter.dropna(subset=["S1_93_11"]).copy()

                gt18 = (
                    inter_temp[inter_temp["S1_93_11"] > 18]
                    .groupby("_bsr_id_", dropna=False)["seg_len_m"]
                    .sum()
                    .rename("len_gt18_m")
                    .reset_index()
                )
                gt22 = (
                    inter_temp[inter_temp["S1_93_11"] > 22]
                    .groupby("_bsr_id_", dropna=False)["seg_len_m"]
                    .sum()
                    .rename("len_gt22_m")
                    .reset_index()
                )
                length_df = total_len.merge(gt18, on="_bsr_id_", how="left").merge(gt22, on="_bsr_id_", how="left")
                length_df["len_gt18_miles"] = length_df["len_gt18_m"].fillna(0.0) * M_TO_MILES
                length_df["len_gt22_miles"] = length_df["len_gt22_m"].fillna(0.0) * M_TO_MILES
            else:
                # No temperature field; conditional lengths are 0
                length_df = total_len.copy()
                length_df["len_gt18_miles"] = 0.0
                length_df["len_gt22_miles"] = 0.0

            # Clean up
            length_df = length_df[["_bsr_id_", "species_stream_length", "total_len_miles", "len_gt18_miles", "len_gt22_miles"]]

        # Merge attributes back onto BSR
        out = bsr.merge(agg_df, on="_bsr_id_", how="left").merge(length_df, on="_bsr_id_", how="left")
        out["species_stream_length"] = out["species_stream_length"].fillna(0.0)
        out["total_len_miles"] = out["total_len_miles"].fillna(0.0)
        out["len_gt18_miles"] = out["len_gt18_miles"].fillna(0.0)
        out["len_gt22_miles"] = out["len_gt22_miles"].fillna(0.0)

        # Percentages (avoid divide-by-zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["percent_above_18C"] = np.where(
                out["total_len_miles"] > 0,
                100.0 * (out["len_gt18_miles"] / out["total_len_miles"]),
                0.0,
            )
            out["percent_above_22C"] = np.where(
                out["total_len_miles"] > 0,
                100.0 * (out["len_gt22_miles"] / out["total_len_miles"]),
                0.0,
            )

        # Clean before writing
        out = out.drop(columns=["_bsr_id_"], errors="ignore")
        out = out.reset_index(drop=True)
        out = out.drop(columns=[c for c in out.columns if c in {"index_right"}], errors="ignore")
        out.columns = [str(c) for c in out.columns]

        out_layer = _sanitize_layer_name(f"{lyr}")
        gpd.GeoDataFrame(out, geometry="geometry", crs=bsr.crs).to_file(
            out_gpkg,
            layer=out_layer,
            driver="GPKG",
            mode=("w" if i == 0 else "a"),
            index=False,
        )

    return out_gpkg



if __name__ == "__main__":
    temp_points_gpkg = r"data\inputs\NorWeST_PredictedStreamTempPoints_MidColumbia_MWMT\NorWeST_PredictedStreamTempPoints_MidColumbia_MWMT.shp"
    fish_dist_gpkg = r"data\inputs\All_Fish_Dist.gpkg"
    fish_dist_path = r"data\outputs\All_Fish_Dist_with_temp.gpkg"
    value_fields = ("S1_93_11", "S30_2040D", "S32_2080D", 
                    #"S41_2080M", "S37_9311M"
                    )
    join_temp_medians_to_fish(temp_points_gpkg, fish_dist_gpkg, fish_dist_path, value_fields = value_fields)

    bsr_path = r"data\inputs\BSR_net_analysis.gpkg"
    fish_dist_path = r"data\outputs\All_Fish_Dist_with_temp.gpkg"
    bsr_temp_path = r"data\outputs\base_bsr_scaled_scores.gpkg"

    # If bsr_path is a .shp, outputs will be written to a sibling .gpkg with the same stem.
    out_gpkg_written = join_fish_fields_to_bsr(fish_dist_path, bsr_path, out_path = bsr_temp_path, value_fields= value_fields)
    print(f"Wrote joined layers to: {out_gpkg_written}")
