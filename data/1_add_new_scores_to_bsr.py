from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import shutil

# =======================
# USER INPUTS
# =======================
INPUT_GPKG  = r"data\outputs\base_bsr_scaled_scores.gpkg"
OUTPUT_GPKG = None              # set to a new path to avoid overwriting; if None, overwrites INPUT_GPKG

CH_FIELD = "Upstream_CH_Miles"
ST_FIELD = "Upstream_ST_Miles"
CH_OUT   = "MScore_CH"
ST_OUT   = "MScore_ST"

P18_FIELD = "percent_above_18C"
P22_FIELD = "percent_above_22C"
C18_OUT   = "CurrTemp18C"  # reverse-normalized: lowest % -> 1, highest % -> 0
C22_OUT   = "CurrTemp22C"  # reverse-normalized: lowest % -> 1, highest % -> 0


def list_layers(path):
    import fiona
    return list(fiona.listlayers(path))


def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not np.any(~np.isnan(s.values)):
        return pd.Series(np.nan, index=series.index)  # all NaN
    vmin = np.nanmin(s.values)
    vmax = np.nanmax(s.values)
    if vmax == vmin:
        return pd.Series(np.where(np.isnan(s.values), np.nan, 0.0), index=series.index)
    return (s - vmin) / (vmax - vmin)


def reverse_minmax_norm(series: pd.Series) -> pd.Series:
    """Map lowest -> 1, highest -> 0; handle NaNs and constants."""
    s = pd.to_numeric(series, errors="coerce")
    if not np.any(~np.isnan(s.values)):
        return pd.Series(np.nan, index=series.index)
    vmin = np.nanmin(s.values)
    vmax = np.nanmax(s.values)
    if vmax == vmin:
        # all same -> treat as all 'lowest'; set 1 for non-NaN
        return pd.Series(np.where(np.isnan(s.values), np.nan, 1.0), index=series.index)
    return 1.0 - ((s - vmin) / (vmax - vmin))


def process_layer(gpkg_path, layer, engine=None):
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer, engine=engine)
    except Exception as e:
        print(f"[skip] Layer '{layer}': cannot read as spatial ({e})")
        return None, False

    # If geometry is missing entirely, skip (attribute-only table)
    if gdf.empty and (gdf.geometry.name not in gdf.columns):
        print(f"[skip] Layer '{layer}': no geometry / empty.")
        return None, False

    # Add CH/ST normalized fields when available
    changed = False
    if CH_FIELD in gdf.columns:
        gdf[CH_OUT] = minmax_norm(gdf[CH_FIELD]).astype("float64")
        changed = True
    else:
        print(f"[note] Layer '{layer}': missing field '{CH_FIELD}'")

    if ST_FIELD in gdf.columns:
        gdf[ST_OUT] = minmax_norm(gdf[ST_FIELD]).astype("float64")
        changed = True
    else:
        print(f"[note] Layer '{layer}': missing field '{ST_FIELD}'")

    # Add reverse-normalized temperature scores when available
    if P18_FIELD in gdf.columns:
        gdf[C18_OUT] = reverse_minmax_norm(gdf[P18_FIELD]).astype("float64")
        changed = True
    else:
        print(f"[note] Layer '{layer}': missing field '{P18_FIELD}'")

    if P22_FIELD in gdf.columns:
        gdf[C22_OUT] = reverse_minmax_norm(gdf[P22_FIELD]).astype("float64")
        changed = True
    else:
        print(f"[note] Layer '{layer}': missing field '{P22_FIELD}'")

    if not changed:
        return None, False

    return gdf, True


def main():
    in_path = Path(INPUT_GPKG)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Choose fastest engine if present
    engine = "pyogrio"
    try:
        import pyogrio  # noqa: F401
    except Exception:
        engine = None

    layers = list_layers(in_path)
    if not layers:
        raise ValueError(f"No layers found in {in_path}")

    # Decide output path (use temp swap if overwriting)
    if OUTPUT_GPKG:
        out_path = Path(OUTPUT_GPKG)
        if out_path.exists():
            out_path.unlink()
        write_mode_first = "w"
    else:
        out_path = in_path.with_suffix(".tmp.gpkg")
        if out_path.exists():
            out_path.unlink()
        write_mode_first = "w"

    wrote_any = False
    mode = write_mode_first

    for lyr in layers:
        gdf, ok = process_layer(in_path, lyr, engine=engine)
        if not ok:
            # pass-through original layer to keep the GPKG intact
            try:
                orig = gpd.read_file(in_path, layer=lyr, engine=engine)
                orig.to_file(out_path, layer=lyr, driver="GPKG", engine=engine, mode=mode)
                mode = "a"
                wrote_any = True
                print(f"[copy] Unchanged layer '{lyr}' written through.")
            except Exception as e:
                print(f"[warn] Could not pass-through layer '{lyr}': {e}")
            continue

        gdf.to_file(out_path, layer=lyr, driver="GPKG", engine=engine, mode=mode)
        mode = "a"
        wrote_any = True
        added = [fld for fld in [CH_OUT, ST_OUT, C18_OUT, C22_OUT] if fld in gdf.columns]
        print(f"[ok] Updated layer '{lyr}' with {added}.")

    if not wrote_any:
        raise RuntimeError("No layers were written; nothing to do.")

    # If overwriting, atomically replace the original
    if OUTPUT_GPKG is None:
        backup = in_path.with_suffix(".bak.gpkg")
        try:
            if backup.exists():
                backup.unlink()
            shutil.move(str(in_path), str(backup))
            shutil.move(str(out_path), str(in_path))
            print(f"Done. Overwrote '{in_path.name}'. Backup at '{backup.name}'.")
        except Exception as e:
            print(f"[error] Swap failed: {e}")
            print(f"Temporary output kept at: {out_path}")
    else:
        print(f"Done. Wrote all layers to: {out_path}")

if __name__ == "__main__":
    main()
