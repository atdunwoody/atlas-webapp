from pathlib import Path
import os
import geopandas as gpd
import fiona

def pack_gpkgs_into_layers(
    gpkg_paths,
    out_gpkg,
    overwrite=True,
    layer_name_prefix=None,
):
    """
    Combine multiple .gpkg files into a single multi-layer .gpkg.

    Parameters
    ----------
    gpkg_paths : Sequence[str | Path]
        Paths to input GeoPackages. If an input has multiple layers,
        each source layer is written separately.
    out_gpkg : str | Path
        Output GeoPackage path to write.
    overwrite : bool, default True
        If True and the output exists, it will be removed first.
    layer_name_prefix : str | None
        Optional string prepended to every output layer name
        (e.g., 'fish_'), useful for namespacing.

    Notes
    -----
    - Output layer names:
        - If an input GPKG has exactly one layer: `{prefix}{stem}`
        - If an input GPKG has multiple layers: `{prefix}{stem}__{src_layer}`
    - If a computed layer name already exists in this run, a numeric
      suffix `_1`, `_2`, ... is appended to keep names unique.
    """
    out_gpkg = Path(out_gpkg)
    if overwrite and out_gpkg.exists():
        out_gpkg.unlink()

    def _unique(name, used):
        base = name
        k = 1
        while name in used:
            name = f"{base}_{k}"
            k += 1
        return name

    used_layer_names = set()

    for p in map(Path, gpkg_paths):
        if not p.exists():
            print(f"Warning: missing file, skipping: {p}")
            continue

        try:
            src_layers = fiona.listlayers(p.as_posix())
        except Exception as e:
            print(f"Warning: could not list layers in {p}: {e}")
            continue

        # If no layers (corrupt/empty container), skip
        if not src_layers:
            print(f"Warning: no layers found in {p}, skipping.")
            continue

        # Decide layer-name template(s)
        file_stem = p.stem
        prefix = layer_name_prefix or ""

        # Write each source layer
        for src_layer in src_layers:
            # Name scheme
            if len(src_layers) == 1:
                dest_layer = f"{prefix}{file_stem}"
            else:
                dest_layer = f"{prefix}{file_stem}__{src_layer}"
            dest_layer = _unique(dest_layer, used_layer_names)

            # Read + write
            gdf = gpd.read_file(p.as_posix(), layer=src_layer)
            mode = "a" if out_gpkg.exists() else "w"
            gdf.to_file(out_gpkg.as_posix(), layer=dest_layer, driver="GPKG", mode=mode, index=False)

            used_layer_names.add(dest_layer)
            print(f"Wrote {p.name}:{src_layer} -> {out_gpkg.name}:{dest_layer}")
inputs = [
    r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\Spring_Chinook.gpkg",
    r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\Summer_Steelhead.gpkg",
    r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\Bull_Trout.gpkg",
    r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\Coho.gpkg",
    r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\Fall_Chinook.gpkg",
]

out_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\data\All_Fish_Dist.gpkg"

pack_gpkgs_into_layers(inputs, out_path, overwrite=True)
