import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
import folium
import fiona
import branca

st.set_page_config(layout="wide")

# -------------------------
# Session init
# -------------------------
def _init_state() -> None:
    """Initialize keys we rely on."""
    for k, v in {
        "gdf_scored": None,     # stores last computed GeoDataFrame
        "last_weights": None,   # stores last weights used to compute
        "last_layer_sig": None, # (gpkg_path, single_layer_name, field choices) tuple
    }.items():
        st.session_state.setdefault(k, v)

_init_state()

# -------------------------
# Tier threshold defaults & UI
# -------------------------
TIER_NS = "tier_cfg_v1"

DEFAULT_TIER_THRESHOLDS = {
    "Upper Grande Ronde": {"t1_min": 85.0, "t2_min": 65.0},
    "Catherine Creek": {"t1_min": 75.0, "t2_min": 50.0},
    "_default": {"t1_min": 85.0, "t2_min": 65.0},  # used for any other basin names
}

def _init_tier_state() -> None:
    ss = st.session_state
    ss.setdefault(TIER_NS, {})
    for basin, vals in DEFAULT_TIER_THRESHOLDS.items():
        ss[TIER_NS].setdefault(basin, dict(vals))  # shallow copy

def tier_inputs() -> Tuple[Dict[str, Dict[str, float]], bool, Optional[str]]:
    """
    Render UI to configure tier thresholds for each basin.
    Returns (cfg, is_valid, error_msg).
    cfg format:
      { 'Upper Grande Ronde': {'t1_min': float, 't2_min': float},
        'Catherine Creek':    {'t1_min': float, 't2_min': float},
    """
    _init_tier_state()
    ss = st.session_state[TIER_NS]

    st.markdown("### Tier thresholds (per basin)")
    st.markdown("""
    **Rule:** All BSRs are scored as:

    - **Tier 1**  Weighted BSR Score ≥ Tier 1 Lower Bound
    - **Tier 2** Tier 1 Lower Bound ≥ Weighted BSR Score ≥ Tier 2 Lower Bound
    - **Tier 3**  Tier 2 Lower Bound ≥ Weighted BSR Score 
    """)


    cols = st.columns(3)
    basins_order = ["Upper Grande Ronde", "Catherine Creek"]
    labels = {
        "Upper Grande Ronde": "Upper Grande Ronde",
        "Catherine Creek": "Catherine Creek",
    }

    for basin, col in zip(basins_order, cols):
        with col:
            st.write(f"**{labels[basin]}**")
            t1 = st.number_input(
                f"Tier 1 Lower Bound",
                min_value=0.0, max_value=100.0, step=1.0,
                value=float(ss[basin]["t1_min"]),
                key=f"{TIER_NS}_{basin}_t1",
                help="Scores at or above this go to Tier 1"
            )
            t2 = st.number_input(
                f"Tier 2 Lower Bound",
                min_value=0.0, max_value=100.0, step=1.0,
                value=float(ss[basin]["t2_min"]),
                key=f"{TIER_NS}_{basin}_t2",
                help="Scores at or above this (but below Tier 1 min) go to Tier 2"
            )
            ss[basin]["t1_min"] = float(t1)
            ss[basin]["t2_min"] = float(t2)

    # Validate
    for basin, vals in ss.items():
        if basin not in labels:  # ignore any stray keys
            continue
        t1, t2 = vals["t1_min"], vals["t2_min"]
        if not (0.0 <= t2 <= t1 <= 100.0):
            return ss, False, f"Invalid thresholds for {labels.get(basin, basin)}: require 0 ≤ Tier2 min ≤ Tier1 min ≤ 100."

    return ss, True, None

# -------------------------
# Path utilities & validation
# -------------------------
def resolve_gpkg_path(raw_path: str) -> Path:
    """Normalize and validate a path across OSes."""
    if not raw_path:
        raise FileNotFoundError("No GeoPackage path provided.")
    p = Path(raw_path.replace("\\", "/")).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"GPKG not found: {p}\n(Working directory: {Path.cwd()})")
    if not p.is_file():
        raise FileNotFoundError(f"Path is not a file: {p}")
    return p

# -------------------------
# Data loading (single layer only)
# -------------------------
@st.cache_data
def get_single_layer_name(gpkg_path: str) -> str:
    """
    Return the sole layer name in a GeoPackage.
    Raises if none exist; if multiple, uses the first but warns in UI.
    """
    p = resolve_gpkg_path(gpkg_path)
    try:
        layers = fiona.listlayers(str(p))
    except Exception as e:
        raise ValueError(f"Unable to read layers from GPKG: {e}") from e
    if not layers:
        raise ValueError(f"No layers found in GeoPackage: {p}")
    if len(layers) > 1:
        st.warning(
            f"Multiple layers detected in {p.name}. This app reads only the first layer: **{layers[0]}**.",
            icon="⚠️",
        )
    return layers[0]

@st.cache_data
def load_single_layer(gpkg_path: str) -> gpd.GeoDataFrame:
    """Read the sole layer and reproject to EPSG:4326 for web mapping."""
    p = resolve_gpkg_path(gpkg_path)
    layer = get_single_layer_name(gpkg_path)
    try:
        gdf = gpd.read_file(str(p), layer=layer)
    except Exception as e:
        raise ValueError(f"Failed to read layer '{layer}' from {p}: {e}") from e

    if gdf.empty:
        raise ValueError(f"Layer '{layer}' is empty in {p}.")

    if gdf.crs is None:
        st.warning("Input layer has no CRS. Assuming it is already in EPSG:4326 for display.")
        gdf_wgs84 = gdf
    else:
        gdf_wgs84 = gdf.to_crs(epsg=4326)

    return gdf_wgs84

# -------------------------
# UI helpers: linked slider + number box with sum constraint (namespaced)
# -------------------------
def _ns_key(ns: str, key: str, kind: str) -> str:
    """Compose a session_state key for a widget under a namespace."""
    return f"{ns}__{key}_{kind}"

def _init_weight_state_ns(ns: str, defaults: Dict[str, int]) -> None:
    """Initialize session_state for weight sliders and numeric boxes under a namespace."""
    for k, default_each in defaults.items():
        s_key, n_key = _ns_key(ns, k, "slider"), _ns_key(ns, k, "num")
        st.session_state.setdefault(s_key, default_each)
        st.session_state.setdefault(n_key, default_each)

def _link_slider_to_box(ns: str, key: str) -> None:
    """Callback: when slider moves, update its paired number input (namespaced)."""
    sk, nk = _ns_key(ns, key, "slider"), _ns_key(ns, key, "num")
    st.session_state[nk] = st.session_state[sk]

def _link_box_to_slider(ns: str, key: str) -> None:
    """Callback: when number changes, update its paired slider and clamp to [0,100] (namespaced)."""
    sk, nk = _ns_key(ns, key, "slider"), _ns_key(ns, key, "num")
    val = max(0, min(100, int(st.session_state[nk])))
    st.session_state[nk] = val
    st.session_state[sk] = val

def weight_inputs(currcond_label: str, currtemp_label: str, mig_label: str, ns: str) -> Tuple[Dict[str, int], int]:
    labels = [
        ("Geomorphic_weight", "Geomorphic"),
        ("PScore_Weight", "PScore"),
        ("UScore_Weight", "UScore"),
        ("CurrCond_Weight", currcond_label),
        ("CurrTemp_Weight", currtemp_label),
        ("Migration_Weight", mig_label),
    ]
    base_keys = [k for k, _ in labels]

    # --- per-key defaults: 17 for 4 categories; 16 for 2 (CurrTemp & Migration)
    default_map = {
        "Geomorphic_weight": 17,
        "PScore_Weight": 17,
        "UScore_Weight": 17,
        "CurrCond_Weight": 17,
        "CurrTemp_Weight": 16,
        "Migration_Weight": 16,
    }
    _init_weight_state_ns(ns, default_map)

    st.markdown("### Weights (must sum to **100**)")

    cols = st.columns(6)
    for (key, label), col in zip(labels, cols):
        with col:
            st.slider(
                label,
                min_value=0, max_value=100, step=1,
                key=_ns_key(ns, key, "slider"),
                on_change=_link_slider_to_box, args=(ns, key),
                help=f"Weight for {label} (0–100)",
            )
            st.number_input(
                "pts", min_value=0, max_value=100, step=1,
                key=_ns_key(ns, key, "num"),
                on_change=_link_box_to_slider, args=(ns, key),
                label_visibility="collapsed",
                help="Type exact points",
            )

    # Collect values and show remaining (for the CURRENT namespace only)
    weights = {k: int(st.session_state[_ns_key(ns, k, "slider")]) for k in base_keys}
    total = sum(weights.values())
    remaining = 100 - total

    if remaining == 0:
        st.success("Remaining points: 0 ✓ (ready to compute)")
    elif remaining > 0:
        st.info(f"Remaining points: {remaining}")
    else:
        st.error(f"Over budget by {-remaining} (reduce weights to proceed)")

    st.caption("Tip: You can type exact values in the small boxes under each slider.")

    # Optional nudge if weights changed compared to last compute
    if st.session_state.last_weights is not None and weights != st.session_state.last_weights:
        st.warning("Weights changed. Click **Compute** to update the map.", icon="⚠️")

    return weights, total

# -------------------------
# Scoring & tiering (fixed)
# -------------------------
BASE_REQUIRED_FIELDS = ["Geomorphic", "PScore", "UScore", "Basin_Name"]
WEIGHTS_NS = "weights_v1"  # stable key namespace for all weight widgets

def _minmax_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = float(s.min(skipna=True))
    mx = float(s.max(skipna=True))
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        # all NaN or constant → return 0s
        return pd.Series(0.0, index=s.index, dtype="float64")
    out = (s - mn) / (mx - mn)
    return out.fillna(0.0).astype("float64")


def compute_weighted_fields(
    gdf: gpd.GeoDataFrame,
    weights: Dict[str, int],
    currcond_field: str,
    currtemp_field: str,
    migration_field: str,
    tier_cfg: Dict[str, Dict[str, float]],
) -> gpd.GeoDataFrame:
    """
    Normalize each metric to [0,1] using min–max, then compute:
      - Per-metric weighted columns W_<field> = weight * norm (0–100 scale contribution)
      - Combined Weighted_Score = sum of per-metric weighted columns (0–100)
      - Weighted_Tier using user-defined basin thresholds in tier_cfg
    """
    required = BASE_REQUIRED_FIELDS + [currcond_field, currtemp_field, migration_field]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    gdf = gdf.copy()

    # Map each weight key to the corresponding data column
    field_map = {
        "Geomorphic_weight": "Geomorphic",
        "PScore_Weight": "PScore",
        "UScore_Weight": "UScore",
        "CurrCond_Weight": currcond_field,
        "CurrTemp_Weight": currtemp_field,
        "Migration_Weight": migration_field,
    }

    # Build normalized and weighted columns
    weighted_cols = []
    for w_key, col in field_map.items():
        ncol = f"{col}_norm"
        gdf[ncol] = _minmax_norm(gdf[col])
        w = float(weights.get(w_key, 0))
        wcol = f"W_{col}"
        gdf[wcol] = (w * gdf[ncol]).astype(float)
        weighted_cols.append(wcol)

    # Weighted sum
    gdf["Weighted_Score"] = gdf[weighted_cols].sum(axis=1).clip(lower=0.0, upper=100.0).round(2)

    # Tiering using user-configured thresholds
    def pick_tier(basin: str, score: float) -> int:
        b = str(basin).strip()
        cfg = tier_cfg.get(b, tier_cfg.get("_default", {"t1_min": 85.0, "t2_min": 65.0}))
        t1, t2 = float(cfg["t1_min"]), float(cfg["t2_min"])
        s = float(score)
        if s >= t1:
            return 1
        elif s >= t2:
            return 2
        else:
            return 3

    gdf["Weighted_Tier"] = [pick_tier(b, s) for b, s in zip(gdf["Basin_Name"], gdf["Weighted_Score"])]
    return gdf



# -------------------------
# Map rendering
# -------------------------
def _tier_map(
    gdf: gpd.GeoDataFrame,
    fill_opacity: float = 0.55,
) -> folium.Map:
    """
    Folium map of Weighted_Tier with pretty colors:
      - 1 -> green, 2 -> blue, 3 -> red (slightly transparent).
    """
    # Pretty ColorBrewer-like palette
    tier_colors = {1: "#1b9e77", 2: "#377eb8", 3: "#e41a1c"}

    def style_fn(feature):
        t = feature["properties"].get("Weighted_Tier", None)
        color = tier_colors.get(int(t) if t is not None else 2, "#377eb8")
        return {"fillColor": color, "color": "#333333", "weight": 0.6, "fillOpacity": fill_opacity}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    fields = [f for f in ["BSR", "Weighted_Tier", "Weighted_Score"] if f in gdf.columns]
    aliases = [f"{f}:" for f in fields]

    folium.GeoJson(
        gdf,
        name="Tier",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 10px; z-index: 9999;
                background: white; padding: 8px 10px; border: 1px solid #bbb;
                border-radius: 4px; font-size: 12px;">
      <div style="margin-bottom:6px;"><b>Weighted Tier</b></div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#1b9e77;border:1px solid #333;margin-right:6px;"></span> 1</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#377eb8;border:1px solid #333;margin-right:6px;"></span> 2</div>
      <div><span style="display:inline-block;width:14px;height:14px;background:#e41a1c;border:1px solid #333;margin-right:6px;"></span> 3</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m

def _score_map(
    gdf: gpd.GeoDataFrame,
    fill_opacity: float = 0.55,
) -> folium.Map:
    """
    Folium choropleth by Weighted_Score:
      - pretty, slightly transparent gradient from white (low) to dark green (high).
    """
    if "Weighted_Score" not in gdf.columns:
        raise ValueError("Weighted_Score not found. Click Compute first.")

    # Build a white -> dark green colormap
    vmin = float(gdf["Weighted_Score"].min())
    vmax = float(gdf["Weighted_Score"].max())
    cmap = branca.colormap.LinearColormap(
        colors=["#ffffff", "#00441b"], vmin=vmin, vmax=vmax
    )
    cmap.caption = "Weighted_Score (white = low, dark green = high)"

    def style_fn(feature):
        val = feature["properties"].get("Weighted_Score", None)
        color = "#cccccc" if val is None else cmap(val)
        return {"fillColor": color, "color": "#333333", "weight": 0.6, "fillOpacity": fill_opacity}

    m = folium.Map(tiles="CartoDB positron", control_scale=True)
    x_min, y_min, x_max, y_max = gdf.total_bounds
    m.fit_bounds([[y_min, x_min], [y_max, x_max]])

    fields = [f for f in ["BSR", "Weighted_Tier", "Weighted_Score"] if f in gdf.columns]
    aliases = [f"{f}:" for f in fields]

    folium.GeoJson(
        gdf,
        name="Weighted Score",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    ).add_to(m)

    m.add_child(cmap)
    folium.LayerControl().add_to(m)
    return m

# -------------------------
# Compute callback
# -------------------------
def _compute_and_store(
    gdf: gpd.GeoDataFrame,
    weights: Dict[str, int],
    layer_sig: Tuple[str, str, str, str, str],
    currcond_field: str,
    currtemp_field: str,
    migration_field: str,
    tier_cfg: Dict[str, Dict[str, float]],
) -> None:
    """Compute fields and store results in session_state."""
    try:
        gdf_scored = compute_weighted_fields(
            gdf, weights, currcond_field, currtemp_field, migration_field, tier_cfg
        )
    except Exception as e:
        st.error(f"Failed to compute weighted fields: {e}")
        return
    st.session_state.gdf_scored = gdf_scored
    st.session_state.last_weights = weights.copy()
    st.session_state.last_layer_sig = layer_sig

# -------------------------
# Documentation Tab content
# -------------------------
def render_docs() -> None:
    st.header("Methodology & Scoring Documentation")
    st.markdown("""
    ### How normalization and user-defined weighting work

    **Purpose:**  
    All input metrics are first standardized to a common 0–1 scale. This allows you to assign your own weights to each of the metrics, producing a final **BSR Tiering Score** that ranges from **0 to 100**.

    ---

    **Step 1 — Normalize each metric (0–1)**  
    Each selected metric (e.g., *Geomorphic*, *PScore*, *UScore*, *Current Condition*, *Temperature*, *Migration Corridor*) is rescaled using **min–max normalization**, where:

    - The **lowest** score for a metric becomes **0**.  
    - The **highest** score becomes **1**.  
    - All other scores fall between 0 and 1.  

    *Example:*  
    The **Geomorphic Potential Score** ranges from 5 to 25, so a BSR with a score of 25 becomes **1.0**, and one with a score of 5 becomes **0.0**.

    ---

    **Step 2 — Apply user-defined weights**  
    After normalization, you use six sliders to assign weights to each metric. These weights determine how much influence each metric has on the final score.

    *Example:*  
    If you assign a **Geomorphic weight of 50**, a BSR with a normalized Geomorphic value of **1.0** contributes **50 points** (50 × 1.0), while a BSR with a normalized value of **0.6** contributes **30 points**.

    ---

    **Step 3 — Compute the combined Weighted_Score (0–100)**  
    The app multiplies each metric’s normalized value by its assigned weight, then sums all results.


    Because the weights always sum to 100, the final score also ranges between **0 and 100**.

    """)

    st.markdown("""
    ### 1) How BSR tiers are determined
    Tiers are derived from the **Weighted_Score (0–100)** using basin-specific thresholds:

    | Basin | Tier 1 | Tier 2 | Tier 3 |
    |---|---|---|---|
    | Upper Grande Ronde | 85 - 100 | 65 - 85 | 0 - 65 |
    | Catherine Creek | 75 - 100 | 50 - 75 | 0 - 50 |

    These thresholds are applied after computing the weighted score from the selected metrics.

    ---

    ### 2) Existing Temperature Score
    If stream temperatures are **poor or lethal**, newly created or existing habitat may not be fully utilized.

    **Primary data layers used by the Science TAC:**
    - **CRITFC:** Temperature model and Chinook extents
    - **U.S. Bureau of Reclamation:** FLIR data

    **Qualitative → numeric conversion**
    - **Catherine Creek:** OK → 0, Lethal → −5  
    - **Upper Grande Ronde:** Excellent → 5, Good → 3, Fair → 0, Poor → −5

    #### 2.1) CurrTemp 18°C Threshold
    Weighted by the **% of Spring Chinook streams above 18°C** within each BSR using **NorWEST** modeled 19-year average August mean temperatures (1993–2011).

    #### 2.2) CurrTemp 22°C Threshold
    Computed analogously to the 18°C metric but using the **22°C** exceedance.

    ---

    ### 3) Geomorphic Potential Score
    Targets areas with higher capacity to achieve geomorphic change. Assumes **moderately confined or unconfined** reaches offer greater opportunity to increase **habitat quantity and quality**.

    **Primary data layers:**
    - **NOAA Science Center:** Chinook intrinsic potential (stream width, valley width, gradient with sediment filter) and **Beechie/Imaki** planform class (confined, island-braided, meandering, straight)
    - **CRITFC:** Valley setting (confined, partly confined, unconfined)

    **Qualitative → numeric conversion:** High → 25, Medium → 15, Low → 5  
    (Original score contributes **up to 25 points** toward the total 100. This is normalzied and weighted by the user as described above.)

    ---

    ### 4) Use (U)-score
    Ranks BSRs by the number of **critical/imperiled life stages** present and their **High/Medium/Low** rankings based on fish utilization.

    **Qualitative → numeric conversion:** High → 5, Medium → 3, Low → 1  
    A calibration factor scales this to **up to 25 points** (implementation differed slightly between **Catherine Creek**, which emphasized Chinook, and the **Upper Grande Ronde**, which used all three focal species).

    ---

    ### 5) Periodicity (P)-score
    Counts the number of life stages present per focal species from periodicity tables (length of time present is **not** weighted).  
    A calibration factor scales the raw count to **up to 25 points** (more species × more life stages → higher score).

    ---

    ### 6) Current Habitat Condition Score
    Targets areas where improving conditions will yield meaningful gains. Assumes:
    - **Fair–Good** habitat = strong opportunity for improvement
    - **Poor** habitat = larger investment for smaller gain
    - **Excellent** habitat = little room to improve

    **Primary data layers:**
    - **ODFW:** HabRate model, redd waypoint data, fish utilization layers

    **Qualitative → numeric conversion:**  
    Excellent → 5, Good → 25, Fair → 25, Poor → 5  
    (Scaled to **up to 25 points** total)

    **6.1) Current Condition RCAT – Prioritizes poor condition**  
    _{placeholder}_

    **6.2) Current Condition RCAT – Prioritizes medium condition**  
    _{placeholder}_

    ---

    ### 7) Migration Corridor Score
    Weighted number of **Chinook** or **Steelhead** stream miles **upstream of each BSR**. Higher values indicate BSRs that are more critical for connecting upstream habitat.

    Reference (conceptual motivation):  
    **Hahlbeck et al. (2023)** *Ecosphere* — “Habitat fragmentation drives divergent survival strategies of a cold-water fish in a warm landscape.”  
    Link: https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecs2.4622
    """)

def main() -> None:
    st.title("BSR Weighted Scoring & Priority Map")

    tab_app, tab_docs = st.tabs(["Map & Scoring", "Documentation"])

    with tab_app:
        uploaded = st.file_uploader("Upload a GeoPackage (optional)", type=["gpkg"])
        gpkg_default = "data/outputs/base_bsr_scaled_scores.gpkg"
        gpkg_input = st.text_input(
            "GeoPackage path",
            value=gpkg_default,
            help="If not uploading, provide a path relative to app root."
        )


        if uploaded is not None:
            tmp_path = Path("/tmp") / uploaded.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            gpkg_path = str(tmp_path)
            st.info(f"Using uploaded file: {tmp_path.name}")
        else:
            gpkg_path = gpkg_input

        # Load the (single) layer
        try:
            single_layer_name = get_single_layer_name(gpkg_path)
            gdf = load_single_layer(gpkg_path)
            st.caption(f"Loaded layer: **{single_layer_name}**")
        except Exception as e:
            st.error(str(e))
            st.stop()

        # -------------------------
        # Tier threshold selectors
        # -------------------------
        tier_cfg, tier_ok, tier_err = tier_inputs()
        if not tier_ok:
            st.error(tier_err)


        # -------------------------
        # Current Condition selector
        # -------------------------
        st.markdown("### Current Condition Field")
        currcond_options = {
            'Existing Current Condition': 'CurrCond',
            'Current Condition RCAT - Prioritizes poor condition': 'CurrCond_RCAT_Linear',
            'Current Condition RCAT - Prioritizes medium condition': 'CurrCond_RCAT_Quad ',
        }
        currcond_label = st.selectbox(
            "Choose the Current Condition metric:",
            list(currcond_options.keys()),
            index=0,
            help="Controls both the weight label and the field used in the weighted score."
        )
        currcond_field = currcond_options[currcond_label]

        # -------------------------
        # CurrTemp selector
        # -------------------------
        st.markdown("### Temperature Score")
        currtemp_options = {
            "Existing CurrTemp": "CurrTemp",
            "CurrTemp - 18C threshold": "CurrTemp18C",
            "CurrTemp - 22C threshold": "CurrTemp22C",
        }
        currtemp_label = st.selectbox(
            "Choose the temperature metric:",
            list(currtemp_options.keys()),
            index=0,
            help="Select which temperature score to include in the weighting."
        )
        currtemp_field = currtemp_options[currtemp_label]

        # -------------------------
        # Migration Corridor selector
        # -------------------------
        st.markdown("### Migration Corridor Score")
        migration_options = {
            "Migration Corridor Score - Chinook": "MScore_CH",
            "Migration Corridor Score - Steelhead": "MScore_ST",
        }
        migration_label = st.selectbox(
            "Choose the migration corridor metric:",
            list(migration_options.keys()),
            index=0,
            help="These metrics represent the weighted number of **Chinook** or **Steelhead** stream miles **upstream of each BSR**. Higher values indicate BSRs that are more critical for connecting upstream habitat."
        )
        migration_field = migration_options[migration_label]

        # Validate required columns before showing weights
        dynamic_required = [currcond_field, currtemp_field, migration_field]
        missing_now = [f for f in (BASE_REQUIRED_FIELDS + dynamic_required) if f not in gdf.columns]
        if missing_now:
            st.error(
                "The layer is missing required fields:\n\n"
                + ", ".join(missing_now)
                + "\n\nProvide a layer containing these fields."
            )
            st.stop()

        # If inputs changed since last compute, clear previous result to prevent mismatch
        current_sig = (gpkg_path, single_layer_name, currcond_field, currtemp_field, migration_field)
        if st.session_state.last_layer_sig is not None and st.session_state.last_layer_sig != current_sig:
            st.session_state.gdf_scored = None
            st.session_state.last_weights = None

        # Weights UI (6 components)
        ns = WEIGHTS_NS
        weights, total = weight_inputs(
            currcond_label=currcond_label,
            currtemp_label=currtemp_label,
            mig_label=migration_label,
            ns=ns,
        )
        ready = (total == 100)

        st.markdown("---")
        st.button(
            "Compute Weighted Score",
            type="primary",
            disabled=not ready or not tier_ok,
            help="Enabled when weights sum to 100 and tier thresholds are valid.",
            on_click=_compute_and_store,
            args=(gdf, weights, current_sig, currcond_field, currtemp_field, migration_field, tier_cfg),
            key="compute_btn",
        )


        # Map style selector (applies to the last computed result)
        st.markdown("### New BSR Tiers")
        map_mode = st.selectbox(
            "Select map display:",
            [
                "Weighted BSR Tier (1 = green, 2 = blue, 3 = red)",
                "Weighted BSR Score (white → dark green)",
            ],
            help="Tier shows BSR rankings resulting from weights; Score shows weighted BSR score.",
        )

        # Persistently display the last computed result (if any)
        if st.session_state.gdf_scored is not None:
            if "Weighted_Score" not in st.session_state.gdf_scored.columns:
                st.error("Weighted_Score not found. Click **Compute**.")
                st.stop()

            if map_mode.startswith("Weighted BSR Tier"):
                m = _tier_map(st.session_state.gdf_scored, fill_opacity=0.55)
            else:
                m = _score_map(st.session_state.gdf_scored, fill_opacity=0.55)

            st_folium(m, use_container_width=True, height=720, key="priority_map")

           # Build preview columns including per-metric weighted contributions
            weighted_cols_preview = [
                f"W_Geomorphic",
                f"W_PScore",
                f"W_UScore",
                f"W_{currcond_field}",
                f"W_{currtemp_field}",
                f"W_{migration_field}",
            ]

            preview_cols = [
                "BSR", 
                f"W_Geomorphic",
                f"W_PScore",
                f"W_UScore",
                f"W_{currcond_field}",
                f"W_{currtemp_field}",
                f"W_{migration_field}",
                "Weighted_Score", "Weighted_Tier"
            ]

            # Only keep columns that exist
            preview_cols = [c for c in preview_cols if c in st.session_state.gdf_scored.columns]

            with st.expander("Table of new weighted metrics:"):
                # Round the W_* contributions for readability in the preview only
                df_prev = st.session_state.gdf_scored[preview_cols].copy()

                for c in weighted_cols_preview + ["Weighted_Score"]:
                    if c in df_prev.columns:
                        df_prev[c] = pd.to_numeric(df_prev[c], errors="coerce").round(2)
                # Drop the index so it doesn’t show in Streamlit
                st.dataframe(df_prev.reset_index(drop=True))

            st.caption(
                f"Notes: Fields are added in memory only; the GeoPackage is not modified. "
                f"Current tier cutoffs — UGR: ≥{tier_cfg['Upper Grande Ronde']['t1_min']:.0f}→Tier 1, "
                f"≥{tier_cfg['Upper Grande Ronde']['t2_min']:.0f}→Tier 2; "
                f"Catherine: ≥{tier_cfg['Catherine Creek']['t1_min']:.0f}→Tier 1, "
                f"≥{tier_cfg['Catherine Creek']['t2_min']:.0f}→Tier 2."
            )

        else:
            st.info("Adjust weights so the total equals 100, then click **Compute**.")

    with tab_docs:
        render_docs()

if __name__ == "__main__":
    main()
