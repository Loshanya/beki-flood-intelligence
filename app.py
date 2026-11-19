import os
import io
from PIL import Image
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage


# =========================================================
# PAGE CONFIGURATION AND PATHS
# =========================================================

st.set_page_config(
    page_title="Beki River | Flood Intelligence",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "ML"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEPLOYMENT_DIR = BASE_DIR / "deployment"
COORD_FILE = DEPLOYMENT_DIR / "coordinate_grids.npz"

ROWS, COLS = 286, 600

COLORS = [
    "#CBD5E1",  # Dry: neutral slate
    "#0284C7",  # Very low: blue
    "#16A34A",  # Low: green
    "#EAB308",  # Moderate: yellow
    "#F97316",  # High: orange
    "#D94646",  # Very high: softened red
]
LABELS = [
    "Dry / unclassified",
    "Very low",
    "Low",
    "Moderate",
    "High",
    "Very high",
]

HAZARD_CMAP = ListedColormap(COLORS)
HAZARD_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), HAZARD_CMAP.N)


# =========================================================
# STYLING
# =========================================================


st.markdown(
    """
    <style>
    /* Main background */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #F3F6FA;
        color: #243447;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.6rem;
        max-width: 1550px;
    }

    /* Hero banner */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102A43, #176B87);
        color: #FFFFFF;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }

    .hero p {
        color: #D9EAF2 !important;
        margin-bottom: 0;
    }

    /* Main text */
    h1, h2, h3, h4, p, label, li,
    .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: #243447;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #E5EDF5;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #243447 !important;
    }

    /* Sidebar input */
    [data-testid="stSidebar"] input {
        background: #FFFFFF !important;
        color: #102A43 !important;
        -webkit-text-fill-color: #102A43 !important;
        border: 1px solid #AABCCB !important;
        border-radius: 8px !important;
    }

    /* Sidebar dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #102A43 !important;
        border-color: #AABCCB !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #102A43 !important;
    }

    /* Sidebar information box */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #D5EAF5;
        border: 1px solid #B8DCE8;
        border-radius: 12px;
    }

    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #526579 !important;
    }

    [data-testid="stMetricValue"] {
        color: #102A43 !important;
    }

    /* Download buttons */
    .stDownloadButton button {
        background: #176B87 !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 9px;
    }

    .stDownloadButton button:hover {
        background: #102A43 !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =========================================================
# HYDRAULIC DATA HELPERS
# =========================================================

def find_average_file(folder: Path, prefix: str):
    """Find the average depth/velocity Excel file in a scenario folder."""
    candidates = sorted(
        p for p in folder.glob("*.xlsx")
        if p.stem.lower().startswith(prefix.lower())
        and "org" not in p.stem.lower()
    )

    standard = folder / f"{prefix}.xlsx"
    if standard.exists():
        return standard

    return candidates[0] if candidates else None


def scenario_folders():
    """Return usable scenario folders keyed by discharge."""
    found = {}

    for folder in ROOT.glob("BEKI_FinalQ_*_GVF*"):
        match = re.search(
            r"BEKI_FinalQ_(\d+)_GVF",
            folder.name,
            re.IGNORECASE,
        )
        if not match:
            continue

        q_value = int(match.group(1))
        h_file = find_average_file(folder, "h_avg")
        v_file = find_average_file(folder, "rv_avg")

        if h_file is not None and v_file is not None:
            found[q_value] = folder

    return dict(sorted(found.items()))


@st.cache_data(show_spinner="Reading hydraulic grids...")
def load_grid(path_string: str):
    return (
        pd.read_excel(path_string, header=None)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )


def load_scenario(q_value: int, folders: dict):
    folder = folders[q_value]

    h_file = find_average_file(folder, "h_avg")
    v_file = find_average_file(folder, "rv_avg")

    if h_file is None or v_file is None:
        raise FileNotFoundError(
            f"Average depth/velocity files unavailable for Q={q_value}."
        )

    h = load_grid(str(h_file))
    v = load_grid(str(v_file))

    if h.shape != v.shape:
        raise ValueError(
            f"Depth and velocity grids differ at Q={q_value}: "
            f"{h.shape} vs {v.shape}"
        )

    if h.shape != (ROWS, COLS):
        raise ValueError(
            f"Unexpected grid shape at Q={q_value}: {h.shape}; "
            f"expected {(ROWS, COLS)}"
        )

    return h, v


# =========================================================
# HAZARD CALCULATION
# =========================================================

def calculate_hazard(h, v):
    valid = np.isfinite(h) & np.isfinite(v)

    fhi = np.full(h.shape, np.nan, dtype=float)
    fhi[valid] = h[valid] * (v[valid] + 0.5)

    wet = valid & (h > 0.1)

    classes = np.zeros(h.shape, dtype=np.uint8)
    classes[wet & (fhi <= 0.5)] = 1
    classes[wet & (fhi > 0.5) & (fhi <= 1.0)] = 2
    classes[wet & (fhi > 1.0) & (fhi <= 2.0)] = 3
    classes[wet & (fhi > 2.0) & (fhi <= 3.0)] = 4
    classes[wet & (fhi > 3.0)] = 5

    return fhi, classes, wet


# =========================================================
# HYDRAULIC INTERPOLATION
# =========================================================

def predict_hydraulic(q: float, folders: dict):
    known = sorted(folders.keys())

    lower = [value for value in known if value <= q]
    upper = [value for value in known if value >= q]

    if not lower or not upper:
        raise ValueError(
            f"Discharge {q:g} is outside the available hydraulic range "
            f"({known[0]}–{known[-1]})."
        )

    q1, q2 = max(lower), min(upper)
    h1, v1 = load_scenario(q1, folders)

    if q1 == q2:
        return h1, v1, "Exact hydraulic scenario", (q1, q2)

    h2, v2 = load_scenario(q2, folders)

    if h1.shape != h2.shape or v1.shape != v2.shape:
        raise ValueError(
            "Grid dimensions differ between bracketing hydraulic scenarios."
        )

    weight = (q - q1) / (q2 - q1)
    h = h1 + weight * (h2 - h1)
    v = v1 + weight * (v2 - v1)

    return h, v, "Linear hydraulic interpolation", (q1, q2)


# =========================================================
# LOAD V3 MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def load_ml_models():

    HF_REPO_ID = "Loshanya/beki-river-flood-models"

    # Get Hugging Face token from Streamlit Secrets or environment
    hf_token = os.getenv("HF_TOKEN")

    try:
        hf_token = hf_token or st.secrets.get("HF_TOKEN")
    except Exception:
        pass

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is missing. Configure it in Streamlit Secrets."
        )

    # Download the three trained V3 models
    wet_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="wet_classifier.joblib",
        token=hf_token,
    )

    depth_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="depth_rf_log.joblib",
        token=hf_token,
    )

    velocity_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="velocity_rf_log.joblib",
        token=hf_token,
    )

    # Load models
    wet_data = joblib.load(wet_path)

    wet_model = wet_data["model"]
    wet_threshold = float(wet_data["threshold"])

    depth_model = joblib.load(depth_path)
    velocity_model = joblib.load(velocity_path)

    return wet_model, wet_threshold, depth_model, velocity_model


def predict_ml_v3(q: float):
    wet_model, wet_threshold, depth_model, velocity_model = load_ml_models()

    row_grid, col_grid = np.indices((ROWS, COLS))
    features = np.column_stack(
        [
            np.full(ROWS * COLS, q),
            row_grid.ravel(),
            col_grid.ravel(),
        ]
    )

    wet_probabilities = wet_model.predict_proba(features)[:, 1]
    wet_mask = wet_probabilities >= wet_threshold

    depth_log = depth_model.predict(features)
    velocity_log = velocity_model.predict(features)

    depth_all = np.maximum(np.expm1(depth_log), 0)
    velocity_all = np.maximum(np.expm1(velocity_log), 0)

    depth = np.zeros(ROWS * COLS, dtype=float)
    velocity = np.zeros(ROWS * COLS, dtype=float)

    depth[wet_mask] = np.maximum(depth_all[wet_mask], 0.1)
    velocity[wet_mask] = velocity_all[wet_mask]

    return (
        depth.reshape(ROWS, COLS),
        velocity.reshape(ROWS, COLS),
        "V3 Machine Learning prediction",
        None,
    )


# =========================================================
# PROJECTED COORDINATES
# =========================================================

@st.cache_data
def load_coordinates():
    if not COORD_FILE.exists():
        return None, None

    with np.load(COORD_FILE) as data:
        x = data["x"]
        y = data["y"]

    expected = (ROWS + 1, COLS + 1)
    if x.shape != expected or y.shape != expected:
        raise ValueError(
            f"Coordinate arrays must both have shape {expected}; "
            f"got {x.shape} and {y.shape}."
        )

    return x, y


# =========================================================
# SHARED RIVER AND DOMAIN OVERLAYS FOR ALL SPATIAL MAPS
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_osm_beki_river(south, west, north, east):
    """Retrieve mapped Beki/Bek(i) river ways. Never infer channel from FHI."""
    import requests
    # OSM river sections may have alternate spellings or unnamed segments.
    # Only return named Beki variants; unrelated rivers must not be relabelled.
    query = ('[out:json][timeout:25];('
             f'way["waterway"~"^(river|stream)$"]["name"~"Bek(i|ee|y)?",i]'
             f'({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
             f'way["waterway"~"^(river|stream)$"]["name:en"~"Bek(i|ee|y)?",i]'
             f'({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
             ');out geom;')
    endpoints = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://overpass.nchc.org.tw/api/interpreter',
    ]
    errors = []
    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, data={'data': query}, timeout=35,
                                     headers={'User-Agent': 'BekiFloodResearchPrototype/1.0'})
            response.raise_for_status()
            elements = response.json().get('elements', [])
            unique = {item['id']: item for item in elements if item.get('type') == 'way'}
            return [[(p['lat'], p['lon']) for p in item.get('geometry', [])]
                    for item in unique.values() if len(item.get('geometry', [])) > 1]
        except (requests.RequestException, ValueError) as exc:
            errors.append(f'{endpoint}: {str(exc)[:100]}')
    raise RuntimeError('All OpenStreetMap river geometry services failed: ' + '; '.join(errors))


def projected_domain_perimeter(x_corners, y_corners):
    px = np.concatenate([x_corners[0, :], x_corners[1:, -1],
                         x_corners[-1, -2::-1], x_corners[-2:0:-1, 0]])
    py = np.concatenate([y_corners[0, :], y_corners[1:, -1],
                         y_corners[-1, -2::-1], y_corners[-2:0:-1, 0]])
    return px, py


# Visual reference digitized from the previously displayed OSM Beki River line.
# This is approximate, for presentation only, not survey-grade geometry.
# Keeping the reference embedded avoids unstable live Overpass API calls.
BEKI_RIVER_REFERENCE_XY = [[296932.3, 2964302.8], [296533.9, 2963506.0], [296454.2, 2962709.2], [296613.5, 2962470.0], [297330.7, 2962549.8], [298047.8, 2962231.0], [298047.8, 2960318.8], [298207.2, 2959920.2], [298844.6, 2959442.2], [299004.0, 2958326.8], [298685.2, 2957928.2], [298764.9, 2956892.5], [298366.5, 2956255.0], [298605.6, 2955537.8], [298366.5, 2954263.0], [298525.9, 2953545.8], [299163.3, 2952908.2], [299243.0, 2952589.8], [299083.7, 2951872.5], [299243.0, 2951235.0], [299004.0, 2950836.8], [299163.3, 2950279.0], [298844.6, 2949641.5], [299482.1, 2948765.0], [299322.7, 2947888.5], [299641.4, 2947569.8], [299641.4, 2947171.2], [298844.6, 2946215.2], [298605.6, 2945498.0], [297888.4, 2944860.5], [297649.4, 2944143.5], [297808.8, 2943745.0], [298525.9, 2943107.5], [298525.9, 2942151.5], [298047.8, 2941753.0], [298207.2, 2939681.2], [297888.4, 2938486.0], [296374.5, 2937689.2], [295099.6, 2937769.0], [293824.7, 2936573.8], [293107.6, 2934183.2], [292310.8, 2933545.8], [292470.1, 2931952.2], [293027.9, 2930677.2], [292629.5, 2927729.0], [291753.0, 2927410.2], [291434.2, 2927091.8], [291434.2, 2926294.8], [291673.3, 2925498.0], [291513.9, 2923585.8], [291912.3, 2918964.2], [289043.8, 2913545.8], [289203.2, 2912510.0], [290239.0, 2911394.5], [290398.4, 2909243.0], [290000.0, 2908685.2], [289123.5, 2908366.5], [288406.4, 2907490.0], [287848.6, 2907330.8], [287450.2, 2906852.5], [286653.4, 2906454.2], [285617.5, 2906454.2], [284900.4, 2906852.5], [284342.6, 2906852.5], [284103.6, 2906613.5], [283864.5, 2905498.0], [283306.8, 2905259.0], [280597.6, 2905179.2], [280278.9, 2906055.8], [279880.5, 2905896.5], [278685.2, 2906055.8], [277011.9, 2905896.5], [276374.5, 2905179.2], [275498.0, 2904860.5], [274382.5, 2903665.2], [273904.4, 2903426.2], [272231.1, 2903107.5], [271274.9, 2903107.5], [270637.4, 2902788.8], [269521.9, 2901593.5], [268804.8, 2900239.0], [267689.2, 2898804.8]]


@st.cache_data(show_spinner=False)
def river_projected_lines(x_corners, y_corners):
    """Return a stable visual reference line in candidate EPSG:32646 coordinates."""
    points = np.asarray(BEKI_RIVER_REFERENCE_XY, dtype=float)
    return [(points[:, 0], points[:, 1])]


def clip_river_to_domain(line_x, line_y, boundary_x, boundary_y, spacing=25.0):
    """Mask projected river geometry outside the curvilinear grid perimeter.

    Densifying before masking avoids long segments cutting across the domain.
    NaN separators prevent Matplotlib from connecting disconnected portions.
    """
    from matplotlib.path import Path as MplPath

    polygon = MplPath(np.column_stack((boundary_x, boundary_y)), closed=True)
    xs, ys = [], []
    for x0, y0, x1, y1 in zip(line_x[:-1], line_y[:-1], line_x[1:], line_y[1:]):
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        length = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(1, int(np.ceil(length / spacing)))
        t = np.linspace(0, 1, steps + 1)
        seg_x = x0 + (x1 - x0) * t
        seg_y = y0 + (y1 - y0) * t
        inside = polygon.contains_points(np.column_stack((seg_x, seg_y)))
        xs.extend(np.where(inside, seg_x, np.nan).tolist())
        ys.extend(np.where(inside, seg_y, np.nan).tolist())
        xs.append(np.nan)
        ys.append(np.nan)
    return np.asarray(xs), np.asarray(ys)


def draw_spatial_reference(ax, x_corners, y_corners, river_lines=None,
                           show_boundary=True):
    """Clipped river overlay on analytical plots; GIS retains full OSM river."""
    if x_corners is None or y_corners is None:
        return
    bx, by = projected_domain_perimeter(x_corners, y_corners)
    if show_boundary:
        ax.plot(np.r_[bx, bx[0]], np.r_[by, by[0]], color='#172554',
                linewidth=1.5, alpha=0.95, zorder=8)
    for line_x, line_y in (river_lines or []):
        cx, cy = clip_river_to_domain(np.asarray(line_x), np.asarray(line_y), bx, by)
        ax.plot(cx, cy, color='white', linewidth=5.4,
                alpha=0.98, zorder=19, solid_capstyle='round')
        ax.plot(cx, cy, color='#087CFF', linewidth=3.5,
                alpha=1, zorder=20, solid_capstyle='round')


# =========================================================
# ADDITIONAL SPATIAL ANALYSIS (NO MODEL CHANGES)
# =========================================================

@st.cache_data(show_spinner=False)
def cell_areas_from_corners(x, y):
    """Quadrilateral cell areas in coordinate-units squared (shoelace formula)."""
    if x is None or y is None:
        return None
    x0, x1, x2, x3 = x[:-1, :-1], x[:-1, 1:], x[1:, 1:], x[1:, :-1]
    y0, y1, y2, y3 = y[:-1, :-1], y[:-1, 1:], y[1:, 1:], y[1:, :-1]
    area = 0.5 * np.abs(x0*y1+x1*y2+x2*y3+x3*y0 - y0*x1-y1*x2-y2*x3-y3*x0)
    return np.where(np.isfinite(area), area, 0.0)


def hotspot_table(classes, fhi, x, y, areas, min_cells=10):
    """Connected high/very-high hazard patches; 8-neighbour connectivity."""
    mask = (classes >= 4)
    components, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if not n:
        return pd.DataFrame(), components
    counts = np.bincount(components.ravel())
    ids = np.flatnonzero(counts >= min_cells)
    ids = ids[ids != 0]
    records = []
    for component_id in ids:
        rr, cc = np.where(components == component_id)
        row, col = int(np.round(rr.mean())), int(np.round(cc.mean()))
        record = {
            "Hotspot ID": int(component_id),
            "Cells": len(rr),
            "Very-high cells": int(np.count_nonzero(classes[rr, cc] == 5)),
            "Maximum FHI": round(float(np.nanmax(fhi[rr, cc])), 3),
            "Centre row": row,
            "Centre column": col,
        }
        if areas is not None:
            record["Area (km², if coordinates in metres)"] = round(float(areas[rr, cc].sum()/1e6), 4)
        if x is not None and y is not None:
            record["Projected X"] = round(float(np.mean((x[rr, cc]+x[rr+1, cc+1])/2)), 1)
            record["Projected Y"] = round(float(np.mean((y[rr, cc]+y[rr+1, cc+1])/2)), 1)
        records.append(record)
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("Cells", ascending=False).reset_index(drop=True)
    return result, components



def figure_png(fig):
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    return output.getvalue()


# =========================================================
# HEADER AND DATA CHECKS
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>BEKI RIVER FLOOD INTELLIGENCE</h1>
        <p>
            Spatial flood hazard assessment · Hydraulic scenario analysis
            · Machine learning-based flood estimation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ROOT.exists():
    st.error(
        f"Cannot find the ML folder: {ROOT}. "
        "Place app.py beside the ML folder."
    )
    st.stop()

folders = scenario_folders()

if not folders:
    st.error(
        "No valid hydraulic scenarios found. Check the ML folder names "
        "and average-grid files."
    )
    st.stop()


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("Simulation controls")
st.sidebar.caption("Configure the flood simulation")

mode = st.sidebar.selectbox(
    "Prediction method",
    [
        "Hydraulic interpolation",
        "Machine Learning (V3)",
        "Hydraulic vs ML comparison",
    ],
)

q_min = min(folders)
q_max = max(folders)
default_q = min(5350, int(q_max))

q = st.sidebar.number_input(
    "Discharge (cumecs)",
    min_value=int(q_min),
    max_value=int(q_max),
    value=int(default_q),
    step=50,
)

if mode == "Hydraulic interpolation":
    st.sidebar.info(
        "Uses available hydraulic scenarios and linear interpolation "
        "for intermediate discharges."
    )
elif mode == "Machine Learning (V3)":
    st.sidebar.info(
        "Uses the trained wet/dry classifier and V3 log-transformed "
        "depth and velocity regressors."
    )
else:
    st.sidebar.info(
        "Runs both methods at the selected discharge and compares "
        "their predicted grids."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Study area: Beki River, Assam")
st.sidebar.caption("Research prototype · simulated-scenario evaluation")


# Load coordinates before any map plotting.
try:
    x, y = load_coordinates()
except Exception as exc:
    st.error(f"Unable to load projected coordinate grids: {exc}")
    st.stop()


# Fetch once for all maps. If OSM is unavailable, keep all maps functional.
river_lines_projected = []
river_overlay_status = None
if x is not None and y is not None:
    try:
        river_lines_projected = river_projected_lines(x, y)
        if not river_lines_projected:
            river_overlay_status = 'No named Beki River geometry was found in OpenStreetMap. Red river-channel hazard cells remain red until a verified river geometry layer is supplied.'
    except Exception as exc:
        river_overlay_status = f'Beki River geometry could not be retrieved ({exc}); hazard values remain unchanged.'

# =========================================================
# COMPUTATION
# =========================================================

comparison_data = None

try:
    with st.spinner("Computing flood hazard prediction..."):

        if mode == "Hydraulic interpolation":
            h, v, method, bracket = predict_hydraulic(float(q), folders)

        elif mode == "Machine Learning (V3)":
            h, v, method, bracket = predict_ml_v3(float(q))

        else:
            h_hyd, v_hyd, hyd_method, bracket = predict_hydraulic(
                float(q), folders
            )
            h_ml, v_ml, ml_method, _ = predict_ml_v3(float(q))

            fhi_hyd, classes_hyd, wet_hyd = calculate_hazard(
                h_hyd, v_hyd
            )
            fhi_ml, classes_ml, wet_ml = calculate_hazard(
                h_ml, v_ml
            )

            comparison_data = {
                "h_hyd": h_hyd,
                "v_hyd": v_hyd,
                "classes_hyd": classes_hyd,
                "wet_hyd": wet_hyd,
                "fhi_hyd": fhi_hyd,
                "h_ml": h_ml,
                "v_ml": v_ml,
                "classes_ml": classes_ml,
                "wet_ml": wet_ml,
                "fhi_ml": fhi_ml,
            }

            # Main dashboard indicators show ML V3 in comparison mode.
            h, v = h_ml, v_ml
            method = "Hydraulic vs ML comparison (main indicators show ML V3)"

        fhi, classes, wet = calculate_hazard(h, v)

except Exception as exc:
    st.error(f"Unable to calculate prediction: {exc}")
    st.exception(exc)
    st.stop()


# =========================================================
# MAIN METRICS
# =========================================================

total = classes.size
counts = np.bincount(classes.ravel(), minlength=6)
high_count = int(counts[4] + counts[5])
wet_count = int(wet.sum())

if bracket is None:
    reference_text = "Trained V3 regression models"
elif bracket[0] == bracket[1]:
    reference_text = f"{bracket[0]} cumecs (exact scenario)"
else:
    reference_text = f"{bracket[0]} and {bracket[1]} cumecs"

st.caption(f"Method: {method} · Reference: {reference_text}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected discharge", f"{q:,.0f} cumecs")
c2.metric("Wet cells", f"{wet_count:,}")
c3.metric("High + very high cells", f"{high_count:,}")
c4.metric("Grid cells", f"{total:,}")


# =========================================================
# HYDRAULIC VS ML COMPARISON
# =========================================================

if comparison_data is not None:
    st.markdown("---")
    st.header("Hydraulic vs Machine Learning Comparison")
    st.caption(
        "Both methods are run at the selected discharge. "
        "Agreement indicates similarity, not independent accuracy."
    )

    d = comparison_data

    wet_agreement = np.mean(d["wet_hyd"] == d["wet_ml"]) * 100
    hazard_agreement = (
        np.mean(d["classes_hyd"] == d["classes_ml"]) * 100
    )

    wet_difference = int(d["wet_ml"].sum() - d["wet_hyd"].sum())

    hyd_mean_depth = (
        float(np.mean(d["h_hyd"][d["wet_hyd"]]))
        if d["wet_hyd"].any() else np.nan
    )
    ml_mean_depth = (
        float(np.mean(d["h_ml"][d["wet_ml"]]))
        if d["wet_ml"].any() else np.nan
    )
    hyd_mean_velocity = (
        float(np.mean(d["v_hyd"][d["wet_hyd"]]))
        if d["wet_hyd"].any() else np.nan
    )
    ml_mean_velocity = (
        float(np.mean(d["v_ml"][d["wet_ml"]]))
        if d["wet_ml"].any() else np.nan
    )

    depth_difference = ml_mean_depth - hyd_mean_depth
    velocity_difference = ml_mean_velocity - hyd_mean_velocity

    a, b, c, e = st.columns(4)
    a.metric("Wet-cell agreement", f"{wet_agreement:.2f}%")
    b.metric("Hazard-class agreement", f"{hazard_agreement:.2f}%")
    c.metric("Wet-cell difference (ML − hydraulic)", f"{wet_difference:+,}")
    e.metric(
        "Mean depth difference",
        f"{depth_difference:+.3f} m"
        if np.isfinite(depth_difference) else "N/A",
    )

    st.metric(
        "Mean velocity difference",
        f"{velocity_difference:+.3f} m/s"
        if np.isfinite(velocity_difference) else "N/A",
    )

    st.subheader("Side-by-side hazard maps")

    fig_compare, axes_compare = plt.subplots(1, 2, figsize=(14, 6))

    if x is not None and y is not None:
        axes_compare[0].pcolormesh(
            x, y, d["classes_hyd"],
            cmap=HAZARD_CMAP, norm=HAZARD_NORM, shading="flat"
        )
        axes_compare[1].pcolormesh(
            x, y, d["classes_ml"],
            cmap=HAZARD_CMAP, norm=HAZARD_NORM, shading="flat"
        )

        for ax in axes_compare:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Projected X")
            ax.set_ylabel("Projected Y")
    else:
        axes_compare[0].imshow(
            d["classes_hyd"], cmap=HAZARD_CMAP,
            norm=HAZARD_NORM, origin="upper"
        )
        axes_compare[1].imshow(
            d["classes_ml"], cmap=HAZARD_CMAP,
            norm=HAZARD_NORM, origin="upper"
        )

        for ax in axes_compare:
            ax.set_xlabel("Grid column")
            ax.set_ylabel("Grid row")

    for spatial_ax in axes_compare:
        draw_spatial_reference(spatial_ax, x, y, river_lines_projected)

    axes_compare[0].set_title("Hydraulic interpolation")
    axes_compare[1].set_title("Machine Learning V3")

    fig_compare.tight_layout()
    st.pyplot(fig_compare, use_container_width=True)
    plt.close(fig_compare)

    st.subheader("Hazard class distribution comparison")

    hyd_counts = np.bincount(d["classes_hyd"].ravel(), minlength=6)
    ml_counts = np.bincount(d["classes_ml"].ravel(), minlength=6)

    comparison_table = pd.DataFrame(
        {
            "Hazard class": LABELS,
            "Hydraulic cells": hyd_counts,
            "ML cells": ml_counts,
            "Difference (ML − hydraulic)": ml_counts - hyd_counts,
        }
    )

    st.dataframe(
        comparison_table,
        hide_index=True,
        use_container_width=True,
    )

    chart_data = comparison_table.set_index("Hazard class")[
        ["Hydraulic cells", "ML cells"]
    ]
    st.bar_chart(chart_data)

    st.info(
        "Agreement measures similarity between the two methods. "
        "It does not establish which method is more accurate. "
        "The separate validation section compares ML predictions "
        "with held-out simulated hydraulic scenarios."
    )


# =========================================================
# SPATIAL DISAGREEMENT BETWEEN HYDRAULIC AND ML
# =========================================================
if comparison_data is not None:
    d = comparison_data
    disagreement = d["classes_ml"].astype(np.int16) - d["classes_hyd"].astype(np.int16)
    changed = disagreement != 0
    either_wet = d["wet_hyd"] | d["wet_ml"]
    st.subheader("Spatial disagreement explorer")
    st.caption("Difference = ML hazard class minus hydraulic hazard class. This is a comparison, not an accuracy map.")
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Different-class cells", f"{changed.sum():,}")
    dc2.metric("Agreement where either method predicts wet", f"{100*np.mean(~changed[either_wet]):.1f}%" if either_wet.any() else "N/A")
    dc3.metric("ML predicts higher class", f"{np.count_nonzero(disagreement > 0):,}")
    fig_diff, ax_diff = plt.subplots(figsize=(8, 8))
    if x is not None and y is not None:
        diff_mesh = ax_diff.pcolormesh(x, y, disagreement, shading="flat", cmap="RdBu_r", vmin=-5, vmax=5)
        ax_diff.set_aspect("equal", adjustable="box")
        ax_diff.set_xlabel("Projected X")
        ax_diff.set_ylabel("Projected Y")
    else:
        diff_mesh = ax_diff.imshow(disagreement, cmap="RdBu_r", vmin=-5, vmax=5)
        ax_diff.set_xlabel("Grid column")
        ax_diff.set_ylabel("Grid row")
    draw_spatial_reference(ax_diff, x, y, river_lines_projected)
    fig_diff.colorbar(diff_mesh, ax=ax_diff, label="ML class − hydraulic class")
    ax_diff.set_title(f"Hazard-class disagreement at Q={q:,} cumecs")
    fig_diff.tight_layout()
    st.pyplot(fig_diff, use_container_width=True)
    st.download_button("Download disagreement map (PNG)", figure_png(fig_diff),
                       file_name=f"hazard_disagreement_Q{q}.png", mime="image/png")
    plt.close(fig_diff)


# =========================================================
# HAZARD MAP AND DISTRIBUTION
# =========================================================

left, right = st.columns([1.8, 1])

with left:
    st.subheader("Flood hazard map")
    st.caption("Blue = approximate Beki River reference line; dark outline = computational grid boundary. Hazard values are unchanged. Candidate CRS EPSG:32646 remains unverified.")
    if river_overlay_status:
        st.caption(river_overlay_status)
    fig, ax = plt.subplots(figsize=(10, 6))

    if x is not None and y is not None:
        mesh = ax.pcolormesh(
            x, y, classes,
            cmap=HAZARD_CMAP,
            norm=HAZARD_NORM,
            shading="flat",
        )
        ax.set_xlabel("Projected X")
        ax.set_ylabel("Projected Y")
        ax.set_aspect("equal", adjustable="box")

        st.caption(
            "Map uses supplied projected coordinate grids. "
            "The coordinate reference system has not been independently confirmed."
        )
    else:
        mesh = ax.imshow(
            classes,
            cmap=HAZARD_CMAP,
            norm=HAZARD_NORM,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

        st.caption(
            "Coordinate file unavailable; displaying grid-index coordinates."
        )

    draw_spatial_reference(ax, x, y, river_lines_projected)
    ax.set_title(f"Beki River Flood Hazard Map - Q={q:,.0f} cumecs")

    fig.colorbar(
        mesh,
        ax=ax,
        ticks=range(6),
        label="Flood hazard class",
    )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.download_button("Download hazard map (PNG)", figure_png(fig),
                       file_name=f"hazard_map_Q{q}.png", mime="image/png")
    plt.close(fig)


with right:
    st.subheader("Hazard distribution")

    distribution = pd.DataFrame(
        {
            "Class": LABELS,
            "Cells": counts,
            "Percentage": (counts / total * 100).round(2),
        }
    )

    st.dataframe(
        distribution,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Hazard legend")

    for index, label in enumerate(LABELS):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:7px 0;">'
            f'<span style="display:inline-block;width:16px;height:16px;'
            f'border-radius:4px;background:{COLORS[index]};"></span>'
            f'{label}</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# PROVISIONAL GEOGRAPHIC GIS PREVIEW
# =========================================================
st.markdown("---")
st.header("GIS preview — geographic flood overlay")
st.warning(
    "Experimental geographic placement: assumes WGS 84 / UTM zone 46N "
    "(EPSG:32646). The source CRS has not been verified. "
    "Do not use this map for official geographic or planning decisions."
)
if x is None or y is None:
    st.info("The geographic preview requires deployment/coordinate_grids.npz.")
else:
    try:
        import folium
        from pyproj import Transformer
        import base64

        @st.cache_data(show_spinner="Preparing geographic overlay...")
        def gis_overlay_png(x_corners, y_corners, class_grid, epsg):
            transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            longitude, latitude = transformer.transform(x_corners, y_corners)
            valid = np.isfinite(longitude) & np.isfinite(latitude)
            if not np.all(valid):
                raise ValueError("Coordinate transformation produced nonfinite values.")
            west, east = float(longitude.min()), float(longitude.max())
            south, north = float(latitude.min()), float(latitude.max())
            # Draw true curvilinear cell geometry in geographic coordinates.
            # Dry cells are transparent so roads and place labels remain visible.
            cmap = ListedColormap(COLORS)
            cmap.set_bad((0, 0, 0, 0))
            wet_classes = np.ma.masked_where(class_grid == 0, class_grid)
            fig_g, ax_g = plt.subplots(figsize=(6, 9), dpi=140)
            fig_g.patch.set_alpha(0)
            ax_g.patch.set_alpha(0)
            ax_g.set_position([0, 0, 1, 1])
            ax_g.pcolormesh(longitude, latitude, wet_classes, cmap=cmap,
                            norm=HAZARD_NORM, shading="flat", rasterized=True,
                            antialiased=False)
            ax_g.set_xlim(west, east)
            ax_g.set_ylim(south, north)
            ax_g.axis("off")
            fig_g.canvas.draw()
            img = Image.frombuffer("RGBA", fig_g.canvas.get_width_height(),
                                   fig_g.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).copy()
            plt.close(fig_g)
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue(), (south, west, north, east)

        opacity = st.slider("Flood overlay opacity", 0.2, 1.0, 0.65, 0.05,
                            key="gis_overlay_opacity")
        show_river = st.checkbox("Show blue Beki River reference line", value=True,
                                 key="gis_show_river")
        png_bytes, (south, west, north, east) = gis_overlay_png(x, y, classes, "EPSG:32646")
        centre = [(south + north) / 2, (west + east) / 2]
        gis_map = folium.Map(location=centre, zoom_start=10, tiles="OpenStreetMap",
                             control_scale=True, prefer_canvas=True)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        folium.raster_layers.ImageOverlay(
            image="data:image/png;base64," + encoded,
            bounds=[[south, west], [north, east]],
            opacity=opacity, name="Flood hazard (wet cells)",
            interactive=False, cross_origin=False,
        ).add_to(gis_map)

        # Trace the genuine computational-domain perimeter from its corner grids.
        transformer = Transformer.from_crs("EPSG:32646", "EPSG:4326", always_xy=True)
        perimeter_x = np.concatenate([x[0, :], x[1:, -1], x[-1, -2::-1], x[-2:0:-1, 0]])
        perimeter_y = np.concatenate([y[0, :], y[1:, -1], y[-1, -2::-1], y[-2:0:-1, 0]])
        perimeter_lon, perimeter_lat = transformer.transform(perimeter_x, perimeter_y)
        perimeter = list(zip(perimeter_lat.tolist(), perimeter_lon.tolist()))
        boundary_layer = folium.FeatureGroup(name="Study-area boundary", show=True)
        folium.PolyLine(perimeter + [perimeter[0]], color="#172554", weight=3.5,
                        opacity=0.95, tooltip="Computational grid boundary").add_to(boundary_layer)
        boundary_layer.add_to(gis_map)

        river_status = None
        if show_river:
            try:
                river_lines = []
                if river_lines_projected:
                    to_geo = Transformer.from_crs('EPSG:32646', 'EPSG:4326', always_xy=True)
                    for line_x, line_y in river_lines_projected:
                        line_lon, line_lat = to_geo.transform(line_x, line_y)
                        river_lines.append(list(zip(line_lat, line_lon)))
                if river_lines:
                    river_layer = folium.FeatureGroup(name="Beki River (approximate reference)", show=True)
                    for line in river_lines:
                        folium.PolyLine(line, color="#FFFFFF", weight=7,
                                        opacity=0.9).add_to(river_layer)
                        folium.PolyLine(line, color="#0066FF", weight=4.5,
                                        opacity=1, tooltip="Beki River — approximate reference").add_to(river_layer)
                    river_layer.add_to(gis_map)
                else:
                    river_status = "No named Beki River line was returned by OpenStreetMap for this extent. The flood hazard colours remain unchanged."
            except Exception:
                river_status = "OpenStreetMap river geometry is temporarily unavailable. The flood overlay and boundary still work."

        folium.LayerControl(collapsed=False).add_to(gis_map)
        gis_map.fit_bounds([[south, west], [north, east]])
        components.html(gis_map.get_root().render(), height=610, scrolling=False)
        if river_status:
            st.info(river_status)
        st.caption(
            "Dark blue outline = computational grid boundary; bright blue line (when available) "
            "= approximate Beki River reference line traced from an earlier OSM display. Red/orange areas "
            "retain their original FHI hazard classes, including within the river channel."
        )
        st.caption(
            "Approximate river geometry; candidate CRS: EPSG:32646 (unverified). Geographic alignment still needs "
            "independent confirmation before planning or engineering use."
        )
    except ImportError:
        st.info("To enable the GIS preview, add `folium` and `pyproj` to requirements.txt, "
                "then redeploy. The existing dashboard will still work without them.")
    except Exception as exc:
        st.warning(f"GIS preview could not be displayed: {exc}")


# =========================================================
# HYDRAULIC INDICATORS
# =========================================================

st.markdown("---")
st.subheader("Hydraulic indicators")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Mean wet-cell depth",
    f"{np.mean(h[wet]):.3f} m" if wet.any() else "N/A",
)
m2.metric(
    "Maximum depth",
    f"{np.nanmax(h):.3f} m" if np.isfinite(h).any() else "N/A",
)
m3.metric(
    "Mean wet-cell velocity",
    f"{np.mean(v[wet]):.3f} m/s" if wet.any() else "N/A",
)
m4.metric(
    "Maximum FHI",
    f"{np.nanmax(fhi):.3f}" if np.isfinite(fhi).any() else "N/A",
)


# =========================================================
# DEPTH AND VELOCITY MAPS
# =========================================================

st.markdown("---")
st.subheader("Depth and velocity distribution")

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

if x is not None and y is not None:
    depth_plot = axes[0].pcolormesh(
        x, y, h, shading="flat", cmap="Blues"
    )
    velocity_plot = axes[1].pcolormesh(
        x, y, v, shading="flat", cmap="YlOrRd"
    )

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Projected X")
        ax.set_ylabel("Projected Y")
else:
    depth_plot = axes[0].imshow(
        h, cmap="Blues", interpolation="nearest", origin="upper"
    )
    velocity_plot = axes[1].imshow(
        v, cmap="YlOrRd", interpolation="nearest", origin="upper"
    )

    for ax in axes:
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

for spatial_ax in axes:
    draw_spatial_reference(spatial_ax, x, y, river_lines_projected)

axes[0].set_title("Flood depth (m)")
axes[1].set_title("Flow velocity (m/s)")

fig2.colorbar(depth_plot, ax=axes[0], fraction=0.046, pad=0.04)
fig2.colorbar(velocity_plot, ax=axes[1], fraction=0.046, pad=0.04)

fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)


# =========================================================
# SPATIAL HAZARD AREAS AND HOTSPOTS
# =========================================================
st.markdown("---")
st.header("Hazard areas and high-risk hotspots")
areas = cell_areas_from_corners(x, y) if x is not None and y is not None else None
if areas is not None:
    st.caption("Provisional area estimates: assumes the projected coordinate units are metres. CRS and units have not yet been verified.")
    class_areas = np.array([float(areas[classes == i].sum() / 1e6) for i in range(6)])
    high_area = float(class_areas[4] + class_areas[5])
    wet_area = float(class_areas[1:].sum())
    area_a, area_b, area_c = st.columns(3)
    area_a.metric("Flooded area (provisional)", f"{wet_area:,.1f} km²")
    area_b.metric("High + very-high area", f"{high_area:,.1f} km²")
    area_c.metric("High-risk share of wet area", f"{100 * high_area / wet_area:.1f}%" if wet_area else "N/A")

    st.subheader("Where is the flood hazard concentrated?")
    fig_area, ax_area = plt.subplots(figsize=(10, 4.2))
    fig_area.patch.set_facecolor("#FFFFFF")
    ax_area.set_facecolor("#FFFFFF")
    ypos = np.arange(6)
    bars = ax_area.barh(ypos, class_areas, color=COLORS, height=0.7)
    ax_area.set_yticks(ypos, LABELS)
    ax_area.invert_yaxis()
    ax_area.set_xlabel("Estimated area (km²; provisional)")
    ax_area.set_xlim(0, max(class_areas.max() * 1.23, 1))
    ax_area.spines[['top', 'right', 'left']].set_visible(False)
    ax_area.grid(axis="x", alpha=0.18)
    ax_area.set_axisbelow(True)
    for bar, value in zip(bars, class_areas):
        ax_area.text(bar.get_width() + max(class_areas.max() * 0.018, 0.05),
                     bar.get_y() + bar.get_height() / 2,
                     f"{value:,.1f} km²", va="center", fontsize=10, color="#172033", fontweight="bold")
    fig_area.tight_layout()
    st.pyplot(fig_area, use_container_width=True)
    plt.close(fig_area)

    with st.expander("View exact hazard-area data"):
        area_summary = pd.DataFrame({
            "Hazard class": LABELS,
            "Cells": counts,
            "Area (km²; provisional)": class_areas,
        })
        st.dataframe(area_summary.style.format({"Cells": "{:,.0f}",
                       "Area (km²; provisional)": "{:,.2f}"}),
                       hide_index=True, use_container_width=True)
else:
    st.info("Coordinate grids are unavailable; physical area estimates cannot be calculated.")

st.subheader("High-risk hotspot explorer")
st.caption("A hotspot is a connected group of orange/red (High or Very high) cells. Increase the minimum size to show only larger clusters.")
minimum_hotspot_size = st.slider("Minimum connected cells per hotspot", min_value=1, max_value=200,
                                 value=20, step=1)
hotspots, hotspot_labels = hotspot_table(classes, fhi, x, y, areas, minimum_hotspot_size)
if hotspots.empty:
    st.info("No connected high-hazard regions meet the selected minimum size.")
else:
    st.metric("High-risk regions found", len(hotspots))
    st.subheader("Find and inspect high-risk areas")
    st.caption(
        "Numbers identify the largest connected high/very-high hazard regions. "
        "Hotspot 1 is the largest. Select a hotspot or hover over its number "
        "to inspect its location and severity. The map labels up to 10 hotspots."
    )
    # Select a region first so its full extent can be outlined on the map.
    if x is not None and y is not None:
        visible_hotspots = hotspots.head(10).reset_index(drop=True)
        selected_rank = st.selectbox(
            "Select a hotspot to highlight its complete boundary",
            range(1, len(visible_hotspots) + 1),
            format_func=lambda n: f"Hotspot {n} — {int(visible_hotspots.iloc[n-1]['Cells']):,} cells",
        )
        selected = visible_hotspots.iloc[selected_rank - 1]
        selected_id = int(selected["Hotspot ID"])
        selected_mask = hotspot_labels == selected_id
        selected_rr, selected_cc = np.where(selected_mask)
        # Maximum FHI is a real cell in the selected connected component.
        maximum_index = int(np.nanargmax(fhi[selected_rr, selected_cc]))
        max_r, max_c = int(selected_rr[maximum_index]), int(selected_cc[maximum_index])

        def cell_centre(r, c):
            return (
                float((x[r,c] + x[r+1,c] + x[r,c+1] + x[r+1,c+1]) / 4),
                float((y[r,c] + y[r+1,c] + y[r,c+1] + y[r+1,c+1]) / 4),
            )

        max_x, max_y = cell_centre(max_r, max_c)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        # Draw the full grid and outline the selected component, not just its marker.
        fig_bg, ax_bg = plt.subplots(figsize=(5, 7), dpi=105)
        fig_bg.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax_bg.set_position([0, 0, 1, 1])
        ax_bg.pcolormesh(x, y, classes, cmap=HAZARD_CMAP,
                         norm=HAZARD_NORM, shading="flat", rasterized=True)
        # Contour uses the projected centres of actual curvilinear cells.
        xc = (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:]) / 4
        yc = (y[:-1,:-1] + y[1:,:-1] + y[:-1,1:] + y[1:,1:]) / 4
        ax_bg.contour(xc, yc, selected_mask.astype(float), levels=[0.5],
                      colors=["#FFFFFF"], linewidths=2.0)
        ax_bg.contour(xc, yc, selected_mask.astype(float), levels=[0.5],
                      colors=["#111827"], linewidths=0.7)
        draw_spatial_reference(ax_bg, x, y, river_lines_projected)
        ax_bg.set_xlim(xmin, xmax)
        ax_bg.set_ylim(ymin, ymax)
        ax_bg.axis("off")
        fig_bg.canvas.draw()
        bg_image = Image.frombuffer(
            "RGBA", fig_bg.canvas.get_width_height(),
            fig_bg.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).convert("RGB")
        plt.close(fig_bg)

        marker_x, marker_y, marker_labels, marker_details = [], [], [], []
        for rank, region in enumerate(visible_hotspots.itertuples(index=False), start=1):
            region_id = int(region[0])
            rr, cc = np.where(hotspot_labels == region_id)
            # Pick a cell that is actually part of the component, near its centroid.
            distance = (rr - rr.mean())**2 + (cc - cc.mean())**2
            k = int(np.argmin(distance))
            r, c = int(rr[k]), int(cc[k])
            px, py = cell_centre(r, c)
            marker_x.append(px)
            marker_y.append(py)
            marker_labels.append(str(rank))
            area_label = f"{float(areas[rr,cc].sum()/1e6):.2f}" if areas is not None else "N/A"
            marker_details.append(
                f"<b>Hotspot {rank}</b><br>High-risk cells: {len(rr):,}"
                f"<br>Very-high cells: {np.count_nonzero(classes[rr,cc] == 5):,}"
                f"<br>Maximum FHI: {np.nanmax(fhi[rr,cc]):.3f}"
                f"<br>Area (provisional): {area_label} km²"
                f"<br>Projected X: {px:,.1f}<br>Projected Y: {py:,.1f}"
                f"<br><i>Latitude/longitude requires verified CRS</i>"
            )

        fig_interactive = go.Figure()
        fig_interactive.add_layout_image(dict(
            source=bg_image, xref="x", yref="y", x=xmin, y=ymax,
            sizex=xmax-xmin, sizey=ymax-ymin, sizing="stretch", layer="below"
        ))
        fig_interactive.add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers+text",
            text=marker_labels, textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            marker=dict(size=[25 if i == selected_rank-1 else 17 for i in range(len(marker_x))],
                        color=["#0F766E" if i == selected_rank-1 else "#15263D" for i in range(len(marker_x))],
                        line=dict(color="white", width=1.2)),
            customdata=marker_details,
            hovertemplate="%{customdata}<extra></extra>",
            name="Hotspot centres", showlegend=False,
        ))
        fig_interactive.add_trace(go.Scatter(
            x=[max_x], y=[max_y], mode="markers",
            marker=dict(size=17, symbol="star", color="#FDE047",
                        line=dict(color="#111827", width=2)),
            hovertemplate=(f"<b>Highest FHI in Hotspot {selected_rank}</b>"
                           f"<br>FHI: {fhi[max_r,max_c]:.3f}"
                           f"<br>Row {max_r}, column {max_c}"
                           f"<br>Projected X: {max_x:,.1f}"
                           f"<br>Projected Y: {max_y:,.1f}<extra></extra>"),
            name="Maximum FHI", showlegend=False,
        ))
        fig_interactive.update_layout(
            width=480, height=590, margin=dict(l=8, r=8, t=32, b=8),
            title=dict(text=f"Selected hotspot boundary | Q = {q:g} cumecs", font=dict(size=15)),
            paper_bgcolor="white", plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white", font=dict(size=13, color="#182536")),
            dragmode="pan", uirevision=f"Q{q}-hotspot-{selected_rank}",
        )
        fig_interactive.update_xaxes(range=[xmin, xmax], visible=False, constrain="domain")
        fig_interactive.update_yaxes(range=[ymin, ymax], visible=False,
                                     scaleanchor="x", scaleratio=1)
        map_col, info_col = st.columns([1.05, 1])
        with map_col:
            st.plotly_chart(fig_interactive, use_container_width=False,
                            config={"displaylogo": False, "scrollZoom": True})
            st.caption("Black-and-white outline = selected hotspot; yellow star = highest FHI.")
        with info_col:
            st.metric("Connected high-risk cells", f"{int(selected['Cells']):,}")
            st.metric("Very-high hazard cells", f"{int(selected['Very-high cells']):,}")
            st.metric("Maximum FHI", f"{float(selected['Maximum FHI']):.3f}")
            if areas is not None:
                st.metric("Area (provisional)", f"{float(areas[selected_rr,selected_cc].sum()/1e6):,.2f} km²")
            st.markdown(f"**Highest-FHI cell:** row {max_r}, column {max_c}")
            st.markdown(f"**Projected position:** X = {max_x:,.1f}, Y = {max_y:,.1f}")
            st.info("Coordinates are projected X/Y. Latitude and longitude require a verified CRS.")
            st.markdown("**Check the extreme FHI value**")
            st.write(
                f"At the highest-FHI cell: depth = {float(h[max_r, max_c]):.3f} m, "
                f"velocity = {float(v[max_r, max_c]):.3f} m/s; "
                f"FHI = {float(fhi[max_r, max_c]):.3f}. "
                "Review unusually large values against the source hydraulic outputs."
            )
            st.caption("The hotspot outline includes both orange (High) and red (Very high) cells. Region numbering is by total connected high-risk cell count, not red area alone.")

        # Export the same selected boundary and maximum-FHI marker.
        fig_export, ax_export = plt.subplots(figsize=(5, 7))
        ax_export.pcolormesh(x, y, classes, cmap=HAZARD_CMAP,
                             norm=HAZARD_NORM, shading="flat", rasterized=True)
        ax_export.contour(xc, yc, selected_mask.astype(float), levels=[0.5],
                          colors=["#111827"], linewidths=1.0)
        ax_export.scatter([max_x], [max_y], marker="*", s=140,
                          c="#FDE047", edgecolors="black", zorder=5)
        draw_spatial_reference(ax_export, x, y, river_lines_projected)
        ax_export.set_aspect("equal")
        ax_export.set_xlabel("Projected X (CRS unverified)")
        ax_export.set_ylabel("Projected Y (CRS unverified)")
        ax_export.set_title(f"Hotspot {selected_rank} extent and maximum FHI | Q={q:g}")
        fig_export.tight_layout()
        st.download_button("Download selected hotspot map (PNG)",
                           data=figure_png(fig_export),
                           file_name=f"beki_hotspot_{selected_rank}_Q{q:g}.png",
                           mime="image/png")
        plt.close(fig_export)
    else:
        st.warning("Interactive projected-coordinate hotspot map requires coordinate_grids.npz.")
    st.caption("Hotspot numbers represent connected high-risk grid regions, not named villages.")
    with st.expander("All hotspot details and coordinates"):
        ranked_hotspots = hotspots.copy()
        ranked_hotspots.insert(0, "Hotspot number (by size)", np.arange(1, len(ranked_hotspots) + 1))
        st.dataframe(ranked_hotspots, hide_index=True, use_container_width=True)
        st.download_button("Download hotspot details (CSV)",
                           ranked_hotspots.to_csv(index=False),
                           file_name=f"beki_hotspots_Q{q}.csv", mime="text/csv")
    st.caption("Hotspots are connected High/Very-high hazard cells (8-neighbour connectivity), not villages. The size filter changes which regions are listed, not the flood predictions.")

# =========================================================
# DISCHARGE SENSITIVITY (HYDRAULIC SCENARIOS)
# =========================================================
st.markdown("---")
st.header("Discharge sensitivity analysis")
st.caption("Calculated from available hydraulic scenarios only; not ML predictions or observed flood records.")
if st.button("Calculate hydraulic scenario trends", help="Reads the available scenario Excel files; this may take some time."):
    trend_rows = []
    progress = st.progress(0)
    for index, discharge in enumerate(sorted(folders)):
        try:
            scenario_h, scenario_v = load_scenario(discharge, folders)
            _, scenario_classes, scenario_wet = calculate_hazard(scenario_h, scenario_v)
            entry = {
                "Discharge (cumecs)": discharge,
                "Wet cells": int(scenario_wet.sum()),
                "High + very-high cells": int(np.count_nonzero(scenario_classes >= 4)),
                "Mean wet depth (m)": float(np.mean(scenario_h[scenario_wet])) if scenario_wet.any() else 0.0,
            }
            if areas is not None:
                entry["High + very-high area (km²; provisional)"] = float(areas[scenario_classes >= 4].sum()/1e6)
            trend_rows.append(entry)
        except Exception as exc:
            st.warning(f"Skipped Q={discharge}: {exc}")
        progress.progress((index+1)/len(folders))
    if trend_rows:
        st.session_state["beki_hydraulic_trends"] = pd.DataFrame(trend_rows)
if "beki_hydraulic_trends" in st.session_state:
    trends = st.session_state["beki_hydraulic_trends"]
    st.line_chart(trends.set_index("Discharge (cumecs)")[["Wet cells", "High + very-high cells"]])
    st.line_chart(trends.set_index("Discharge (cumecs)")[["Mean wet depth (m)"]])
    if "High + very-high area (km²; provisional)" in trends:
        st.line_chart(trends.set_index("Discharge (cumecs)")[["High + very-high area (km²; provisional)"]])
    st.download_button("Download sensitivity statistics (CSV)", trends.to_csv(index=False),
                       file_name="hydraulic_discharge_sensitivity.csv", mime="text/csv")


# =========================================================
# MODEL PERFORMANCE DASHBOARD
# =========================================================

st.markdown("---")
st.header("Machine Learning Model Performance")

validation_file = OUTPUT_DIR / "multi_discharge_validation_v3.csv"

if validation_file.exists():
    validation = pd.read_csv(validation_file)

    required_columns = [
        "Discharge",
        "Depth_MAE",
        "Velocity_MAE",
        "FHI_MAE",
        "Hazard_Accuracy",
        "Hazard_Macro_F1",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in validation.columns
    ]

    if missing_columns:
        st.warning(
            "Validation CSV is missing expected columns: "
            + ", ".join(missing_columns)
        )
    else:
        validation = validation.sort_values("Discharge")

        st.caption(
            "V3 evaluation against eight held-out simulated hydraulic "
            "scenarios. These are not independent field-observation "
            "validation results."
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Mean hazard accuracy",
            f"{validation['Hazard_Accuracy'].mean() * 100:.2f}%",
        )
        p2.metric(
            "Mean hazard Macro F1",
            f"{validation['Hazard_Macro_F1'].mean():.3f}",
        )
        p3.metric(
            "Mean depth MAE",
            f"{validation['Depth_MAE'].mean():.3f} m",
        )

        st.subheader("Regression errors across discharge")
        st.line_chart(
            validation.set_index("Discharge")[
                ["Depth_MAE", "Velocity_MAE", "FHI_MAE"]
            ]
        )

        st.subheader("Hazard classification performance")
        st.line_chart(
            validation.set_index("Discharge")[
                ["Hazard_Accuracy", "Hazard_Macro_F1"]
            ]
        )

        st.subheader("Detailed validation results")
        st.dataframe(
            validation.round(4),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download model validation results",
            data=validation.to_csv(index=False),
            file_name="v3_model_validation.csv",
            mime="text/csv",
        )
else:
    st.warning(
        f"V3 validation CSV not found: {validation_file}. "
        "Run evaluate_v3_multiple_discharges.py first."
    )


# =========================================================
# EXPORT
# =========================================================

st.markdown("---")
st.subheader("Export prediction results")

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.download_button(
        "Download hazard classes (CSV)",
        data=pd.DataFrame(classes).to_csv(index=False, header=False),
        file_name=f"hazard_classes_Q{q:g}.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download depth grid (CSV)",
        data=pd.DataFrame(h).to_csv(index=False, header=False),
        file_name=f"depth_Q{q:g}.csv",
        mime="text/csv",
    )

with export_col2:
    st.download_button(
        "Download FHI grid (CSV)",
        data=pd.DataFrame(fhi).to_csv(index=False, header=False),
        file_name=f"fhi_Q{q:g}.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download velocity grid (CSV)",
        data=pd.DataFrame(v).to_csv(index=False, header=False),
        file_name=f"velocity_Q{q:g}.csv",
        mime="text/csv",
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")
st.caption(
    "Research prototype. Hydraulic predictions use scenario-based "
    "interpolation. ML predictions use trained Random Forest models. "
    "The FHI formula and hazard thresholds are applied consistently "
    "in both modes. ML results are evaluated against held-out simulated "
    "hydraulic scenarios and have not been independently validated "
    "against observed flood measurements. The coordinate reference "
    "system also requires confirmation."
)
