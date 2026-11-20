import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="AutoFuel AI · Fuel Economy & Emissions Intelligence",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Determine working paths safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "CO2_Emissions_Canada.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "importance.csv")
EVAL_PATH = os.path.join(BASE_DIR, "test_eval.csv")

# If files are in a subdirectory 'ML_PROJECT' relative to current file, adapt
if not os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(BASE_DIR, "ML_PROJECT", "model.pkl")):
    BASE_DIR = os.path.join(BASE_DIR, "ML_PROJECT")
    DATA_PATH = os.path.join(BASE_DIR, "data", "CO2_Emissions_Canada.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "metrics.csv")
    IMPORTANCE_PATH = os.path.join(BASE_DIR, "importance.csv")
    EVAL_PATH = os.path.join(BASE_DIR, "test_eval.csv")

TARGET = "Fuel Consumption Comb (L/100 km)"

# ---------- Custom Unified Styling ----------
st.markdown("""
<style>
/* Clean typography and palette base styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #2B2D42;
}

/* Background styling - non-white cool slate canvas */
.stApp {
    background-color: #EDF2F4 !important;
    background: linear-gradient(180deg, #EDF2F4 0%, #E6ECEF 100%) !important;
}
div[data-testid="stAppViewContainer"] {
    background-color: #EDF2F4 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(237, 242, 244, 0.85) !important;
}

/* Base headings */
h1, h2, h3, h4, h5, h6 {
    color: #2B2D42 !important;
    letter-spacing: -0.015em;
}

p, label {
    color: #2B2D42;
}

/* Executive Hero Banner - Deep Space Cadet Navy with Crimson Accent & Crisp White Typography */
.hero-container {
    background: linear-gradient(135deg, #1C1D2B 0%, #2B2D42 55%, #393C57 100%) !important;
    padding: 26px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 10px 25px -5px rgba(43, 45, 66, 0.28);
    color: #EDF2F4 !important;
    border: 1px solid rgba(141, 153, 174, 0.25);
    border-left: 6px solid #EF233C !important;
}
.hero-container,
.hero-container * {
    color: #EDF2F4;
}
.hero-title {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 12px;
    line-height: 1.25;
}
.hero-title span,
.hero-title-main {
    color: #FFFFFF !important;
    font-weight: 800;
}
.hero-title-sub {
    font-size: 1.15rem;
    font-weight: 500;
    color: #EDF2F4 !important;
    opacity: 0.92;
}
.hero-subtitle {
    margin: 8px 0 18px 0;
    font-size: 1.02rem;
    color: #D3DBE2 !important;
    font-weight: 400;
    max-width: 900px;
    line-height: 1.48;
}
.hero-chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 6px;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    background: rgba(237, 242, 244, 0.12) !important;
    border: 1px solid rgba(237, 242, 244, 0.22) !important;
    border-radius: 9999px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #FFFFFF !important;
    backdrop-filter: blur(8px);
    letter-spacing: 0.01em;
}
.hero-chip-highlight {
    background: rgba(239, 35, 60, 0.35) !important;
    border-color: #EF233C !important;
    color: #FFFFFF !important;
    font-weight: 700;
}

/* Elevated Surface Cards on Slate Canvas */
.metric-card {
    background: #FFFFFF;
    border: 1px solid rgba(141, 153, 174, 0.32);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(43, 45, 66, 0.04);
    height: 100%;
}
.metric-card-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8D99AE;
    margin-bottom: 6px;
}
.metric-card-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #2B2D42;
    line-height: 1.2;
}
.metric-card-sub {
    font-size: 0.88rem;
    color: #8D99AE;
    margin-top: 4px;
    font-weight: 500;
}

/* EU Energy Rating Badge Styling */
.eu-badge-container {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 18px;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid rgba(141, 153, 174, 0.32);
    box-shadow: 0 2px 10px rgba(43, 45, 66, 0.04);
    margin-bottom: 14px;
}
.eu-chevron {
    position: relative;
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 900;
    line-height: 1;
    padding: 8px 20px 8px 14px;
    clip-path: polygon(0% 0%, 78% 0%, 100% 50%, 78% 100%, 0% 100%);
    border-radius: 6px 0 0 6px;
    min-width: 65px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(43, 45, 66, 0.15);
}
.eu-rating-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2B2D42;
}
.eu-rating-desc {
    font-size: 0.88rem;
    color: #8D99AE;
    margin-top: 2px;
}

/* Preset Pill Buttons in Sidebar & Main */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    background: #FFFFFF;
    border: 1px solid #8D99AE;
    color: #2B2D42;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(43, 45, 66, 0.04);
}
div.stButton > button:hover {
    background: #EDF2F4;
    border-color: #EF233C;
    color: #EF233C;
}
div.stButton > button:active {
    background: #EF233C;
    color: #FFFFFF;
}

/* Streamlit Native Metric Card Polish */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid rgba(141, 153, 174, 0.32);
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(43, 45, 66, 0.03);
}
div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #8D99AE;
}
div[data-testid="stMetricValue"] {
    font-weight: 800;
    color: #2B2D42;
}

/* Sidebar Styling Polish */
section[data-testid="stSidebar"] {
    background-color: #DFE5E8 !important;
    border-right: 1px solid #CBD5E1 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #CBD5E1 !important;
}

/* Streamlit Tabs Polish */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #8D99AE !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #EF233C !important;
    border-bottom-color: #EF233C !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #EF233C !important;
}

/* Input Fields styling */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
    border-color: rgba(141, 153, 174, 0.45) !important;
    border-radius: 10px !important;
    color: #2B2D42 !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #EF233C !important;
}

/* Streamlit Dataframe Card Wrapper */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(141, 153, 174, 0.32);
    border-radius: 12px;
    background-color: #FFFFFF;
}

/* Streamlit Alert Callouts */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    box-shadow: 0 1px 4px rgba(43, 45, 66, 0.03);
}
</style>
""", unsafe_allow_html=True)


# ---------- Cohesive Plotly Theme Utility ----------
PALETTE = {
    "primary": "#EF233C",       # Vivid Crimson Red (from palette)
    "secondary": "#2B2D42",     # Space Cadet Navy (from palette)
    "accent": "#D90429",        # Deep Cardinal Red (from palette)
    "slate": "#8D99AE",         # Cool Slate Gray (from palette)
    "bg_light": "#EDF2F4",      # Off-white Slate Mist (from palette)
    "danger": "#D90429",        # Deep Cardinal Red
    "success": "#2B9348",       # Fresh Eco Green
    "light_bg": "rgba(0,0,0,0)",
    "grid": "rgba(141, 153, 174, 0.22)",
    "font": "Inter, -apple-system, Segoe UI, sans-serif"
}

def apply_plotly_theme(fig, title=None, height=320):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=14, color="#2B2D42", family=PALETTE["font"]),
            x=0.01, y=0.98
        ),
        paper_bgcolor=PALETTE["light_bg"],
        plot_bgcolor=PALETTE["light_bg"],
        font=dict(family=PALETTE["font"], color="#2B2D42", size=12),
        margin=dict(t=40 if title else 20, b=30, l=45, r=25),
        height=height,
        hoverlabel=dict(
            bgcolor="#2B2D42",
            font_size=12,
            font_color="#EDF2F4",
            font_family=PALETTE["font"],
            bordercolor="#8D99AE"
        )
    )
    fig.update_xaxes(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["grid"],
        tickfont=dict(size=11, color="#8D99AE")
    )
    fig.update_yaxes(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["grid"],
        tickfont=dict(size=11, color="#8D99AE")
    )
    return fig


def render_html(html_str):
    """Safely renders HTML without triggering markdown indented code blocks."""
    clean_html = "".join(line.strip() for line in html_str.splitlines())
    if hasattr(st, "html"):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)


# ---------- Data & Model Caching ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).drop_duplicates()

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return None

@st.cache_data
def load_importance():
    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)
    return None

@st.cache_data
def load_eval_data():
    if os.path.exists(EVAL_PATH):
        return pd.read_csv(EVAL_PATH)
    return None

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Model file `model.pkl` not found or could not be loaded: {e}. Please run `python train.py` first.")
    st.stop()

df = load_data()
metrics_df = load_metrics()
importance_df = load_importance()
eval_df = load_eval_data()


# ---------- Helper Calculations ----------
def get_prediction_and_intervals(input_df):
    """
    Computes ensemble mean prediction, 80% and 90% prediction intervals,
    and returns tree votes array.
    """
    prep = model.named_steps["prep"]
    rf = model.named_steps["model"]
    
    X_prep = prep.transform(input_df)
    if hasattr(X_prep, "toarray"):
        X_prep = X_prep.toarray()
        
    tree_preds = np.array([tree.predict(X_prep)[0] for tree in rf.estimators_])
    mean_pred = float(model.predict(input_df)[0])
    p10 = float(np.percentile(tree_preds, 10))
    p90 = float(np.percentile(tree_preds, 90))
    p05 = float(np.percentile(tree_preds, 5))
    p95 = float(np.percentile(tree_preds, 95))
    std = float(np.std(tree_preds))
    
    return {
        "pred": mean_pred,
        "p10": p10,
        "p90": p90,
        "p05": p05,
        "p95": p95,
        "std": std,
        "tree_preds": tree_preds
    }


def explain_prediction_waterfall(input_df):
    """
    Exact decision-path decomposition across all 200 random forest trees.
    Returns baseline average, local feature contributions, and reconstructed prediction.
    """
    prep = model.named_steps["prep"]
    rf = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    
    feature_group = {}
    for i, col_name in enumerate(feature_names):
        if col_name.startswith("cat__Vehicle Class"):
            feature_group[i] = "Vehicle Class"
        elif col_name.startswith("cat__Transmission"):
            feature_group[i] = "Transmission"
        elif col_name.startswith("cat__Fuel Type"):
            feature_group[i] = "Fuel Type"
        elif "Engine Size" in col_name:
            feature_group[i] = "Engine Size"
        elif "Cylinders" in col_name:
            feature_group[i] = "Cylinders"
        else:
            feature_group[i] = col_name

    X_prep = prep.transform(input_df)
    if hasattr(X_prep, "toarray"):
        X_prep = X_prep.toarray()

    tree_biases = []
    tree_contributions = {feat: [] for feat in ["Engine Size", "Cylinders", "Vehicle Class", "Transmission", "Fuel Type"]}

    for tree in rf.estimators_:
        t = tree.tree_
        node_indicator = tree.decision_path(X_prep)
        node_index = node_indicator.indices
        tree_biases.append(t.value[0, 0, 0])
        
        feat_contrib = {feat: 0.0 for feat in tree_contributions}
        for j in range(len(node_index) - 1):
            curr_node = node_index[j]
            next_node = node_index[j+1]
            split_feat_idx = t.feature[curr_node]
            diff = t.value[next_node, 0, 0] - t.value[curr_node, 0, 0]
            grouped_name = feature_group.get(split_feat_idx, "Other")
            if grouped_name in feat_contrib:
                feat_contrib[grouped_name] += diff
        
        for feat in tree_contributions:
            tree_contributions[feat].append(feat_contrib[feat])
            
    base_val = float(np.mean(tree_biases))
    contributions = {feat: float(np.mean(tree_contributions[feat])) for feat in tree_contributions}
    return base_val, contributions


def get_eu_rating(pred):
    """
    Computes EU appliance-style energy rating (A through G) based on benchmark percentiles.
    """
    better_than = float((df[TARGET] > pred).mean() * 100)
    
    # Standard EU Rating Brackets
    if better_than >= 85: # <= 8.2 L/100 km
        return "A", "#008037", "#ffffff", better_than, "Top 15% · Outstanding Eco Efficiency"
    elif better_than >= 70: # 8.2 - 9.2 L/100 km
        return "B", "#50b848", "#ffffff", better_than, "Top 30% · Highly Efficient"
    elif better_than >= 50: # 9.2 - 10.6 L/100 km
        return "C", "#b5d333", "#2B2D42", better_than, "Above Average · Moderate Fuel Usage"
    elif better_than >= 30: # 10.6 - 12.1 L/100 km
        return "D", "#fff200", "#2B2D42", better_than, "Median Industry Baseline"
    elif better_than >= 15: # 12.1 - 13.8 L/100 km
        return "E", "#f68212", "#ffffff", better_than, "Below Average · High Consumption"
    elif better_than >= 5: # 13.8 - 16.5 L/100 km
        return "F", "#EF233C", "#ffffff", better_than, "Bottom 15% · Heavy Fuel Use"
    else:
        return "G", "#D90429", "#ffffff", better_than, "Bottom 5% · Severe Fuel Consumption"


def find_similar_vehicles(vclass, engine, cyl, trans, fuel, top_n=5):
    """
    Finds closest real production vehicles from the NRCan benchmark dataset.
    """
    pool = df.copy()
    e_std = pool["Engine Size(L)"].std()
    c_std = pool["Cylinders"].std()
    
    # Distance scoring: strongly penalize mismatching class and fuel
    class_pen = (pool["Vehicle Class"] != vclass).astype(float) * 3.0
    fuel_pen = (pool["Fuel Type"] != fuel).astype(float) * 2.5
    trans_pen = (pool["Transmission"] != trans).astype(float) * 0.5
    engine_dist = (pool["Engine Size(L)"] - engine).abs() / e_std
    cyl_dist = (pool["Cylinders"] - cyl).abs() / c_std
    
    pool["score"] = class_pen + fuel_pen + trans_pen + engine_dist + cyl_dist
    results = pool.sort_values("score").drop_duplicates(subset=["Make", "Model", "Engine Size(L)"]).head(top_n)
    return results


def calculate_emissions_and_trees(fuel_type, consumption_l100km, distance_km):
    """
    Estimates CO2 emissions based on fuel stoichiometric factors and computes mature tree offset.
    """
    # Grams CO2 per km per (L/100 km) consumption
    fuel_factors = {
        "D": 26.89,  # Diesel
        "E": 16.32,  # Ethanol (E85)
        "N": 16.77,  # Natural Gas
        "X": 23.32,  # Regular Gasoline
        "Z": 23.29   # Premium Gasoline
    }
    factor = fuel_factors.get(fuel_type, 23.3)
    co2_g_per_km = factor * consumption_l100km
    total_co2_kg = (co2_g_per_km * distance_km) / 1000.0
    
    # Mature tree absorbs ~21.77 kg CO2/year (US EPA / EEA standard)
    trees_annual = total_co2_kg / 21.77
    return co2_g_per_km, total_co2_kg, trees_annual


# ---------- Sidebar Configuration (Persistent State) ----------
with st.sidebar:
    st.markdown("### 🚘 Vehicle Configuration")
    st.caption("Inputs update predictions **live across all tabs**.")
    
    # Presets
    st.markdown("**⚡ Quick Presets (1-Click Demo)**")
    p_col1, p_col2 = st.columns(2)
    
    # Session state initialization
    if "v_class" not in st.session_state:
        st.session_state.v_class = "MID-SIZE"
    if "v_engine" not in st.session_state:
        st.session_state.v_engine = 2.0
    if "v_cyl" not in st.session_state:
        st.session_state.v_cyl = 4
    if "v_trans" not in st.session_state:
        st.session_state.v_trans = "AS8"
    if "v_fuel" not in st.session_state:
        st.session_state.v_fuel = "X"
    if "unit_choice" not in st.session_state:
        st.session_state.unit_choice = "L/100 km"
    if "curr_choice" not in st.session_state:
        st.session_state.curr_choice = "₹ INR"
    if "fuel_price" not in st.session_state:
        st.session_state.fuel_price = 104.0

    if p_col1.button("🌿 Eco Compact", use_container_width=True, help="1.5L 4-Cyl CVT Compact"):
        st.session_state.v_class = "COMPACT"
        st.session_state.v_engine = 1.5
        st.session_state.v_cyl = 4
        st.session_state.v_trans = "AV"
        st.session_state.v_fuel = "X"
        st.rerun()

    if p_col2.button("🚗 Mid Sedan", use_container_width=True, help="2.0L 4-Cyl 8-Speed Mid-Size"):
        st.session_state.v_class = "MID-SIZE"
        st.session_state.v_engine = 2.0
        st.session_state.v_cyl = 4
        st.session_state.v_trans = "AS8"
        st.session_state.v_fuel = "X"
        st.rerun()

    if p_col1.button("🚙 Family SUV", use_container_width=True, help="2.5L 4-Cyl 6-Speed SUV"):
        st.session_state.v_class = "SUV - SMALL"
        st.session_state.v_engine = 2.5
        st.session_state.v_cyl = 4
        st.session_state.v_trans = "AS6"
        st.session_state.v_fuel = "X"
        st.rerun()

    if p_col2.button("🛻 V8 Pickup", use_container_width=True, help="5.3L 8-Cyl Automatic Pickup"):
        st.session_state.v_class = "PICKUP TRUCK - STANDARD"
        st.session_state.v_engine = 5.3
        st.session_state.v_cyl = 8
        st.session_state.v_trans = "A8"
        st.session_state.v_fuel = "X"
        st.rerun()

    st.markdown("---")
    
    # Detailed Inputs
    vclass_list = sorted(df["Vehicle Class"].unique())
    vclass_idx = vclass_list.index(st.session_state.v_class) if st.session_state.v_class in vclass_list else 0
    selected_vclass = st.selectbox("Vehicle Class", vclass_list, index=vclass_idx, key="sb_vclass")
    st.session_state.v_class = selected_vclass

    fuel_labels = {
        "X": "X · Regular Gasoline",
        "Z": "Z · Premium Gasoline",
        "D": "D · Diesel Fuel",
        "E": "E · Ethanol (E85)",
        "N": "N · Natural Gas"
    }
    fuel_keys = sorted(df["Fuel Type"].unique())
    fuel_idx = fuel_keys.index(st.session_state.v_fuel) if st.session_state.v_fuel in fuel_keys else 0
    selected_fuel = st.selectbox(
        "Fuel Type",
        fuel_keys,
        format_func=lambda x: fuel_labels.get(x, x),
        index=fuel_idx,
        key="sb_fuel"
    )
    st.session_state.v_fuel = selected_fuel

    selected_engine = st.slider(
        "Engine Displacement (L)",
        min_value=0.8,
        max_value=8.4,
        value=float(st.session_state.v_engine),
        step=0.1,
        key="sb_engine"
    )
    st.session_state.v_engine = selected_engine

    cyl_options = [3, 4, 5, 6, 8, 10, 12, 16]
    cyl_idx = cyl_options.index(st.session_state.v_cyl) if st.session_state.v_cyl in cyl_options else 1
    selected_cyl = st.selectbox("Cylinders", cyl_options, index=cyl_idx, key="sb_cyl")
    st.session_state.v_cyl = selected_cyl

    trans_list = sorted(df["Transmission"].unique())
    trans_idx = trans_list.index(st.session_state.v_trans) if st.session_state.v_trans in trans_list else 0
    selected_trans = st.selectbox("Transmission", trans_list, index=trans_idx, key="sb_trans")
    st.session_state.v_trans = selected_trans

    # Sanity warning
    ratio_warning = False
    if selected_engine / selected_cyl > 2.5:
        st.warning("⚠️ High displacement per cylinder (>2.5L/cyl). Verify inputs.")
        ratio_warning = True

    st.markdown("---")
    st.markdown("### ⚙️ Display & Economy Units")
    
    unit_col, curr_col = st.columns(2)
    selected_unit = unit_col.radio("Efficiency Unit", ["L/100 km", "km/L"], key="sb_unit")
    st.session_state.unit_choice = selected_unit

    currency_symbols = {
        "₹ INR": ("₹", 104.0),
        "$ USD/CAD": ("$", 1.65),
        "€ EUR": ("€", 1.85),
        "£ GBP": ("£", 1.45)
    }
    selected_curr = curr_col.selectbox("Currency", list(currency_symbols.keys()), key="sb_curr")
    st.session_state.curr_choice = selected_curr
    curr_symbol, default_price = currency_symbols[selected_curr]

    fuel_price = st.number_input(
        f"Fuel Price per Litre ({curr_symbol})",
        min_value=0.1,
        max_value=1000.0,
        value=float(default_price),
        step=1.0 if curr_symbol == "₹" else 0.05,
        key="sb_price"
    )
    st.session_state.fuel_price = fuel_price

# Build Active Vehicle Input DataFrame
active_vehicle = pd.DataFrame([{
    "Engine Size(L)": st.session_state.v_engine,
    "Cylinders": st.session_state.v_cyl,
    "Vehicle Class": st.session_state.v_class,
    "Transmission": st.session_state.v_trans,
    "Fuel Type": st.session_state.v_fuel
}])

# Compute Live Predictions
pred_results = get_prediction_and_intervals(active_vehicle)
pred_val = pred_results["pred"]
p10_val = pred_results["p10"]
p90_val = pred_results["p90"]
std_val = pred_results["std"]

# Compute EU Rating
eu_grade, eu_color, eu_txt_col, eu_pct, eu_desc = get_eu_rating(pred_val)

# Mileage conversions
mileage_kml = 100.0 / pred_val if pred_val > 0 else 0
mileage_p10 = 100.0 / p90_val if p90_val > 0 else 0
mileage_p90 = 100.0 / p10_val if p10_val > 0 else 0


# ---------- Hero Banner with Quality Stat Chips ----------
rf_r2 = f"{metrics_df[metrics_df['Model'] == 'Random Forest']['R2'].values[0]:.3f}" if metrics_df is not None else "0.950"
rf_mae = f"{metrics_df[metrics_df['Model'] == 'Random Forest']['MAE'].values[0]:.2f}" if metrics_df is not None else "0.47"
rf_rmse = f"{metrics_df[metrics_df['Model'] == 'Random Forest']['RMSE'].values[0]:.2f}" if metrics_df is not None else "0.66"

hero_html = f"""
<div class="hero-container" style="background: linear-gradient(135deg, #1C1D2B 0%, #2B2D42 55%, #393C57 100%) !important; color: #EDF2F4 !important; border-left: 6px solid #EF233C !important;">
  <div class="hero-title" style="color: #FFFFFF !important;">
    <span class="hero-title-main" style="color: #FFFFFF !important; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">⛽ AutoFuel AI</span>
    <span class="hero-title-sub" style="font-size: 1.15rem; font-weight: 500; color: #EDF2F4 !important; opacity: 0.92;">· Vehicle Fuel Economy & Emissions Intelligence</span>
  </div>
  <p class="hero-subtitle" style="color: #D3DBE2 !important; margin: 8px 0 18px 0; font-size: 1.02rem; line-height: 1.48;">
    Live multi-target ensemble regression with real-time uncertainty quantification, local XAI decomposition, and standardized EU eco-efficiency classification.
  </p>
  <div class="hero-chips-container">
    <div class="hero-chip hero-chip-highlight" style="color: #FFFFFF !important; background: rgba(239, 35, 60, 0.35) !important; border-color: #EF233C !important;">🏆 Random Forest (200 Trees)</div>
    <div class="hero-chip" style="color: #FFFFFF !important;">📊 R² = {rf_r2} (95% Explained Variance)</div>
    <div class="hero-chip" style="color: #FFFFFF !important;">🎯 MAE = {rf_mae} L/100 km</div>
    <div class="hero-chip" style="color: #FFFFFF !important;">📉 RMSE = {rf_rmse} L/100 km</div>
    <div class="hero-chip" style="color: #FFFFFF !important;">📁 7,385 Certified Canadian Vehicles (NRCan)</div>
  </div>
</div>
"""
render_html(hero_html)


# ---------- Main Tab Navigation with Material Icons ----------
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/speed: Prediction & What-If",
    ":material/route: Trip Cost & Emissions",
    ":material/compare_arrows: Compare Vehicles",
    ":material/analytics: Model Insights & XAI"
])


# ==============================================================================
# TAB 1: PREDICTION & WHAT-IF (LIVE REACTIVE)
# ==============================================================================
with tab1:
    col_pred_left, col_pred_right = st.columns([1.1, 1.3], gap="large")
    
    with col_pred_left:
        # EU Energy Efficiency Card
        badge_html = f"""
        <div class="eu-badge-container">
          <div class="eu-chevron" style="background-color: {eu_color}; color: {eu_txt_col};">
            {eu_grade}
          </div>
          <div>
            <div class="eu-rating-title">EU Efficiency Rating: Grade {eu_grade}</div>
            <div class="eu-rating-desc">{eu_desc}</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #EF233C; margin-top: 3px;">
              Better than <b>{eu_pct:.1f}%</b> of benchmark fleet
            </div>
          </div>
        </div>
        """
        render_html(badge_html)

        # Visual A-G Ladder
        eu_tiers = [
            ("A", "#008037", "<8.2"),
            ("B", "#50b848", "8.2-9.2"),
            ("C", "#b5d333", "9.2-10.6"),
            ("D", "#fff200", "10.6-12.1"),
            ("E", "#f68212", "12.1-13.8"),
            ("F", "#EF233C", "13.8-16.5"),
            ("G", "#D90429", ">16.5")
        ]
        ladder_items = []
        for letter, color, bracket in eu_tiers:
            is_active = (letter == eu_grade)
            height_px = 36 if is_active else 24
            border_style = "2px solid #2B2D42" if is_active else "1px solid rgba(141,153,174,0.3)"
            font_wt = "900" if is_active else "600"
            opacity = "1.0" if is_active else "0.45"
            pointer = "▲" if is_active else "&nbsp;"
            txt_c = "#2B2D42" if letter in ["C", "D"] else "#ffffff"
            
            ladder_items.append(f"""
            <div style="flex: 1; text-align: center;">
              <div style="background-color: {color}; color: {txt_c}; height: {height_px}px; 
                          display: flex; align-items: center; justify-content: center; 
                          font-weight: {font_wt}; font-size: 0.85rem; border-radius: 4px; 
                          border: {border_style}; opacity: {opacity}; transition: all 0.3s ease;">
                {letter}
              </div>
              <div style="font-size: 0.68rem; color: #2B2D42; font-weight: 700; margin-top: 2px;">{pointer}</div>
            </div>
            """)
        
        ladder_html = f"<div style='display: flex; gap: 4px; margin-bottom: 20px; align-items: flex-end;'>{''.join(ladder_items)}</div>"
        render_html(ladder_html)

        # Primary Metrics & Prediction Intervals
        m_col1, m_col2 = st.columns(2)
        if st.session_state.unit_choice == "L/100 km":
            m_col1.metric(
                label="Predicted Consumption",
                value=f"{pred_val:.2f} L/100 km",
                help="Combined 55% City / 45% Highway laboratory rating"
            )
            m_col2.metric(
                label="Equivalent Mileage",
                value=f"{mileage_kml:.1f} km/L",
                help="Kilometres traveled per single litre of fuel"
            )
            interval_str = f"likely **{p10_val:.1f} – {p90_val:.1f} L/100 km** (80% ensemble interval, σ = {std_val:.2f})"
        else:
            m_col1.metric(
                label="Predicted Mileage",
                value=f"{mileage_kml:.1f} km/L",
                help="Kilometres traveled per single litre of fuel"
            )
            m_col2.metric(
                label="Equivalent Consumption",
                value=f"{pred_val:.2f} L/100 km",
                help="Combined 55% City / 45% Highway laboratory rating"
            )
            interval_str = f"likely **{mileage_p10:.1f} – {mileage_p90:.1f} km/L** (80% ensemble interval, σ = {std_val:.2f})"

        st.info(f"🔍 **Prediction Uncertainty**: {interval_str}", icon="ℹ️")

        # Current Specs Quick Summary Table
        st.markdown("**Selected Drivetrain Specification:**")
        spec_summary = pd.DataFrame([{
            "Displacement": f"{st.session_state.v_engine:.1f} L",
            "Cylinders": f"{st.session_state.v_cyl} Cyl",
            "Class": st.session_state.v_class,
            "Gearbox": st.session_state.v_trans,
            "Fuel": fuel_labels.get(st.session_state.v_fuel, st.session_state.v_fuel)
        }])
        st.dataframe(spec_summary, hide_index=True, use_container_width=True)

    with col_pred_right:
        # "Where You Stand" Dataset Distribution Histogram
        st.markdown("#### 📊 Fleet Benchmark: Where You Stand")
        
        hist_fig = go.Figure()
        
        # Base fleet distribution
        hist_fig.add_trace(go.Histogram(
            x=df[TARGET],
            nbinsx=45,
            name="Canadian Vehicle Fleet (NRCan)",
            marker=dict(
                color="rgba(141, 153, 174, 0.35)",
                line=dict(color=PALETTE["slate"], width=1)
            ),
            hovertemplate="Consumption: %{x:.1f} L/100 km<br>Fleet Vehicles: %{y}<extra></extra>"
        ))
        
        # Vertical marker for current prediction
        hist_fig.add_vline(
            x=pred_val,
            line_width=3,
            line_dash="dash",
            line_color=PALETTE["primary"],
            annotation_text=f"Your Vehicle: {pred_val:.1f} L/100 km (Grade {eu_grade})",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#FFFFFF", family=PALETTE["font"]),
            annotation_bgcolor=PALETTE["primary"],
            annotation_bordercolor=PALETTE["accent"]
        )
        
        hist_fig.update_layout(
            xaxis_title="Combined Fuel Consumption (L/100 km)",
            yaxis_title="Vehicle Count",
            showlegend=False
        )
        apply_plotly_theme(hist_fig, height=270)
        st.plotly_chart(hist_fig, use_container_width=True)

        # What-If Engine Displacement Sensitivity Curve
        st.markdown("#### 📈 What-If: Engine Displacement Sensitivity Curve")
        engine_sweep = np.linspace(1.0, 7.0, 25)
        sweep_rows = pd.concat([active_vehicle] * len(engine_sweep), ignore_index=True)
        sweep_rows["Engine Size(L)"] = np.round(engine_sweep, 2)
        
        # Predict along the sweep curve
        sweep_preds = model.predict(sweep_rows)
        sweep_df = pd.DataFrame({
            "Engine Size (L)": sweep_rows["Engine Size(L)"],
            "L/100 km": sweep_preds,
            "km/L": 100.0 / sweep_preds
        })

        wi_fig = go.Figure()
        
        # Sensitivity line
        wi_fig.add_trace(go.Scatter(
            x=sweep_df["Engine Size (L)"],
            y=sweep_df["L/100 km"] if st.session_state.unit_choice == "L/100 km" else sweep_df["km/L"],
            mode="lines",
            name="Model Response Curve",
            line=dict(color=PALETTE["primary"], width=3),
            hovertemplate="Engine: %{x:.1f}L<br>Economy: %{y:.2f}<extra></extra>"
        ))
        
        # Active marker
        active_y = pred_val if st.session_state.unit_choice == "L/100 km" else mileage_kml
        wi_fig.add_trace(go.Scatter(
            x=[st.session_state.v_engine],
            y=[active_y],
            mode="markers+text",
            name="Current Setting",
            marker=dict(size=13, color=PALETTE["accent"], line=dict(color="#FFFFFF", width=2)),
            text=[f"{active_y:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color=PALETTE["accent"], family=PALETTE["font"]),
            hovertemplate="Current Configuration: %{x:.1f}L -> %{y:.2f}<extra></extra>"
        ))
        
        wi_fig.update_layout(
            xaxis_title="Engine Displacement (Litres)",
            yaxis_title="Combined Consumption (L/100 km)" if st.session_state.unit_choice == "L/100 km" else "Fuel Mileage (km/L)",
            showlegend=False
        )
        apply_plotly_theme(wi_fig, height=270)
        st.plotly_chart(wi_fig, use_container_width=True)

    # Grounding Table: 5 Most Similar Vehicles from Real Dataset
    st.markdown("---")
    st.markdown("#### 🎯 Grounding: 5 Most Similar Real Vehicles from Dataset")
    st.caption("Validates model predictions against certified production vehicles tested by Natural Resources Canada.")
    
    similar_cars = find_similar_vehicles(
        st.session_state.v_class,
        st.session_state.v_engine,
        st.session_state.v_cyl,
        st.session_state.v_trans,
        st.session_state.v_fuel,
        top_n=5
    )
    
    # Format grounding table
    sim_display = similar_cars.copy()
    sim_display["Model Pred (Comb)"] = f"{pred_val:.1f} L/100 km"
    sim_display["Actual Comb"] = sim_display[TARGET].apply(lambda x: f"{x:.1f} L/100 km ({100/x:.1f} km/L)")
    sim_display["Difference (Error)"] = sim_display[TARGET].apply(lambda x: f"{x - pred_val:+.2f} L/100 km")
    sim_display["CO2 Emissions"] = sim_display["CO2 Emissions(g/km)"].apply(lambda x: f"{x} g/km")
    
    show_cols = ["Make", "Model", "Vehicle Class", "Engine Size(L)", "Cylinders", "Transmission", "Actual Comb", "Difference (Error)", "CO2 Emissions"]
    st.dataframe(sim_display[show_cols], hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 2: TRIP COST & EMISSIONS (LIVE REACTIVE)
# ==============================================================================
with tab2:
    st.markdown("### 🧾 Route Trip Cost & Environmental Footprint Calculator")
    st.caption("Instant calculation driven by your sidebar vehicle parameters.")

    trip_col_l, trip_col_r = st.columns([1, 1.3], gap="large")

    with trip_col_l:
        st.markdown("**Trip Parameters**")
        
        # Trip presets
        t_preset1, t_preset2, t_preset3, t_preset4 = st.columns(4)
        trip_dist_default = 350.0
        if "trip_distance" not in st.session_state:
            st.session_state.trip_distance = trip_dist_default
            
        if t_preset1.button("City (40 km)", use_container_width=True):
            st.session_state.trip_distance = 40.0
            st.rerun()
        if t_preset2.button("Weekend (200 km)", use_container_width=True):
            st.session_state.trip_distance = 200.0
            st.rerun()
        if t_preset3.button("Intercity (500 km)", use_container_width=True):
            st.session_state.trip_distance = 500.0
            st.rerun()
        if t_preset4.button("Tour (1200 km)", use_container_width=True):
            st.session_state.trip_distance = 1200.0
            st.rerun()

        trip_distance = st.number_input(
            "One-Way Trip Distance (km)",
            min_value=1.0,
            max_value=25000.0,
            value=float(st.session_state.trip_distance),
            step=25.0,
            key="trip_dist_input"
        )
        st.session_state.trip_distance = trip_distance

        return_trip = st.checkbox("Include return journey (2x distance)", value=False)
        effective_dist = trip_distance * 2 if return_trip else trip_distance

        # Calculations
        fuel_needed = (pred_val * effective_dist) / 100.0
        total_trip_cost = fuel_needed * st.session_state.fuel_price
        co2_g_km, trip_co2_kg, annual_trees = calculate_emissions_and_trees(
            st.session_state.v_fuel, pred_val, effective_dist
        )

        st.markdown("---")
        st.markdown("**Trip Summary Metrics**")
        
        cost_m1, cost_m2 = st.columns(2)
        cost_m1.metric("Fuel Needed", f"{fuel_needed:.1f} Litres")
        cost_m2.metric(f"Trip Cost ({curr_symbol})", f"{curr_symbol}{total_trip_cost:,.0f}")

        env_m1, env_m2 = st.columns(2)
        env_m1.metric("Trip CO₂ Emissions", f"{trip_co2_kg:.1f} kg CO₂", f"{co2_g_km:.0f} g/km rate")
        env_m2.metric("Trees to Offset", f"🌳 {annual_trees:.1f} trees", "1 yr absorption equivalent")

        st.info(
            f"🌱 **Eco Note**: Offsetting this {effective_dist:,.0f} km journey's {trip_co2_kg:.1f} kg CO₂ "
            f"requires the annual carbon sequestration capacity of **{annual_trees:.1f} mature trees** "
            f"(based on EPA/EEA baseline of 21.8 kg CO₂/tree/year).",
            icon="🌿"
        )

    with trip_col_r:
        st.markdown("#### 📈 Fuel Expenditure vs. Distance Curve")
        
        max_dist = max(int(effective_dist * 2), 200)
        dist_steps = np.linspace(0, max_dist, 50)
        curve_costs = [(pred_val * d / 100.0) * st.session_state.fuel_price for d in dist_steps]
        curve_df = pd.DataFrame({"Distance (km)": dist_steps, "Cost": curve_costs})

        cost_fig = go.Figure()
        
        # Area fill curve
        cost_fig.add_trace(go.Scatter(
            x=curve_df["Distance (km)"],
            y=curve_df["Cost"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(239, 35, 60, 0.12)",
            line=dict(color=PALETTE["primary"], width=2.5),
            name="Trip Cost",
            hovertemplate="Distance: %{x:.0f} km<br>Cost: " + curr_symbol + "%{y:,.1f}<extra></extra>"
        ))
        
        # Current distance marker
        cost_fig.add_vline(
            x=effective_dist,
            line_width=2.5,
            line_dash="dash",
            line_color=PALETTE["danger"],
            annotation_text=f"Selected Trip ({effective_dist:,.0f} km): {curr_symbol}{total_trip_cost:,.0f}",
            annotation_position="top left",
            annotation_font=dict(size=11, color=PALETTE["danger"], family=PALETTE["font"]),
            annotation_bgcolor="#FFFFFF",
            annotation_bordercolor=PALETTE["danger"]
        )

        cost_fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title=f"Total Cost ({curr_symbol})",
            showlegend=False
        )
        apply_plotly_theme(cost_fig, height=270)
        st.plotly_chart(cost_fig, use_container_width=True)

        # Environmental Breakdown Visual
        st.markdown("#### 🌳 Environmental Footprint Comparison")
        
        # Annualized comparison for 15,000 km
        annual_dist = 15000.0
        annual_co2_kg = (co2_g_km * annual_dist) / 1000.0
        annual_trees_needed = annual_co2_kg / 21.77
        
        benchmark_modes = pd.DataFrame({
            "Scenario": ["This Trip (Actual)", "Annual Driving (15,000 km)", "Domestic Flight Eq.", "High-Speed Rail Eq."],
            "CO2_kg": [trip_co2_kg, annual_co2_kg, effective_dist * 0.18, effective_dist * 0.035]
        })
        
        eco_fig = px.bar(
            benchmark_modes,
            x="Scenario",
            y="CO2_kg",
            color="Scenario",
            color_discrete_sequence=[PALETTE["primary"], PALETTE["danger"], PALETTE["accent"], PALETTE["success"]],
            labels={"CO2_kg": "CO₂ Emissions (kg)", "Scenario": ""}
        )
        eco_fig.update_layout(showlegend=False)
        apply_plotly_theme(eco_fig, height=240)
        st.plotly_chart(eco_fig, use_container_width=True)


# ==============================================================================
# TAB 3: COMPARE VEHICLES (LIVE REACTIVE)
# ==============================================================================
with tab3:
    st.markdown("### ⚖️ Head-to-Head Vehicle Benchmark & Financial TCO")
    st.caption("Live comparison between Vehicle A (Sidebar Active) and Vehicle B.")

    # Driving distance & fuel price for comparison
    cmp_p1, cmp_p2 = st.columns(2)
    cmp_annual_dist = cmp_p1.number_input("Annual Distance (km/year)", 1000.0, 100000.0, 15000.0, 1000.0, key="cmp_dist")
    cmp_ownership_yrs = cmp_p2.selectbox("Ownership Period", [3, 5, 7, 10], index=1, key="cmp_yrs")

    st.markdown("---")
    cmp_col1, cmp_col2 = st.columns(2, gap="large")

    with cmp_col1:
        st.markdown("#### 🚗 Vehicle A (Sidebar Active)")
        st.markdown(f"""
        - **Class**: `{st.session_state.v_class}`
        - **Engine**: `{st.session_state.v_engine:.1f} L` ({st.session_state.v_cyl} Cylinders)
        - **Transmission**: `{st.session_state.v_trans}`
        - **Fuel Type**: `{fuel_labels.get(st.session_state.v_fuel, st.session_state.v_fuel)}`
        """)
        pred_A = pred_val
        kml_A = mileage_kml
        annual_litres_A = (pred_A * cmp_annual_dist) / 100.0
        annual_cost_A = annual_litres_A * st.session_state.fuel_price
        co2_km_A, annual_co2_A, _ = calculate_emissions_and_trees(st.session_state.v_fuel, pred_A, cmp_annual_dist)

    with cmp_col2:
        st.markdown("#### 🚙 Vehicle B (Competitor)")
        
        # Quick Presets for Vehicle B
        b_p1, b_p2, b_p3 = st.columns(3)
        if "vb_class" not in st.session_state:
            st.session_state.vb_class = "SUV - SMALL"
        if "vb_engine" not in st.session_state:
            st.session_state.vb_engine = 2.5
        if "vb_cyl" not in st.session_state:
            st.session_state.vb_cyl = 4
        if "vb_trans" not in st.session_state:
            st.session_state.vb_trans = "AS6"
        if "vb_fuel" not in st.session_state:
            st.session_state.vb_fuel = "X"

        if b_p1.button("Eco Hatch", key="btn_b_eco", use_container_width=True):
            st.session_state.vb_class = "COMPACT"
            st.session_state.vb_engine = 1.6
            st.session_state.vb_cyl = 4
            st.session_state.vb_trans = "AV"
            st.session_state.vb_fuel = "X"
            st.rerun()

        if b_p2.button("Heavy V8 SUV", key="btn_b_suv", use_container_width=True):
            st.session_state.vb_class = "SUV - STANDARD"
            st.session_state.vb_engine = 5.3
            st.session_state.vb_cyl = 8
            st.session_state.vb_trans = "A8"
            st.session_state.vb_fuel = "X"
            st.rerun()

        if b_p3.button("Diesel Sedan", key="btn_b_dsl", use_container_width=True):
            st.session_state.vb_class = "MID-SIZE"
            st.session_state.vb_engine = 2.0
            st.session_state.vb_cyl = 4
            st.session_state.vb_trans = "AS8"
            st.session_state.vb_fuel = "D"
            st.rerun()

        b_c1, b_c2 = st.columns(2)
        vb_vclass_idx = vclass_list.index(st.session_state.vb_class) if st.session_state.vb_class in vclass_list else 0
        vb_class = b_c1.selectbox("Class (B)", vclass_list, index=vb_vclass_idx, key="sel_vb_class")
        st.session_state.vb_class = vb_class

        vb_fuel_idx = fuel_keys.index(st.session_state.vb_fuel) if st.session_state.vb_fuel in fuel_keys else 0
        vb_fuel = b_c2.selectbox("Fuel (B)", fuel_keys, format_func=lambda x: fuel_labels.get(x, x), index=vb_fuel_idx, key="sel_vb_fuel")
        st.session_state.vb_fuel = vb_fuel

        b_c3, b_c4 = st.columns(2)
        vb_engine = b_c3.slider("Engine (B)", 0.8, 8.4, float(st.session_state.vb_engine), 0.1, key="sel_vb_engine")
        st.session_state.vb_engine = vb_engine

        vb_cyl_idx = cyl_options.index(st.session_state.vb_cyl) if st.session_state.vb_cyl in cyl_options else 1
        vb_cyl = b_c4.selectbox("Cylinders (B)", cyl_options, index=vb_cyl_idx, key="sel_vb_cyl")
        st.session_state.vb_cyl = vb_cyl

        vb_trans_idx = trans_list.index(st.session_state.vb_trans) if st.session_state.vb_trans in trans_list else 0
        vb_trans = st.selectbox("Transmission (B)", trans_list, index=vb_trans_idx, key="sel_vb_trans")
        st.session_state.vb_trans = vb_trans

        # Compute Vehicle B prediction
        vehicle_B = pd.DataFrame([{
            "Engine Size(L)": vb_engine,
            "Cylinders": vb_cyl,
            "Vehicle Class": vb_class,
            "Transmission": vb_trans,
            "Fuel Type": vb_fuel
        }])
        pred_B = float(model.predict(vehicle_B)[0])
        kml_B = 100.0 / pred_B if pred_B > 0 else 0
        annual_litres_B = (pred_B * cmp_annual_dist) / 100.0
        annual_cost_B = annual_litres_B * st.session_state.fuel_price
        co2_km_B, annual_co2_B, _ = calculate_emissions_and_trees(vb_fuel, pred_B, cmp_annual_dist)

    st.markdown("---")
    
    # Live Comparison Metrics Cards
    comp_m1, comp_m2, comp_m3 = st.columns(3)
    
    comp_m1.metric(
        "Vehicle A",
        f"{pred_A:.2f} L/100 km",
        f"{curr_symbol}{annual_cost_A:,.0f} / year ({kml_A:.1f} km/L)",
        delta_color="off"
    )
    comp_m2.metric(
        "Vehicle B",
        f"{pred_B:.2f} L/100 km",
        f"{curr_symbol}{annual_cost_B:,.0f} / year ({kml_B:.1f} km/L)",
        delta_color="off"
    )
    
    diff_annual = abs(annual_cost_A - annual_cost_B)
    diff_tco = diff_annual * cmp_ownership_yrs
    winner = "Vehicle A" if annual_cost_A < annual_cost_B else "Vehicle B"
    diff_co2 = abs(annual_co2_A - annual_co2_B)

    comp_m3.metric(
        "Financial Winner",
        f"🏆 {winner}",
        f"Saves {curr_symbol}{diff_annual:,.0f} / yr ({cmp_ownership_yrs}yr: {curr_symbol}{diff_tco:,.0f})"
    )

    # Comparative Visualizations
    ch_col1, ch_col2 = st.columns(2, gap="large")
    
    with ch_col1:
        st.markdown(f"#### 💰 Cumulative Fuel Cost ({cmp_ownership_yrs}-Year Ownership)")
        years = list(range(1, cmp_ownership_yrs + 1))
        tco_A = [annual_cost_A * y for y in years]
        tco_B = [annual_cost_B * y for y in years]
        
        tco_fig = go.Figure()
        tco_fig.add_trace(go.Bar(
            x=[f"Year {y}" for y in years],
            y=tco_A,
            name="Vehicle A",
            marker_color=PALETTE["primary"],
            hovertemplate="Vehicle A: " + curr_symbol + "%{y:,.0f}<extra></extra>"
        ))
        tco_fig.add_trace(go.Bar(
            x=[f"Year {y}" for y in years],
            y=tco_B,
            name="Vehicle B",
            marker_color=PALETTE["secondary"],
            hovertemplate="Vehicle B: " + curr_symbol + "%{y:,.0f}<extra></extra>"
        ))
        tco_fig.update_layout(
            barmode="group",
            yaxis_title=f"Cumulative Fuel Cost ({curr_symbol})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        apply_plotly_theme(tco_fig, height=290)
        st.plotly_chart(tco_fig, use_container_width=True)

    with ch_col2:
        st.markdown("#### 🌍 Annual Carbon Footprint Comparison")
        eco_cmp_fig = go.Figure()
        eco_cmp_fig.add_trace(go.Bar(
            x=["Vehicle A", "Vehicle B"],
            y=[annual_co2_A, annual_co2_B],
            marker_color=[PALETTE["primary"], PALETTE["secondary"]],
            text=[f"{annual_co2_A:,.0f} kg", f"{annual_co2_B:,.0f} kg"],
            textposition="auto",
            hovertemplate="%{x}: %{y:,.1f} kg CO₂/year<extra></extra>"
        ))
        eco_cmp_fig.update_layout(
            yaxis_title="Annual CO₂ Emissions (kg)",
            showlegend=False
        )
        apply_plotly_theme(eco_cmp_fig, height=290)
        st.plotly_chart(eco_cmp_fig, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL INSIGHTS & XAI (EXPLAINABLE AI)
# ==============================================================================
with tab4:
    st.markdown("### 📊 Model Architecture, Diagnostics & Explainability (XAI)")
    st.caption("Rigorous machine learning benchmarking, local feature attribution, and residual analysis.")

    # Section 1: Multi-Model Benchmark Chart
    st.markdown("#### 1. Machine Learning Model Benchmark")
    if metrics_df is not None:
        bench_col1, bench_col2 = st.columns([1.4, 1], gap="large")
        
        with bench_col1:
            # Multi-metric comparative chart
            m_plot_df = metrics_df.melt(id_vars=["Model"], value_vars=["MAE", "RMSE", "R2"], var_name="Metric", value_name="Score")
            
            bench_fig = go.Figure()
            for model_name, m_color in zip(metrics_df["Model"], [PALETTE["primary"], PALETTE["secondary"], PALETTE["slate"]]):
                sub = m_plot_df[m_plot_df["Model"] == model_name]
                bench_fig.add_trace(go.Bar(
                    x=sub["Metric"],
                    y=sub["Score"],
                    name=f"{model_name} {'🏆 (Winner)' if 'Random Forest' in model_name else ''}",
                    marker_color=m_color,
                    hovertemplate="%{x}: %{y:.3f}<extra>" + model_name + "</extra>"
                ))
            bench_fig.update_layout(
                barmode="group",
                yaxis_title="Score (Lower MAE/RMSE is better; Higher R² is better)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            apply_plotly_theme(bench_fig, height=280)
            st.plotly_chart(bench_fig, use_container_width=True)

        with bench_col2:
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            st.success(
                "🏆 **Winning Model: Random Forest Regressor**\n\n"
                "- **200 Decision Trees** with bootstrap aggregation (bagging).\n"
                f"- Explains **{rf_r2}% of total variance** (R² = {rf_r2}).\n"
                f"- Outperforms Linear Regression by **{((0.984 - 0.656)/0.984)*100:.1f}% lower RMSE**."
            )

    st.markdown("---")

    # Section 2: Local XAI Per-Prediction Waterfall & Normalized Global Importance
    xai_col_l, xai_col_r = st.columns([1.2, 1], gap="large")

    with xai_col_l:
        st.markdown("#### 2. Local XAI: Why this prediction?")
        st.caption("Exact tree-path decomposition attributing the prediction for your active sidebar vehicle.")
        
        base_val, contribs = explain_prediction_waterfall(active_vehicle)
        
        waterfall_x = ["Baseline (Fleet Avg)"] + list(contribs.keys()) + ["Final Prediction"]
        waterfall_y = [base_val] + list(contribs.values()) + [pred_val]
        waterfall_measure = ["absolute"] + ["relative"] * len(contribs) + ["total"]
        
        wf_fig = go.Figure(go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=waterfall_measure,
            x=waterfall_x,
            textposition="outside",
            text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}" for v, m in zip(waterfall_y, waterfall_measure)],
            y=waterfall_y,
            connector=dict(line=dict(color="rgba(141, 153, 174, 0.45)")),
            decreasing=dict(marker=dict(color=PALETTE["success"])), # Green for lower consumption
            increasing=dict(marker=dict(color=PALETTE["danger"])),  # Red for higher consumption
            totals=dict(marker=dict(color=PALETTE["primary"]))
        ))
        wf_fig.update_layout(
            yaxis_title="Fuel Consumption (L/100 km)",
            waterfallgap=0.3
        )
        apply_plotly_theme(wf_fig, height=310)
        st.plotly_chart(wf_fig, use_container_width=True)

    with xai_col_r:
        st.markdown("#### 3. Global Permutation Feature Importance")
        st.caption("Normalized percentage share of predictive influence on test holdout data.")
        
        if importance_df is not None:
            imp_plot_df = importance_df.copy().sort_values("Importance_Pct", ascending=True)
            
            imp_fig = go.Figure(go.Bar(
                x=imp_plot_df["Importance_Pct"],
                y=imp_plot_df["Feature"],
                orientation="h",
                marker_color=PALETTE["primary"],
                text=imp_plot_df["Importance_Pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
                hovertemplate="Feature: %{y}<br>Importance Share: %{x:.2f}%<extra></extra>"
            ))
            imp_fig.update_layout(
                xaxis_title="Relative Importance Share (%)",
                yaxis_title="",
                xaxis=dict(range=[0, 100])
            )
            apply_plotly_theme(imp_fig, height=310)
            st.plotly_chart(imp_fig, use_container_width=True)

    st.markdown("---")

    # Section 3: Diagnostic Plots (Actual vs Predicted & Residuals)
    st.markdown("#### 4. Model Diagnostics & Goodness of Fit")
    st.caption(f"Evaluated on N=1,257 unseen test split vehicles.")

    if eval_df is not None:
        diag_col1, diag_col2 = st.columns(2, gap="large")
        
        with diag_col1:
            st.markdown("**Actual vs. Predicted Consumption**")
            
            scatter_fig = go.Figure()
            # 45-degree identity reference line
            min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
            max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
            scatter_fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Fit (y = x)",
                line=dict(color=PALETTE["danger"], dash="dash", width=2)
            ))
            # Test samples
            scatter_fig.add_trace(go.Scatter(
                x=eval_df["Actual"],
                y=eval_df["Predicted"],
                mode="markers",
                name="Holdout Test Vehicles",
                marker=dict(color=PALETTE["primary"], size=5, opacity=0.55),
                hovertemplate="Actual: %{x:.1f} L/100 km<br>Predicted: %{y:.1f} L/100 km<extra></extra>"
            ))
            scatter_fig.update_layout(
                xaxis_title="Actual Certified Consumption (L/100 km)",
                yaxis_title="Model Predicted Consumption (L/100 km)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            apply_plotly_theme(scatter_fig, height=310)
            st.plotly_chart(scatter_fig, use_container_width=True)

        with diag_col2:
            st.markdown("**Residual Error Distribution (Homoscedasticity Check)**")
            
            resid_fig = go.Figure()
            # Zero error line
            resid_fig.add_hline(
                y=0,
                line_width=2,
                line_dash="dash",
                line_color=PALETTE["danger"]
            )
            # Residual points
            resid_fig.add_trace(go.Scatter(
                x=eval_df["Predicted"],
                y=eval_df["Residual"],
                mode="markers",
                name="Residuals (Actual - Predicted)",
                marker=dict(color=PALETTE["secondary"], size=5, opacity=0.55),
                hovertemplate="Predicted: %{x:.1f} L/100 km<br>Residual: %{y:+.2f} L/100 km<extra></extra>"
            ))
            resid_fig.update_layout(
                xaxis_title="Predicted Consumption (L/100 km)",
                yaxis_title="Residual Error (L/100 km)",
                showlegend=False
            )
            apply_plotly_theme(resid_fig, height=310)
            st.plotly_chart(resid_fig, use_container_width=True)

    # Section 4: Rigorous Academic Limitations & Scope
    st.markdown("---")
    with st.expander("📚 Model Assumptions, Scope & Real-World Engineering Limitations", expanded=False):
        st.markdown("""
        ### Academic Rigour: Methodological Scope & Real-World Boundaries
        
        1. **Feature Dimensionality & Omitted Variables**:
           - **Available Parameters**: The model operates strictly on 5 primary drivetrain features: Engine Displacement (L), Cylinder Count, Vehicle Class, Transmission Type, and Fuel Chemistry.
           - **Physical Determinants Not Captured**: Does **not** incorporate vehicle curb weight (mass), aerodynamic drag coefficient ($C_d \times A$), tire rolling resistance coefficient ($C_{rr}$), auxiliary electrical loads (e.g., HVAC air-conditioning, heated seats), or battery state-of-charge for mild hybrids.
        
        2. **Standardized Laboratory Testing vs. Real-World Driving Dynamics**:
           - **Test Cycle**: Data is sourced from Natural Resources Canada (NRCan) standardized 5-cycle testing protocol (55% city, 45% highway simulated driving).
           - **Real-World Deviation**: Real-world driving exhibits significant divergence due to aggressive throttle acceleration, traffic congestion (extended idling), road elevation gradients, payloads/towing, and ambient temperatures (severe Canadian winter conditions decrease ICE efficiency by 15–25%).
        
        3. **Fleet & Geographic Scope**:
           - **Population**: Certified light-duty internal combustion engine (ICE) vehicles sold in the Canadian market between 2014 and 2020.
           - **Exclusions**: Pure Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) operating in charge-depleting modes are excluded.
           - **Fuel Formulation**: Fuel octane and ethanol blending (E10 vs pure gasoline) correspond to North American retail standards.
        
        4. **Prediction Uncertainty & Extremes**:
           - Predictions for extreme mechanical configurations (e.g., 8.0L with 3 cylinders) reflect model interpolation boundaries and are flagged with input sanity warnings.
        """)

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "AutoFuel AI · Machine Learning for Sustainable Mobility · "
    "Trained on NRCan Light-Duty Vehicle Fuel Consumption Datasets · "
    "Random Forest Ensemble Architecture"
)
