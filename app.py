"""
Household Electricity Consumption Prediction Dashboard
--------------------------------------------------------
Author: Built for Chintan | BreinyBeam Technologies
Model : household_energy_model.pkl (XGBoost / sklearn-compatible pipeline)

SETUP (important):
    Put these two files in the SAME folder as this app.py:
        household_energy_model.pkl
        household_energy_data.csv
    The app loads them automatically from disk. There is no "upload model"
    button anymore on purpose — uploading large .pkl files through the
    browser is what caused the "STACK_GLOBAL requires str" crash before
    (partial/corrupted transfer, or a joblib-saved file being read with
    raw pickle). Loading from disk avoids both problems.

Run with:
    streamlit run app.py
"""

import io
import pickle
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import joblib
except ImportError:
    joblib = None

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Household Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# Note: base theme colors now live in .streamlit/config.toml — that's
# what fixes native widgets (sidebar radio, captions, file-uploader
# labels, headers) that this CSS block never touched before. This CSS
# only adds the extra card/box styling on top.
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    /* Belt-and-braces: force readable text color on every common
       Streamlit text element, in case a browser/theme override
       ever fights the config.toml theme again. */
    body, p, label, li,
    .stMarkdown, .stCaption, .stRadio label,
    section[data-testid="stSidebar"] * {
        color: #f0f1f5 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #9aa0ac !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1c24 0%, #21242f 100%);
        border: 1px solid #2d3040;
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .metric-card h3 {
        color: #8b8fa3 !important;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-card p {
        color: #ffffff !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .metric-card span {
        font-size: 13px;
        color: #57e389 !important;
    }

    .prediction-box {
        background: linear-gradient(135deg, #113a2e 0%, #0e2921 100%);
        border: 1px solid #1f6f4a;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 10px;
    }
    .prediction-box h2 {
        color: #57e389 !important;
        font-size: 46px;
        margin: 0;
    }
    .prediction-box p {
        color: #9aa0ac !important;
        font-size: 14px;
        margin-top: 6px;
    }

    section[data-testid="stSidebar"] {
        background-color: #12141c;
        border-right: 1px solid #23262f;
        height: 100vh;
        overflow: hidden;
    }

    /* Keep sidebar content scrollable while the navigation control stays
       available at the top of the viewport. */
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-width: thin;
        scrollbar-color: #596273 #12141c;
        padding-top: 3.5rem;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
        width: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: #596273;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
        background: #12141c;
    }
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarExpandButton"] {
        position: fixed;
        top: 0.75rem;
        z-index: 100001;
        color: #f0f1f5 !important;
        background: #202532 !important;
        border: 1px solid #596273 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }
    button[data-testid="stSidebarCollapseButton"] {
        left: 0.75rem;
    }
    button[data-testid="stSidebarExpandButton"] {
        left: 0.75rem;
    }
    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="stSidebarExpandButton"] svg {
        color: #f0f1f5 !important;
        stroke: #f0f1f5 !important;
    }

    /* BaseWeb menus use a light surface. Keep their values readable when
       the app's dark text theme is active. */
    [data-baseweb="select"] input,
    [data-baseweb="select"] [role="combobox"],
    [data-baseweb="select"] [role="listbox"],
    [data-baseweb="select"] [role="option"],
    [data-baseweb="select"] [role="option"] *,
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] * {
        color: #1f2937 !important;
    }
    [data-baseweb="select"] [role="listbox"],
    [data-baseweb="popover"] [role="listbox"] {
        background-color: #ffffff !important;
    }
    [data-baseweb="select"] [role="option"]:hover,
    [data-baseweb="select"] [aria-selected="true"] {
        background-color: #e8eef7 !important;
    }
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] *,
    [data-baseweb="calendar"] button {
        color: #1f2937 !important;
    }
    [data-baseweb="calendar"] {
        background-color: #ffffff !important;
    }

    div[data-testid="stMetricValue"] { color: #57e389 !important; }

    h1, h2, h3 { color: #f0f1f5 !important; }

    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }

    .model-status-ok {
        background: #113a2e; border: 1px solid #1f6f4a; color: #57e389 !important;
        border-radius: 10px; padding: 10px 14px; font-size: 13px;
    }
    .model-status-bad {
        background: #3a1a1a; border: 1px solid #6f1f1f; color: #f28b82 !important;
        border-radius: 10px; padding: 10px 14px; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — must match training exactly
# ============================================================
FEATURE_ORDER = [
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC",
    "Peak_Hours_Usage_kWh",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "IsWeekend",
]
TARGET_COL = "Energy_Consumption_kWh"
MODEL_PATH = "household_energy_model.pkl"
DATA_PATH = "household_energy_consumption.csv"

# ============================================================
# LOADERS (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_from_path(path):
    """Try joblib first (most XGBoost/sklearn pipelines are saved this
    way), then fall back to plain pickle. Raises with a clear message
    instead of the opaque 'STACK_GLOBAL requires str' error."""
    last_err = None

    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception as e:
            last_err = e

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{path}' with joblib or pickle. "
            f"This usually means the file is not a valid pickle/joblib "
            f"file for this Python/sklearn/xgboost version, or it was "
            f"only partially written to disk. Re-export the model with "
            f"`joblib.dump(model, '{path}')` from the same environment "
            f"you trained it in. (Last error: {last_err or e})"
        )

@st.cache_data(show_spinner=False)
def load_data_from_path(path):
    df = pd.read_csv(path)
    return prep_dataframe(df)

@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return prep_dataframe(df)

def prep_dataframe(df):
    """Standardize columns, parse date, engineer time features if missing."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Year" not in df.columns:
            df["Year"] = df["Date"].dt.year
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
        if "Day" not in df.columns:
            df["Day"] = df["Date"].dt.day
        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = df["Date"].dt.dayofweek
        if "IsWeekend" not in df.columns:
            df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df

# ============================================================
# SIDEBAR — data & model status (model is LOCAL ONLY, no upload)
# ============================================================
st.sidebar.title("⚡ Energy Predictor")
st.sidebar.caption("Household Electricity Consumption")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔮 Predict Consumption", "ℹ️ About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model & Data")

# --- Model: load ONLY from the local file next to app.py ---
model = None
model_error = None
try:
    model = load_model_from_path(MODEL_PATH)
except FileNotFoundError:
    model_error = (
        f"`{MODEL_PATH}` not found next to app.py. Place the trained "
        f"model file in this same folder (in VS Code, drag it into the "
        f"project directory) — there's no in-app upload for the model "
        f"on purpose, to avoid corrupted browser uploads."
    )
except Exception as e:
    model_error = str(e)

if model is not None:
    st.sidebar.markdown(
        f'<div class="model-status-ok">✅ Model loaded from <code>{MODEL_PATH}</code></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        f'<div class="model-status-bad">❌ {model_error}</div>',
        unsafe_allow_html=True,
    )

# --- Data: local file preferred, CSV upload still allowed as a convenience ---
data = None
data_source = None
try:
    data = load_data_from_path(DATA_PATH)
    data_source = f"Loaded `{DATA_PATH}` from app folder ✅"
except Exception:
    uploaded_data = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded_data is not None:
        try:
            data = load_data_from_bytes(uploaded_data.getvalue())
            data_source = "Dataset uploaded ✅"
        except Exception as e:
            st.sidebar.error(f"Couldn't load dataset: {e}")

if data is not None:
    st.sidebar.success(data_source or "Dataset ready ✅")
else:
    st.sidebar.info("⬆️ Upload your dataset CSV to see the dashboard")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · XGBoost pipeline")

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.title("📊 Household Energy Consumption Dashboard")
    st.caption("Exploratory overview of household electricity usage patterns")

    if data is None:
        st.warning("Upload your dataset CSV from the sidebar to populate this dashboard.")
        st.stop()

    # ---- KPI cards ----
    n_records = len(data)
    n_households = data["Household_ID"].nunique() if "Household_ID" in data.columns else "—"
    avg_consumption = data[TARGET_COL].mean() if TARGET_COL in data.columns else np.nan
    avg_temp = data["Avg_Temperature_C"].mean() if "Avg_Temperature_C" in data.columns else np.nan
    pct_ac = (
        (data["Has_AC"].astype(str).str.lower() == "yes").mean() * 100
        if "Has_AC" in data.columns else np.nan
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, suffix in zip(
        [c1, c2, c3, c4, c5],
        ["Total Records", "Households", "Avg Consumption", "Avg Temperature", "Homes with AC"],
        [f"{n_records:,}", f"{n_households}", f"{avg_consumption:,.2f}", f"{avg_temp:,.1f}", f"{pct_ac:,.1f}"],
        ["", "", " kWh", " °C", " %"],
    ):
        col.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p>{value}{suffix}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Data preview ----
    with st.expander("🔍 Preview Dataset", expanded=True):
        st.dataframe(data.head(20), use_container_width=True)
        st.caption(f"Showing 20 of {n_records:,} rows · {data.shape[1]} columns")

    st.markdown("---")

    # ---- Charts ----
    if "Date" in data.columns and TARGET_COL in data.columns:
        st.subheader("📈 Consumption Trend Over Time")
        trend = data.groupby("Date", as_index=False)[TARGET_COL].mean()
        fig = px.line(trend, x="Date", y=TARGET_COL, template="plotly_dark")
        fig.update_traces(line_color="#57e389", line_width=2.5)
        fig.update_layout(height=380, margin=dict(t=20, b=10),
                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if "Household_Size" in data.columns and TARGET_COL in data.columns:
            st.subheader("🏠 Consumption by Household Size")
            fig = px.box(data, x="Household_Size", y=TARGET_COL, template="plotly_dark",
                         color_discrete_sequence=["#2563eb"])
            fig.update_layout(height=350, margin=dict(t=10, b=10),
                               plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "Has_AC" in data.columns and TARGET_COL in data.columns:
            st.subheader("❄️ AC vs Non-AC Consumption")
            fig = px.box(data, x="Has_AC", y=TARGET_COL, template="plotly_dark",
                         color="Has_AC", color_discrete_sequence=["#f97316", "#57e389"])
            fig.update_layout(height=350, margin=dict(t=10, b=10),
                               plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        if "Avg_Temperature_C" in data.columns and TARGET_COL in data.columns:
            st.subheader("🌡️ Temperature vs Consumption")
            fig = px.scatter(data, x="Avg_Temperature_C", y=TARGET_COL, template="plotly_dark",
                              color=data["Has_AC"] if "Has_AC" in data.columns else None,
                              opacity=0.6, color_discrete_sequence=["#f97316", "#57e389"])
            fig.update_layout(height=350, margin=dict(t=10, b=10),
                               plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if "DayOfWeek" in data.columns and TARGET_COL in data.columns:
            st.subheader("📅 Avg Consumption by Day of Week")
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            tmp = data.copy()
            tmp["DayName"] = tmp["DayOfWeek"].map(dow_map)
            dow_avg = tmp.groupby("DayName", as_index=False)[TARGET_COL].mean()
            order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_avg["DayName"] = pd.Categorical(dow_avg["DayName"], categories=order, ordered=True)
            dow_avg = dow_avg.sort_values("DayName")
            fig = px.bar(dow_avg, x="DayName", y=TARGET_COL, template="plotly_dark",
                         color_discrete_sequence=["#a78bfa"])
            fig.update_layout(height=350, margin=dict(t=10, b=10),
                               plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

    # ---- Correlation heatmap ----
    num_cols = [c for c in FEATURE_ORDER + [TARGET_COL] if c in data.columns and
                pd.api.types.is_numeric_dtype(data[c])]
    if len(num_cols) > 2:
        st.subheader("🔗 Feature Correlation")
        corr = data[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", template="plotly_dark",
                         color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(height=420, margin=dict(t=10, b=10),
                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2: PREDICTION
# ============================================================
elif page == "🔮 Predict Consumption":
    st.title("🔮 Predict Household Energy Consumption")
    st.caption("Enter household details to estimate daily electricity consumption (kWh)")

    if model is None:
        st.warning(model_error or "Model not available.")
        st.stop()

    # Derive sensible input ranges/options directly from the dataset,
    # so the user can only enter values consistent with what the model
    # actually saw during training. Falls back to fixed defaults if no
    # dataset is loaded.
    if data is not None and "Household_Size" in data.columns:
        hh_min, hh_max = int(data["Household_Size"].min()), int(data["Household_Size"].max())
        hh_default = int(data["Household_Size"].median())
    else:
        hh_min, hh_max, hh_default = 1, 15, 4

    if data is not None and "Avg_Temperature_C" in data.columns:
        temp_min = float(data["Avg_Temperature_C"].min())
        temp_max = float(data["Avg_Temperature_C"].max())
        temp_default = float(data["Avg_Temperature_C"].median())
    else:
        temp_min, temp_max, temp_default = -5.0, 50.0, 25.0

    if data is not None and "Peak_Hours_Usage_kWh" in data.columns:
        peak_min = float(data["Peak_Hours_Usage_kWh"].min())
        peak_max = float(data["Peak_Hours_Usage_kWh"].max())
        peak_default = float(data["Peak_Hours_Usage_kWh"].median())
    else:
        peak_min, peak_max, peak_default = 0.0, 50.0, 3.0

    ac_options = ["Yes", "No"]
    if data is not None and "Has_AC" in data.columns:
        seen = [str(v) for v in data["Has_AC"].dropna().unique()]
        # Keep Yes/No order if that's what the data uses; otherwise use as-is
        if set(v.lower() for v in seen) <= {"yes", "no"}:
            ac_options = ["Yes", "No"]
        else:
            ac_options = seen

    household_ids = None
    if data is not None and "Household_ID" in data.columns:
        household_ids = sorted(data["Household_ID"].dropna().unique().tolist())

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Household Inputs")

        if household_ids:
            st.selectbox(
                "Household ID (reference only — not used by the model)",
                household_ids,
            )

        c1, c2 = st.columns(2)
        with c1:
            household_size = st.number_input(
                "Household Size (people)",
                min_value=hh_min, max_value=hh_max, value=hh_default, step=1,
                help=f"Based on your dataset, sizes range {hh_min}–{hh_max}.",
            )
            has_ac = st.selectbox("Has Air Conditioning?", ac_options)
        with c2:
            avg_temp = st.slider(
                "Avg Temperature (°C)",
                min_value=temp_min, max_value=temp_max, value=temp_default, step=0.1,
                help=f"Dataset range: {temp_min:.1f}°C to {temp_max:.1f}°C.",
            )
            peak_usage = st.number_input(
                "Peak Hours Usage (kWh)",
                min_value=peak_min, max_value=peak_max, value=peak_default, step=0.1,
                help=f"Dataset range: {peak_min:.1f}–{peak_max:.1f} kWh.",
            )

        st.markdown("**Date**")
        input_date = st.date_input("Prediction Date", value=date.today())

        year = input_date.year
        month = input_date.month
        day = input_date.day
        day_of_week = input_date.weekday()  # 0=Mon
        is_weekend = 1 if day_of_week in (5, 6) else 0

        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        st.caption(f"📅 {dow_names[day_of_week]}"
                   f"{' · Weekend' if is_weekend else ' · Weekday'} · "
                   f"Year={year} · Month={month} · Day={day}")

        predict_btn = st.button("⚡ Predict Consumption", use_container_width=True)

    with right:
        st.subheader("Prediction Result")

        if predict_btn:
            input_row = pd.DataFrame([{
                "Household_Size": household_size,
                "Avg_Temperature_C": avg_temp,
                "Has_AC": has_ac,
                "Peak_Hours_Usage_kWh": peak_usage,
                "Year": year,
                "Month": month,
                "Day": day,
                "DayOfWeek": day_of_week,
                "IsWeekend": is_weekend,
            }])[FEATURE_ORDER]

            try:
                prediction = model.predict(input_row)[0]

                st.markdown(f"""
                    <div class="prediction-box">
                        <p>Estimated Daily Consumption</p>
                        <h2>{prediction:,.2f} kWh</h2>
                        <p>≈ {prediction * 30:,.1f} kWh / month</p>
                    </div>
                """, unsafe_allow_html=True)

                # Comparison gauge vs dataset average, if data available
                if data is not None and TARGET_COL in data.columns:
                    avg_val = data[TARGET_COL].mean()
                    max_val = max(data[TARGET_COL].max(), prediction) * 1.15
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        delta={"reference": avg_val, "increasing": {"color": "#f97316"},
                               "decreasing": {"color": "#57e389"}},
                        gauge={
                            "axis": {"range": [0, max_val]},
                            "bar": {"color": "#57e389"},
                            "steps": [
                                {"range": [0, avg_val], "color": "#1a2e24"},
                                {"range": [avg_val, max_val], "color": "#2e1a1a"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "value": avg_val,
                            },
                        },
                        title={"text": "vs. Dataset Average (kWh)"},
                    ))
                    fig.update_layout(height=320, margin=dict(t=40, b=10),
                                       paper_bgcolor="#0e1117", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Dataset average: {avg_val:,.2f} kWh/day")

                with st.expander("🧾 Model Input Vector"):
                    st.dataframe(input_row, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "This usually means the feature order or input types don't match how "
                    "the model was trained. The training pipeline expects Has_AC as the "
                    "original Yes/No text value."
                )
        else:
            st.info("Fill in the inputs and click **Predict Consumption** to see the result.")

# ============================================================
# PAGE 3: ABOUT
# ============================================================
else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Household Electricity Consumption Predictor

    This dashboard estimates a household's daily electricity consumption (kWh)
    using an XGBoost-based regression pipeline trained on historical household
    energy usage data.

    **Model input features (in order):**
    1. `Household_Size` — number of people in the household
    2. `Avg_Temperature_C` — average daily temperature
    3. `Has_AC` — whether the household has air conditioning (`Yes` or `No`)
    4. `Peak_Hours_Usage_kWh` — electricity used during peak hours
    5. `Year`, `Month`, `Day`, `DayOfWeek`, `IsWeekend` — engineered date features

    **How to use:**
    - Place `household_energy_model.pkl` **and** `household_energy_data.csv`
      directly in the same folder as `app.py` in VS Code — the model is
      loaded from disk only, there is no in-app model upload.
    - Use the **Dashboard** tab to explore consumption patterns.
    - Use the **Predict Consumption** tab to get a live prediction for a
      specific household profile and date. Input ranges are pulled
      automatically from your dataset so you can't enter values the
      model never saw during training.

    ---
    Built with **Streamlit**, **Plotly**, and **XGBoost**.
    """)