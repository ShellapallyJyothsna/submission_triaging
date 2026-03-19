
######with the bronze<silver<gold<platinum logic fix
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from openai import AzureOpenAI
import random
import plotly.express as px


# --- Page Configuration and Styling (Apply globally) ---
st.set_page_config(layout="wide")
##########white themed
# st.markdown("""
# <style>
# /* Main app background */
# .stApp {
#     background-color: #f0f8ff; /* Light AliceBlue background */
# }

# /* Main title */
# h1 {
#     color: #004080; /* Dark blue */
# }

# /* Subheaders */
# h2, h3 {
#     color: #0055a4; /* Slightly lighter blue */
# }

# /* Style for buttons */
# .stButton>button {
#     color: #ffffff;
#     background: linear-gradient(45deg, #007bff, #0056b3);
#     border: none;
#     border-radius: 12px;
#     padding: 12px 28px;
#     font-size: 16px;
#     font-weight: bold;
#     box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
#     transition: 0.3s;
#     width: 100%; /* Make buttons fill column width */
# }

# .stButton>button:hover {
#     background: linear-gradient(45deg, #0056b3, #007bff);
#     box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
#     transform: translateY(-2px);
# }

# /* Styling for the main content container */
# .main .block-container {
#     border-radius: 20px;
#     background-color: #ffffff;
#     padding: 2rem;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.1);
# }
# </style>
# """, unsafe_allow_html=True)








st.markdown("""
<style>

/* FULL PAGE BACKGROUND FIX */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0E1117 !important;
}

/* Remove top white bar padding */
[data-testid="stHeader"] {
    background-color: #0E1117 !important;
}

/* Sidebar (if used) */
[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}

/* Main app background */
.stApp {
    background-color: #0E1117 !important;
    color: #FAFAFA;
}

/* Labels */
label, [data-testid="stWidgetLabel"] p {
    color: #FAFAFA !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #00D4FF !important;
}

/* Containers */
[data-testid="stVerticalBlock"] > div > div > div[style*="background-color"] {
    background-color: #161B22 !important; 
    border: 1px solid #30363D;
    border-radius: 15px;
}

/* Buttons */
.stButton>button {
    color: #ffffff;
    background: linear-gradient(45deg, #1f6feb, #00d4ff);
    border: none;
    border-radius: 8px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    transform: scale(1.02);
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #00D4FF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent !important;
    color: white !important;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: #00D4FF !important;
}

/* Markdown */
.stMarkdown p {
    color: #FAFAFA;
}

</style>
""", unsafe_allow_html=True)
# --- Tier offsets + helpers 
TIER_OFFSETS = {
    "Bronze":   -0.04,
    "Silver":    0.00,
    "Gold":     +0.03,
    "Platinum": +0.06,
}

def _signature_without_tier(row_dict: dict) -> tuple:
    """Stable signature of inputs except Broker Tier, used for caching base score."""
    items = [(k, v) for k, v in row_dict.items() if k != "Broker Tier"]
    return tuple(sorted(items))




def create_gauge_chart(score, level):
    """Creates a more polished and eye-catching Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0.1, 0.9], 'y': [0, 0.9]},
        title={
            'text': f"<b>Bind Propensity Score</b><br><span style='font-size:1.2em;color:white'>{level}</span>",
            'font': {'size': 24, 'color': 'white'}
        },
        number={
            'valueformat': '.2f',
            'font': {'color': 'white'}
        },
        gauge={
            'axis': {
                'range': [0, 1],
                'tickwidth': 1,
                'tickcolor': "white"
            },
            'bar': {'color': "darkslategray", 'thickness': 0.4},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E0E0E0",
            'steps': [
                {'range': [0, 0.4], 'color': '#FF7979'},
                {'range': [0.4, 0.65], 'color': '#FFC312'},
                {'range': [0.65, 1], 'color': '#009432'}
            ],
        }))
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial, Helvetica, sans-serif"},
        height=350,
        margin=dict(l=20, r=20, b=20, t=50)
    )

    return fig

# --- Business-Friendly Feature Aggregation and Formatting ---
def aggregate_shap_values(shap_values, feature_names, original_features):
    """Aggregates SHAP values for one-hot encoded features back to their original parent feature."""
    shap_series = pd.Series(shap_values, index=feature_names)
    
    original_feature_map = {}
    for encoded_col in feature_names:
        original_col_found = False
        for original_col in original_features:
            if encoded_col.startswith(original_col + '_'):
                original_feature_map[encoded_col] = original_col
                original_col_found = True
                break
        if not original_col_found:
            original_feature_map[encoded_col] = encoded_col
            
    aggregated_shaps = shap_series.groupby(original_feature_map).sum()
    return aggregated_shaps










@st.cache_data(show_spinner=False)
def get_level_explanation_text(level: str, score: float) -> str:
    """
    Returns a fixed, human-readable explanation based on the likelihood level.
    Uses your requested wording and shows the model's score as an approximate %.
    """
    pct = f"{score*100:.0f}%"
    lvl = (level or "").strip().lower()

    if lvl == "high":
        return (
            f"Strong binding potential backed by historical success in similar profiles. "
            "Clear broker engagement, complete and consistent submission data, and values within underwriting appetite "
            "mirror past wins. Expected to convert smoothly with minimal follow-ups."
        )
    elif lvl == "medium":
        return (
            f"Moderate chance of binding. Shows positive indicators but also uncertainties such as partial data, "
            "borderline pricing, or mixed historical outcomes. With quick clarification or additional context, "
            "this opportunity remains promising."
        )
    else:
        # default to Low
        return (
            f"Low probability of binding. Weak broker traction, incomplete information, and off-appetite parameters "
            "align with past declines. Significant data or risk adjustments are required to improve conversion odds."
        )










def create_top_5_shap_plot(shap_values, feature_names, input_data):
    """
    Creates an interactive Plotly bar chart for the top 5 aggregated SHAP features.

    Rules:
      A) Only 'Historical Bind Rate' is forced by Broker Tier
         - Bronze/Silver => negative
         - Gold => positive, random value between 0.03–0.06
         - Platinum => positive, random value between 0.04–0.08
      B) Case-level modifier (if available): Low vs Medium
         - 'low'   => stronger negative impact (more magnitude)
         - 'medium'=> weaker negative impact (less magnitude)
    """
    # Aggregate SHAP values
    aggregated_shaps = aggregate_shap_values(shap_values[0], feature_names, input_data.columns)

    # --- RULE OVERRIDE for Historical Bind Rate ---
    if "Historical Bind Rate" in aggregated_shaps.index:
        broker_tier = input_data["Broker Tier"].iloc[0]

        # Helper: stable RNG seeded by other inputs (prevents flicker)
        row = input_data.iloc[0].to_dict()
        seed_tuple = tuple(sorted((k, v) for k, v in row.items() if k != "Broker Tier"))
        r = random.Random(hash(str(seed_tuple)) & 0xFFFFFFFF)

        if broker_tier in ("Gold", "Platinum"):
            rng = (0.03, 0.06) if broker_tier == "Gold" else (0.04, 0.08)
            aggregated_shaps.loc["Historical Bind Rate"] = r.uniform(*rng)

        elif broker_tier in ("Bronze", "Silver"):
            # start from model value (negative) + jitter
            hb_val = float(aggregated_shaps.loc["Historical Bind Rate"])
            jitter = r.uniform(0.92, 1.08)  # small variation around model magnitude
            base_negative = -abs(hb_val) * jitter

            # ---- Case-Level Modifier (Low vs Medium) ----
            # Try to find a column that indicates case level.
            case_col_candidates = [
                "Case Level", "Case", "Case Complexity", "Submission Case",
                "Risk Case", "Opportunity Case", "Opportunity Size", "Deal Size"
            ]
            case_level_value = None
            for col in case_col_candidates:
                if col in input_data.columns:
                    v = str(input_data[col].iloc[0]).strip().lower()
                    if v in ("low", "medium"):
                        case_level_value = v
                        break

            # Default multipliers keep behavior unchanged if no case column is present
            # Ensure: |low| > |medium|
            if case_level_value == "low":
                # stronger negative (e.g., 1.15–1.40x)
                factor = r.uniform(1.15, 1.40)
            elif case_level_value == "medium":
                # weaker negative (e.g., 0.60–0.90x)
                factor = r.uniform(0.60, 0.90)
            else:
                factor = 1.0  # no case info → keep base behavior

            aggregated_shaps.loc["Historical Bind Rate"] = base_negative * factor

    # 2. Select Top-5
    top_5_series = aggregated_shaps.abs().nlargest(5)
    top_5_shap_series = aggregated_shaps[top_5_series.index].sort_values(ascending=True)
    
    values = top_5_shap_series.values
    
    # 3. Define the "Tiled" Color Palette
    
    # 1. Vibrant Tile Sets (starting from visible mid-tones)
    green_tiles = ['#A5D6A7', '#81C784', '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20']
    red_tiles = ['#EF9A9A', '#E57373', '#EF5350', '#F44336', '#E53935', '#D32F2F', '#C62828', '#B71C1C']

    fig = go.Figure()

    for i, val in enumerate(values):
        feature_name = top_5_shap_series.index[i]
        feature_val = input_data[feature_name].iloc[0] if feature_name in input_data.columns else ""
        label = f"<b>{feature_name}</b><br><span style='font-size:10px; color:#A1A1A1;'>{feature_val}</span>"
        
        color_scale = green_tiles if val >= 0 else red_tiles
        
        # Color intensity calculation
        max_range = 0.15
        normalized_val = min(abs(val) / max_range, 1.0)
        intensity_idx = int((normalized_val ** 0.6) * (len(color_scale) - 1))
        color = color_scale[intensity_idx]

        fig.add_trace(go.Bar(
            x=[val],
            y=[label],
            orientation='h',
            marker=dict(
                color=color,
                
                line=dict(color=color, width=0)
            ),
            text=f"<b>{val:.3f}</b>",
            textposition='outside',
            cliponaxis=False,
            showlegend=False
        ))

    
    fig.update_layout(
    xaxis_title="<b>Contribution to Score</b>",
    yaxis_title="",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#FAFAFA", family="sans-serif"),
    height=450,
    margin=dict(l=10, r=60, t=10, b=10),

    xaxis=dict(
        range=[-0.18, 0.18],
        showgrid=False,
        zeroline=True,
        zerolinecolor='white',
        zerolinewidth=2,
        showticklabels=True,
        tickfont=dict(color="white")   
    ),

    yaxis=dict(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(color="white")   
    )
    )
    
    return fig




# --- Data and Model Loading (Load once at the start) ---
@st.cache_resource
def load_models_and_explainer():
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        explainer = shap.TreeExplainer(rf_model)
        return lr_model, rf_model, explainer
    except FileNotFoundError:
        st.error("Model or explainer files not found. Please ensure all necessary files are present.")
        return None, None, None

@st.cache_data
def load_encoded_dataframe():
    try:
        df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
        X = df.drop(columns=["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"])
        X_encoded = pd.get_dummies(X, drop_first=True)
        return X_encoded
    except FileNotFoundError:
        st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for encoding.")
        return None

@st.cache_data
def load_full_dataframe():
    try:
        df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
        
        if 'Expected Value' in df.columns:
            df['Expected Value Numeric'] = df['Expected Value'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
        if 'Total Insured Value ($)' in df.columns:
            df['TIV_Numeric'] = df['Total Insured Value ($)'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        else:
             st.error("'Total Insured Value ($)' column not found in the dataset.")
             return None
             
        return df
    except FileNotFoundError:
        st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for visualization.")
        return None
# --- Add this helper near your other cached helpers (top of file) ---
@st.cache_data
def compute_shap_plot_cached(_explainer, _aligned_df, _input_df):
    shap_values = _explainer.shap_values(_aligned_df)
    fig = create_top_5_shap_plot(shap_values, _aligned_df.columns, _input_df)
    return fig, shap_values


lr_model, rf_model, explainer = load_models_and_explainer()
X_encoded = load_encoded_dataframe()
df_full = load_full_dataframe()

@st.cache_data
def align_full_df_for_model(df_full, _X_columns):   # <-- underscore here
    drop_cols = ["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"]
    X_all = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns], errors="ignore")
    X_all_enc = pd.get_dummies(X_all, drop_first=True)
    # cast to list to be safe even outside caching
    X_all_aligned = X_all_enc.reindex(columns=list(_X_columns), fill_value=0)
    return X_all_aligned

@st.cache_data
def compute_broker_summary(_rf_model, df_full, _X_columns):  # <-- underscore here too
    df = df_full.copy()

    if "TIV_Numeric" not in df.columns and "Total Insured Value ($)" in df.columns:
        df["TIV_Numeric"] = (
            df["Total Insured Value ($)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
        )

    X_all_aligned = align_full_df_for_model(df, _X_columns)  # pass through
    try:
        df["predicted_propensity"] = _rf_model.predict(X_all_aligned).clip(0, 1)
    except Exception:
        if hasattr(_rf_model, "predict_proba"):
            df["predicted_propensity"] = _rf_model.predict_proba(X_all_aligned)[:, 1]
        else:
            df["predicted_propensity"] = np.nan

    if "Bind Propensity Score" not in df.columns:
        df["Bind Propensity Score"] = df["predicted_propensity"]

    gb = df.groupby("Broker Name", dropna=False)
    summary = gb.agg(
        historical_avg_propensity=("Bind Propensity Score", "mean"),
        volume=("Broker Name", "count"),
        avg_TIV=("TIV_Numeric", "mean"),
    ).reset_index()

    if "Bind_Flag" in df.columns:
        summary = summary.merge(gb["Bind_Flag"].mean().reset_index(name="win_rate"), on="Broker Name", how="left")
    else:
        summary["win_rate"] = np.nan

    summary = summary.merge(gb["predicted_propensity"].mean().reset_index(name="predicted_propensity_mean"),
                            on="Broker Name", how="left")
    summary = summary.merge(gb["predicted_propensity"].sum().reset_index(name="predicted_expected_wins"),
                            on="Broker Name", how="left")

    for c in ["historical_avg_propensity", "win_rate", "predicted_propensity_mean"]:
        if c in summary.columns:
            summary[c] = summary[c].astype(float)

    summary["avg_TIV"] = summary.get("avg_TIV", 0.0).fillna(0.0)

    return summary, df


def human_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%"

def make_kpi_triplet(col_a, col_b, col_c, title_a, val_a, title_b, val_b, title_c, val_c):
    with col_a:
        st.metric(title_a, val_a)
    with col_b:
        st.metric(title_b, val_b)
    with col_c:
        st.metric(title_c, val_c)



col1, col2 = st.columns([3, 1])
with col1:
    st.title("Submission Triage Application")
with col2:
    try:
        st.image("logo.png", caption="Drive Value | Drive Momentum", width=200)
    except Exception:
        st.warning("logo.png not found.")


# --- Tab Definitions ---
tab1, tab2, tab3 = st.tabs(["Bind Propensity Score Prediction", "Submissions Prioritization", "Broker Performance Insights"])




with tab1:
    if lr_model is None or rf_model is None or X_encoded is None or explainer is None:
        st.warning("Cannot proceed with prediction due to missing files.")
    else:
        st.subheader("Select a Scenario to Pre-fill Form")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1: low_button = st.button("Low Bind Propensity")
        with b_col2: medium_button = st.button("Medium Bind Propensity")
        with b_col3: high_button = st.button("High Bind Propensity")

        if 'scenario' not in st.session_state: st.session_state['scenario'] = None

        if low_button:
            # FIX: Clear previous results when a new scenario is selected
            # st.session_state.pop('prediction_results', None)
            # Clear previous results AND tier cache (ADD THESE)
            st.session_state.pop('prediction_results', None)
            st.session_state.pop("last_signature", None)
            st.session_state.pop("last_base_score", None)
            st.session_state.pop("last_tier", None)
            st.session_state.update({
                'scenario': "Low", 'broker_name': "Delta Insure", 'channel': "Email", 'broker_tier': "Bronze",
                'industry': "Manufacturing", 'client_size': 30, 'locations': 1, 'state': "IL",
                'building_value': 5_000_000, 'contents_value': 1_000_000, 'bi_value': 500_000,
                'historical_bind_rate': 0.35, 'days_to_quote': 9, 'prior_claims': "Yes",
                'submission_complete': "No", 'cat_zone': "Yes"
            })
            st.toast("Low Bind Propensity Scenario Loaded!")
        elif medium_button:
            # FIX: Clear previous results when a new scenario is selected
            # st.session_state.pop('prediction_results', None)
            # Clear previous results AND tier cache (ADD THESE)
            st.session_state.pop('prediction_results', None)
            st.session_state.pop("last_signature", None)
            st.session_state.pop("last_base_score", None)
            st.session_state.pop("last_tier", None)
            st.session_state.update({
                'scenario': "Medium", 'broker_name': "CoreTrust", 'channel': "Wholesaler", 'broker_tier': "Silver",
                'industry': "Retail", 'client_size': 120, 'locations': 3, 'state': "NY",
                'building_value': 40_000_000, 'contents_value': 6_000_000, 'bi_value': 2_500_000,
                'historical_bind_rate': 0.55, 'days_to_quote': 6, 'prior_claims': "No",
                'submission_complete': "Yes", 'cat_zone': "No"
            })
            st.toast("Medium Bind Propensity Scenario Loaded!")
        elif high_button:
            # FIX: Clear previous results when a new scenario is selected
            # st.session_state.pop('prediction_results', None)
            # Clear previous results AND tier cache (ADD THESE)
            st.session_state.pop('prediction_results', None)
            st.session_state.pop("last_signature", None)
            st.session_state.pop("last_base_score", None)
            st.session_state.pop("last_tier", None)
            st.session_state.update({
                'scenario': "High", 'broker_name': "Alpha Risk", 'channel': "Portal", 'broker_tier': "Platinum",
                'industry': "Real Estate", 'client_size': 250, 'locations': 5, 'state': "TX",
                'building_value': 60_000_000, 'contents_value': 10_000_000, 'bi_value': 5_000_000,
                'historical_bind_rate': 0.80, 'days_to_quote': 3, 'prior_claims': "No",
                'submission_complete': "Yes", 'cat_zone': "No"
            })
            st.toast("High Bind Propensity Scenario Loaded!")

        if st.session_state['scenario'] is not None:
            st.markdown("---")
            st.subheader("Submission & Broker Information")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1: broker_name = st.selectbox("Broker Name", ["Alpha Risk", "Beta Cover", "CoreTrust", "FastBind", "Delta Insure"], key='broker_name')
            with f_col2: channel = st.selectbox("Channel", ["Portal", "Email", "Wholesaler", "API"], key='channel')
            with f_col3: broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key='broker_tier')
            with f_col4: historical_bind_rate = st.number_input("Historical Bind Rate", 0.0, 1.0, step=0.01, key='historical_bind_rate')
            
            st.markdown("---")
            st.subheader("Client & Policy Information")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Warehousing", "Real Estate", "Hospitality"], key='industry')
                building_value = st.number_input("Building Value ($)", 1000000, 100000000, step=100000, key='building_value')
                days_to_quote = st.number_input("Days to Quote", 1, 30, key='days_to_quote')
            with f_col2:
                client_size = st.number_input("Client Size (Revenue $M)", 10, 300, key='client_size')
                contents_value = st.number_input("Contents Value ($)", 100000, 20000000, step=100000, key='contents_value')
                prior_claims = st.selectbox("Prior Claims", ["Yes", "No"], key='prior_claims')
            with f_col3:
                locations = st.number_input("Number of Locations", 1, 10, key='locations')
                bi_value = st.number_input("BI Value ($)", 500000, 10000000, step=50000, key='bi_value')
                submission_complete = st.selectbox("Submission Complete", ["Yes", "No"], key='submission_complete')
            state_col, cat_col = st.columns(2)
            with state_col: state = st.selectbox("State", ["TX", "FL", "NY", "CA", "IL"], key='state')
            with cat_col: cat_zone = st.selectbox("CAT Zone", ["Yes", "No"], key='cat_zone')

            st.markdown("---")
            pred_col, _, clear_col = st.columns([2, 12, 2])
            with pred_col:
                if st.button("Submit"):
                    input_data = pd.DataFrame([{
                        "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
                        "Historical Bind Rate": historical_bind_rate, "Industry": industry,
                        "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
                        "Building Value ($)": building_value, "Contents Value ($)": contents_value,
                        "BI Value ($)": bi_value, "Submission Complete": submission_complete,
                        "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
                    }])

                    # Align for model
                    input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)

                    # Fresh model prediction (used for base when non-tier inputs change)
                    rf_pred = rf_model.predict(input_df_aligned)
                    predicted_value = float(rf_pred[0])
                    TIV = float(building_value + contents_value + bi_value)

                    # ----- TIER-AWARE CACHING -----
                    this_signature = _signature_without_tier(input_data.iloc[0].to_dict())
                    current_tier = broker_tier
                    current_offset = float(TIER_OFFSETS.get(current_tier, 0.0))

                    # Initialize session keys if missing
                    if "last_signature" not in st.session_state:
                        st.session_state["last_signature"] = None
                    if "last_base_score" not in st.session_state:
                        st.session_state["last_base_score"] = None
                    if "last_tier" not in st.session_state:
                        st.session_state["last_tier"] = None

                    # If only Broker Tier changed, reuse cached base and just adjust
                    if (
                        st.session_state["last_signature"] == this_signature
                        and st.session_state["last_base_score"] is not None
                    ):
                        base_score = float(st.session_state["last_base_score"])
                        adjusted_score = float(np.clip(base_score + current_offset, 0.0, 1.0))

                        old_tier = st.session_state.get("last_tier", current_tier)
                        old_offset = float(TIER_OFFSETS.get(old_tier, 0.0))
                        # if old_tier != current_tier:
                        #     # st.toast(f"Broker tier changed: {old_tier} → {current_tier} (Δ {current_offset - old_offset:+.02f})")
                        st.session_state["last_tier"] = current_tier

                    else:
                        # Inputs changed (or first run): compute fresh base from model prediction
                        # base = model_pred - current_offset  → so adjusted = base + offset == model_pred
                        base_score = float(np.clip(predicted_value - current_offset, 0.0, 1.0))
                        adjusted_score = float(np.clip(base_score + current_offset, 0.0, 1.0))

                        # Refresh cache
                        st.session_state["last_signature"] = this_signature
                        st.session_state["last_base_score"] = base_score
                        st.session_state["last_tier"] = current_tier
                        # st.toast("Inputs changed. Cached a new base score for tier adjustments.")
                    # ----- END TIER-AWARE CACHING -----

                    # SHAP + visuals (unchanged)
                    shap_values = explainer.shap_values(input_df_aligned)
                    top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)

                    level = "Low"
                    if adjusted_score > 0.65:
                        level = "High"
                    elif adjusted_score > 0.4:
                        level = "Medium"

                    # llm_explanation = get_llm_explanation(adjusted_score, level, input_data, shap_values, input_df_aligned.columns)
                    llm_explanation = get_level_explanation_text(level, adjusted_score)


                    # Save results (include base + offset for transparency)
                    st.session_state['prediction_results'] = {
                        "score": adjusted_score,
                        "tiv": f"${TIV:,.0f}",
                        "level": level,
                        "top_5_plot": top_5_plot,
                        "llm_explanation": llm_explanation,
                        "base_score": base_score,
                        "tier_offset": current_offset,
                    }

           
            with clear_col:
                if st.button("Clear"):
                    for k in ["prediction_results", "scenario", "last_signature", "last_base_score", "last_tier"]:
                        st.session_state.pop(k, None)
                    st.rerun()


            if 'prediction_results' in st.session_state and st.session_state['scenario'] is not None:
                st.markdown("---")
                st.subheader("Prediction Results")
                
                results = st.session_state['prediction_results']
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.plotly_chart(create_gauge_chart(results['score'], results['level']), use_container_width=True)
                with res_col2:
                    st.metric("Total Insured Value (TIV)", results['tiv'])
                
                st.markdown("---")
                st.subheader("Model Prediction Explainability")
                
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.write("#### Top 5 SHAP Features Influencing Prediction")
                    st.plotly_chart(results['top_5_plot'], use_container_width=True)
                with exp_col2:
                    st.markdown(f"""
                    <div style="border: 1px solid #0055a4; border-radius: 10px; padding: 15px; background-color: #f0f8ff; height: 100%;">
                        <h4 style="color: #004080; margin-bottom: 10px;">Insights</h4>
                        <p style="color: #333;">{results['llm_explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)









import pandas as pd
import plotly.graph_objects as go
import streamlit as st

with tab2:
    st.header("Submissions Prioritization using Strike Zone")

    # ---------- CSS ----------
    st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            white-space: nowrap;
            height: 2.6rem !important;
            min-width: 170px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            padding: 0.4rem 1rem !important;
            border-radius: 10px !important;
        }
        .kpi-wrap {display:flex; gap:14px; flex-wrap:wrap;}
        .kpi-card {
          flex:1 1 260px; padding:16px 18px; border-radius:14px;
          background: linear-gradient(135deg, #f7f9ff 0%, #eef2ff 100%);
          border: 1px solid #e6ebff;
          box-shadow: 0 2px 10px rgba(30, 64, 175, .06);
        }
        .kpi-title {font-size:13px; color:#475569; margin-bottom:6px; letter-spacing:.2px;}
        .kpi-value {font-size:34px; font-weight:700; color:#0f172a; line-height:1.15;}
        .kpi-sub {font-size:12px; color:#6b7280; margin-top:4px;}
        .kpi-badge {
          display:inline-flex; align-items:center; gap:6px;
          font-size:12px; font-weight:600; padding:4px 8px; border-radius:999px;
        }
        .kpi-badge.up {color:#065f46; background:#ecfdf5; border:1px solid #34d399;}
        .kpi-badge.down {color:#7f1d1d; background:#fef2f2; border:1px solid #fca5a5;}
        .kpi-dot {width:8px; height:8px; border-radius:50%;}
        .dot-strike {background:#2563eb;}
        .dot-nonstrike {background:#a1a1aa;}
        .section-h {
          display:flex; align-items:center; gap:10px; margin:8px 0 14px;
          font-size:22px; font-weight:800; color:#0f172a;
        }
        .section-h .emoji {font-size:20px;}
        </style>
    """, unsafe_allow_html=True)

    # ---------- Helpers ----------
    @st.cache_data(show_spinner=False)
    def load_triaging_csv(path: str):
        return pd.read_csv(path)

    def _fmt_pct(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"{x*100:.1f}%"
    def _fmt_delta(x):
        """Format plain delta numbers without % sign."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x*100:.1f}"  # multiply by 100, but don't add '%'


    def _fmt_money(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
        return f"${x:,.0f}"

    # ---------- READ CSV ----------
    try:
        df = load_triaging_csv("Triaging_Data_Comprehensive.csv")
    except Exception as e:
        st.error(f"Could not read Triaging_Data_Comprehensive.csv — {e}")
        st.stop()

    # ---------- Guardrails ----------
    required_cols = {"Bind Propensity Score", "Expected_Profitability_Ratio"}
    missing = required_cols - set(df.columns)
    if missing:
        st.warning(f"CSV is missing columns: {', '.join(missing)}")
        st.stop()

    # ---------- Clean + numeric ----------
    plot_df = df.copy()
    plot_df = plot_df[
        plot_df["Bind Propensity Score"].notna()
        & plot_df["Expected_Profitability_Ratio"].notna()
    ].copy()

    plot_df["Expected_Profitability_Ratio"] = pd.to_numeric(
        plot_df["Expected_Profitability_Ratio"], errors="coerce"
    )

    if "Expected_Profit_Dollars_With_Bind" in plot_df.columns:
        plot_df["Expected_Profit_Dollars_With_Bind"] = pd.to_numeric(
            plot_df["Expected_Profit_Dollars_With_Bind"], errors="coerce"
        )
        # Winsorize 5–95% and min–max normalize to 0..1 for % display & diff calc
        prof = plot_df["Expected_Profit_Dollars_With_Bind"]
        q05, q95 = prof.quantile(0.05), prof.quantile(0.95)
        if pd.notna(q05) and pd.notna(q95) and q95 > q05:
            prof_clip = prof.clip(lower=q05, upper=q95)
            denom = (q95 - q05)
            plot_df["Profit_Norm"] = (prof_clip - q05) / denom
        else:
            maxv = prof.max() if pd.notna(prof.max()) and prof.max() != 0 else 1.0
            plot_df["Profit_Norm"] = prof / maxv

    plot_df = plot_df.dropna(subset=["Expected_Profitability_Ratio"])
    if plot_df.empty:
        st.warning("No valid rows to plot after cleaning the CSV.")
        st.stop()

    x_data = plot_df["Bind Propensity Score"]
    y_data = plot_df["Expected_Profitability_Ratio"]
    min_y = float(y_data.min())
    max_y = float(y_data.max())

    # ---------- Sliders (strike zone) ----------
    values_x = st.slider(
        "Select a Bind Propensity range for Strike Zone",
        0.2, 1.0, (0.5, 1.0), key="slider_x_final"
    )
    values_y = st.slider(
        "Select an Expected Profitability Ratio range for Strike Zone",
        float(min_y), float(max_y),
        (float(max_y * 0.25), float(max_y)),
        key="slider_y_final"
    )

    # ---------- Scatter + strike box ----------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode="markers",
        marker=dict(
            size=8,
            color=y_data,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Expected Profitability Ratio"),
            opacity=0.7
        ),
        text=plot_df.apply(
            lambda r: (
                f"Broker: {r.get('Broker Name', '')}"
                f"<br>Expected Profitability Ratio: {r['Expected_Profitability_Ratio']:.3f}"
            ),
            axis=1
        ),
        hoverinfo="text+x+y"
    ))
    fig.add_shape(
        type="rect",
        x0=values_x[0], x1=values_x[1], y0=values_y[0], y1=values_y[1],
        line=dict(color="red", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer="above"
    )
    fig.update_layout(
        title="Submissions Segmented by Bind Propensity and Expected Profitability Ratio",
        xaxis_title="Bind Propensity Score",
        yaxis_title="Expected Profitability Ratio",
        xaxis_range=[0.2, 1.0],
        showlegend=False,
        height=600,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ---------- Strike zone slice ----------
    strike_zone_df = plot_df[
        (plot_df["Bind Propensity Score"].between(values_x[0], values_x[1]))
        & (plot_df["Expected_Profitability_Ratio"].between(values_y[0], values_y[1]))
    ].copy()

    st.write("### Simulate the Impact of Submission Prioritization")
    st.write(f"There are **{len(strike_zone_df)}** submissions in the selected Strike Zone.")

    # ---------- Button row ----------
    left_btn_col, spacer, right_btn_col = st.columns([1, 6, 1])
    with left_btn_col:
        run_clicked = st.button("Run Simulation", key="run_simulation_btn")
    with right_btn_col:
        show_metrics_clicked = st.button("Show Metrics", key="show_zone_metrics_btn")

    # ---------- Run Simulation (Top-10 in strike zone) ----------
    if run_clicked and not strike_zone_df.empty:
        strike_top10 = (
            strike_zone_df
            .sort_values(by=["Expected_Profitability_Ratio", "Bind Propensity Score"],
                         ascending=[False, False])
            .head(10)
            .copy()
        )

        display_cols = [
            "Submission ID", "Broker Name", "Industry",
            "Bind Propensity Score", "Expected_Profitability_Ratio"
        ]
        display_cols = [c for c in display_cols if c in strike_top10.columns]
        top_show = strike_top10[display_cols].rename(columns={
            "Expected_Profitability_Ratio": "Expected Profitability Ratio"
        })

        st.write("#### Top 10 Submissions (Ranked by Expected Profitability Ratio and Bind Propensity)")
        st.dataframe(
            top_show.style.format({
                "Bind Propensity Score": "{:.1%}",
                "Expected Profitability Ratio": "{:.1%}"
            }),
            use_container_width=True
        )

        if "Expected Value Numeric" in strike_top10.columns:
            total_expected_value = float(strike_top10["Expected Value Numeric"].sum())
            st.metric("Total Expected Value (Top 10)", f"${total_expected_value:,.2f}")
        
        st.download_button(
            "⬇️ Download CSV",
            data=top_show.to_csv(index=False).encode("utf-8"),
            file_name="top10_strike_zone.csv",
            mime="text/csv",
            key="download_top10"
        )
        

        st.session_state["strike_top10_cached"] = strike_top10.index.tolist()

    # ---------- Show Metrics (Top-10 Strike vs Top-10 Non-Strike) ----------
    if show_metrics_clicked:
        if "strike_top10_cached" in st.session_state:
            strike_idx = st.session_state["strike_top10_cached"]
            strike_top10 = plot_df.loc[strike_idx].copy()
        else:
            strike_top10 = (
                strike_zone_df
                .sort_values(by=["Expected_Profitability_Ratio", "Bind Propensity Score"],
                             ascending=[False, False])
                .head(10)
                .copy()
            )

        non_strike_df = plot_df.drop(strike_zone_df.index) if not strike_zone_df.empty else plot_df
        non_strike_top10 = (
            non_strike_df
            .sort_values(by=["Expected_Profitability_Ratio", "Bind Propensity Score"],
                         ascending=[False, False])
            .head(10)
            .copy()
        )

        # ---------- Bind Propensity: % point difference ----------
        avg_bp_in  = float(strike_top10["Bind Propensity Score"].mean()) if not strike_top10.empty else np.nan
        avg_bp_out = float(non_strike_top10["Bind Propensity Score"].mean()) if not non_strike_top10.empty else np.nan
        diff_bp    = None if (np.isnan(avg_bp_in) or np.isnan(avg_bp_out)) else (avg_bp_in - avg_bp_out)
        bp_badge_cls  = "up" if (diff_bp or 0) >= 0 else "down"
        # bp_badge_txt  = _fmt_pct(diff_bp)
        bp_badge_txt = _fmt_delta(diff_bp)

        st.markdown('<div class="section-h"><span class="emoji">📊</span>Bind Propensity — Top 10 (Strike Zone) vs Top 10 (Non-Strike)</div>', unsafe_allow_html=True)
        bp_col1, bp_col2 = st.columns([2, 1])
        with bp_col1:
            st.markdown(
                f"""
                <div class="kpi-wrap">
                  <div class="kpi-card">
                    <div class="kpi-title"><span class="kpi-dot dot-strike"></span> Avg (Top 10 in Strike Zone)</div>
                    <div class="kpi-value">{_fmt_pct(avg_bp_in)}</div>
                    <div class="kpi-sub">Higher is better</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-title"><span class="kpi-dot dot-nonstrike"></span> Avg (Top 10 in Non-Strike)</div>
                    <div class="kpi-value">{_fmt_pct(avg_bp_out)}</div>
                    <div class="kpi-sub">Comparison baseline</div>
                  </div>
                  <div class="kpi-card">
                    <div class="kpi-title">Delta</div>
                    <div class="kpi-value">{bp_badge_txt}</div>
                    <div class="kpi-sub"><span class="kpi-badge {bp_badge_cls}">{'▲' if (diff_bp or 0) >= 0 else '▼'} {bp_badge_txt}</span></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with bp_col2:
            fig_bp = go.Figure()
            fig_bp.add_bar(
                x=[avg_bp_in or 0, avg_bp_out or 0],
                y=["Strike", "Non-Strike"],
                orientation="h",
                text=[_fmt_pct(avg_bp_in), _fmt_pct(avg_bp_out)],
                textposition="auto",
                marker_color=["#2563eb", "#a1a1aa"],
                hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            )
            fig_bp.update_layout(
                height=160, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[0, 1], tickformat=".0%"),
                yaxis=dict(showgrid=False),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bp, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # ---------- Expected Profit: show % (from normalized) + % point difference ----------
        profit_col_dollars = "Expected_Profit_Dollars_With_Bind"
        profit_col_norm    = "Profit_Norm"

        if profit_col_norm in strike_top10.columns and profit_col_norm in non_strike_top10.columns:
            avg_profit_in_pct  = float(strike_top10[profit_col_norm].mean())
            avg_profit_out_pct = float(non_strike_top10[profit_col_norm].mean())
            diff_profit        = avg_profit_in_pct - avg_profit_out_pct  # percentage-point difference

            # Optional context in dollars
            avg_profit_in_dollars  = float(strike_top10[profit_col_dollars].mean())  if profit_col_dollars in strike_top10.columns else np.nan
            avg_profit_out_dollars = float(non_strike_top10[profit_col_dollars].mean()) if profit_col_dollars in non_strike_top10.columns else np.nan

            p_badge_cls = "up" if (diff_profit or 0) >= 0 else "down"
            # p_badge_txt = _fmt_pct(diff_profit)
            p_badge_txt  = _fmt_delta(diff_profit)

            st.markdown('<div class="section-h"><span class="emoji">💹</span>Expected Profit (%) — Top 10 (Strike Zone) vs Top 10 (Non-Strike)</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns([2, 1])
            with p_col1:
                st.markdown(
                    f"""
                    <div class="kpi-wrap">
                      <div class="kpi-card">
                        <div class="kpi-title"><span class="kpi-dot dot-strike"></span> Avg (Top 10 in Strike Zone)</div>
                        <div class="kpi-value">{_fmt_pct(avg_profit_in_pct)}</div>
                        
                      </div>
                      <div class="kpi-card">
                        <div class="kpi-title"><span class="kpi-dot dot-nonstrike"></span> Avg (Top 10 in Non-Strike)</div>
                        <div class="kpi-value">{_fmt_pct(avg_profit_out_pct)}</div>
                        
                      </div>
                      <div class="kpi-card">
                        <div class="kpi-title">Delta</div>
                        <div class="kpi-value">{p_badge_txt}</div>
                        <div class="kpi-sub"><span class="kpi-badge {p_badge_cls}">{'▲' if (diff_profit or 0) >= 0 else '▼'} {p_badge_txt}</span></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with p_col2:
                fig_pr = go.Figure()
                fig_pr.add_bar(
                    x=[avg_profit_in_pct or 0, avg_profit_out_pct or 0],
                    y=["Strike", "Non-Strike"],
                    orientation="h",
                    text=[_fmt_pct(avg_profit_in_pct), _fmt_pct(avg_profit_out_pct)],
                    textposition="auto",
                    marker_color=["#2563eb", "#a1a1aa"],
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                )
                fig_pr.update_layout(
                    height=160, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    yaxis=dict(showgrid=False),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_pr, use_container_width=True, config={"displayModeBar": False})

            # st.caption("Percentages are from winsorized (5–95%) and min–max normalized profit dollars (0–1). Difference is Strike − Non-Strike (percentage points).")
            st.divider()

        # ---------- Two Top-10 tables ----------
        profit_col = "Expected_Profit_Dollars_With_Bind"
        use_profit = profit_col in strike_top10.columns and profit_col in non_strike_top10.columns

        cols_keep = ["Submission ID", "Broker Name", "Bind Propensity Score"]
        if use_profit:
            cols_keep.append(profit_col)
        elif "Expected_Profitability_Ratio" in strike_top10.columns:
            cols_keep.append("Expected_Profitability_Ratio")

        left_tbl, right_tbl = st.columns(2)

        with left_tbl:
            st.write("#### 🏅 Top 10 — Strike Zone")
            df_show = strike_top10[[c for c in cols_keep if c in strike_top10.columns]].rename(columns={
                "Bind Propensity Score": "Bind Propensity",
                profit_col: "Expected Profit ($)",
                "Expected_Profitability_Ratio": "Expected Profitability Ratio"
            })
            sty = {"Bind Propensity": "{:.1%}"}
            if "Expected Profit ($)" in df_show.columns:
                sty["Expected Profit ($)"] = "${:,.0f}"
            if "Expected Profitability Ratio" in df_show.columns:
                sty["Expected Profitability Ratio"] = "{:.1%}"
            st.dataframe(df_show.style.format(sty), use_container_width=True)

        with right_tbl:
            st.write("#### 📄 Top 10 — Non-Strike Zone")
            df_show = non_strike_top10[[c for c in cols_keep if c in non_strike_top10.columns]].rename(columns={
                "Bind Propensity Score": "Bind Propensity",
                profit_col: "Expected Profit ($)",
                "Expected_Profitability_Ratio": "Expected Profitability Ratio"
            })
            sty = {"Bind Propensity": "{:.1%}"}
            if "Expected Profit ($)" in df_show.columns:
                sty["Expected Profit ($)"] = "${:,.0f}"
            if "Expected Profitability Ratio" in df_show.columns:
                sty["Expected Profitability Ratio"] = "{:.1%}"
            st.dataframe(df_show.style.format(sty), use_container_width=True)










# --- Tab 3: Broker Performance Insights (Power BI-style, 2×2 layout) ---
with tab3:
    st.header("📊 Broker Performance Insights")

    # Load dataset with dates for this tab
    try:
        df3 = pd.read_csv("Triaging_Data_Expanded_Controlled_Variation.csv")
    except Exception:
        df3 = df_full.copy() if df_full is not None else None

    if df3 is None or X_encoded is None or rf_model is None:
        st.warning("Insights unavailable. Ensure the dataset and model are loaded.")
    else:
        if "submission_date" not in df3.columns:
            st.error("submission_date not found in the CSV. Please use the file with dates.")
        else:
            df3["submission_date"] = pd.to_datetime(df3["submission_date"], errors="coerce")

            # Compute broker summary + scored rows based on this CSV
            summary3, df_scored3 = compute_broker_summary(rf_model, df3, X_encoded.columns)

            # --- BROKER FILTER ---
            brokers = sorted(summary3["Broker Name"].dropna().unique().tolist())
            selected_brokers = st.multiselect(
                "Filter Brokers",
                options=brokers,
                default=brokers
            )

            # Apply filter
            if selected_brokers:
                summary_f = summary3[summary3["Broker Name"].isin(selected_brokers)].copy()
                df_scored_f = df_scored3[df_scored3["Broker Name"].isin(selected_brokers)].copy()
            else:
                summary_f = summary3.copy()
                df_scored_f = df_scored3.copy()

            # --- KPI METRICS ---
            # --- KPI METRICS (Submission→Quote, Quote→Bind only) ---
            # --- KPI METRICS (S→Q with threshold, Q→B simple) ---
            # --- KPI METRICS (S→Q with threshold, Q→B simple) ---
            st.markdown("---")
            st.subheader("Key Performance Indicators")

            overall_volume = int(summary_f["volume"].sum())
            if overall_volume > 0:
                avg_propensity = (summary_f["predicted_propensity_mean"] * summary_f["volume"]).sum() / overall_volume
                avg_win_rate   = (summary_f["win_rate"] * summary_f["volume"]).sum() / overall_volume
            else:
                avg_propensity, avg_win_rate = 0.0, 0.0

            # Base filtered rows
            base_df = df_scored_f.copy()
            submissions = int(len(base_df))

            def _norm_bool(series):
                return series.astype(str).str.strip().str.lower().isin(["1","true","yes","y"])

            # --- Submission → Quote (fixed threshold ≤ 7 days) ---
            quote_threshold = 7
            if "Days to Quote" in base_df.columns:
                d2q = pd.to_numeric(base_df["Days to Quote"], errors="coerce")
                quoted_mask_sla = d2q.notna() & (d2q <= quote_threshold)
            else:
                quoted_mask_sla = pd.Series([False]*submissions, index=base_df.index)

            # --- Quote → Bind (simple: all non-null Days to Quote count as quotes) ---
            if "Days to Quote" in base_df.columns:
                quoted_mask_all = pd.to_numeric(base_df["Days to Quote"], errors="coerce").notna()
            else:
                quoted_mask_all = pd.Series([False]*submissions, index=base_df.index)

            # Detect binds
            if "Bind_Flag" in base_df.columns:
                bind_mask = _norm_bool(base_df["Bind_Flag"])
            else:
                bind_mask = pd.Series([False]*submissions, index=base_df.index)

            # Counts
            quotes_sla = int(quoted_mask_sla.sum())      # For S→Q
            quotes_all = int(quoted_mask_all.sum())      # For Q→B
            binds      = int(bind_mask.sum())

            def _safe_rate(n, d):
                return (n / d) if (d and d > 0) else None

            s_to_q = _safe_rate(quotes_sla, submissions)   # S→Q (≤7 days)
            q_to_b = _safe_rate(binds, quotes_all)         # Q→B (simple, all quotes)

            # --- NEW: Predicted Wins (Expected) ---
            if "predicted_propensity" in base_df.columns:
                predicted_expected_wins = float(base_df["predicted_propensity"].sum())
            else:
                predicted_expected_wins = None

            # --- Display ---
            # Row 1 (now 4 KPIs including Predicted Wins)
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Submissions", f"{submissions:,}")
            k2.metric("Predicted Wins (Expected)", f"{predicted_expected_wins:.1f}" if predicted_expected_wins is not None else "N/A")
            k3.metric("Avg Bind Propensity", f"{avg_propensity:.1%}")
            

            # Row 2 (conversion KPIs)
            r1, r2,r3 = st.columns(3)
            r1.metric("Submission → Quote", f"{s_to_q:.1%}" if s_to_q is not None else "N/A")
            r2.metric("Quote → Bind", f"{q_to_b:.1%}" if q_to_b is not None else "N/A")
            r3.metric("Avg Historical Win Rate", f"{avg_win_rate:.1%}" if pd.notna(avg_win_rate) else "N/A")

            st.markdown("---")


            # -------- Charts (unchanged except filtered data) --------
            YELLOW = "#FDB913"
            BLACK  = "#111111"
            RED    = "#C00000"

            temp = df_scored_f.dropna(subset=["submission_date"]).copy()
            temp["YYYY_MM"] = temp["submission_date"].dt.to_period("M").dt.to_timestamp()
            monthly = (
                temp.groupby("YYYY_MM")
                    .agg(
                        volume=("Broker Name", "count"),
                        predicted_wins=("predicted_propensity", "sum"),
                        avg_propensity=("predicted_propensity", "mean"),
                    )
                    .reset_index()
            ).sort_values("YYYY_MM")

            # Top-Left: Area
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=monthly["YYYY_MM"], y=monthly["volume"],
                mode="lines", line=dict(width=2, color=BLACK, shape="spline"),
                fill="tozeroy", name="Submissions (Volume)",
                hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
            ))
            fig_area.add_trace(go.Scatter(
                x=monthly["YYYY_MM"], y=monthly["predicted_wins"],
                mode="lines", line=dict(width=2, color=YELLOW, shape="spline"),
                fill="tozeroy", name="Predicted Wins",
                hovertemplate="Month: %{x|%b %Y}<br>Predicted Wins: %{y:.1f}<extra></extra>", opacity=0.95
            ))
            fig_area.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title=None, showgrid=False),
                yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )

            # Top-Right: Donut
            donut_df = summary_f.sort_values("volume", ascending=False).copy()
            fig_donut = go.Figure(go.Pie(
                labels=donut_df["Broker Name"],
                values=donut_df["volume"],
                hole=0.55, textinfo="label+percent", insidetextorientation="radial",
                hovertemplate="<b>%{label}</b><br>Volume: %{value}<extra></extra>"
            ))
            fig_donut.update_traces(marker=dict(colors=[YELLOW, BLACK, "#F5C542", "#4D4D4D", "#FFE08A", "#7A7A7A"]),
                                    showlegend=False)
            fig_donut.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))

            # Bottom-Left: Expected Wins per Month
            # Bottom-Left: Broker Scatter Plot (Volume vs. Avg. Propensity)
            # Use the pre-computed summary_f data which already contains "volume" and "predicted_propensity_mean"
            plot_df_scatter = summary_f.copy()

            # Filter out brokers with 0 volume to clean up the plot (optional, but recommended for visual clarity)
            plot_df_scatter = plot_df_scatter[plot_df_scatter["volume"] > 0].sort_values("volume", ascending=False)

            # Add a text column for rich hover/text label
            plot_df_scatter["Broker Label"] = (
                plot_df_scatter["Broker Name"] + "<br>Volume: " + plot_df_scatter["volume"].astype(int).astype(str) + 
                "<br>Avg. Propensity: " + (plot_df_scatter["predicted_propensity_mean"] * 100).round(1).astype(str) + "%"
            )

            fig_broker_conv = go.Figure(go.Scatter(
                x=plot_df_scatter["volume"],
                y=plot_df_scatter["predicted_propensity_mean"],
                mode='markers+text',
                marker=dict(
                    size=plot_df_scatter["volume"] / plot_df_scatter["volume"].max() * 40, # Size markers by volume
                    sizemode='area',
                    color=YELLOW,
                    line=dict(width=1, color=BLACK)
                ),
                text=plot_df_scatter["Broker Name"], # Use Broker Name as label
                textposition="middle right",
                textfont=dict(color=BLACK, size=10),
                hovertemplate=(
                    "<b>%{text}</b><br>" + 
                    "Volume: %{x:,}<br>" + 
                    "Avg. Propensity: %{y:.1%}<extra></extra>"
                ),
                name="Brokers"
            ))

            # Calculate the overall average propensity line for comparison
            overall_avg_propensity = avg_propensity

            # Add a horizontal line for the overall average propensity
            fig_broker_conv.add_hline(
                y=overall_avg_propensity, 
                line_dash="dot", line_color=RED, 
                annotation_text=f"Overall Avg. Propensity ({overall_avg_propensity:.1%})", 
                annotation_position="top right"
            )


            fig_broker_conv.update_layout(
                template="plotly_white",
                margin=dict(l=10,r=10,t=30,b=10),
                title_text="Broker Comparison: Submission Volume vs. Avg. Bind Propensity", # Added a specific title
                title_x=0.5,
                xaxis=dict(title="Total Submission Volume", showgrid=True),
                yaxis=dict(title="Avg. Predicted Bind Propensity", tickformat=".1%"),
                showlegend=False
            )
                        


            # Bottom-Right: Broker bars + Predicted Wins line
            per_broker = summary_f.sort_values("volume", ascending=False)
            fig_brokers = go.Figure()
            fig_brokers.add_trace(go.Bar(
                x=per_broker["Broker Name"], y=per_broker["volume"],
                name="Volume", marker=dict(color=YELLOW), opacity=0.95,
                hovertemplate="Broker: %{x}<br>Volume: %{y}<extra></extra>"
            ))
            fig_brokers.add_trace(go.Scatter(
                x=per_broker["Broker Name"], y=per_broker["predicted_expected_wins"],
                name="Predicted Wins", mode="lines+markers",
                line=dict(color=BLACK, width=2, shape="spline"), yaxis="y2",
                hovertemplate="Broker: %{x}<br>Predicted Wins: %{y:.1f}<extra></extra>"
            ))
            fig_brokers.update_layout(
                template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title=None, showgrid=False),
                yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                yaxis2=dict(title="Predicted Wins", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            

            # --- Predicted Wins (Expected) by Broker (new row) ---
            if "predicted_propensity" in df_scored_f.columns:
                broker_expected = (
                    df_scored_f.groupby("Broker Name", dropna=False)["predicted_propensity"]
                    .sum()
                    .reset_index()
                    .rename(columns={"predicted_propensity": "predicted_wins"})
                    .sort_values("predicted_wins", ascending=True)  # ascending for nicer horizontal bar
                )

                fig_predwins = go.Figure(go.Bar(
                    x=broker_expected["predicted_wins"],
                    y=broker_expected["Broker Name"],
                    orientation="h",
                    marker=dict(color="#FDB913"),
                    text=[f"{v:.1f}" for v in broker_expected["predicted_wins"]],
                    textposition="outside",
                    hovertemplate="Broker: %{y}<br>Predicted Wins: %{x:.1f}<extra></extra>"
                ))

                fig_predwins.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Predicted Wins (Expected)"),
                    yaxis=dict(title=None, automargin=True),
                    height=420
                )

                # New full-width row under the 2×2
                # st.write("#### Predicted Wins (Expected) by Broker")
                # st.plotly_chart(fig_predwins, use_container_width=True)
            else:
                st.info("Column `predicted_propensity` not found; cannot compute Predicted Wins per Broker.")
            # Uniform chart heights
            for f in (fig_area, fig_donut, fig_broker_conv, fig_brokers):
                f.update_layout(height=360)

            
            # -------- 2×2 MATRIX LAYOUT --------
            r1c1, r1c2 = st.columns(2, gap="large")
            with r1c1:
                st.write("#### Pipeline Over Time")
                st.plotly_chart(fig_area, use_container_width=True, key="fig_area")

            with r1c2:
                st.write("#### Broker Share (Volume)")
                st.plotly_chart(fig_donut, use_container_width=True, key="fig_donut")

            r2c1, r2c2 = st.columns(2, gap="large")
            with r2c1:
                st.write("#### Broker Comparison")
                st.plotly_chart(fig_broker_conv, use_container_width=True, key="fig_broker_conv")

            with r2c2:
                st.write("#### Submissions vs Predicted Wins by Broker")
                st.plotly_chart(fig_brokers, use_container_width=True, key="fig_brokers")

            # # -------- 5th Chart (Full Width Row) --------
            # st.write("#### Predicted Wins (Expected) by Broker")
            # st.plotly_chart(fig_predwins, use_container_width=True, key="fig_expected_broker")

























