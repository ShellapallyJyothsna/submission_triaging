import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Risk Assessment Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* New background color with hex code E7EDF2 */
    body {
        background-color: #E7EDF2;
    }
    .main > div {
        background: none;
        padding: 1.5rem;
    }

    /* Card styling with glass morphism effect */
    .custom-card {
    background: rgba(255, 255, 255, 0.8);
    /* ... all other styles ... */
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    /* height: 100%; DELETED */
    border-top: 4px solid #3b82f6;
    }

    .custom-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-left: 0.5rem;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Risk segment badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        text-align: center;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid transparent;
    }

    .risk-badge:hover {
        transform: scale(1.05);
    }

    .badge-preferred { background-color: #dcfce7; color: #166534; border-color: #bbf7d0; }
    .badge-standard { background-color: #dbeafe; color: #1e40af; border-color: #bfdbfe; }
    .badge-high { background-color: #fed7aa; color: #c2410c; border-color: #fdba74; }
    .badge-declined { background-color: #fecaca; color: #991b1b; border-color: #fca5a5; }

    .badge-active {
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 2px solid #3b82f6;
    }

    /* Action items styling */
    .action-item {
        background: linear-gradient(90deg, #eff6ff, #f3e8ff);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #eff6ff, #e0e7ff);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        height: 100%;
    }

    .metric-card-green {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border-color: rgba(34, 197, 94, 0.3);
    }

    .metric-card-orange {
        background: linear-gradient(135deg, #fff7ed, #fed7aa);
        border-color: rgba(249, 115, 22, 0.3);
    }

    /* Header styling */
    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Custom progress bars */
    .custom-progress {
        width: 100%;
        height: 12px;
        background-color: #e5e7eb;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }

    .progress-blue { background: #3b82f6; }
    .progress-green { background: #10b981; }
    .progress-orange { background: #f97316; }

    /* Fix for Streamlit columns spacing */
    .st-emotion-cache-1kyx411 {
        gap: 1.5rem; /* Adjusted gap */
    }
    
    /* Override for main container padding to make it full width */
    .st-emotion-cache-18ni7ap {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

def create_risk_score_gauge(score):
    """Create a circular gauge for risk score, matching the provided image."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "RISK SCORE", 'font': {'size': 14, 'color': '#6b7280'}},
        gauge = {
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 0, 'visible': False},
            'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
            'bgcolor': "rgba(255, 255, 255, 0.5)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#dcfce7'},
                {'range': [50, 80], 'color': '#fed7aa'},
                {'range': [80, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        number = {'suffix': "%", 'font': {'size': 32, 'color': '#111827'}, 'valueformat': '.0f'}
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#6b7280"},
        height = 280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    return fig

def create_progress_bar(value, color_class):
    """Create a custom progress bar"""
    return f"""
    <div class="custom-progress">
        <div class="progress-fill {color_class}" style="width: {value}%;"></div>
    </div>
    """

# Dashboard Data
risk_score = 72
risk_segment = 'High'

# Header Section with logo
col_logo, col_header, col_status = st.columns([1, 4, 1])

with col_logo:
    st.image("logo.png", width=250)

with col_header:
    st.markdown("""
        <h1 style="color: #111827; margin: 0; font-size: 2rem; font-weight: 600;">Risk Assessment Dashboard</h1>
        <p style="color: #6b7280; margin: 0;">Comprehensive risk analysis and underwriting guidance</p>
    """, unsafe_allow_html=True)

with col_status:
    st.markdown("""
        <div style="color: #6b7280; font-size: 0.875rem; text-align: right;">
            🕐 Last Updated: 2min ago
            <span style="color: #dc2626; margin-left: 1rem; font-size: 1.5rem;">🔔</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Stats Bar
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: #fed7aa; padding: 0.5rem; border-radius: 8px;">
                🛡️
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Risk Level</p>
                <p style="color: #ea580c; font-weight: 600; margin: 0;">High</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: #bbf7d0; padding: 0.5rem; border-radius: 8px;">
                📈
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Premium Change</p>
                <p style="color: #059669; font-weight: 600; margin: 0;">+12%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: #bbf7d0; padding: 0.5rem; border-radius: 8px;">
                📈
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">testing</p>
                <p style="color: #059669; font-weight: 600; margin: 0;">+12%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


with col3:
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: #fed7aa; padding: 0.5rem; border-radius: 8px;">
                ⚠️
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Actions Needed</p>
                <p style="color: #ea580c; font-weight: 600; margin: 0;">4 Items</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="custom-card">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: #e9d5ff; padding: 0.5rem; border-radius: 8px;">
                ✅
            </div>
            <div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">Compliance</p>
                <p style="color: #7c3aed; font-weight: 600; margin: 0;">98%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Dashboard Content
col1, col2 = st.columns([1, 2])

# Risk Score Card
# Risk
    # # Define and display the chart (the "item" in the box)
    # risk_score = 72 # Assuming this variable is defined
    # risk_color = "#ea580c"
    # fig = go.Figure(go.Indicator(
    #     mode="gauge+number",
    #     value=risk_score,
    #     domain={'x': [0, 1], 'y': [0, 1]},
    #     number={
    #         'valueformat': '.0f',
    #         'font': {'size': 40, 'color': '#111827'},
    #         'suffix': f"%<br><span style='font-size:16px;color:#6b7280;'>RISK SCORE</span>"
    #     },
    #     gauge={
    #         'axis': {'range': [None, 100], 'visible': False},
    #         'bar': {'thickness': 0.15},
    #         'borderwidth': 0,
    #         'steps': [
    #             {'range': [0, risk_score], 'color': risk_color},
    #             {'range': [risk_score, 100], 'color': '#e5e7eb'}
    #         ],
    #     }
    # ))
    # fig.update_layout(
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     height=280,
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     showlegend=False
    # )
    # st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Step 3: Close the HTML container (put the "lid" on the box)


# Risk Segment Classification
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk Segment Classification")
    
    col_p, col_s, col_h, col_d = st.columns(4)

    segments = ['Preferred', 'Standard', 'High', 'Declined']
    colors = ['preferred', 'standard', 'high', 'declined']
    
    for i, (segment, color) in enumerate(zip(segments, colors)):
        with [col_p, col_s, col_h, col_d][i]:
            active_class = "badge-active" if segment == risk_segment else ""
            st.markdown(f"""
            <div class="risk-badge badge-{color} {active_class}" style="width: 100%;">
                {segment}
            </div>
            """, unsafe_allow_html=True)
    
    # <-- Change: Removed the separate title and placed it inside the div below.
    st.markdown("""
    <div style="background: linear-gradient(90deg, #fff7ed, #fecaca); padding: 1rem; border-radius: 8px; border-left: 4px solid #f97316; margin-top: 1rem;">
        <h4 style="margin-top: 0; margin-bottom: 0.5rem; color: #374151;">Submission Summary</h4>
        <p style="color: #374151; line-height: 1.6; margin: 0;">
            Loss history shows significant volatility with multiple claims. Fleet maintenance needs improvement,
            and telematics adoption is limited. Requires comprehensive risk mitigation measures.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Second Row
col1, col2 = st.columns([2, 1])

# Recommended Underwriter Actions
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Recommended Underwriter Actions")
    
    underwriter_actions = [
        "Request details of safety program improvements.",
        "Suggest quarterly fleet inspections.",
        "Verify subcontractor arrangements.",
        "Consider higher deductible options"
    ]
    
    for action in underwriter_actions:
        st.markdown(f"""
        <div class="action-item">
            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                <div style="background: #dbeafe; padding: 0.25rem; border-radius: 50%; margin-top: 0.125rem;">
                    <span style="color: #1e40af; font-size: 0.75rem;">✓</span>
                </div>
                <p style="color: #374151; margin: 0; line-height: 1.5;">{action}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pricing Guidance
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Pricing Guidance")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 1rem; border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.3); position: relative; margin-bottom: 1rem;">
        <div style="position: absolute; top: 0.5rem; right: 0.5rem; width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: pulse 2s infinite;"></div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="background: #bbf7d0; color: #059669; padding: 0.25rem; border-radius: 50%; font-size: 0.75rem;">💰</span>
            <span style="color: #065f46; font-weight: 600; font-size: 0.875rem;">Pricing Recommendation</span>
        </div>
        <p style="color: #047857; font-size: 0.875rem; margin: 0;">
            Current pricing is slightly inadequate. Increase by 10-12%.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pricing details
    st.markdown("""
    <div style="background: rgba(249, 250, 251, 0.5); padding: 0.75rem; border-radius: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.875rem; margin-bottom: 0.5rem;">
            <span style="color: #6b7280;">Current Premium</span>
            <span style="font-weight: 600;">$24,500</span>
        </div>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #d1d5db, transparent); margin: 0.5rem 0;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 0.875rem; margin-bottom: 0.5rem;">
            <span style="color: #6b7280;">Recommended Premium</span>
            <span style="font-weight: 600; color: #059669;">$27,440</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
            <span style="color: #6b7280;">Adjustment</span>
            <span style="font-weight: 600; color: #059669; display: flex; align-items: center; gap: 0.25rem;">
                📈 +12%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Key Risk Factors
st.markdown('<div class="custom-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True) # Adjusted margin
st.markdown("### 📈 Key Risk Factors")

col1, col2, col3 = st.columns(3)

# Fleet Size
with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="width: 8px; height: 8px; background: #3b82f6; border-radius: 50%;"></div>
            <span style="color: #6b7280; font-weight: 500; font-size: 0.875rem;">Fleet Size</span>
        </div>
        <p style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0; color: #111827;">45 Vehicles</p>
    """, unsafe_allow_html=True)
    st.markdown(create_progress_bar(75, "progress-blue"), unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
            <span>Small</span>
            <span>Large</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Safety Score
with col2:
    st.markdown("""
    <div class="metric-card-green">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></div>
            <span style="color: #6b7280; font-weight: 500; font-size: 0.875rem;">Safety Score</span>
        </div>
        <p style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0; color: #111827;">8.2/10</p>
    """, unsafe_allow_html=True)
    st.markdown(create_progress_bar(82, "progress-green"), unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
            <span>Poor</span>
            <span>Excellent</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Claims Frequency
with col3:
    st.markdown("""
    <div class="metric-card-orange">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <div style="width: 8px; height: 8px; background: #f97316; border-radius: 50%;"></div>
            <span style="color: #6b7280; font-weight: 500; font-size: 0.875rem;">Claims Frequency</span>
        </div>
        <p style="font-size: 2rem; font-weight: 600; margin: 0.5rem 0; color: #111827;">0.15</p>
    """, unsafe_allow_html=True)
    st.markdown(create_progress_bar(35, "progress-orange"), unsafe_allow_html=True)
    st.markdown("""
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">
            <span>Low</span>
            <span>High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer with additional info
st.markdown("""
<div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.6); border-radius: 8px; text-align: center; color: #6b7280; font-size: 0.875rem;">
    Dashboard last refreshed: {} | Data accuracy: 99.2% | Processing time: 0.3s
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)