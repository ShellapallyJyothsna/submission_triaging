




















































############demoed version
# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import shap
# import matplotlib.pyplot as plt
# from openai import AzureOpenAI

# # --- Page Configuration and Styling (Apply globally) ---
# st.set_page_config(layout="wide")

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


# # --- Helper Function for Gauge Chart ---
# def create_gauge_chart(score, level):
#     """Creates a more polished and eye-catching Plotly gauge chart."""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=score,
#         domain={'x': [0.1, 0.9], 'y': [0, 0.9]},
#         title={'text': f"<b>Bind Propensity Score</b><br><span style='font-size:1.2em;color:grey'>{level}</span>", 'font': {'size': 24}},
#         number={'valueformat': '.2f'},
#         gauge={
#             'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkslategray"},
#             'bar': {'color': "darkslategray", 'thickness': 0.4},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "#E0E0E0",
#             'steps': [
#                 {'range': [0, 0.4], 'color': '#FF7979'},
#                 {'range': [0.4, 0.65], 'color': '#FFC312'},
#                 {'range': [0.65, 1], 'color': '#009432'}
#             ],
#         }))
        
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': "darkslategray", 'family': "Arial, Helvetica, sans-serif"},
#         height=350,
#         margin=dict(l=20, r=20, b=20, t=50)
#     )
#     return fig

# # --- Business-Friendly Feature Aggregation and Formatting ---
# def aggregate_shap_values(shap_values, feature_names, original_features):
#     """Aggregates SHAP values for one-hot encoded features back to their original parent feature."""
#     shap_series = pd.Series(shap_values, index=feature_names)
    
#     original_feature_map = {}
#     for encoded_col in feature_names:
#         original_col_found = False
#         for original_col in original_features:
#             if encoded_col.startswith(original_col + '_'):
#                 original_feature_map[encoded_col] = original_col
#                 original_col_found = True
#                 break
#         if not original_col_found:
#             original_feature_map[encoded_col] = encoded_col
            
#     aggregated_shaps = shap_series.groupby(original_feature_map).sum()
#     return aggregated_shaps

# # --- Azure OpenAI LLM Integration ---
# @st.cache_data
# def get_llm_explanation(score, level, _input_data, _shap_values, _feature_names):
#     """Generates a natural language explanation for a prediction using Azure OpenAI."""
#     try:
#         client = AzureOpenAI(
#             azure_endpoint="https://advancedanalyticsopenaikey.openai.azure.com/",
#             api_key="FqFd4DBx1W97MSVjcZvdQsmQlhI80hXjl48iWYmZ4W3NutUlWvf0JQQJ99BDACYeBjFXJ3w3AAABACOGl3xo",
#             api_version="2024-02-15-preview"
#         )
        
#         aggregated_shaps = aggregate_shap_values(_shap_values[0], _feature_names, _input_data.columns)
#         top_features_series = aggregated_shaps.abs().nlargest(5)
        
#         feature_summary = ""
#         for feature_name, _ in top_features_series.items():
#             shap_val = aggregated_shaps[feature_name]
#             impact = "positively" if shap_val > 0 else "negatively"
            
#             value = _input_data[feature_name].iloc[0]
#             if isinstance(value, str):
#                 display_name = f"{feature_name} = {value}"
#             else:
#                 display_name = feature_name

#             feature_summary += f"- **{display_name}** influenced the score **{impact}**.\n"

#         prompt = f"""
#         You are an expert underwriting assistant. A machine learning model predicted a Bind Propensity Score of {score:.2f}, which is considered '{level}'.
#         The top factors influencing this prediction were:
#         {feature_summary}
#         Based on this, provide a concise, easy-to-understand "Key Drivers" summary for an underwriter in 2-3 sentences.
#         Explain WHY the submission likely received this score in business terms. Do not just list the features.
#         """

#         response = client.chat.completions.create(
#             model="gpt-4o-mini", 
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5,
#             max_tokens=200
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Could not generate AI explanation. Error: {e}"

# # --- SHAP Plot Functions ---
# def create_top_5_shap_plot(shap_values, feature_names, input_data):
#     """Creates an interactive Plotly bar chart for the top 5 aggregated SHAP features."""
#     aggregated_shaps = aggregate_shap_values(shap_values[0], feature_names, input_data.columns)
    
#     top_5_series = aggregated_shaps.abs().nlargest(5)
#     top_5_shap_series = aggregated_shaps[top_5_series.index].sort_values(ascending=True)

#     colors = ['#007bff' if val > 0 else '#dc3545' for val in top_5_shap_series.values]
    
#     renamed_index = []
#     for feature_name in top_5_shap_series.index:
#         value = input_data[feature_name].iloc[0]
#         if isinstance(value, str):
#             renamed_index.append(f"{feature_name} = {value}")
#         else:
#             renamed_index.append(feature_name)

#     fig = go.Figure(go.Bar(
#         x=top_5_shap_series.values,
#         y=renamed_index,
#         orientation='h',
#         marker_color=colors,
#         text=np.round(top_5_shap_series.values, 3),
#         textposition='auto'
#     ))

#     fig.update_layout(
#         xaxis_title="SHAP Value (Impact on Prediction)",
#         yaxis_title="Feature",
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         height=400,
#         margin=dict(l=10, r=10, t=10, b=10)
#     )
#     return fig


# # --- Data and Model Loading (Load once at the start) ---
# @st.cache_resource
# def load_models_and_explainer():
#     try:
#         lr_model = joblib.load('linear_regression_model.pkl')
#         rf_model = joblib.load('random_forest_model.pkl')
#         explainer = shap.TreeExplainer(rf_model)
#         return lr_model, rf_model, explainer
#     except FileNotFoundError:
#         st.error("Model or explainer files not found. Please ensure all necessary files are present.")
#         return None, None, None

# @st.cache_data
# def load_encoded_dataframe():
#     try:
#         df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
#         X = df.drop(columns=["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"])
#         X_encoded = pd.get_dummies(X, drop_first=True)
#         return X_encoded
#     except FileNotFoundError:
#         st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for encoding.")
#         return None

# @st.cache_data
# def load_full_dataframe():
#     try:
#         df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
        
#         if 'Expected Value' in df.columns:
#             df['Expected Value Numeric'] = df['Expected Value'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
#         if 'Total Insured Value ($)' in df.columns:
#             df['TIV_Numeric'] = df['Total Insured Value ($)'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
#         else:
#              st.error("'Total Insured Value ($)' column not found in the dataset.")
#              return None
             
#         return df
#     except FileNotFoundError:
#         st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for visualization.")
#         return None


# lr_model, rf_model, explainer = load_models_and_explainer()
# X_encoded = load_encoded_dataframe()
# df_full = load_full_dataframe()

# @st.cache_data
# def align_full_df_for_model(df_full, _X_columns):   # <-- underscore here
#     drop_cols = ["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"]
#     X_all = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns], errors="ignore")
#     X_all_enc = pd.get_dummies(X_all, drop_first=True)
#     # cast to list to be safe even outside caching
#     X_all_aligned = X_all_enc.reindex(columns=list(_X_columns), fill_value=0)
#     return X_all_aligned

# @st.cache_data
# def compute_broker_summary(_rf_model, df_full, _X_columns):  # <-- underscore here too
#     df = df_full.copy()

#     if "TIV_Numeric" not in df.columns and "Total Insured Value ($)" in df.columns:
#         df["TIV_Numeric"] = (
#             df["Total Insured Value ($)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
#         )

#     X_all_aligned = align_full_df_for_model(df, _X_columns)  # pass through
#     try:
#         df["predicted_propensity"] = _rf_model.predict(X_all_aligned).clip(0, 1)
#     except Exception:
#         if hasattr(_rf_model, "predict_proba"):
#             df["predicted_propensity"] = _rf_model.predict_proba(X_all_aligned)[:, 1]
#         else:
#             df["predicted_propensity"] = np.nan

#     if "Bind Propensity Score" not in df.columns:
#         df["Bind Propensity Score"] = df["predicted_propensity"]

#     gb = df.groupby("Broker Name", dropna=False)
#     summary = gb.agg(
#         historical_avg_propensity=("Bind Propensity Score", "mean"),
#         volume=("Broker Name", "count"),
#         avg_TIV=("TIV_Numeric", "mean"),
#     ).reset_index()

#     if "Bind_Flag" in df.columns:
#         summary = summary.merge(gb["Bind_Flag"].mean().reset_index(name="win_rate"), on="Broker Name", how="left")
#     else:
#         summary["win_rate"] = np.nan

#     summary = summary.merge(gb["predicted_propensity"].mean().reset_index(name="predicted_propensity_mean"),
#                             on="Broker Name", how="left")
#     summary = summary.merge(gb["predicted_propensity"].sum().reset_index(name="predicted_expected_wins"),
#                             on="Broker Name", how="left")

#     for c in ["historical_avg_propensity", "win_rate", "predicted_propensity_mean"]:
#         if c in summary.columns:
#             summary[c] = summary[c].astype(float)

#     summary["avg_TIV"] = summary.get("avg_TIV", 0.0).fillna(0.0)

#     return summary, df


# def human_pct(x):
#     if pd.isna(x):
#         return "—"
#     return f"{x*100:.1f}%"

# def make_kpi_triplet(col_a, col_b, col_c, title_a, val_a, title_b, val_b, title_c, val_c):
#     with col_a:
#         st.metric(title_a, val_a)
#     with col_b:
#         st.metric(title_b, val_b)
#     with col_c:
#         st.metric(title_c, val_c)


# # --- Main App Title and Logo (Displayed above tabs) ---
# col1, col2 = st.columns([3, 1])
# with col1:
#     st.title("Submission Triage Application")
# with col2:
#     try:
#         st.image("logo.png", caption="Drive Value | Drive Momentum", width=200)
#     except Exception:
#         st.warning("logo.png not found.")


# # --- Tab Definitions ---
# tab1, tab2, tab3 = st.tabs(["Bind Propensity Score Prediction", "Submissions Prioritization", "Broker Performance Insights"])



# # --- Tab 1: Bind Propensity Score Prediction ---
# with tab1:
#     if lr_model is None or rf_model is None or X_encoded is None or explainer is None:
#         st.warning("Cannot proceed with prediction due to missing files.")
#     else:
#         st.subheader("Select a Scenario to Pre-fill Form")
#         b_col1, b_col2, b_col3 = st.columns(3)
#         with b_col1: low_button = st.button("Low Bind Propensity")
#         with b_col2: medium_button = st.button("Medium Bind Propensity")
#         with b_col3: high_button = st.button("High Bind Propensity")

#         if 'scenario' not in st.session_state: st.session_state['scenario'] = None

#         if low_button:
#             st.session_state.update({
#                 'scenario': "Low", 'broker_name': "Delta Insure", 'channel': "Email", 'broker_tier': "Bronze",
#                 'industry': "Manufacturing", 'client_size': 30, 'locations': 1, 'state': "IL",
#                 'building_value': 5_000_000, 'contents_value': 1_000_000, 'bi_value': 500_000,
#                 'historical_bind_rate': 0.35, 'days_to_quote': 9, 'prior_claims': "Yes",
#                 'submission_complete': "No", 'cat_zone': "Yes"
#             })
#             st.toast("Low Bind Propensity Scenario Loaded!")
#         elif medium_button:
#             st.session_state.update({
#                 'scenario': "Medium", 'broker_name': "CoreTrust", 'channel': "Wholesaler", 'broker_tier': "Silver",
#                 'industry': "Retail", 'client_size': 120, 'locations': 3, 'state': "NY",
#                 'building_value': 40_000_000, 'contents_value': 6_000_000, 'bi_value': 2_500_000,
#                 'historical_bind_rate': 0.55, 'days_to_quote': 6, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("Medium Bind Propensity Scenario Loaded!")
#         elif high_button:
#             st.session_state.update({
#                 'scenario': "High", 'broker_name': "Alpha Risk", 'channel': "Portal", 'broker_tier': "Platinum",
#                 'industry': "Real Estate", 'client_size': 250, 'locations': 5, 'state': "TX",
#                 'building_value': 60_000_000, 'contents_value': 10_000_000, 'bi_value': 5_000_000,
#                 'historical_bind_rate': 0.80, 'days_to_quote': 3, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("High Bind Propensity Scenario Loaded!")

#         if st.session_state['scenario'] is not None:
#             st.markdown("---")
#             st.subheader("Submission & Broker Information")
#             f_col1, f_col2, f_col3, f_col4 = st.columns(4)
#             with f_col1: broker_name = st.selectbox("Broker Name", ["Alpha Risk", "Beta Cover", "CoreTrust", "FastBind", "Delta Insure"], key='broker_name')
#             with f_col2: channel = st.selectbox("Channel", ["Portal", "Email", "Wholesaler", "API"], key='channel')
#             with f_col3: broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key='broker_tier')
#             with f_col4: historical_bind_rate = st.number_input("Historical Bind Rate", 0.0, 1.0, step=0.01, key='historical_bind_rate')
            
#             st.markdown("---")
#             st.subheader("Client & Policy Information")
#             f_col1, f_col2, f_col3 = st.columns(3)
#             with f_col1:
#                 industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Warehousing", "Real Estate", "Hospitality"], key='industry')
#                 building_value = st.number_input("Building Value ($)", 1000000, 100000000, step=100000, key='building_value')
#                 days_to_quote = st.number_input("Days to Quote", 1, 30, key='days_to_quote')
#             with f_col2:
#                 client_size = st.number_input("Client Size (Revenue $M)", 10, 300, key='client_size')
#                 contents_value = st.number_input("Contents Value ($)", 100000, 20000000, step=100000, key='contents_value')
#                 prior_claims = st.selectbox("Prior Claims", ["Yes", "No"], key='prior_claims')
#             with f_col3:
#                 locations = st.number_input("Number of Locations", 1, 10, key='locations')
#                 bi_value = st.number_input("BI Value ($)", 500000, 10000000, step=50000, key='bi_value')
#                 submission_complete = st.selectbox("Submission Complete", ["Yes", "No"], key='submission_complete')
#             state_col, cat_col = st.columns(2)
#             with state_col: state = st.selectbox("State", ["TX", "FL", "NY", "CA", "IL"], key='state')
#             with cat_col: cat_zone = st.selectbox("CAT Zone", ["Yes", "No"], key='cat_zone')

#             st.markdown("---")
#             pred_col, _, clear_col = st.columns([2, 12, 2])
#             with pred_col:
#                 if st.button("Submit"):
#                     input_data = pd.DataFrame([{
#                         "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
#                         "Historical Bind Rate": historical_bind_rate, "Industry": industry,
#                         "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
#                         "Building Value ($)": building_value, "Contents Value ($)": contents_value,
#                         "BI Value ($)": bi_value, "Submission Complete": submission_complete,
#                         "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
#                     }])
#                     input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
                    
#                     rf_pred = rf_model.predict(input_df_aligned)
#                     predicted_value = rf_pred[0]
#                     TIV = building_value + contents_value + bi_value
                    
#                     shap_values = explainer.shap_values(input_df_aligned)
#                     top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)
                    
#                     level = "Low"
#                     if predicted_value > 0.65: level = "High"
#                     elif predicted_value > 0.4: level = "Medium"
                    
#                     llm_explanation = get_llm_explanation(predicted_value, level, input_data, shap_values, input_df_aligned.columns)

#                     st.session_state['prediction_results'] = {
#                         "score": predicted_value, "tiv": f"${TIV:,.0f}",
#                         "level": level, "top_5_plot": top_5_plot, "llm_explanation": llm_explanation
#                     }
#             with clear_col:
#                 if st.button("Clear"):
#                     st.session_state.pop('prediction_results', None)
#                     st.session_state['scenario'] = None
#                     st.rerun()

#             if 'prediction_results' in st.session_state and st.session_state['scenario'] is not None:
#                 st.markdown("---")
#                 st.subheader("Prediction Results")
                
#                 results = st.session_state['prediction_results']
#                 res_col1, res_col2 = st.columns(2)
#                 with res_col1:
#                     st.plotly_chart(create_gauge_chart(results['score'], results['level']), use_container_width=True)
#                 with res_col2:
#                     st.metric("Total Insured Value (TIV)", results['tiv'])
                
#                 st.markdown("---")
#                 st.subheader("Model Prediction Explainability")
                
#                 exp_col1, exp_col2 = st.columns(2)
#                 with exp_col1:
#                     st.write("#### Top 5 SHAP Features Influencing Prediction")
#                     st.plotly_chart(results['top_5_plot'], use_container_width=True)
#                 with exp_col2:
#                     st.markdown(f"""
#                     <div style="border: 1px solid #0055a4; border-radius: 10px; padding: 15px; background-color: #f0f8ff; height: 100%;">
#                         <h4 style="color: #004080; margin-bottom: 10px;">Key Drivers</h4>
#                         <p style="color: #333;">{results['llm_explanation']}</p>
#                     </div>
#                     """, unsafe_allow_html=True)

# # --- Tab 2: Submissions Prioritization using Strike Zone ---
# with tab2:
#     st.header("Submissions Prioritization using Strike Zone")
#     if df_full is not None and 'TIV_Numeric' in df_full.columns:
#         plot_df = df_full[df_full['TIV_Numeric'] > 0].copy()

#         if not plot_df.empty:
#             x_data = plot_df['Bind Propensity Score']
#             y_data = plot_df['TIV_Numeric']

#             min_tiv = int(y_data.min())
#             max_tiv = int(y_data.max())

#             values_x = st.slider("Select a Bind Propensity range for Strike Zone", 0.2, 1.0, (0.5, 1.0), key="slider_x_final")
#             values_y = st.slider("Select a Total Insured Value ($) range for Strike Zone", min_tiv, max_tiv, (int(max_tiv * 0.25), max_tiv), step=1000000, key="slider_y_final", format="$%d")

#             fig = go.Figure()

#             # Add the scatter plot with color scale
#             fig.add_trace(go.Scatter(
#                 x=x_data, 
#                 y=y_data, 
#                 mode='markers',
#                 marker=dict(
#                     size=8,
#                     color=y_data, # Color points by TIV
#                     colorscale='Viridis', # A nice color scale
#                     showscale=True,
#                     colorbar=dict(title="TIV ($)"),
#                     opacity=0.7
#                 ),
#                 text=plot_df.apply(lambda row: f"Broker: {row['Broker Name']}<br>TIV: ${row['TIV_Numeric']:,.0f}", axis=1),
#                 hoverinfo='text+x+y'
#             ))

#             # Add the strike zone rectangle on top
#             fig.add_shape(type="rect",
#                           x0=values_x[0], x1=values_x[1], y0=values_y[0], y1=values_y[1],
#                           line=dict(color="red", width=2),
#                           fillcolor="rgba(0,0,0,0)", # transparent fill
#                           layer="above")
            
#             fig.update_layout(
#                 title="Submissions Segmented by Bind Propensity and Total Insured Value",
#                 xaxis_title="Bind Propensity Score",
#                 yaxis_title="Total Insured Value ($) (Log Scale)",
#                 yaxis_type="log",
#                 xaxis_range=[0.2, 1.0],
#                 showlegend=False,
#                 height=600,
#                 template="plotly_white"
#             )
#             st.plotly_chart(fig, use_container_width=True)
#             st.divider()
#             st.write("### Simulate the Impact of Submission Prioritization")
            
#             strike_zone_df = plot_df[
#                 (plot_df['Bind Propensity Score'].between(values_x[0], values_x[1])) &
#                 (plot_df['TIV_Numeric'].between(values_y[0], values_y[1]))
#             ]

#             st.write(f"There are **{len(strike_zone_df)}** submissions in the selected Strike Zone.")

#             if 'Expected Value Numeric' in strike_zone_df.columns:
#                 if st.button("Run Simulation", key="sim_button_final"):
#                     if len(strike_zone_df) >= 10:
#                         sampled_df = strike_zone_df.sample(10)
#                         display_cols = ['Submission ID', 'Broker Name', 'Industry', 'Bind Propensity Score', 'Expected Value Numeric']
#                         st.write("Randomly selected 10 submissions from the Strike Zone:")
#                         st.dataframe(sampled_df[display_cols].rename(columns={'Expected Value Numeric': 'Expected Value ($)'}))
#                         total_expected_value = sampled_df['Expected Value Numeric'].sum()
#                         # st.subheader(f"Total Expected Prospect Value from Sample: ${total_expected_value:,.2f}")
#                     else:
#                         st.warning("Not enough submissions in the Strike Zone to sample 10. Please expand the ranges using the sliders.")
#             else:
#                 st.warning("Simulation requires the 'Expected Value' column in the data.")
#         else:
#             st.warning("No data with positive Total Insured Value to display on the chart.")
#     else:
#         st.warning("Data for visualization could not be loaded. Please check the CSV file.")

# # --- Tab 3: Broker Performance Insights ---
# # --- Tab 3: Broker Performance Insights ---
# # --- Tab 3: Broker Performance Insights ---
# with tab3:
#     st.header("📊 Broker Performance Insights")

#     if df_full is None or X_encoded is None or rf_model is None:
#         st.warning("Insights unavailable. Ensure the dataset and model are loaded.")
#     else:
#         # Compute summaries and the scored dataframe once
#         summary, df_scored = compute_broker_summary(rf_model, df_full, X_encoded.columns)

#         # Broker filter only
#         brokers = sorted(summary["Broker Name"].dropna().unique().tolist())
#         selected_brokers = st.multiselect(
#             "Filter Brokers",
#             options=brokers,
#             default=brokers[: min(6, len(brokers))]
#         )

#         # Apply broker filter
#         summary_f = summary[summary["Broker Name"].isin(selected_brokers)].copy() if selected_brokers else summary.copy()

#         # --- KPI Top Row (overall across selected brokers) ---
#         st.subheader("Key Performance Indicators")

#         overall = {
#             "avg_propensity": summary_f["predicted_propensity_mean"].mean(),
#             "win_rate": summary_f["win_rate"].mean(skipna=True),
#             "volume": int(summary_f["volume"].sum()),
#             "predicted_expected_wins": summary_f["predicted_expected_wins"].sum(),
#         }

#         k1, k2, k3 = st.columns(3)
#         make_kpi_triplet(
#             k1, k2, k3,
#             "Avg Bind Propensity", human_pct(overall["avg_propensity"]),
#             "Historical Win Rate", human_pct(overall["win_rate"]),
#             "Total Submissions", f"{overall['volume']:,}"
#         )

#         k4, _ = st.columns([3, 9])
#         with k4:
#             st.metric("Predicted Wins (Expected)", f"{overall['predicted_expected_wins']:.1f}")

#         st.markdown("---")

#         # --- Chart 1: Volume vs Predicted Wins (bar + line) ---
#         st.write("#### Volume and Predicted Wins by Broker")
#         vol_df = summary_f.sort_values("volume", ascending=False)
#         fig_v = go.Figure()
#         fig_v.add_trace(go.Bar(
#             x=vol_df["Broker Name"],
#             y=vol_df["volume"],
#             name="Submissions (Volume)",
#             opacity=0.85
#         ))
#         fig_v.add_trace(go.Scatter(
#             x=vol_df["Broker Name"],
#             y=vol_df["predicted_expected_wins"],
#             name="Predicted Wins (Expected)",
#             mode="lines+markers",
#             yaxis="y2"
#         ))
#         fig_v.update_layout(
#             height=450,
#             template="plotly_white",
#             xaxis=dict(title="Broker"),
#             yaxis=dict(title="Volume"),
#             yaxis2=dict(title="Predicted Wins", overlaying="y", side="right"),
#             legend=dict(orientation="h")
#         )
#         st.plotly_chart(fig_v, use_container_width=True)

#         # --- Chart 2: Propensity vs Win Rate bubble (size = volume, color = avg propensity) ---
#         st.write("#### Propensity vs. Win Rate (bubble size = volume, color = avg propensity)")
#         pr_col = "predicted_propensity_mean"
#         bub = summary_f.dropna(subset=[pr_col, "win_rate"])
#         fig_b = go.Figure()
#         fig_b.add_trace(go.Scatter(
#             x=bub[pr_col],
#             y=bub["win_rate"],
#             mode="markers+text",
#             text=bub["Broker Name"],
#             textposition="top center",
#             marker=dict(
#                 size=np.clip(np.sqrt(bub["volume"]) * 6, 8, 50),
#                 color=bub[pr_col],
#                 showscale=True,
#                 colorbar=dict(title="Avg Propensity"),
#                 opacity=0.85
#             )
#         ))
#         fig_b.update_layout(
#             height=500,
#             template="plotly_white",
#             xaxis=dict(title="Avg Bind Propensity (Predicted)"),
#             yaxis=dict(title="Historical Win Rate"),
#         )
#         st.plotly_chart(fig_b, use_container_width=True)

#         # --- Chart 3: Avg Propensity by Broker (horizontal bar) ---
#         st.write("#### Avg Propensity by Broker")
#         pdf = summary_f.sort_values(pr_col, ascending=True)
#         fig_p = go.Figure(go.Bar(
#             x=pdf[pr_col],
#             y=pdf["Broker Name"],
#             orientation="h",
#             text=[human_pct(x) for x in pdf[pr_col]],
#             textposition="auto"
#         ))
#         fig_p.update_layout(
#             height=500,
#             template="plotly_white",
#             xaxis=dict(title="Avg Bind Propensity"),
#             yaxis=dict(title="Broker"),
#             margin=dict(l=10, r=10, t=10, b=10)
#         )
#         st.plotly_chart(fig_p, use_container_width=True)

#         # --- Optional: Monthly trend if a date exists ---
#         date_cols = [c for c in df_full.columns if "date" in c.lower()]
#         if date_cols:
#             date_col = date_cols[0]
#             try:
#                 temp = df_scored.copy()
#                 temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
#                 temp = temp.dropna(subset=[date_col])
#                 temp["YYYY-MM"] = temp[date_col].dt.to_period("M").dt.to_timestamp()

#                 st.write("#### Volume Trend (Monthly)")
#                 trend = temp.groupby("YYYY-MM").size().reset_index(name="count")
#                 fig_t = go.Figure(go.Scatter(
#                     x=trend["YYYY-MM"], y=trend["count"], mode="lines+markers", name="Monthly Volume"
#                 ))
#                 fig_t.update_layout(
#                     height=380,
#                     template="plotly_white",
#                     xaxis_title="Month",
#                     yaxis_title="Submissions"
#                 )
#                 st.plotly_chart(fig_t, use_container_width=True)
#             except Exception:
#                 pass





















###########with fixes
# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import shap
# import matplotlib.pyplot as plt
# from openai import AzureOpenAI

# # --- Page Configuration and Styling (Apply globally) ---
# st.set_page_config(layout="wide")

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


# # --- Helper Function for Gauge Chart ---
# def create_gauge_chart(score, level):
#     """Creates a more polished and eye-catching Plotly gauge chart."""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=score,
#         domain={'x': [0.1, 0.9], 'y': [0, 0.9]},
#         title={'text': f"<b>Bind Propensity Score</b><br><span style='font-size:1.2em;color:grey'>{level}</span>", 'font': {'size': 24}},
#         number={'valueformat': '.2f'},
#         gauge={
#             'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkslategray"},
#             'bar': {'color': "darkslategray", 'thickness': 0.4},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "#E0E0E0",
#             'steps': [
#                 {'range': [0, 0.4], 'color': '#FF7979'},
#                 {'range': [0.4, 0.65], 'color': '#FFC312'},
#                 {'range': [0.65, 1], 'color': '#009432'}
#             ],
#         }))
        
#     fig.update_layout(
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         font={'color': "darkslategray", 'family': "Arial, Helvetica, sans-serif"},
#         height=350,
#         margin=dict(l=20, r=20, b=20, t=50)
#     )
#     return fig

# # --- Business-Friendly Feature Aggregation and Formatting ---
# def aggregate_shap_values(shap_values, feature_names, original_features):
#     """Aggregates SHAP values for one-hot encoded features back to their original parent feature."""
#     shap_series = pd.Series(shap_values, index=feature_names)
    
#     original_feature_map = {}
#     for encoded_col in feature_names:
#         original_col_found = False
#         for original_col in original_features:
#             if encoded_col.startswith(original_col + '_'):
#                 original_feature_map[encoded_col] = original_col
#                 original_col_found = True
#                 break
#         if not original_col_found:
#             original_feature_map[encoded_col] = encoded_col
            
#     aggregated_shaps = shap_series.groupby(original_feature_map).sum()
#     return aggregated_shaps

# # --- Azure OpenAI LLM Integration ---
# @st.cache_data(show_spinner=False)
# def get_llm_explanation(score, level, _input_data, _shap_values, _feature_names):
#     """Generates a natural language explanation for a prediction using Azure OpenAI."""
#     try:
#         client = AzureOpenAI(
#             azure_endpoint="https://advancedanalyticsopenaikey.openai.azure.com/",
#             api_key="FqFd4DBx1W97MSVjcZvdQsmQlhI80hXjl48iWYmZ4W3NutUlWvf0JQQJ99BDACYeBjFXJ3w3AAABACOGl3xo",
#             api_version="2024-02-15-preview"
#         )
        
#         aggregated_shaps = aggregate_shap_values(_shap_values[0], _feature_names, _input_data.columns)
#         top_features_series = aggregated_shaps.abs().nlargest(5)
        
#         feature_summary = ""
#         for feature_name, _ in top_features_series.items():
#             shap_val = aggregated_shaps[feature_name]
#             impact = "positively" if shap_val > 0 else "negatively"
            
#             value = _input_data[feature_name].iloc[0]
#             if isinstance(value, str):
#                 display_name = f"{feature_name} = {value}"
#             else:
#                 display_name = feature_name

#             feature_summary += f"- **{display_name}** influenced the score **{impact}**.\n"

#         prompt = f"""
#         You are an expert underwriting assistant. A machine learning model predicted a Bind Propensity Score of {score:.2f}, which is considered '{level}'.
#         The top factors influencing this prediction were:
#         {feature_summary}
#         Based on this, provide a concise, easy-to-understand "Key Drivers" summary for an underwriter in 2-3 sentences.
#         Explain WHY the submission likely received this score in business terms. Do not just list the features.
#         """

#         response = client.chat.completions.create(
#             model="gpt-4o-mini", 
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5,
#             max_tokens=200
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Could not generate AI explanation. Error: {e}"

# # --- SHAP Plot Functions ---
# def create_top_5_shap_plot(shap_values, feature_names, input_data):
#     """Creates an interactive Plotly bar chart for the top 5 aggregated SHAP features."""
#     aggregated_shaps = aggregate_shap_values(shap_values[0], feature_names, input_data.columns)
    
#     top_5_series = aggregated_shaps.abs().nlargest(5)
#     top_5_shap_series = aggregated_shaps[top_5_series.index].sort_values(ascending=True)

#     colors = ['#007bff' if val > 0 else '#dc3545' for val in top_5_shap_series.values]
    
#     renamed_index = []
#     for feature_name in top_5_shap_series.index:
#         value = input_data[feature_name].iloc[0]
#         if isinstance(value, str):
#             renamed_index.append(f"{feature_name} = {value}")
#         else:
#             renamed_index.append(feature_name)

#     fig = go.Figure(go.Bar(
#         x=top_5_shap_series.values,
#         y=renamed_index,
#         orientation='h',
#         marker_color=colors,
#         text=np.round(top_5_shap_series.values, 3),
#         textposition='auto'
#     ))

#     fig.update_layout(
#         xaxis_title="SHAP Value (Impact on Prediction)",
#         yaxis_title="Feature",
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         height=400,
#         margin=dict(l=10, r=10, t=10, b=10)
#     )
#     return fig


# # --- Data and Model Loading (Load once at the start) ---
# @st.cache_resource
# def load_models_and_explainer():
#     try:
#         lr_model = joblib.load('linear_regression_model.pkl')
#         rf_model = joblib.load('random_forest_model.pkl')
#         explainer = shap.TreeExplainer(rf_model)
#         return lr_model, rf_model, explainer
#     except FileNotFoundError:
#         st.error("Model or explainer files not found. Please ensure all necessary files are present.")
#         return None, None, None

# @st.cache_data
# def load_encoded_dataframe():
#     try:
#         df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
#         X = df.drop(columns=["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"])
#         X_encoded = pd.get_dummies(X, drop_first=True)
#         return X_encoded
#     except FileNotFoundError:
#         st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for encoding.")
#         return None

# @st.cache_data
# def load_full_dataframe():
#     try:
#         df = pd.read_csv('Triaging_Data_Expanded_Complete.csv')
        
#         if 'Expected Value' in df.columns:
#             df['Expected Value Numeric'] = df['Expected Value'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
#         if 'Total Insured Value ($)' in df.columns:
#             df['TIV_Numeric'] = df['Total Insured Value ($)'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
#         else:
#              st.error("'Total Insured Value ($)' column not found in the dataset.")
#              return None
             
#         return df
#     except FileNotFoundError:
#         st.error("Dataset 'Triaging_Data_Expanded_Complete.csv' not found for visualization.")
#         return None
# # --- Add this helper near your other cached helpers (top of file) ---
# @st.cache_data
# def compute_shap_plot_cached(_explainer, _aligned_df, _input_df):
#     shap_values = _explainer.shap_values(_aligned_df)
#     fig = create_top_5_shap_plot(shap_values, _aligned_df.columns, _input_df)
#     return fig, shap_values


# lr_model, rf_model, explainer = load_models_and_explainer()
# X_encoded = load_encoded_dataframe()
# df_full = load_full_dataframe()

# @st.cache_data
# def align_full_df_for_model(df_full, _X_columns):   # <-- underscore here
#     drop_cols = ["Submission ID", "Bind Propensity Score", "Bind_Flag", "Total Insured Value ($)", "Expected Value"]
#     X_all = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns], errors="ignore")
#     X_all_enc = pd.get_dummies(X_all, drop_first=True)
#     # cast to list to be safe even outside caching
#     X_all_aligned = X_all_enc.reindex(columns=list(_X_columns), fill_value=0)
#     return X_all_aligned

# @st.cache_data
# def compute_broker_summary(_rf_model, df_full, _X_columns):  # <-- underscore here too
#     df = df_full.copy()

#     if "TIV_Numeric" not in df.columns and "Total Insured Value ($)" in df.columns:
#         df["TIV_Numeric"] = (
#             df["Total Insured Value ($)"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
#         )

#     X_all_aligned = align_full_df_for_model(df, _X_columns)  # pass through
#     try:
#         df["predicted_propensity"] = _rf_model.predict(X_all_aligned).clip(0, 1)
#     except Exception:
#         if hasattr(_rf_model, "predict_proba"):
#             df["predicted_propensity"] = _rf_model.predict_proba(X_all_aligned)[:, 1]
#         else:
#             df["predicted_propensity"] = np.nan

#     if "Bind Propensity Score" not in df.columns:
#         df["Bind Propensity Score"] = df["predicted_propensity"]

#     gb = df.groupby("Broker Name", dropna=False)
#     summary = gb.agg(
#         historical_avg_propensity=("Bind Propensity Score", "mean"),
#         volume=("Broker Name", "count"),
#         avg_TIV=("TIV_Numeric", "mean"),
#     ).reset_index()

#     if "Bind_Flag" in df.columns:
#         summary = summary.merge(gb["Bind_Flag"].mean().reset_index(name="win_rate"), on="Broker Name", how="left")
#     else:
#         summary["win_rate"] = np.nan

#     summary = summary.merge(gb["predicted_propensity"].mean().reset_index(name="predicted_propensity_mean"),
#                             on="Broker Name", how="left")
#     summary = summary.merge(gb["predicted_propensity"].sum().reset_index(name="predicted_expected_wins"),
#                             on="Broker Name", how="left")

#     for c in ["historical_avg_propensity", "win_rate", "predicted_propensity_mean"]:
#         if c in summary.columns:
#             summary[c] = summary[c].astype(float)

#     summary["avg_TIV"] = summary.get("avg_TIV", 0.0).fillna(0.0)

#     return summary, df


# def human_pct(x):
#     if pd.isna(x):
#         return "—"
#     return f"{x*100:.1f}%"

# def make_kpi_triplet(col_a, col_b, col_c, title_a, val_a, title_b, val_b, title_c, val_c):
#     with col_a:
#         st.metric(title_a, val_a)
#     with col_b:
#         st.metric(title_b, val_b)
#     with col_c:
#         st.metric(title_c, val_c)


# # --- Main App Title and Logo (Displayed above tabs) ---
# col1, col2 = st.columns([3, 1])
# with col1:
#     st.title("Submission Triage Application")
# with col2:
#     try:
#         st.image("logo.png", caption="Drive Value | Drive Momentum", width=200)
#     except Exception:
#         st.warning("logo.png not found.")


# # --- Tab Definitions ---
# tab1, tab2, tab3 = st.tabs(["Bind Propensity Score Prediction", "Submissions Prioritization", "Broker Performance Insights"])



# # # # --- Tab 1: Bind Propensity Score Prediction ---
# # with tab1:
# #     if lr_model is None or rf_model is None or X_encoded is None or explainer is None:
# #         st.warning("Cannot proceed with prediction due to missing files.")
# #     else:
# #         st.subheader("Select a Scenario to Pre-fill Form")
# #         b_col1, b_col2, b_col3 = st.columns(3)
# #         with b_col1: low_button = st.button("Low Bind Propensity")
# #         with b_col2: medium_button = st.button("Medium Bind Propensity")
# #         with b_col3: high_button = st.button("High Bind Propensity")

# #         if 'scenario' not in st.session_state: st.session_state['scenario'] = None

# #         if low_button:
# #             st.session_state.update({
# #                 'scenario': "Low", 'broker_name': "Delta Insure", 'channel': "Email", 'broker_tier': "Bronze",
# #                 'industry': "Manufacturing", 'client_size': 30, 'locations': 1, 'state': "IL",
# #                 'building_value': 5_000_000, 'contents_value': 1_000_000, 'bi_value': 500_000,
# #                 'historical_bind_rate': 0.35, 'days_to_quote': 9, 'prior_claims': "Yes",
# #                 'submission_complete': "No", 'cat_zone': "Yes"
# #             })
# #             st.toast("Low Bind Propensity Scenario Loaded!")
# #         elif medium_button:
# #             st.session_state.update({
# #                 'scenario': "Medium", 'broker_name': "CoreTrust", 'channel': "Wholesaler", 'broker_tier': "Silver",
# #                 'industry': "Retail", 'client_size': 120, 'locations': 3, 'state': "NY",
# #                 'building_value': 40_000_000, 'contents_value': 6_000_000, 'bi_value': 2_500_000,
# #                 'historical_bind_rate': 0.55, 'days_to_quote': 6, 'prior_claims': "No",
# #                 'submission_complete': "Yes", 'cat_zone': "No"
# #             })
# #             st.toast("Medium Bind Propensity Scenario Loaded!")
# #         elif high_button:
# #             st.session_state.update({
# #                 'scenario': "High", 'broker_name': "Alpha Risk", 'channel': "Portal", 'broker_tier': "Platinum",
# #                 'industry': "Real Estate", 'client_size': 250, 'locations': 5, 'state': "TX",
# #                 'building_value': 60_000_000, 'contents_value': 10_000_000, 'bi_value': 5_000_000,
# #                 'historical_bind_rate': 0.80, 'days_to_quote': 3, 'prior_claims': "No",
# #                 'submission_complete': "Yes", 'cat_zone': "No"
# #             })
# #             st.toast("High Bind Propensity Scenario Loaded!")

# #         if st.session_state['scenario'] is not None:
# #             st.markdown("---")
# #             st.subheader("Submission & Broker Information")
# #             f_col1, f_col2, f_col3, f_col4 = st.columns(4)
# #             with f_col1: broker_name = st.selectbox("Broker Name", ["Alpha Risk", "Beta Cover", "CoreTrust", "FastBind", "Delta Insure"], key='broker_name')
# #             with f_col2: channel = st.selectbox("Channel", ["Portal", "Email", "Wholesaler", "API"], key='channel')
# #             with f_col3: broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key='broker_tier')
# #             with f_col4: historical_bind_rate = st.number_input("Historical Bind Rate", 0.0, 1.0, step=0.01, key='historical_bind_rate')
            
# #             st.markdown("---")
# #             st.subheader("Client & Policy Information")
# #             f_col1, f_col2, f_col3 = st.columns(3)
# #             with f_col1:
# #                 industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Warehousing", "Real Estate", "Hospitality"], key='industry')
# #                 building_value = st.number_input("Building Value ($)", 1000000, 100000000, step=100000, key='building_value')
# #                 days_to_quote = st.number_input("Days to Quote", 1, 30, key='days_to_quote')
# #             with f_col2:
# #                 client_size = st.number_input("Client Size (Revenue $M)", 10, 300, key='client_size')
# #                 contents_value = st.number_input("Contents Value ($)", 100000, 20000000, step=100000, key='contents_value')
# #                 prior_claims = st.selectbox("Prior Claims", ["Yes", "No"], key='prior_claims')
# #             with f_col3:
# #                 locations = st.number_input("Number of Locations", 1, 10, key='locations')
# #                 bi_value = st.number_input("BI Value ($)", 500000, 10000000, step=50000, key='bi_value')
# #                 submission_complete = st.selectbox("Submission Complete", ["Yes", "No"], key='submission_complete')
# #             state_col, cat_col = st.columns(2)
# #             with state_col: state = st.selectbox("State", ["TX", "FL", "NY", "CA", "IL"], key='state')
# #             with cat_col: cat_zone = st.selectbox("CAT Zone", ["Yes", "No"], key='cat_zone')

# #             st.markdown("---")
# #             pred_col, _, clear_col = st.columns([2, 12, 2])
# #             with pred_col:
# #                 if st.button("Submit"):
# #                     input_data = pd.DataFrame([{
# #                         "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
# #                         "Historical Bind Rate": historical_bind_rate, "Industry": industry,
# #                         "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
# #                         "Building Value ($)": building_value, "Contents Value ($)": contents_value,
# #                         "BI Value ($)": bi_value, "Submission Complete": submission_complete,
# #                         "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
# #                     }])
# #                     input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
                    
# #                     rf_pred = rf_model.predict(input_df_aligned)
# #                     predicted_value = rf_pred[0]
# #                     TIV = building_value + contents_value + bi_value
                    
# #                     shap_values = explainer.shap_values(input_df_aligned)
# #                     top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)
                    
# #                     level = "Low"
# #                     if predicted_value > 0.65: level = "High"
# #                     elif predicted_value > 0.4: level = "Medium"
                    
# #                     llm_explanation = get_llm_explanation(predicted_value, level, input_data, shap_values, input_df_aligned.columns)

# #                     st.session_state['prediction_results'] = {
# #                         "score": predicted_value, "tiv": f"${TIV:,.0f}",
# #                         "level": level, "top_5_plot": top_5_plot, "llm_explanation": llm_explanation
# #                     }
# #             with clear_col:
# #                 if st.button("Clear"):
# #                     st.session_state.pop('prediction_results', None)
# #                     st.session_state['scenario'] = None
# #                     st.rerun()

# #             if 'prediction_results' in st.session_state and st.session_state['scenario'] is not None:
# #                 st.markdown("---")
# #                 st.subheader("Prediction Results")
                
# #                 results = st.session_state['prediction_results']
# #                 res_col1, res_col2 = st.columns(2)
# #                 with res_col1:
# #                     st.plotly_chart(create_gauge_chart(results['score'], results['level']), use_container_width=True)
# #                 with res_col2:
# #                     st.metric("Total Insured Value (TIV)", results['tiv'])
                
# #                 st.markdown("---")
# #                 st.subheader("Model Prediction Explainability")
                
# #                 exp_col1, exp_col2 = st.columns(2)
# #                 with exp_col1:
# #                     st.write("#### Top 5 SHAP Features Influencing Prediction")
# #                     st.plotly_chart(results['top_5_plot'], use_container_width=True)
# #                 with exp_col2:
# #                     st.markdown(f"""
# #                     <div style="border: 1px solid #0055a4; border-radius: 10px; padding: 15px; background-color: #f0f8ff; height: 100%;">
# #                         <h4 style="color: #004080; margin-bottom: 10px;">Key Drivers</h4>
# #                         <p style="color: #333;">{results['llm_explanation']}</p>
# #                     </div>
# #                     """, unsafe_allow_html=True)

# with tab1:
#     if lr_model is None or rf_model is None or X_encoded is None or explainer is None:
#         st.warning("Cannot proceed with prediction due to missing files.")
#     else:
#         st.subheader("Select a Scenario to Pre-fill Form")
#         b_col1, b_col2, b_col3 = st.columns(3)
#         with b_col1: low_button = st.button("Low Bind Propensity")
#         with b_col2: medium_button = st.button("Medium Bind Propensity")
#         with b_col3: high_button = st.button("High Bind Propensity")

#         if 'scenario' not in st.session_state: st.session_state['scenario'] = None

#         if low_button:
#             # FIX: Clear previous results when a new scenario is selected
#             st.session_state.pop('prediction_results', None)
#             st.session_state.update({
#                 'scenario': "Low", 'broker_name': "Delta Insure", 'channel': "Email", 'broker_tier': "Bronze",
#                 'industry': "Manufacturing", 'client_size': 30, 'locations': 1, 'state': "IL",
#                 'building_value': 5_000_000, 'contents_value': 1_000_000, 'bi_value': 500_000,
#                 'historical_bind_rate': 0.35, 'days_to_quote': 9, 'prior_claims': "Yes",
#                 'submission_complete': "No", 'cat_zone': "Yes"
#             })
#             st.toast("Low Bind Propensity Scenario Loaded!")
#         elif medium_button:
#             # FIX: Clear previous results when a new scenario is selected
#             st.session_state.pop('prediction_results', None)
#             st.session_state.update({
#                 'scenario': "Medium", 'broker_name': "CoreTrust", 'channel': "Wholesaler", 'broker_tier': "Silver",
#                 'industry': "Retail", 'client_size': 120, 'locations': 3, 'state': "NY",
#                 'building_value': 40_000_000, 'contents_value': 6_000_000, 'bi_value': 2_500_000,
#                 'historical_bind_rate': 0.55, 'days_to_quote': 6, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("Medium Bind Propensity Scenario Loaded!")
#         elif high_button:
#             # FIX: Clear previous results when a new scenario is selected
#             st.session_state.pop('prediction_results', None)
#             st.session_state.update({
#                 'scenario': "High", 'broker_name': "Alpha Risk", 'channel': "Portal", 'broker_tier': "Platinum",
#                 'industry': "Real Estate", 'client_size': 250, 'locations': 5, 'state': "TX",
#                 'building_value': 60_000_000, 'contents_value': 10_000_000, 'bi_value': 5_000_000,
#                 'historical_bind_rate': 0.80, 'days_to_quote': 3, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("High Bind Propensity Scenario Loaded!")

#         if st.session_state['scenario'] is not None:
#             st.markdown("---")
#             st.subheader("Submission & Broker Information")
#             f_col1, f_col2, f_col3, f_col4 = st.columns(4)
#             with f_col1: broker_name = st.selectbox("Broker Name", ["Alpha Risk", "Beta Cover", "CoreTrust", "FastBind", "Delta Insure"], key='broker_name')
#             with f_col2: channel = st.selectbox("Channel", ["Portal", "Email", "Wholesaler", "API"], key='channel')
#             with f_col3: broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key='broker_tier')
#             with f_col4: historical_bind_rate = st.number_input("Historical Bind Rate", 0.0, 1.0, step=0.01, key='historical_bind_rate')
            
#             st.markdown("---")
#             st.subheader("Client & Policy Information")
#             f_col1, f_col2, f_col3 = st.columns(3)
#             with f_col1:
#                 industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Warehousing", "Real Estate", "Hospitality"], key='industry')
#                 building_value = st.number_input("Building Value ($)", 1000000, 100000000, step=100000, key='building_value')
#                 days_to_quote = st.number_input("Days to Quote", 1, 30, key='days_to_quote')
#             with f_col2:
#                 client_size = st.number_input("Client Size (Revenue $M)", 10, 300, key='client_size')
#                 contents_value = st.number_input("Contents Value ($)", 100000, 20000000, step=100000, key='contents_value')
#                 prior_claims = st.selectbox("Prior Claims", ["Yes", "No"], key='prior_claims')
#             with f_col3:
#                 locations = st.number_input("Number of Locations", 1, 10, key='locations')
#                 bi_value = st.number_input("BI Value ($)", 500000, 10000000, step=50000, key='bi_value')
#                 submission_complete = st.selectbox("Submission Complete", ["Yes", "No"], key='submission_complete')
#             state_col, cat_col = st.columns(2)
#             with state_col: state = st.selectbox("State", ["TX", "FL", "NY", "CA", "IL"], key='state')
#             with cat_col: cat_zone = st.selectbox("CAT Zone", ["Yes", "No"], key='cat_zone')

#             st.markdown("---")
#             pred_col, _, clear_col = st.columns([2, 12, 2])
#             with pred_col:
#                 if st.button("Submit"):
#                     input_data = pd.DataFrame([{
#                         "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
#                         "Historical Bind Rate": historical_bind_rate, "Industry": industry,
#                         "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
#                         "Building Value ($)": building_value, "Contents Value ($)": contents_value,
#                         "BI Value ($)": bi_value, "Submission Complete": submission_complete,
#                         "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
#                     }])
#                     input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
                    
#                     rf_pred = rf_model.predict(input_df_aligned)
#                     predicted_value = rf_pred[0]
#                     TIV = building_value + contents_value + bi_value
                    
#                     shap_values = explainer.shap_values(input_df_aligned)
#                     top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)
                    
#                     level = "Low"
#                     if predicted_value > 0.65: level = "High"
#                     elif predicted_value > 0.4: level = "Medium"
                    
#                     llm_explanation = get_llm_explanation(predicted_value, level, input_data, shap_values, input_df_aligned.columns)

#                     st.session_state['prediction_results'] = {
#                         "score": predicted_value, "tiv": f"${TIV:,.0f}",
#                         "level": level, "top_5_plot": top_5_plot, "llm_explanation": llm_explanation
#                     }
#             with clear_col:
#                 if st.button("Clear"):
#                     st.session_state.pop('prediction_results', None)
#                     st.session_state['scenario'] = None
#                     st.rerun()

#             if 'prediction_results' in st.session_state and st.session_state['scenario'] is not None:
#                 st.markdown("---")
#                 st.subheader("Prediction Results")
                
#                 results = st.session_state['prediction_results']
#                 res_col1, res_col2 = st.columns(2)
#                 with res_col1:
#                     st.plotly_chart(create_gauge_chart(results['score'], results['level']), use_container_width=True)
#                 with res_col2:
#                     st.metric("Total Insured Value (TIV)", results['tiv'])
                
#                 st.markdown("---")
#                 st.subheader("Model Prediction Explainability")
                
#                 exp_col1, exp_col2 = st.columns(2)
#                 with exp_col1:
#                     st.write("#### Top 5 SHAP Features Influencing Prediction")
#                     st.plotly_chart(results['top_5_plot'], use_container_width=True)
#                 with exp_col2:
#                     st.markdown(f"""
#                     <div style="border: 1px solid #0055a4; border-radius: 10px; padding: 15px; background-color: #f0f8ff; height: 100%;">
#                         <h4 style="color: #004080; margin-bottom: 10px;">Key Drivers</h4>
#                         <p style="color: #333;">{results['llm_explanation']}</p>
#                     </div>
#                     """, unsafe_allow_html=True)



# # --- Tab 2: Submissions Prioritization using Strike Zone ---
# with tab2:
#     st.header("Submissions Prioritization using Strike Zone")
#     if df_full is not None and 'TIV_Numeric' in df_full.columns:
#         plot_df = df_full[df_full['TIV_Numeric'] > 0].copy()

#         if not plot_df.empty:
#             x_data = plot_df['Bind Propensity Score']
#             y_data = plot_df['TIV_Numeric']

#             min_tiv = int(y_data.min())
#             max_tiv = int(y_data.max())

#             values_x = st.slider("Select a Bind Propensity range for Strike Zone", 0.2, 1.0, (0.5, 1.0), key="slider_x_final")
#             values_y = st.slider("Select a Total Insured Value ($) range for Strike Zone", min_tiv, max_tiv, (int(max_tiv * 0.25), max_tiv), step=1000000, key="slider_y_final", format="$%d")

#             fig = go.Figure()

#             # Add the scatter plot with color scale
#             fig.add_trace(go.Scatter(
#                 x=x_data, 
#                 y=y_data, 
#                 mode='markers',
#                 marker=dict(
#                     size=8,
#                     color=y_data, # Color points by TIV
#                     colorscale='Viridis', # A nice color scale
#                     showscale=True,
#                     colorbar=dict(title="TIV ($)"),
#                     opacity=0.7
#                 ),
#                 text=plot_df.apply(lambda row: f"Broker: {row['Broker Name']}<br>TIV: ${row['TIV_Numeric']:,.0f}", axis=1),
#                 hoverinfo='text+x+y'
#             ))

#             # Add the strike zone rectangle on top
#             fig.add_shape(type="rect",
#                           x0=values_x[0], x1=values_x[1], y0=values_y[0], y1=values_y[1],
#                           line=dict(color="red", width=2),
#                           fillcolor="rgba(0,0,0,0)", # transparent fill
#                           layer="above")
            
#             fig.update_layout(
#                 title="Submissions Segmented by Bind Propensity and Total Insured Value",
#                 xaxis_title="Bind Propensity Score",
#                 yaxis_title="Total Insured Value ($) (Log Scale)",
#                 yaxis_type="log",
#                 xaxis_range=[0.2, 1.0],
#                 showlegend=False,
#                 height=600,
#                 template="plotly_white"
#             )
#             st.plotly_chart(fig, use_container_width=True)
#             st.divider()
            
#             st.write("### Simulate the Impact of Submission Prioritization")

#             # Filter to strike-zone
#             strike_zone_df = plot_df[
#                 (plot_df['Bind Propensity Score'].between(values_x[0], values_x[1])) &
#                 (plot_df['TIV_Numeric'].between(values_y[0], values_y[1]))
#             ].copy()

#             st.write(f"There are **{len(strike_zone_df)}** submissions in the selected Strike Zone.")

#             # Guard: nothing to show
#             if strike_zone_df.empty:
#                 st.warning("No submissions fall inside the Strike Zone. Try expanding the ranges.")
#             else:
#                 if st.button("Run Simulation", key="top10_button_final"):
#                     # Sort by TIV desc, then Bind Propensity desc, and take top 10
#                     top_df = (
#                         strike_zone_df
#                         .sort_values(by=["TIV_Numeric", "Bind Propensity Score"], ascending=[False, False])
#                         .head(10)
#                         .copy()
#                     )

#                     # Choose columns to display
#                     display_cols = ["Submission ID", "Broker Name", "Industry",
#                                     "Bind Propensity Score", "TIV_Numeric"]
#                     display_cols = [c for c in display_cols if c in top_df.columns]

#                     # Nicely rename for UI
#                     rename_map = {"TIV_Numeric": "Total Insured Value ($)"}
#                     top_show = top_df[display_cols].rename(columns=rename_map)

#                     st.write("#### Top 10 Submissions")
#                     st.dataframe(
#                         top_show.style.format({
#                             "Bind Propensity Score": "{:.3f}",
#                             "Total Insured Value ($)": lambda v: f"${v:,.0f}"
#                         }),
#                         use_container_width=True
#                     )

#                     # Optional: show total expected value if present
#                     if "Expected Value Numeric" in top_df.columns:
#                         total_expected_value = float(top_df["Expected Value Numeric"].sum())
#                         st.metric("Total Expected Value (Top 10)",
#                                 f"${total_expected_value:,.2f}")

#                     # --- Export button (Download as CSV) ---
#                     # --- Export button (compact style) ---
#                     csv_data = top_show.to_csv(index=False).encode("utf-8")

#                     col1, col2, col3 = st.columns([4, 2, 4])  # center it nicely
#                     with col2:
#                         st.download_button(
#                             label="⬇️ Download CSV",
#                             data=csv_data,
#                             file_name="top10_strike_zone.csv",
#                             mime="text/csv",
#                             key="download_top10"
#                         )

#         else:
#             st.warning("No data with positive Total Insured Value to display on the chart.")
#     else:
#         st.warning("Data for visualization could not be loaded. Please check the CSV file.")

# # --- Tab 3: Broker Performance Insights ---
# # --- Tab 3: Broker Performance Insights ---
# # # --- Tab 3: Broker Performance Insights (Power BI-style, 2×2 layout) ---
# # with tab3:
# #     st.header("📊 Broker Performance Insights")

# #     # Load dataset with dates for this tab (uses the file we created)
# #     try:
# #         df3 = pd.read_csv("Triaging_Data_Expanded_Complete_with_dates.csv")
# #     except Exception:
# #         df3 = df_full.copy() if df_full is not None else None

# #     if df3 is None or X_encoded is None or rf_model is None:
# #         st.warning("Insights unavailable. Ensure the dataset and model are loaded.")
# #     else:
# #         if "submission_date" not in df3.columns:
# #             st.error("submission_date not found in the CSV. Please use the file with dates.")
# #         else:
# #             df3["submission_date"] = pd.to_datetime(df3["submission_date"], errors="coerce")

# #             # Compute broker summary + scored rows based on this CSV
# #             summary3, df_scored3 = compute_broker_summary(rf_model, df3, X_encoded.columns)

# #             # Theme colors (Power BI vibe)
# #             YELLOW = "#FDB913"
# #             BLACK  = "#111111"
# #             RED    = "#C00000"

# #             # Monthly aggregations
# #             temp = df_scored3.dropna(subset=["submission_date"]).copy()
# #             temp["YYYY_MM"] = temp["submission_date"].dt.to_period("M").dt.to_timestamp()
# #             monthly = (
# #                 temp.groupby("YYYY_MM")
# #                     .agg(
# #                         volume=("Broker Name", "count"),
# #                         predicted_wins=("predicted_propensity", "sum"),
# #                         avg_propensity=("predicted_propensity", "mean"),
# #                     )
# #                     .reset_index()
# #             ).sort_values("YYYY_MM")

# #             # -------- Build Figures --------
# #             # 1) Top-Left: Layered Area (Pipeline Over Time)
# #             fig_area = go.Figure()
# #             fig_area.add_trace(go.Scatter(
# #                 x=monthly["YYYY_MM"], y=monthly["volume"],
# #                 mode="lines", line=dict(width=2, color=BLACK, shape="spline"),
# #                 fill="tozeroy", name="Submissions (Volume)",
# #                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
# #             ))
# #             fig_area.add_trace(go.Scatter(
# #                 x=monthly["YYYY_MM"], y=monthly["predicted_wins"],
# #                 mode="lines", line=dict(width=2, color=YELLOW, shape="spline"),
# #                 fill="tozeroy", name="Predicted Wins",
# #                 hovertemplate="Month: %{x|%b %Y}<br>Predicted Wins: %{y:.1f}<extra></extra>", opacity=0.95
# #             ))
# #             fig_area.update_layout(
# #                 template="plotly_white",
# #                 margin=dict(l=10, r=10, t=10, b=10),
# #                 xaxis=dict(title=None, showgrid=False),
# #                 yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
# #                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
# #             )

# #             # 2) Top-Right: Donut (Broker Share by Volume)
# #             donut_df = summary3.sort_values("volume", ascending=False).copy()
# #             fig_donut = go.Figure(go.Pie(
# #                 labels=donut_df["Broker Name"],
# #                 values=donut_df["volume"],
# #                 hole=0.55, textinfo="label+percent", insidetextorientation="radial",
# #                 hovertemplate="<b>%{label}</b><br>Volume: %{value}<extra></extra>"
# #             ))
# #             fig_donut.update_traces(marker=dict(colors=[YELLOW, BLACK, "#F5C542", "#4D4D4D", "#FFE08A", "#7A7A7A"]),
# #                                     showlegend=False)
# #             fig_donut.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))

# #             # 3) Bottom-Left: Combo (Monthly Volume columns + Avg Propensity line)
# #             fig_combo = go.Figure()
# #             fig_combo.add_trace(go.Bar(
# #                 x=monthly["YYYY_MM"], y=monthly["volume"], name="Volume",
# #                 marker=dict(color=YELLOW),
# #                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
# #             ))
# #             fig_combo.add_trace(go.Scatter(
# #                 x=monthly["YYYY_MM"], y=monthly["avg_propensity"],
# #                 name="Avg Propensity", mode="lines+markers",
# #                 line=dict(color=RED, width=2, shape="spline"), yaxis="y2",
# #                 hovertemplate="Month: %{x|%b %Y}<br>Avg Propensity: %{y:.2f}<extra></extra>"
# #             ))
# #             fig_combo.update_layout(
# #                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
# #                 xaxis=dict(title=None, showgrid=False),
# #                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
# #                 yaxis2=dict(title="Avg Propensity", overlaying="y", side="right", range=[0, 1]),
# #                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
# #             )

# #             # 4) Bottom-Right: Broker bars (Volume) + line (Predicted Wins)
# #             per_broker = summary3.sort_values("volume", ascending=False)
# #             fig_brokers = go.Figure()
# #             fig_brokers.add_trace(go.Bar(
# #                 x=per_broker["Broker Name"], y=per_broker["volume"],
# #                 name="Volume", marker=dict(color=YELLOW), opacity=0.95,
# #                 hovertemplate="Broker: %{x}<br>Volume: %{y}<extra></extra>"
# #             ))
# #             fig_brokers.add_trace(go.Scatter(
# #                 x=per_broker["Broker Name"], y=per_broker["predicted_expected_wins"],
# #                 name="Predicted Wins", mode="lines+markers",
# #                 line=dict(color=BLACK, width=2, shape="spline"), yaxis="y2",
# #                 hovertemplate="Broker: %{x}<br>Predicted Wins: %{y:.1f}<extra></extra>"
# #             ))
# #             fig_brokers.update_layout(
# #                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
# #                 xaxis=dict(title=None, showgrid=False),
# #                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
# #                 yaxis2=dict(title="Predicted Wins", overlaying="y", side="right"),
# #                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
# #             )

# #             # Uniform chart heights
# #             target_h = 360
# #             for f in (fig_area, fig_donut, fig_combo, fig_brokers):
# #                 f.update_layout(height=target_h)

# #             # -------- 2×2 MATRIX LAYOUT --------
# #             r1c1, r1c2 = st.columns(2, gap="large")
# #             with r1c1:
# #                 st.write("#### Pipeline Over Time")
# #                 st.plotly_chart(fig_area, use_container_width=True)
# #             with r1c2:
# #                 st.write("#### Broker Share (Volume)")
# #                 st.plotly_chart(fig_donut, use_container_width=True)

# #             r2c1, r2c2 = st.columns(2, gap="large")
# #             with r2c1:
# #                 st.write("#### Monthly Volume & Avg Propensity")
# #                 st.plotly_chart(fig_combo, use_container_width=True)
# #             with r2c2:
# #                 st.write("#### Submissions vs Predicted Wins by Broker")
# #                 st.plotly_chart(fig_brokers, use_container_width=True)



# # --- Tab 3: Broker Performance Insights (Power BI-style, 2×2 layout) ---
# with tab3:
#     st.header("📊 Broker Performance Insights")

#     # Load dataset with dates for this tab
#     try:
#         df3 = pd.read_csv("Triaging_Data_Expanded_Complete_with_dates.csv")
#     except Exception:
#         df3 = df_full.copy() if df_full is not None else None

#     if df3 is None or X_encoded is None or rf_model is None:
#         st.warning("Insights unavailable. Ensure the dataset and model are loaded.")
#     else:
#         if "submission_date" not in df3.columns:
#             st.error("submission_date not found in the CSV. Please use the file with dates.")
#         else:
#             df3["submission_date"] = pd.to_datetime(df3["submission_date"], errors="coerce")

#             # Compute broker summary + scored rows based on this CSV
#             summary3, df_scored3 = compute_broker_summary(rf_model, df3, X_encoded.columns)

#             # --- NEW: BROKER FILTER (from previous code) ---
#             brokers = sorted(summary3["Broker Name"].dropna().unique().tolist())
#             selected_brokers = st.multiselect(
#                 "Filter Brokers",
#                 options=brokers,
#                 default=brokers # Default to all brokers selected
#             )

#             # --- NEW: APPLY FILTER TO DATAFRAMES ---
#             if selected_brokers:
#                 summary_f = summary3[summary3["Broker Name"].isin(selected_brokers)].copy()
#                 df_scored_f = df_scored3[df_scored3["Broker Name"].isin(selected_brokers)].copy()
#             else: # Handle case where nothing is selected by showing all data
#                 summary_f = summary3.copy()
#                 df_scored_f = df_scored3.copy()

#             # --- NEW: KPI METRICS (from previous code, adapted for new layout) ---
#             st.markdown("---")
#             st.subheader("Key Performance Indicators")

#             # Calculate overall KPIs from the filtered summary
#             overall_volume = int(summary_f["volume"].sum())
#             overall_predicted_wins = summary_f["predicted_expected_wins"].sum()
#             # Calculate weighted average for propensity and win rate for accuracy
#             if overall_volume > 0:
#                 avg_propensity = (summary_f["predicted_propensity_mean"] * summary_f["volume"]).sum() / overall_volume
#                 avg_win_rate = (summary_f["win_rate"] * summary_f["volume"]).sum() / overall_volume
#             else:
#                 avg_propensity, avg_win_rate = 0, 0

#             k1, k2, k3, k4 = st.columns(4)
#             k1.metric("Total Submissions", f"{overall_volume:,}")
#             k2.metric("Predicted Wins (Expected)", f"{overall_predicted_wins:.1f}")
#             k3.metric("Avg Bind Propensity", f"{avg_propensity:.1%}")
#             k4.metric("Avg Historical Win Rate", f"{avg_win_rate:.1%}" if not pd.isna(avg_win_rate) else "N/A")
#             st.markdown("---")


#             # Theme colors (Power BI vibe)
#             YELLOW = "#FDB913"
#             BLACK  = "#111111"
#             RED    = "#C00000"

#             # Monthly aggregations (NOW USES FILTERED DATA)
#             temp = df_scored_f.dropna(subset=["submission_date"]).copy()
#             temp["YYYY_MM"] = temp["submission_date"].dt.to_period("M").dt.to_timestamp()
#             monthly = (
#                 temp.groupby("YYYY_MM")
#                     .agg(
#                         volume=("Broker Name", "count"),
#                         predicted_wins=("predicted_propensity", "sum"),
#                         avg_propensity=("predicted_propensity", "mean"),
#                     )
#                     .reset_index()
#             ).sort_values("YYYY_MM")

#             # -------- Build Figures (all charts now respect the filter) --------
#             # 1) Top-Left: Layered Area (Pipeline Over Time)
#             fig_area = go.Figure()
#             fig_area.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["volume"],
#                 mode="lines", line=dict(width=2, color=BLACK, shape="spline"),
#                 fill="tozeroy", name="Submissions (Volume)",
#                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_area.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["predicted_wins"],
#                 mode="lines", line=dict(width=2, color=YELLOW, shape="spline"),
#                 fill="tozeroy", name="Predicted Wins",
#                 hovertemplate="Month: %{x|%b %Y}<br>Predicted Wins: %{y:.1f}<extra></extra>", opacity=0.95
#             ))
#             fig_area.update_layout(
#                 template="plotly_white",
#                 margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # 2) Top-Right: Donut (Broker Share by Volume)
#             donut_df = summary_f.sort_values("volume", ascending=False).copy()
#             fig_donut = go.Figure(go.Pie(
#                 labels=donut_df["Broker Name"],
#                 values=donut_df["volume"],
#                 hole=0.55, textinfo="label+percent", insidetextorientation="radial",
#                 hovertemplate="<b>%{label}</b><br>Volume: %{value}<extra></extra>"
#             ))
#             fig_donut.update_traces(marker=dict(colors=[YELLOW, BLACK, "#F5C542", "#4D4D4D", "#FFE08A", "#7A7A7A"]),
#                                       showlegend=False)
#             fig_donut.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))

#             # 3) Bottom-Left: Combo (Monthly Volume columns + Avg Propensity line)
#             fig_combo = go.Figure()
#             fig_combo.add_trace(go.Bar(
#                 x=monthly["YYYY_MM"], y=monthly["volume"], name="Volume",
#                 marker=dict(color=YELLOW),
#                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_combo.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["avg_propensity"],
#                 name="Avg Propensity", mode="lines+markers",
#                 line=dict(color=RED, width=2, shape="spline"), yaxis="y2",
#                 hovertemplate="Month: %{x|%b %Y}<br>Avg Propensity: %{y:.2f}<extra></extra>"
#             ))
#             fig_combo.update_layout(
#                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 yaxis2=dict(title="Avg Propensity", overlaying="y", side="right", range=[0, 1]),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # 4) Bottom-Right: Broker bars (Volume) + line (Predicted Wins)
#             per_broker = summary_f.sort_values("volume", ascending=False)
#             fig_brokers = go.Figure()
#             fig_brokers.add_trace(go.Bar(
#                 x=per_broker["Broker Name"], y=per_broker["volume"],
#                 name="Volume", marker=dict(color=YELLOW), opacity=0.95,
#                 hovertemplate="Broker: %{x}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_brokers.add_trace(go.Scatter(
#                 x=per_broker["Broker Name"], y=per_broker["predicted_expected_wins"],
#                 name="Predicted Wins", mode="lines+markers",
#                 line=dict(color=BLACK, width=2, shape="spline"), yaxis="y2",
#                 hovertemplate="Broker: %{x}<br>Predicted Wins: %{y:.1f}<extra></extra>"
#             ))
#             fig_brokers.update_layout(
#                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 yaxis2=dict(title="Predicted Wins", overlaying="y", side="right"),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # Uniform chart heights
#             target_h = 360
#             for f in (fig_area, fig_donut, fig_combo, fig_brokers):
#                 f.update_layout(height=target_h)

#             # -------- 2×2 MATRIX LAYOUT --------
#             r1c1, r1c2 = st.columns(2, gap="large")
#             with r1c1:
#                 st.write("#### Pipeline Over Time")
#                 st.plotly_chart(fig_area, use_container_width=True)
#             with r1c2:
#                 st.write("#### Broker Share (Volume)")
#                 st.plotly_chart(fig_donut, use_container_width=True)

#             r2c1, r2c2 = st.columns(2, gap="large")
#             with r2c1:
#                 st.write("#### Monthly Volume & Avg Propensity")
#                 st.plotly_chart(fig_combo, use_container_width=True)
#             with r2c2:
#                 st.write("#### Submissions vs Predicted Wins by Broker")
            #    st.plotly_chart(fig_brokers, use_container_width=True)














######with the bronze<silver<gold<platinum logic fix
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from openai import AzureOpenAI

# --- Page Configuration and Styling (Apply globally) ---
st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #f0f8ff; /* Light AliceBlue background */
}

/* Main title */
h1 {
    color: #004080; /* Dark blue */
}

/* Subheaders */
h2, h3 {
    color: #0055a4; /* Slightly lighter blue */
}

/* Style for buttons */
.stButton>button {
    color: #ffffff;
    background: linear-gradient(45deg, #007bff, #0056b3);
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 100%; /* Make buttons fill column width */
}

.stButton>button:hover {
    background: linear-gradient(45deg, #0056b3, #007bff);
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    transform: translateY(-2px);
}

/* Styling for the main content container */
.main .block-container {
    border-radius: 20px;
    background-color: #ffffff;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
# --- Tier offsets + helpers (ADD THIS) ---
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


# --- Helper Function for Gauge Chart ---
def create_gauge_chart(score, level):
    """Creates a more polished and eye-catching Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0.1, 0.9], 'y': [0, 0.9]},
        title={'text': f"<b>Bind Propensity Score</b><br><span style='font-size:1.2em;color:grey'>{level}</span>", 'font': {'size': 24}},
        number={'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkslategray"},
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
        font={'color': "darkslategray", 'family': "Arial, Helvetica, sans-serif"},
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

# --- Azure OpenAI LLM Integration ---
@st.cache_data(show_spinner=False)
def get_llm_explanation(score, level, _input_data, _shap_values, _feature_names):
    """Generates a natural language explanation for a prediction using Azure OpenAI."""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://advancedanalyticsopenaikey.openai.azure.com/",
            api_key="FqFd4DBx1W97MSVjcZvdQsmQlhI80hXjl48iWYmZ4W3NutUlWvf0JQQJ99BDACYeBjFXJ3w3AAABACOGl3xo",
            api_version="2024-02-15-preview"
        )
        
        aggregated_shaps = aggregate_shap_values(_shap_values[0], _feature_names, _input_data.columns)
        top_features_series = aggregated_shaps.abs().nlargest(5)
        
        feature_summary = ""
        for feature_name, _ in top_features_series.items():
            shap_val = aggregated_shaps[feature_name]
            impact = "positively" if shap_val > 0 else "negatively"
            
            value = _input_data[feature_name].iloc[0]
            if isinstance(value, str):
                display_name = f"{feature_name} = {value}"
            else:
                display_name = feature_name

            feature_summary += f"- **{display_name}** influenced the score **{impact}**.\n"

        prompt = f"""
        You are an expert underwriting assistant. A machine learning model predicted a Bind Propensity Score of {score:.2f}, which is considered '{level}'.
        The top factors influencing this prediction were:
        {feature_summary}
        Based on this, provide a concise, easy-to-understand "Key Drivers" summary for an underwriter in 2-3 sentences.
        Explain WHY the submission likely received this score in business terms. Do not just list the features.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI explanation. Error: {e}"

# --- SHAP Plot Functions ---
def create_top_5_shap_plot(shap_values, feature_names, input_data):
    """Creates an interactive Plotly bar chart for the top 5 aggregated SHAP features."""
    aggregated_shaps = aggregate_shap_values(shap_values[0], feature_names, input_data.columns)
    
    top_5_series = aggregated_shaps.abs().nlargest(5)
    top_5_shap_series = aggregated_shaps[top_5_series.index].sort_values(ascending=True)

    colors = ['#007bff' if val > 0 else '#dc3545' for val in top_5_shap_series.values]
    
    renamed_index = []
    for feature_name in top_5_shap_series.index:
        value = input_data[feature_name].iloc[0]
        if isinstance(value, str):
            renamed_index.append(f"{feature_name} = {value}")
        else:
            renamed_index.append(feature_name)

    fig = go.Figure(go.Bar(
        x=top_5_shap_series.values,
        y=renamed_index,
        orientation='h',
        marker_color=colors,
        text=np.round(top_5_shap_series.values, 3),
        textposition='auto'
    ))

    fig.update_layout(
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=10, r=10, t=10, b=10)
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


# --- Main App Title and Logo (Displayed above tabs) ---
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



# # # --- Tab 1: Bind Propensity Score Prediction ---
# with tab1:
#     if lr_model is None or rf_model is None or X_encoded is None or explainer is None:
#         st.warning("Cannot proceed with prediction due to missing files.")
#     else:
#         st.subheader("Select a Scenario to Pre-fill Form")
#         b_col1, b_col2, b_col3 = st.columns(3)
#         with b_col1: low_button = st.button("Low Bind Propensity")
#         with b_col2: medium_button = st.button("Medium Bind Propensity")
#         with b_col3: high_button = st.button("High Bind Propensity")

#         if 'scenario' not in st.session_state: st.session_state['scenario'] = None

#         if low_button:
#             st.session_state.update({
#                 'scenario': "Low", 'broker_name': "Delta Insure", 'channel': "Email", 'broker_tier': "Bronze",
#                 'industry': "Manufacturing", 'client_size': 30, 'locations': 1, 'state': "IL",
#                 'building_value': 5_000_000, 'contents_value': 1_000_000, 'bi_value': 500_000,
#                 'historical_bind_rate': 0.35, 'days_to_quote': 9, 'prior_claims': "Yes",
#                 'submission_complete': "No", 'cat_zone': "Yes"
#             })
#             st.toast("Low Bind Propensity Scenario Loaded!")
#         elif medium_button:
#             st.session_state.update({
#                 'scenario': "Medium", 'broker_name': "CoreTrust", 'channel': "Wholesaler", 'broker_tier': "Silver",
#                 'industry': "Retail", 'client_size': 120, 'locations': 3, 'state': "NY",
#                 'building_value': 40_000_000, 'contents_value': 6_000_000, 'bi_value': 2_500_000,
#                 'historical_bind_rate': 0.55, 'days_to_quote': 6, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("Medium Bind Propensity Scenario Loaded!")
#         elif high_button:
#             st.session_state.update({
#                 'scenario': "High", 'broker_name': "Alpha Risk", 'channel': "Portal", 'broker_tier': "Platinum",
#                 'industry': "Real Estate", 'client_size': 250, 'locations': 5, 'state': "TX",
#                 'building_value': 60_000_000, 'contents_value': 10_000_000, 'bi_value': 5_000_000,
#                 'historical_bind_rate': 0.80, 'days_to_quote': 3, 'prior_claims': "No",
#                 'submission_complete': "Yes", 'cat_zone': "No"
#             })
#             st.toast("High Bind Propensity Scenario Loaded!")

#         if st.session_state['scenario'] is not None:
#             st.markdown("---")
#             st.subheader("Submission & Broker Information")
#             f_col1, f_col2, f_col3, f_col4 = st.columns(4)
#             with f_col1: broker_name = st.selectbox("Broker Name", ["Alpha Risk", "Beta Cover", "CoreTrust", "FastBind", "Delta Insure"], key='broker_name')
#             with f_col2: channel = st.selectbox("Channel", ["Portal", "Email", "Wholesaler", "API"], key='channel')
#             with f_col3: broker_tier = st.selectbox("Broker Tier", ["Platinum", "Gold", "Silver", "Bronze"], key='broker_tier')
#             with f_col4: historical_bind_rate = st.number_input("Historical Bind Rate", 0.0, 1.0, step=0.01, key='historical_bind_rate')
            
#             st.markdown("---")
#             st.subheader("Client & Policy Information")
#             f_col1, f_col2, f_col3 = st.columns(3)
#             with f_col1:
#                 industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Warehousing", "Real Estate", "Hospitality"], key='industry')
#                 building_value = st.number_input("Building Value ($)", 1000000, 100000000, step=100000, key='building_value')
#                 days_to_quote = st.number_input("Days to Quote", 1, 30, key='days_to_quote')
#             with f_col2:
#                 client_size = st.number_input("Client Size (Revenue $M)", 10, 300, key='client_size')
#                 contents_value = st.number_input("Contents Value ($)", 100000, 20000000, step=100000, key='contents_value')
#                 prior_claims = st.selectbox("Prior Claims", ["Yes", "No"], key='prior_claims')
#             with f_col3:
#                 locations = st.number_input("Number of Locations", 1, 10, key='locations')
#                 bi_value = st.number_input("BI Value ($)", 500000, 10000000, step=50000, key='bi_value')
#                 submission_complete = st.selectbox("Submission Complete", ["Yes", "No"], key='submission_complete')
#             state_col, cat_col = st.columns(2)
#             with state_col: state = st.selectbox("State", ["TX", "FL", "NY", "CA", "IL"], key='state')
#             with cat_col: cat_zone = st.selectbox("CAT Zone", ["Yes", "No"], key='cat_zone')

#             st.markdown("---")
#             pred_col, _, clear_col = st.columns([2, 12, 2])
#             with pred_col:
#                 if st.button("Submit"):
#                     input_data = pd.DataFrame([{
#                         "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
#                         "Historical Bind Rate": historical_bind_rate, "Industry": industry,
#                         "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
#                         "Building Value ($)": building_value, "Contents Value ($)": contents_value,
#                         "BI Value ($)": bi_value, "Submission Complete": submission_complete,
#                         "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
#                     }])
#                     input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
                    
#                     rf_pred = rf_model.predict(input_df_aligned)
#                     predicted_value = rf_pred[0]
#                     TIV = building_value + contents_value + bi_value
                    
#                     shap_values = explainer.shap_values(input_df_aligned)
#                     top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)
                    
#                     level = "Low"
#                     if predicted_value > 0.65: level = "High"
#                     elif predicted_value > 0.4: level = "Medium"
                    
#                     llm_explanation = get_llm_explanation(predicted_value, level, input_data, shap_values, input_df_aligned.columns)

#                     st.session_state['prediction_results'] = {
#                         "score": predicted_value, "tiv": f"${TIV:,.0f}",
#                         "level": level, "top_5_plot": top_5_plot, "llm_explanation": llm_explanation
#                     }
#             with clear_col:
#                 if st.button("Clear"):
#                     st.session_state.pop('prediction_results', None)
#                     st.session_state['scenario'] = None
#                     st.rerun()

#             if 'prediction_results' in st.session_state and st.session_state['scenario'] is not None:
#                 st.markdown("---")
#                 st.subheader("Prediction Results")
                
#                 results = st.session_state['prediction_results']
#                 res_col1, res_col2 = st.columns(2)
#                 with res_col1:
#                     st.plotly_chart(create_gauge_chart(results['score'], results['level']), use_container_width=True)
#                 with res_col2:
#                     st.metric("Total Insured Value (TIV)", results['tiv'])
                
#                 st.markdown("---")
#                 st.subheader("Model Prediction Explainability")
                
#                 exp_col1, exp_col2 = st.columns(2)
#                 with exp_col1:
#                     st.write("#### Top 5 SHAP Features Influencing Prediction")
#                     st.plotly_chart(results['top_5_plot'], use_container_width=True)
#                 with exp_col2:
#                     st.markdown(f"""
#                     <div style="border: 1px solid #0055a4; border-radius: 10px; padding: 15px; background-color: #f0f8ff; height: 100%;">
#                         <h4 style="color: #004080; margin-bottom: 10px;">Key Drivers</h4>
#                         <p style="color: #333;">{results['llm_explanation']}</p>
#                     </div>
#                     """, unsafe_allow_html=True)

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

                    llm_explanation = get_llm_explanation(adjusted_score, level, input_data, shap_values, input_df_aligned.columns)

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

            # with pred_col:
            #     if st.button("Submit"):
            #         input_data = pd.DataFrame([{
            #             "Broker Name": broker_name, "Channel": channel, "Broker Tier": broker_tier,
            #             "Historical Bind Rate": historical_bind_rate, "Industry": industry,
            #             "Client Revenue ($M)": client_size, "Locations": locations, "State": state,
            #             "Building Value ($)": building_value, "Contents Value ($)": contents_value,
            #             "BI Value ($)": bi_value, "Submission Complete": submission_complete,
            #             "CAT Zone": cat_zone, "Days to Quote": days_to_quote, "Prior Claims": prior_claims
            #         }])
            #         input_df_aligned = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
                    
            #         rf_pred = rf_model.predict(input_df_aligned)
            #         predicted_value = rf_pred[0]
            #         TIV = building_value + contents_value + bi_value
                    
            #         shap_values = explainer.shap_values(input_df_aligned)
            #         top_5_plot = create_top_5_shap_plot(shap_values, input_df_aligned.columns, input_data)
                    
            #         level = "Low"
            #         if predicted_value > 0.65: level = "High"
            #         elif predicted_value > 0.4: level = "Medium"
                    
            #         llm_explanation = get_llm_explanation(predicted_value, level, input_data, shap_values, input_df_aligned.columns)

            #         st.session_state['prediction_results'] = {
            #             "score": predicted_value, "tiv": f"${TIV:,.0f}",
            #             "level": level, "top_5_plot": top_5_plot, "llm_explanation": llm_explanation
            #         }
            # with clear_col:
            #     if st.button("Clear"):
            #         st.session_state.pop('prediction_results', None)
            #         st.session_state['scenario'] = None
            #         st.rerun()
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
                        <h4 style="color: #004080; margin-bottom: 10px;">Key Drivers</h4>
                        <p style="color: #333;">{results['llm_explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)



# --- Tab 2: Submissions Prioritization using Strike Zone ---
with tab2:
    st.header("Submissions Prioritization using Strike Zone")
    if df_full is not None and 'TIV_Numeric' in df_full.columns:
        plot_df = df_full[df_full['TIV_Numeric'] > 0].copy()

        if not plot_df.empty:
            x_data = plot_df['Bind Propensity Score']
            y_data = plot_df['TIV_Numeric']

            min_tiv = int(y_data.min())
            max_tiv = int(y_data.max())

            values_x = st.slider("Select a Bind Propensity range for Strike Zone", 0.2, 1.0, (0.5, 1.0), key="slider_x_final")
            values_y = st.slider("Select a Total Insured Value ($) range for Strike Zone", min_tiv, max_tiv, (int(max_tiv * 0.25), max_tiv), step=1000000, key="slider_y_final", format="$%d")

            fig = go.Figure()

            # Add the scatter plot with color scale
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=y_data, 
                mode='markers',
                marker=dict(
                    size=8,
                    color=y_data, # Color points by TIV
                    colorscale='Viridis', # A nice color scale
                    showscale=True,
                    colorbar=dict(title="TIV ($)"),
                    opacity=0.7
                ),
                text=plot_df.apply(lambda row: f"Broker: {row['Broker Name']}<br>TIV: ${row['TIV_Numeric']:,.0f}", axis=1),
                hoverinfo='text+x+y'
            ))

            # Add the strike zone rectangle on top
            fig.add_shape(type="rect",
                          x0=values_x[0], x1=values_x[1], y0=values_y[0], y1=values_y[1],
                          line=dict(color="red", width=2),
                          fillcolor="rgba(0,0,0,0)", # transparent fill
                          layer="above")
            
            fig.update_layout(
                title="Submissions Segmented by Bind Propensity and Total Insured Value",
                xaxis_title="Bind Propensity Score",
                yaxis_title="Total Insured Value ($) (Log Scale)",
                yaxis_type="log",
                xaxis_range=[0.2, 1.0],
                showlegend=False,
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
            
            st.write("### Simulate the Impact of Submission Prioritization")

            # Filter to strike-zone
            strike_zone_df = plot_df[
                (plot_df['Bind Propensity Score'].between(values_x[0], values_x[1])) &
                (plot_df['TIV_Numeric'].between(values_y[0], values_y[1]))
            ].copy()

            st.write(f"There are **{len(strike_zone_df)}** submissions in the selected Strike Zone.")

            # Guard: nothing to show
            if strike_zone_df.empty:
                st.warning("No submissions fall inside the Strike Zone. Try expanding the ranges.")
            else:
                if st.button("Run Simulation", key="top10_button_final"):
                    # Sort by TIV desc, then Bind Propensity desc, and take top 10
                    top_df = (
                        strike_zone_df
                        .sort_values(by=["TIV_Numeric", "Bind Propensity Score"], ascending=[False, False])
                        .head(10)
                        .copy()
                        .reset_index(drop=True)  # Reset the index
                    )
                    top_df.index = top_df.index + 1  # Start numbering from 1 instead of 0


                    # Choose columns to display
                    display_cols = ["Submission ID", "Broker Name", "Industry",
                                    "Bind Propensity Score", "TIV_Numeric"]
                    display_cols = [c for c in display_cols if c in top_df.columns]

                    # Nicely rename for UI
                    rename_map = {"TIV_Numeric": "Total Insured Value ($)"}
                    top_show = top_df[display_cols].rename(columns=rename_map)

                    st.write("#### Top 10 Submissions (Ranked by TIV and Bind Propensity)")
                    st.dataframe(
                        top_show.style.format({
                            "Bind Propensity Score": "{:.3f}",
                            "Total Insured Value ($)": lambda v: f"${v:,.0f}"
                        }),
                        use_container_width=True
                    )

                    # Optional: show total expected value if present
                    if "Expected Value Numeric" in top_df.columns:
                        total_expected_value = float(top_df["Expected Value Numeric"].sum())
                        st.metric("Total Expected Value (Top 10)",
                                f"${total_expected_value:,.2f}")

                    # --- Export button (Download as CSV) ---
                    # --- Export button (compact style) ---
                    csv_data = top_show.to_csv(index=False).encode("utf-8")

                    col1, col2, col3 = st.columns([4, 2, 4])  # center it nicely
                    with col2:
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_data,
                            file_name="top10_strike_zone.csv",
                            mime="text/csv",
                            key="download_top10"
                        )

        else:
            st.warning("No data with positive Total Insured Value to display on the chart.")
    else:
        st.warning("Data for visualization could not be loaded. Please check the CSV file.")

# --- Tab 3: Broker Performance Insights ---
# --- Tab 3: Broker Performance Insights ---
# # --- Tab 3: Broker Performance Insights (Power BI-style, 2×2 layout) ---
# with tab3:
#     st.header("📊 Broker Performance Insights")

#     # Load dataset with dates for this tab (uses the file we created)
#     try:
#         df3 = pd.read_csv("Triaging_Data_Expanded_Complete_with_dates.csv")
#     except Exception:
#         df3 = df_full.copy() if df_full is not None else None

#     if df3 is None or X_encoded is None or rf_model is None:
#         st.warning("Insights unavailable. Ensure the dataset and model are loaded.")
#     else:
#         if "submission_date" not in df3.columns:
#             st.error("submission_date not found in the CSV. Please use the file with dates.")
#         else:
#             df3["submission_date"] = pd.to_datetime(df3["submission_date"], errors="coerce")

#             # Compute broker summary + scored rows based on this CSV
#             summary3, df_scored3 = compute_broker_summary(rf_model, df3, X_encoded.columns)

#             # Theme colors (Power BI vibe)
#             YELLOW = "#FDB913"
#             BLACK  = "#111111"
#             RED    = "#C00000"

#             # Monthly aggregations
#             temp = df_scored3.dropna(subset=["submission_date"]).copy()
#             temp["YYYY_MM"] = temp["submission_date"].dt.to_period("M").dt.to_timestamp()
#             monthly = (
#                 temp.groupby("YYYY_MM")
#                     .agg(
#                         volume=("Broker Name", "count"),
#                         predicted_wins=("predicted_propensity", "sum"),
#                         avg_propensity=("predicted_propensity", "mean"),
#                     )
#                     .reset_index()
#             ).sort_values("YYYY_MM")

#             # -------- Build Figures --------
#             # 1) Top-Left: Layered Area (Pipeline Over Time)
#             fig_area = go.Figure()
#             fig_area.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["volume"],
#                 mode="lines", line=dict(width=2, color=BLACK, shape="spline"),
#                 fill="tozeroy", name="Submissions (Volume)",
#                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_area.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["predicted_wins"],
#                 mode="lines", line=dict(width=2, color=YELLOW, shape="spline"),
#                 fill="tozeroy", name="Predicted Wins",
#                 hovertemplate="Month: %{x|%b %Y}<br>Predicted Wins: %{y:.1f}<extra></extra>", opacity=0.95
#             ))
#             fig_area.update_layout(
#                 template="plotly_white",
#                 margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title=None, showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # 2) Top-Right: Donut (Broker Share by Volume)
#             donut_df = summary3.sort_values("volume", ascending=False).copy()
#             fig_donut = go.Figure(go.Pie(
#                 labels=donut_df["Broker Name"],
#                 values=donut_df["volume"],
#                 hole=0.55, textinfo="label+percent", insidetextorientation="radial",
#                 hovertemplate="<b>%{label}</b><br>Volume: %{value}<extra></extra>"
#             ))
#             fig_donut.update_traces(marker=dict(colors=[YELLOW, BLACK, "#F5C542", "#4D4D4D", "#FFE08A", "#7A7A7A"]),
#                                     showlegend=False)
#             fig_donut.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))

#             # 3) Bottom-Left: Combo (Monthly Volume columns + Avg Propensity line)
#             fig_combo = go.Figure()
#             fig_combo.add_trace(go.Bar(
#                 x=monthly["YYYY_MM"], y=monthly["volume"], name="Volume",
#                 marker=dict(color=YELLOW),
#                 hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_combo.add_trace(go.Scatter(
#                 x=monthly["YYYY_MM"], y=monthly["avg_propensity"],
#                 name="Avg Propensity", mode="lines+markers",
#                 line=dict(color=RED, width=2, shape="spline"), yaxis="y2",
#                 hovertemplate="Month: %{x|%b %Y}<br>Avg Propensity: %{y:.2f}<extra></extra>"
#             ))
#             fig_combo.update_layout(
#                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 yaxis2=dict(title="Avg Propensity", overlaying="y", side="right", range=[0, 1]),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # 4) Bottom-Right: Broker bars (Volume) + line (Predicted Wins)
#             per_broker = summary3.sort_values("volume", ascending=False)
#             fig_brokers = go.Figure()
#             fig_brokers.add_trace(go.Bar(
#                 x=per_broker["Broker Name"], y=per_broker["volume"],
#                 name="Volume", marker=dict(color=YELLOW), opacity=0.95,
#                 hovertemplate="Broker: %{x}<br>Volume: %{y}<extra></extra>"
#             ))
#             fig_brokers.add_trace(go.Scatter(
#                 x=per_broker["Broker Name"], y=per_broker["predicted_expected_wins"],
#                 name="Predicted Wins", mode="lines+markers",
#                 line=dict(color=BLACK, width=2, shape="spline"), yaxis="y2",
#                 hovertemplate="Broker: %{x}<br>Predicted Wins: %{y:.1f}<extra></extra>"
#             ))
#             fig_brokers.update_layout(
#                 template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
#                 xaxis=dict(title=None, showgrid=False),
#                 yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
#                 yaxis2=dict(title="Predicted Wins", overlaying="y", side="right"),
#                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
#             )

#             # Uniform chart heights
#             target_h = 360
#             for f in (fig_area, fig_donut, fig_combo, fig_brokers):
#                 f.update_layout(height=target_h)

#             # -------- 2×2 MATRIX LAYOUT --------
#             r1c1, r1c2 = st.columns(2, gap="large")
#             with r1c1:
#                 st.write("#### Pipeline Over Time")
#                 st.plotly_chart(fig_area, use_container_width=True)
#             with r1c2:
#                 st.write("#### Broker Share (Volume)")
#                 st.plotly_chart(fig_donut, use_container_width=True)

#             r2c1, r2c2 = st.columns(2, gap="large")
#             with r2c1:
#                 st.write("#### Monthly Volume & Avg Propensity")
#                 st.plotly_chart(fig_combo, use_container_width=True)
#             with r2c2:
#                 st.write("#### Submissions vs Predicted Wins by Broker")
#                 st.plotly_chart(fig_brokers, use_container_width=True)


def format_human_readable(value):
    if value >= 1e9:
        return f"{value/1e9:.1f} B"
    elif value >= 1e6:
        return f"{value/1e6:.0f} M"  # rounded to whole millions
    else:
        return f"{value:,.0f}"


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

            # --- NEW: BROKER FILTER (from previous code) ---
            brokers = sorted(summary3["Broker Name"].dropna().unique().tolist())
            selected_brokers = st.multiselect(
                "Filter Brokers",
                options=brokers,
                default=brokers # Default to all brokers selected
            )

            # --- NEW: APPLY FILTER TO DATAFRAMES ---
            if selected_brokers:
                summary_f = summary3[summary3["Broker Name"].isin(selected_brokers)].copy()
                df_scored_f = df_scored3[df_scored3["Broker Name"].isin(selected_brokers)].copy()
            else: # Handle case where nothing is selected by showing all data
                summary_f = summary3.copy()
                df_scored_f = df_scored3.copy()

            # --- NEW: KPI METRICS (from previous code, adapted for new layout) ---
            st.markdown("---")
            st.subheader("Key Performance Indicators")

            # Calculate overall KPIs from the filtered summary
            overall_volume = int(summary_f["volume"].sum())
            # overall_predicted_wins = summary_f["predicted_expected_wins"].sum()
            # Compute summed TIV
            overall_tiv = df_scored_f["TIV_Numeric"].sum() if "TIV_Numeric" in df_scored_f.columns else 0

            # Calculate weighted average for propensity and win rate for accuracy
            if overall_volume > 0:
                avg_propensity = (summary_f["predicted_propensity_mean"] * summary_f["volume"]).sum() / overall_volume
                avg_win_rate = (summary_f["win_rate"] * summary_f["volume"]).sum() / overall_volume
            else:
                avg_propensity, avg_win_rate = 0, 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Submissions", f"{overall_volume:,}")
            # k2.metric("Predicted Wins (Expected)", f"{overall_predicted_wins:.1f}")
            k2.metric("Total Insured Value (TIV)", format_human_readable(overall_tiv))
            k3.metric("Avg Bind Propensity", f"{avg_propensity:.1%}")
            k4.metric("Avg Historical Win Rate", f"{avg_win_rate:.1%}" if not pd.isna(avg_win_rate) else "N/A")
            st.markdown("---")


            # Theme colors (Power BI vibe)
            YELLOW = "#FDB913"
            BLACK  = "#111111"
            RED    = "#C00000"

            # Monthly aggregations (NOW USES FILTERED DATA)
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

            # -------- Build Figures (all charts now respect the filter) --------
            # 1) Top-Left: Layered Area (Pipeline Over Time)
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

            # 2) Top-Right: Donut (Broker Share by Volume)
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

            # # 3) Bottom-Left: Combo (Monthly Volume columns + Avg Propensity line)
            # fig_combo = go.Figure()
            # fig_combo.add_trace(go.Bar(
            #     x=monthly["YYYY_MM"], y=monthly["volume"], name="Volume",
            #     marker=dict(color=YELLOW),
            #     hovertemplate="Month: %{x|%b %Y}<br>Volume: %{y}<extra></extra>"
            # ))
            # fig_combo.add_trace(go.Scatter(
            #     x=monthly["YYYY_MM"], y=monthly["avg_propensity"],
            #     name="Avg Propensity", mode="lines+markers",
            #     line=dict(color=RED, width=2, shape="spline"), yaxis="y2",
            #     hovertemplate="Month: %{x|%b %Y}<br>Avg Propensity: %{y:.2f}<extra></extra>"
            # ))
            # fig_combo.update_layout(
            #     template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
            #     xaxis=dict(title=None, showgrid=False),
            #     yaxis=dict(title="Volume", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
            #     yaxis2=dict(title="Avg Propensity", overlaying="y", side="right", range=[0, 1]),
            #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            # )
            # 3) Bottom-Left: Expected Wins per Month (New Chart)
            monthly["expected_wins"] = monthly["volume"] * monthly["avg_propensity"]

            fig_expected = go.Figure()
            fig_expected.add_trace(go.Bar(
                x=monthly["YYYY_MM"],
                y=monthly["expected_wins"],
                name="Expected Wins",
                marker=dict(color="#FDB913"),
                hovertemplate="Month: %{x|%b %Y}<br>Expected Wins: %{y:.1f}<extra></extra>"
            ))
            fig_expected.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title=None, showgrid=False),
                yaxis=dict(title="Expected Wins", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                showlegend=False
            )


            # 4) Bottom-Right: Broker bars (Volume) + line (Predicted Wins)
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

            # Uniform chart heights
            target_h = 360
            for f in (fig_area, fig_donut, fig_expected, fig_brokers):
                f.update_layout(height=target_h)

            # -------- 2×2 MATRIX LAYOUT --------
            r1c1, r1c2 = st.columns(2, gap="large")
            with r1c1:
                st.write("#### Pipeline Over Time")
                st.plotly_chart(fig_area, use_container_width=True)
            with r1c2:
                st.write("#### Broker Share (Volume)")
                st.plotly_chart(fig_donut, use_container_width=True)

            r2c1, r2c2 = st.columns(2, gap="large")
            with r2c1:
                st.write("#### Expected Wins per Month")
                st.plotly_chart(fig_expected, use_container_width=True)

            with r2c2:
                st.write("#### Submissions vs Predicted Wins by Broker")
                st.plotly_chart(fig_brokers, use_container_width=True)


























