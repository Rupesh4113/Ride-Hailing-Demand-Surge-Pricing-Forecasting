import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
from datetime import datetime
import h3

# Page configuration
st.set_page_config(
    page_title="Ride-Hailing Spatio-Temporal Demand Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_data
def load_predictions(source):
    preds_path = os.path.join(DATA_DIR, f"predictions_with_surge_{source}.parquet")
    if not os.path.exists(preds_path):
        st.error(f"Predictions data not found at {preds_path}. Please run the pipeline for this source first: `python run_pipeline.py --source {source}`")
        return None
    df = pd.read_parquet(preds_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_metrics(source):
    metrics_path = os.path.join(DATA_DIR, f"metrics_report_{source}.csv")
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("Data Configuration")

# 1. Source Selection
source_display = st.sidebar.selectbox(
    "Data Source",
    options=["NYC TLC Cloud (Real)", "Chicago Synthetic (Mock)"]
)
source_map = {
    "NYC TLC Cloud (Real)": "nyc_cloud",
    "Chicago Synthetic (Mock)": "synthetic"
}
source = source_map[source_display]

df = load_predictions(source)
metrics_df = load_metrics(source)

if df is not None:
    st.title("🚖 Ride-Hailing Spatio-Temporal Demand & Surge Pricing Dashboard")
    st.markdown(f"Currently viewing predictions and metrics for **{source_display}** mapping pickups on H3 resolution 8.")
    
    st.sidebar.header("Time Filters")
    
    # 2. Select Timestamp
    timestamps = sorted(df['timestamp'].unique())
    min_date = timestamps[0].date()
    max_date = timestamps[-1].date()
    
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter by date first
    df_date = df[df['timestamp'].dt.date == selected_date]
    available_hours = sorted(df_date['timestamp'].dt.time.unique())
    
    if available_hours:
        selected_time = st.sidebar.select_slider(
            "Select Time Window (30-min intervals)",
            options=available_hours,
            format_func=lambda t: t.strftime("%H:%M")
        )
        target_dt = datetime.combine(selected_date, selected_time)
        df_filtered = df_date[df_date['timestamp'].dt.time == selected_time].copy()
    else:
        st.warning("No records available for the selected date. Showing first available date.")
        # Fallback to first timestamp
        target_dt = timestamps[0]
        df_filtered = df[df['timestamp'] == target_dt].copy()
    
    # 3. Select Metric to Visualize
    metric_choice = st.sidebar.selectbox(
        "Map Visualization Metric",
        options=["Predicted Demand", "Actual Demand (Ground Truth)", "Surge Multiplier", "Driver Supply Proxy"]
    )
    
    # 4. Model performance reporting
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    if metrics_df is not None:
        st.sidebar.table(metrics_df)
    else:
        st.sidebar.info(f"Run the training pipeline (`python run_pipeline.py --source {source}`) to generate evaluation metrics.")
        
    # ------------------ MAP LAYER STYLING ------------------
    max_val = df['target_demand'].max() if df['target_demand'].max() > 0 else 1.0
    
    def get_color_and_elevation(row, metric):
        if metric == "Predicted Demand":
            val = row['predicted_demand']
            ratio = val / max_val
            color = [int(255), int(255 * (1 - ratio)), int(0), 160] # Yellow to Red
            elevation = val * 80
        elif metric == "Actual Demand (Ground Truth)":
            val = row['target_demand']
            ratio = val / max_val
            color = [int(0), int(255 * ratio), int(255), 160] # Cyan to Blue
            elevation = val * 80
        elif metric == "Surge Multiplier":
            val = row['surge_multiplier']
            if val <= 1.0:
                color = [50, 200, 50, 140] # Green (No surge)
            else:
                ratio = (val - 1.0) / 2.5 # normalized between 0 and 1 (cap 3.5)
                color = [int(100 + 155 * ratio), int(200 * (1 - ratio)), int(50), 180] # Orange to Purple-Red
            elevation = (val - 1.0) * 800
        else: # Supply
            val = row['supply_lag_1']
            ratio = val / max_val
            color = [int(100 * (1 - ratio)), int(100), int(255), 160] # Indigo gradient
            elevation = val * 80
        return pd.Series([color, elevation], index=['color', 'elevation'])

    df_filtered[['color', 'elevation']] = df_filtered.apply(lambda r: get_color_and_elevation(r, metric_choice), axis=1)
    
    # ------------------ CORE METRICS DASHBOARD ------------------
    st.subheader(f"System State at {target_dt.strftime('%Y-%m-%d %H:%M')}")
    
    col1, col2, col3, col4 = st.columns(4)
    total_predicted = df_filtered['predicted_demand'].sum()
    total_actual = df_filtered['target_demand'].sum()
    avg_surge = df_filtered['surge_multiplier'].mean()
    total_supply = df_filtered['supply_lag_1'].sum()
    
    col1.metric("Predicted Total Pickups", f"{total_predicted:.1f}")
    col2.metric("Actual Total Pickups", f"{total_actual}")
    col3.metric("Average Surge Multiplier", f"{avg_surge:.2f}x")
    col4.metric("Estimated Supply (Prior Dropoffs)", f"{total_supply:.0f}")
    
    # ------------------ DYNAMIC MAP VISUALIZATION ------------------
    # Center map on the data coordinates
    if not df_filtered.empty:
        sample_hex = df_filtered['h3_index'].iloc[0]
        lat, lon = h3.cell_to_latlng(sample_hex)
        view_state = pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=11.5 if source == 'nyc_cloud' else 10.5,
            pitch=45,
            bearing=0
        )
    else:
        # Default Chicago
        view_state = pdk.ViewState(
            latitude=41.8781,
            longitude=-87.6298,
            zoom=10.5,
            pitch=45,
            bearing=0
        )
    
    # Define deck layer
    layer = pdk.Layer(
        "H3HexagonLayer",
        df_filtered,
        pickable=True,
        get_hexagon="h3_index",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 50],
        line_width_min_pixels=1,
        filled=True,
        extruded=True,
        get_elevation="elevation",
        elevation_scale=1.5,
    )
    
    tooltip = {
        "html": "<b>Hex ID:</b> {h3_index}<br/>"
                "<b>Predicted Pickups:</b> {predicted_demand}<br/>"
                "<b>Actual Pickups:</b> {target_demand}<br/>"
                "<b>Estimated Supply:</b> {supply_lag_1}<br/>"
                "<b>Surge Multiplier:</b> {surge_multiplier}x",
        "style": {"backgroundColor": "black", "color": "white"}
    }
    
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        )
    )
    
    # ------------------ REBALANCING RECOMMENDATIONS ------------------
    st.markdown("---")
    st.subheader("💡 Drivers Rebalancing Recommendations (High Deficit Zones)")
    
    df_filtered['deficit'] = df_filtered['predicted_demand'] - df_filtered['supply_lag_1']
    rebalancing_df = df_filtered[df_filtered['deficit'] > 0].sort_values(by='deficit', ascending=False).head(5)
    
    if not rebalancing_df.empty:
        rec_cols = st.columns(len(rebalancing_df))
        for idx, (_, row) in enumerate(rebalancing_df.iterrows()):
            with rec_cols[idx]:
                st.info(
                    f"**Rank {idx+1}: Hex {row['h3_index']}**\n\n"
                    f"• Deficit: **+{row['deficit']:.1f}** rides\n"
                    f"• Predicted Demand: **{row['predicted_demand']:.1f}**\n"
                    f"• Available Supply: **{row['supply_lag_1']:.0f}**\n"
                    f"• Surge: **{row['surge_multiplier']:.2f}x**"
                )
    else:
        st.success("No deficit zones identified. Driver supply matches or exceeds predicted demand everywhere!")
        
    # ------------------ TIMELINE CHART ------------------
    st.markdown("---")
    st.subheader(f"Temporal Demand Profile (Top Hexagon in {source_display})")
    
    top_hex = df.groupby('h3_index', observed=False)['target_demand'].sum().idxmax()
    top_hex_df = df[df['h3_index'] == top_hex].sort_values(by='timestamp')
    
    chart_data = top_hex_df.set_index('timestamp')[['target_demand', 'predicted_demand']]
    chart_data.columns = ['Actual Pickups', 'Predicted Pickups']
    st.line_chart(chart_data)
    st.caption(f"Historical vs Predicted Pickups in the highest demand H3 cell: **{top_hex}**.")
