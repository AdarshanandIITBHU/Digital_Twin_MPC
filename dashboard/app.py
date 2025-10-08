# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import joblib # Import joblib for loading the models

# Add project root to path for correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reactor_model.kalman_filter import run_ukf_tracker

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Ammonia Reactor Digital Twin")

# --- Constants and File Paths ---
LIVE_DATA_FILE = "live_data.csv"
GOLDEN_BATCH_FILE = "data/processed/golden_batch.csv"
TEMP_MODEL_FILE = "reactor_model/temp_model.joblib"
PRESS_MODEL_FILE = "reactor_model/press_model.joblib"
# --- FIX #2: Lowered the threshold to make the system more sensitive ---
DEVIATION_THRESHOLD = 0.01 # A 1% deviation from the target
DURATION_THRESHOLD = 5     # Deviation must last 5 seconds to trigger a major warning

# --- Initialize Session State ---
if 'deviation_counter' not in st.session_state:
    st.session_state.deviation_counter = 0

# --- NEW: Load the trained ML models ---
# Use st.cache_resource to load the models only once
@st.cache_resource
def load_models():
    """Loads the trained recommender models from disk."""
    try:
        temp_model = joblib.load(TEMP_MODEL_FILE)
        press_model = joblib.load(PRESS_MODEL_FILE)
        return temp_model, press_model
    except FileNotFoundError:
        return None, None

temp_model, press_model = load_models()
if not temp_model or not press_model:
    st.error("ML models not found! Please run 'scripts/train_model.py' first.")
    st.stop()


# === Main App UI Setup ===
st.title("🏭 Ammonia Reactor Digital Twin with ML Control")

# --- Load Golden Batch Data ---
try:
    golden_batch_df = pd.read_csv(GOLDEN_BATCH_FILE)
except FileNotFoundError:
    st.error(f"Error: '{GOLDEN_BATCH_FILE}' not found. Please run 'scripts/generate_datasets.py' first.")
    st.stop()

# --- Main App Body ---
st.header("Live Process Monitoring")

try:
    live_data = pd.read_csv(LIVE_DATA_FILE)
    if live_data.empty:
        st.warning("⏳ Waiting for data feed... Please run `feed_simulator.py` in a separate terminal.")
        st.stop()
except (FileNotFoundError, pd.errors.EmptyDataError):
    st.warning("⏳ Waiting for data feed... Please run `feed_simulator.py` in a separate terminal.")
    st.stop()

# --- Run UKF and Comparison Logic ---
current_T = live_data['Temperature'].iloc[-1]
current_P = live_data['Pressure'].iloc[-1]
initial_conditions = [1.0, 3.0, 0.0]
noise_levels = {'measurement': 0.01, 'process': 0.0001}
measurements = live_data['NH3'].values
ukf_estimates = run_ukf_tracker(measurements, initial_conditions, current_T, current_P, noise_levels)
latest_estimate_NH3 = ukf_estimates[-1, 2]
current_time = live_data['Time'].iloc[-1]
target_value_NH3 = np.interp(current_time, golden_batch_df['Time'], golden_batch_df['NH3'])
deviation = latest_estimate_NH3 - target_value_NH3

if abs(deviation) > DEVIATION_THRESHOLD:
    st.session_state.deviation_counter += 1
else:
    st.session_state.deviation_counter = 0

# --- Create the Dashboard Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Process Status")
    if st.session_state.deviation_counter >= DURATION_THRESHOLD:
        st.error(f"🔴 WARNING: Process deviated for {st.session_state.deviation_counter}s!")
        
        # --- NEW: Get and Display ML Recommendations ---
        st.subheader("🤖 ML Recommendations")
        
        # 1. Prepare the input data for the models in a DataFrame
        model_input = pd.DataFrame([{
            'Time': current_time,
            'Current_NH3': latest_estimate_NH3,
            'Deviation': deviation,
            'Current_Temp': current_T,
            'Current_Press': current_P
        }])
        
        # 2. Get predictions from the loaded models
        temp_adjustment = temp_model.predict(model_input)[0]
        press_adjustment = press_model.predict(model_input)[0]
        
        # 3. Display the recommendations
        rec_col1, rec_col2 = st.columns(2)
        rec_col1.metric("Temp Adjustment", f"{temp_adjustment:+.0f} K")
        rec_col2.metric("Press Adjustment", f"{press_adjustment:+.0f} atm")
        
    else:
        st.success("✅ STATUS: Process is on track.")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Live Estimate (NH₃)", f"{latest_estimate_NH3:.3f}")
    m_col2.metric("Target Value (NH₃)", f"{target_value_NH3:.3f}")
    m_col3.metric("Deviation", f"{deviation:+.3f}", delta_color="inverse")

    st.subheader("Current Conditions")
    golden_conditions = golden_batch_df.iloc[0]
    comparison_df = pd.DataFrame({
        'Parameter': ['Temperature (K)', 'Pressure (atm)'],
        'Current': [f"{current_T:.0f}", f"{current_P:.0f}"],
        'Optimal': [f"{golden_conditions['Temperature']:.0f}", f"{golden_conditions['Pressure']:.0f}"]
    })
    st.table(comparison_df.set_index('Parameter'))


with col2:
    st.subheader("Live Trajectory vs. Optimal Path")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(golden_batch_df['Time'], golden_batch_df['NH3'], label='Golden Batch (Optimal Path)', color='gold', linewidth=4, alpha=0.7)
    ax.scatter(live_data['Time'], measurements, label='Live Sensor Data (Noisy)', color='red', marker='x', s=20)
    ax.plot(live_data['Time'], ukf_estimates[:, 2], label='UKF Estimate (True State)', color='limegreen', linestyle='--', linewidth=2.5)
    ax.set_title("Ammonia Concentration over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Concentration of NH₃")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(1.0, golden_batch_df['NH3'].max() * 1.2))
    st.pyplot(fig)
    # --- FIX #1: Close the figure to prevent duplicates on rerun ---
    plt.close(fig)

# --- Auto-refresh the page ---
time.sleep(1)
st.rerun()

