
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import shap
import xgboost
from streamlit_autorefresh import st_autorefresh

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="F&B Predictive Maintenance System")

# --- Load Models & Static Assets ---
@st.cache_resource
def load_models():
    return {
        "xgb_model": joblib.load('xgb_model.pkl'), "scaler": joblib.load('scaler.pkl'),
        "inv_cov_matrix": joblib.load('inv_cov_matrix.pkl'), "features": joblib.load('features.pkl'),
        "t2_threshold": joblib.load('t2_threshold.pkl'), "explainer": shap.TreeExplainer(joblib.load('xgb_model.pkl'))
    }
models = load_models()

# --- Dynamic Data Loading based on SKU ---
@st.cache_data
def load_sku_data(sku_name):
    sku_prefix = sku_name.lower().replace(" ", "_")
    data_files = {
        "live_demo_batch": pd.read_csv(f'demo_{sku_prefix}.csv', parse_dates=['timestamp']),
        "golden_batch_df": pd.read_csv(f'golden_{sku_prefix}.csv', parse_dates=['timestamp'])
    }
    for name, df in data_files.items():
        if 'temp_deviation' not in df.columns: df['temp_deviation'] = df['temp_actual'] - df['temp_setpoint']
        if 'airflow_deviation' not in df.columns: df['airflow_deviation'] = df['airflow_actual'] - df['airflow_setpoint']
    return data_files

# --- Helper Functions ---
def calculate_health_score(t2_score, t2_threshold):
    health = 100 * (1 - (t2_score / (t2_threshold * 3)))
    return max(0, min(100, health))

def get_alert_details(data_point):
    top_driver_index = np.argmax(np.abs(models["explainer"](data_point[models["features"]]).values))
    top_driver = models["features"][top_driver_index]
    t2_score = (models['scaler'].transform(data_point[models['features']]) @ models['inv_cov_matrix'] @ models['scaler'].transform(data_point[models['features']]).T)[0][0]

    if "protein" in top_driver: impact = f"**Projected Yield Loss: {(0.5 + (t2_score / models['t2_threshold']) * 2):.1f}%**"
    else: impact = f"**Projected Cpk Drift: {max(0.8, 1.33 - (t2_score / models['t2_threshold']) * 0.2):.2f}** (Target: >1.33)"

    cause = f"Primary Driver: **{top_driver.replace('_', ' ').title()}**."
    playbook = {"step_1": "Acknowledge the alert and verify process area.", "step_2": "Notify Shift Supervisor of the deviation."}
    if "temp" in top_driver: playbook["step_3"] = "Check oven heating elements and thermostats."
    elif "protein" in top_driver: playbook["step_3"] = "Contact QA to verify raw material lot ID."
    elif "airflow" in top_driver: playbook["step_3"] = "Inspect fans, filters, and ductwork for blockages."
    else: playbook["step_3"] = "Follow standard procedure for pressure deviations."
    return {"cause": cause, "impact": impact, "playbook": playbook}

# --- Initialize Session State ---
if 'sku' not in st.session_state:
    st.session_state.sku = "Chocolate Chip"
    st.session_state.running = False
    st.session_state.step = 0
    st.session_state.history = pd.DataFrame()
    st.session_state.is_in_anomaly = False
    st.session_state.last_alert = None
    st.session_state.event_log = []

# --- Auto-Refresh Component ---
if st.session_state.running:
    st_autorefresh(interval=2000, limit=None, key="auto_refresher")

# --- Sidebar ---
st.sidebar.title("Configuration & Controls")
sku_list = ["Chocolate Chip", "Oatmeal Raisin"]
selected_sku = st.sidebar.selectbox("Select Product (SKU)", sku_list, index=sku_list.index(st.session_state.sku))
if selected_sku != st.session_state.sku:
    st.session_state.sku = selected_sku; st.session_state.running = False; st.session_state.step = 0; st.session_state.history = pd.DataFrame(); st.session_state.is_in_anomaly = False; st.session_state.last_alert = None; st.rerun()

sku_data = load_sku_data(st.session_state.sku)
st.sidebar.markdown("---")
button_label = "▶️ Start"
if st.session_state.running: button_label = "Running..."
elif 0 < st.session_state.step < len(sku_data['live_demo_batch']): button_label = "▶️ Resume"
if st.sidebar.button(button_label, key="start_resume", disabled=st.session_state.running): st.session_state.running = True; st.rerun()
if st.sidebar.button("⏹️ Stop", key="stop"): st.session_state.running = False; st.rerun()
if st.sidebar.button("🔁 Reset", key="reset"): st.session_state.running = False; st.session_state.step = 0; st.session_state.history = pd.DataFrame(); st.session_state.is_in_anomaly = False; st.session_state.last_alert = None; st.rerun()

# --- Main App Logic ---
if st.session_state.running and st.session_state.step < len(sku_data['live_demo_batch']):
    st.session_state.step += 1

current_step_index = st.session_state.step -1 if st.session_state.step > 0 else 0
current_data = sku_data['live_demo_batch'].iloc[current_step_index:current_step_index+1]
if st.session_state.step > 0 and st.session_state.history.empty:
    st.session_state.history = sku_data['live_demo_batch'].iloc[0:st.session_state.step]
elif st.session_state.running:
    st.session_state.history = sku_data['live_demo_batch'].iloc[0:st.session_state.step]

# --- Main App Display ---
st.title(f"🏭 Predictive Maintenance System: {st.session_state.sku}")
main_tabs = st.tabs(["📊 Live Dashboard", "📂 Event Log & Reporting"])

with main_tabs[0]:
    status_placeholder = st.empty()
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    t2_score = (models['scaler'].transform(current_data[models['features']]) @ models['inv_cov_matrix'] @ models['scaler'].transform(current_data[models['features']]).T)[0][0]
    health_score = calculate_health_score(t2_score, models['t2_threshold'])
    if health_score < 70 and not st.session_state.is_in_anomaly:
        st.session_state.is_in_anomaly = True; st.session_state.last_alert = get_alert_details(current_data); st.session_state.running = False
    elif health_score >= 70: st.session_state.is_in_anomaly = False

    with col1:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=health_score, title={'text': "Process Health Score"}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "green" if health_score > 80 else "orange" if health_score > 65 else "red"}}))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50, pad=4)); st.plotly_chart(fig_gauge, use_container_width=True)
        predicted_moisture = models['xgb_model'].predict(current_data[models['features']])[0]
        st.metric("💧 Predicted Moisture", f"{predicted_moisture:.2f} %", f"{predicted_moisture - 8.0:.2f} vs Target")
        st.metric("📈 Anomaly Score (T²)", f"{t2_score:.2f}", f"Threshold: {models['t2_threshold']:.1f}")

    with col2:
        history = st.session_state.history
        if not history.empty:
            history['Health Score'] = history.apply(lambda row: calculate_health_score((models['scaler'].transform(row[models['features']].values.reshape(1, -1)) @ models['inv_cov_matrix'] @ models['scaler'].transform(row[models['features']].values.reshape(1, -1)).T)[0][0], models['t2_threshold']), axis=1)
            anomaly_points = history[history['Health Score'] < 70]
            fig_chart = go.Figure(); fig_chart.add_trace(go.Scatter(x=history['timestamp'], y=history['Health Score'], mode='lines', name='Health Score', line=dict(color='royalblue')))
            if not anomaly_points.empty: fig_chart.add_trace(go.Scatter(x=anomaly_points['timestamp'], y=anomaly_points['Health Score'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
            fig_chart.update_layout(title="Health Score Over Time", yaxis_range=[0,105]); st.plotly_chart(fig_chart, use_container_width=True)

    if st.session_state.last_alert:
        with st.expander("🚨 ALERT: Corrective Action Required", expanded=True):
             # --- THIS IS THE FIX ---
             st.error(st.session_state.last_alert['impact']); st.warning(st.session_state.last_alert['cause'])
             # --- END OF FIX ---
             with st.form("playbook_form"):
                 responses = {step: st.checkbox(label=st.session_state.last_alert['playbook'][step]) for step in st.session_state.last_alert['playbook']}
                 submitted = st.form_submit_button("Acknowledge & Log Event")
                 if submitted and all(responses.values()):
                     st.session_state.event_log.append({"Timestamp": pd.Timestamp.now(), "SKU": st.session_state.sku, "Alert Cause": st.session_state.last_alert['cause'], "Quality Impact": st.session_state.last_alert['impact'], "Actions Taken": ", ".join([st.session_state.last_alert['playbook'][s] for s,r in responses.items() if r]), "Acknowledged By": "Operator_01"})
                     st.success("Event logged successfully!"); st.session_state.last_alert = None; time.sleep(1); st.rerun()
                 elif submitted: st.error("Please complete all playbook steps before logging.")

    with st.expander("Deep Dive: Process Parameter Charts", expanded=True):
        if not st.session_state.history.empty:
            param_tabs = st.tabs(["Temperature", "Airflow", "Pressure", "Raw Material"])
            with param_tabs[0]:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['temp_actual'], name="Live", line=dict(color='red'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['temp_setpoint'], name="Setpoint", line=dict(color='gray', dash='dash'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=sku_data['golden_batch_df']['temp_actual'], name="Golden Batch", line=dict(color='gold', dash='dot'))); st.plotly_chart(fig, use_container_width=True)
            with param_tabs[1]:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['airflow_actual'], name="Live", line=dict(color='deepskyblue'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['airflow_setpoint'], name="Setpoint", line=dict(color='gray', dash='dash'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=sku_data['golden_batch_df']['airflow_actual'], name="Golden Batch", line=dict(color='gold', dash='dot'))); st.plotly_chart(fig, use_container_width=True)
            with param_tabs[2]:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['pressure_actual'], name="Live", line=dict(color='blueviolet'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=sku_data['golden_batch_df']['pressure_actual'], name="Golden Batch", line=dict(color='gold', dash='dot'))); st.plotly_chart(fig, use_container_width=True)
            with param_tabs[3]:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=history['timestamp'], y=history['raw_material_protein'], name="Live", line=dict(color='brown'))); fig.add_trace(go.Scatter(x=history['timestamp'], y=sku_data['golden_batch_df']['raw_material_protein'], name="Golden Batch", line=dict(color='gold', dash='dot'))); st.plotly_chart(fig, use_container_width=True)

    if st.session_state.running: status_placeholder.header(f"▶️ Running... (Health: {health_score:.0f}%)")
    elif st.session_state.step >= len(sku_data['live_demo_batch']): status_placeholder.success("✅ Simulation Finished.")
    elif st.session_state.is_in_anomaly: status_placeholder.error("🚨 Simulation Paused on Anomaly. Review details and complete playbook.")
    elif st.session_state.step > 0: status_placeholder.info("ℹ️ Simulation is paused. Press 'Resume' to continue.")
    else: status_placeholder.info("ℹ️ Simulation is ready. Press 'Start' to begin.")

with main_tabs[1]:
    st.header("📂 Historical Event Log")
    if not st.session_state.event_log: st.info("No events have been logged during this session.")
    else:
        log_df = pd.DataFrame(st.session_state.event_log).sort_values(by="Timestamp", ascending=False)
        st.dataframe(log_df, use_container_width=True)
        csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Log as CSV", data=csv, file_name='anomaly_event_log.csv', mime='text/csv')
