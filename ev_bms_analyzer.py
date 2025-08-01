import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import time
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# --- App Configuration ---
st.set_page_config(
    page_title="🔋 Advanced EV Battery BMS", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .cell-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .alert-good {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .sidebar .stSelectbox, .sidebar .stSlider {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []

if 'refresh_time' not in st.session_state:
    st.session_state.refresh_time = datetime.now()

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'cell_health' not in st.session_state:
    st.session_state.cell_health = {}

# --- Header ---
st.markdown("""
    <div class="main-header">
        <h1>🔋 Advanced EV Battery Management System (BMS)</h1>
        <p>Real-time monitoring, safety analysis, and predictive maintenance for EV battery packs</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("## 🔧 BMS Configuration")
st.sidebar.markdown("---")

# Battery Pack Configuration
st.sidebar.subheader("📦 Battery Pack Setup")
num_cells = st.sidebar.slider("Number of cells in pack", 4, 20, 12)
pack_voltage = st.sidebar.selectbox("Pack nominal voltage", ["48V", "72V", "96V", "144V"], index=2)
pack_capacity = st.sidebar.slider("Pack capacity (kWh)", 10, 100, 50)

# Cell Configuration
st.sidebar.subheader("🔋 Cell Configuration")
default_cell_type = st.sidebar.selectbox("Default cell type", ['LiFePO4 (LFP)', 'NMC', 'LTO', 'NCA'], key="default_type")
cell_types = []
cell_names = []

with st.sidebar.expander("Individual Cell Settings", expanded=False):
    for i in range(num_cells):
        col1, col2 = st.columns(2)
        with col1:
            cell_name = st.text_input(f"Cell {i+1} name", f"Cell_{i+1:02d}", key=f"name_{i}")
        with col2:
            cell_type = st.selectbox(f"Type", ['LiFePO4 (LFP)', 'NMC', 'LTO', 'NCA'], 
                                   index=['LiFePO4 (LFP)', 'NMC', 'LTO', 'NCA'].index(default_cell_type),
                                   key=f"type_{i}")
        cell_names.append(cell_name)
        cell_types.append(cell_type)

# Monitoring Settings
st.sidebar.subheader("📊 Monitoring Settings")
auto_refresh = st.sidebar.checkbox("🔁 Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, 5)
enable_alerts = st.sidebar.checkbox("🚨 Safety Alerts", value=True)
simulation_mode = st.sidebar.selectbox("Simulation Mode", ["Normal Operation", "Charging", "Discharging", "Fault Simulation"])

# --- Generate Enhanced Cell Data ---
def get_cell_specs(cell_type):
    specs = {
        'LiFePO4 (LFP)': {'nominal_v': 3.2, 'max_v': 3.6, 'min_v': 2.5, 'max_temp': 60, 'max_current': 100},
        'NMC': {'nominal_v': 3.6, 'max_v': 4.2, 'min_v': 2.8, 'max_temp': 45, 'max_current': 80},
        'LTO': {'nominal_v': 2.4, 'max_v': 2.8, 'min_v': 1.5, 'max_temp': 55, 'max_current': 120},
        'NCA': {'nominal_v': 3.7, 'max_v': 4.3, 'min_v': 2.7, 'max_temp': 50, 'max_current': 90}
    }
    return specs.get(cell_type, specs['LiFePO4 (LFP)'])

def generate_cell_data(cell_name, cell_type, simulation_mode, cell_index):
    specs = get_cell_specs(cell_type)
    temp =25.0  # Default temperature
    # Base parameters with simulation variations
    base_voltage = specs['nominal_v']
    
    if simulation_mode == "Charging":
        voltage = round(random.uniform(base_voltage, specs['max_v'] * 0.95), 3)
        current = round(random.uniform(20, specs['max_current'] * 0.8), 2)
        soc = round(random.uniform(70, 95), 1)
        power = voltage * current
    elif simulation_mode == "Discharging":
        voltage = round(random.uniform(specs['min_v'] * 1.1, base_voltage), 3)
        current = -round(random.uniform(10, specs['max_current'] * 0.6), 2)  # Negative for discharge
        soc = round(random.uniform(20, 80), 1)
        power = voltage * abs(current)
    elif simulation_mode == "Fault Simulation":
        # Introduce some faults randomly
        if random.random() < 0.3:  # 30% chance of fault
            voltage = round(random.uniform(specs['min_v'] * 0.9, specs['max_v'] * 1.1), 3)
            current = round(random.uniform(-50, specs['max_current'] * 1.2), 2)
            temp = round(random.uniform(45, 70), 1)
        else:
            voltage = round(random.uniform(base_voltage * 0.95, base_voltage * 1.05), 3)
            current = round(random.uniform(-30, 50), 2)
            temp = round(random.uniform(25, 40), 1)
        soc = round(random.uniform(10, 100), 1)
        power = voltage * abs(current)
    else:  # Normal Operation
        voltage = round(random.uniform(base_voltage * 0.95, base_voltage * 1.05), 3)
        current = round(random.uniform(-20, 30), 2)
        temp = round(random.uniform(25, 40), 1)
        soc = round(random.uniform(40, 90), 1)
        power = voltage * abs(current)
    
    # Additional BMS parameters
    internal_resistance = round(random.uniform(0.5, 2.0), 3)
    cycle_count = random.randint(100, 5000)
    health = max(80, 100 - (cycle_count / 50))  # Health degrades with cycles
    
    # Calculate capacity based on SOC and nominal capacity
    nominal_capacity = 50  # Ah
    available_capacity = round((soc / 100) * nominal_capacity, 2)
    
    # Time calculations
    if current != 0:
        time_to_full = round(((100 - soc) / 100) * nominal_capacity / abs(current), 2) if current > 0 else 0
        time_to_empty = round((soc / 100) * nominal_capacity / abs(current), 2) if current < 0 else 0
    else:
        time_to_full = 0
        time_to_empty = 0
    
    return {
        "Cell_Name": cell_name,
        "Cell_Type": cell_type,
        "Voltage (V)": voltage,
        "Current (A)": current,
        "Temperature (°C)": temp,
        "SOC (%)": soc,
        "Power (W)": round(power, 2),
        "Internal_Resistance (mΩ)": internal_resistance,
        "Available_Capacity (Ah)": available_capacity,
        "Health (%)": round(health, 1),
        "Cycle_Count": cycle_count,
        "Time_to_Full (h)": time_to_full,
        "Time_to_Empty (h)": time_to_empty,
        "Status": "Normal",
        "Timestamp": datetime.now()
    }

# Generate data for all cells
cells_data = {}
current_alerts = []

for idx, (cell_name, cell_type) in enumerate(zip(cell_names, cell_types)):
    cell_data = generate_cell_data(cell_name, cell_type, simulation_mode, idx)
    cells_data[cell_name] = cell_data
    
    # Safety checks and alerts
    if enable_alerts:
        specs = get_cell_specs(cell_type)
        
        # Voltage alerts
        if cell_data["Voltage (V)"] > specs['max_v']:
            current_alerts.append(f"⚠️ {cell_name}: Overvoltage ({cell_data['Voltage (V)']}V)")
            cell_data["Status"] = "CRITICAL"
        elif cell_data["Voltage (V)"] < specs['min_v']:
            current_alerts.append(f"⚠️ {cell_name}: Undervoltage ({cell_data['Voltage (V)']}V)")
            cell_data["Status"] = "CRITICAL"
        
        # Temperature alerts
        if cell_data["Temperature (°C)"] > specs['max_temp']:
            current_alerts.append(f"🌡️ {cell_name}: Overtemperature ({cell_data['Temperature (°C)']}°C)")
            cell_data["Status"] = "WARNING" if cell_data["Status"] == "Normal" else cell_data["Status"]
        
        # Current alerts
        if abs(cell_data["Current (A)"]) > specs['max_current']:
            current_alerts.append(f"⚡ {cell_name}: Overcurrent ({cell_data['Current (A)']}A)")
            cell_data["Status"] = "WARNING" if cell_data["Status"] == "Normal" else cell_data["Status"]
        
        # Health alerts
        if cell_data["Health (%)"] < 80:
            current_alerts.append(f"🔋 {cell_name}: Low health ({cell_data['Health (%)']}%)")
            cell_data["Status"] = "WARNING" if cell_data["Status"] == "Normal" else cell_data["Status"]

# Update session state
st.session_state.alerts = current_alerts
df_cells = pd.DataFrame.from_dict(cells_data, orient='index')

# --- Main Dashboard ---
# System Overview Metrics
st.subheader("📊 System Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_voltage = df_cells["Voltage (V)"].mean()
    st.metric("Avg Voltage", f"{avg_voltage:.2f}V", f"{avg_voltage - 3.5:.2f}V")

with col2:
    total_current = df_cells["Current (A)"].sum()
    st.metric("Total Current", f"{total_current:.1f}A", f"{'Charging' if total_current > 0 else 'Discharging'}")

with col3:
    avg_temp = df_cells["Temperature (°C)"].mean()
    st.metric("Avg Temperature", f"{avg_temp:.1f}°C", f"{avg_temp - 30:.1f}°C")

with col4:
    avg_soc = df_cells["SOC (%)"].mean()
    st.metric("Pack SOC", f"{avg_soc:.1f}%", f"{avg_soc - 50:.1f}%")

with col5:
    avg_health = df_cells["Health (%)"].mean()
    st.metric("Pack Health", f"{avg_health:.1f}%", f"{100 - avg_health:.1f}%")

# Safety Alerts
if current_alerts:
    st.subheader("🚨 Safety Alerts")
    for alert in current_alerts:
        if "CRITICAL" in alert or "Overvoltage" in alert or "Undervoltage" in alert:
            st.markdown(f'<div class="alert-critical">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-good">✅ All systems normal - No alerts</div>', unsafe_allow_html=True)

# Individual Cell Data Table
st.subheader("📋 Individual Cell Monitoring")
# Color code the dataframe based on status
def highlight_status(row):
    if row['Status'] == 'CRITICAL':
        return ['background-color: #ff6b6b'] * len(row)
    elif row['Status'] == 'WARNING':
        return ['background-color: #ffa726'] * len(row)
    else:
        return ['background-color: #66bb6a'] * len(row)

styled_df = df_cells.style.apply(highlight_status, axis=1)
st.dataframe(styled_df, use_container_width=True)

# --- Advanced Visualizations ---
st.subheader("📈 Real-time Cell Analysis")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Parameters", "🔄 Balance", "🌡️ Thermal", "⚡ Power"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Voltage distribution
        fig_voltage = px.bar(df_cells, x=df_cells.index, y="Voltage (V)", 
                           color="Status", title="Cell Voltages",
                           color_discrete_map={"Normal": "#66bb6a", "WARNING": "#ffa726", "CRITICAL": "#ff6b6b"})
        fig_voltage.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_voltage, use_container_width=True)
    
    with col2:
        # SOC distribution
        fig_soc = px.bar(df_cells, x=df_cells.index, y="SOC (%)", 
                        color="SOC (%)", title="State of Charge",
                        color_continuous_scale="viridis")
        fig_soc.update_layout(height=400)
        st.plotly_chart(fig_soc, use_container_width=True)

with tab2:
    # Cell balance analysis
    voltage_std = df_cells["Voltage (V)"].std()
    soc_std = df_cells["SOC (%)"].std()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Voltage Imbalance", f"{voltage_std:.3f}V", "Lower is better")
    with col2:
        st.metric("SOC Imbalance", f"{soc_std:.2f}%", "Lower is better")
    
    # Balance visualization
    fig_balance = make_subplots(rows=2, cols=1, 
                               subplot_titles=('Voltage Balance', 'SOC Balance'),
                               vertical_spacing=0.12)
    
    fig_balance.add_trace(go.Scatter(x=df_cells.index, y=df_cells["Voltage (V)"], 
                                   mode='lines+markers', name='Voltage'), row=1, col=1)
    fig_balance.add_trace(go.Scatter(x=df_cells.index, y=df_cells["SOC (%)"], 
                                   mode='lines+markers', name='SOC'), row=2, col=1)
    
    fig_balance.update_layout(height=500)
    st.plotly_chart(fig_balance, use_container_width=True)

with tab3:
    # Thermal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp = px.bar(df_cells, x=df_cells.index, y="Temperature (°C)", 
                         color="Temperature (°C)", title="Cell Temperatures",
                         color_continuous_scale="thermal")
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Temperature distribution
        fig_temp_dist = px.histogram(df_cells, x="Temperature (°C)", 
                                   title="Temperature Distribution", nbins=10)
        st.plotly_chart(fig_temp_dist, use_container_width=True)

with tab4:
    # Power analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_power = px.bar(df_cells, x=df_cells.index, y="Power (W)", 
                          color="Current (A)", title="Cell Power Output",
                          color_continuous_scale="RdYlBu")
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col2:
        # Current vs Power scatter
        fig_scatter = px.scatter(df_cells, x="Current (A)", y="Power (W)", 
                               color="Temperature (°C)", size="SOC (%)",
                               title="Current vs Power Analysis")
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Predictive Analytics ---
st.subheader("🤖 Predictive Analytics & Diagnostics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔮 Health Prediction")
    
    # Simple health prediction based on cycle count and temperature
    avg_cycles = df_cells["Cycle_Count"].mean()
    avg_temp_exposure = df_cells["Temperature (°C)"].mean()
    
    # Simplified prediction model
    predicted_life = max(0, 10000 - avg_cycles - (avg_temp_exposure - 25) * 100)
    health_trend = "Declining" if avg_temp_exposure > 40 else "Stable"
    
    st.metric("Predicted Remaining Cycles", f"{predicted_life:.0f}", health_trend)
    st.metric("Current Average Cycles", f"{avg_cycles:.0f}")

with col2:
    st.markdown("#### ⚡ Charging Time Estimation")
    
    target_soc = st.slider("Target SOC (%)", int(avg_soc), 100, 90)
    
    # Calculate charging time based on current SOC and charging current
    charging_cells = df_cells[df_cells["Current (A)"] > 0]
    if not charging_cells.empty:
        avg_charge_current = charging_cells["Current (A)"].mean()
        soc_diff = target_soc - avg_soc
        estimated_time = (soc_diff / 100) * (pack_capacity * 1000) / (avg_charge_current * avg_voltage)
        st.metric("Estimated Time to Target", f"{estimated_time:.1f} hours")
    else:
        st.info("No cells currently charging")

# --- Data Export and History ---
st.subheader("📤 Data Management")

col1, col2, col3 = st.columns(3)

# Save current data to history
current_data = df_cells.copy()
current_data['Pack_Voltage'] = pack_voltage
current_data['Pack_Capacity'] = pack_capacity
current_data['Simulation_Mode'] = simulation_mode

if len(st.session_state.history) == 0 or len(st.session_state.history) % 10 == 0:
    st.session_state.history.append(current_data)

with col1:
    if st.session_state.history:
        export_df = pd.concat(st.session_state.history).reset_index()
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Historical CSV", csv, 
                         f"bms_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                         "text/csv")

with col2:
    current_csv = df_cells.to_csv(index=True).encode('utf-8')
    st.download_button("📋 Download Current Data", current_csv,
                     f"current_bms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                     "text/csv")

with col3:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared!")

# History summary
if st.session_state.history:
    st.info(f"📊 Historical records: {len(st.session_state.history)} snapshots")

# --- Auto Refresh Logic ---
if auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.refresh_time).seconds
    if time_since_refresh >= refresh_interval:
        st.session_state.refresh_time = datetime.now()
        st.rerun()
    
    # Progress bar for next refresh
    progress = min(time_since_refresh / refresh_interval, 1.0)
    st.progress(progress, f"Next refresh in {refresh_interval - time_since_refresh} seconds")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 2rem;'>
        <h4>🔧 Advanced EV Battery Management System</h4>
        <p>Real-time monitoring • Safety analysis • Predictive maintenance • Data export</p>
        <p><em>Developed for comprehensive EV battery pack management and diagnostics</em></p>
    </div>
""", unsafe_allow_html=True)