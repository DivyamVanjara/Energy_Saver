import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import random
import re

# --- Page Configuration (SHEOS) ---
st.set_page_config(
    page_title="SHEOS - Energy Dashboard",
    page_icon="☀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lighter-themed CSS for improved readability ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

/* Lighter background gradient for readability */
.stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #eef6ff 50%, #ffffff 100%);
    color: #0b2545;
}

/* Soft decorative orb - much lighter */
.stApp::before {
    content: '';
    position: fixed;
    top: 6%;
    left: 4%;
    width: 360px;
    height: 360px;
    background: radial-gradient(circle, rgba(255, 223, 93, 0.12) 0%, transparent 70%);
    border-radius: 50%;
    z-index: 0;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #fff7ed 0%, #fff1d6 100%);
    color: #0b2545;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 1.3rem;
    box-shadow: 0 8px 24px rgba(11,37,69,0.06);
}

/* Card styling - light and readable */
.metric-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(11,37,69,0.06);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #0b2545;
}

/* Hover effect */
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(11,37,69,0.06);
}

/* Metric value styling - darker for contrast */
.big-metric {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0b6b3a;
    margin: 0;
}

/* Buttons - slightly darker accent on light background */
.stButton>button {
    background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600;
}

/* Sidebar - lighter but distinct */
.css-1d391kg, .css-1v3fvcr {
    background: linear-gradient(180deg, #ffffff, #f6f8ff) !important;
    border-right: 1px solid rgba(11,37,69,0.04);
}

/* Tabs - light theme */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(11,37,69,0.03);
    padding: 0.4rem;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.95);
    border-radius: 8px;
    color: #0b2545;
    font-weight: 600;
    padding: 0.5rem 1rem;
    border: 1px solid rgba(11,37,69,0.04);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ffedd5 0%, #fff1e0 100%);
    color: #0b2545;
    border: 1px solid rgba(11,37,69,0.06);
}

/* Success/Warning boxes - adjusted colors for light bg */
.success-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(52,211,153,0.05) 100%);
    border-radius: 10px;
    padding: 0.8rem;
    color: #065f46;
    font-weight: 600;
}
.warning-box {
    background: linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(248,113,113,0.03) 100%);
    border-radius: 10px;
    padding: 0.8rem;
    color: #7f1d1d;
    font-weight: 600;
}

/* Hide Streamlit branding area but keep accessible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Dataframe / table wrapping and responsiveness for light theme */
section[aria-label="Main"] * {
    box-sizing: border-box;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    white-space: normal !important;
    color: #0b2545;
}
.stDataFrame, .stTable, table {
    table-layout: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
    background: rgba(255,255,255,0.98) !important;
}

/* Inputs style - darker text on light inputs */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    background: rgba(246,248,255,1) !important;
    border: 1px solid rgba(11,37,69,0.06) !important;
    border-radius: 6px !important;
    color: #0b2545 !important;
}

/* Small helper */
.long-label { font-size: 0.95rem; line-height: 1.1; color: #0b2545; }

</style>
""",
    unsafe_allow_html=True,
)

# --- Constants & Load Profile ---
PANEL_AREA_M2 = 2.0
GRID_RATE = 7.0  # ₹ per kWh
LOAD_PROFILE = {
    "AC (1.5 Ton)": {"qty": 1, "kwh": 1.5},
    "Fans": {"qty": 5, "kwh": 0.075},
    "LEDs": {"qty": 10, "kwh": 0.01},
    "Washing Machine": {"qty": 1, "kwh": 0.5}
}

# --- Helpers to prevent overflow in strings ---
def soft_wrap_token(s: str, max_token_len: int = 30) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r'(\S{' + str(max_token_len) + r',})',
                  lambda m: '\u200b'.join([m.group(0)[i:i+max_token_len] for i in range(0, len(m.group(0)), max_token_len)]),
                  s)

def shorten_label(s: str, width: int = 60) -> str:
    s = soft_wrap_token(s, 30)
    if not isinstance(s, str):
        return s
    if len(s) > width:
        return s[: width - 3] + "..."
    return s

# --- Solar system core class (constructor fixed) ---
class SolarHomeSystem:
    def __init__(self, df: pd.DataFrame, num_panels: int = 10):
        self.df = df.reset_index(drop=True).copy()
        self.num_panels = int(num_panels)
        self.scaler = StandardScaler()
        self.ml_model = None
        if 'datetime' in self.df.columns:
            if not np.issubdtype(self.df['datetime'].dtype, np.datetime64):
                self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')
        if len(self.df) > 0:
            self.current_index = random.randint(0, max(0, len(self.df) - 1))
        else:
            self.current_index = 0

    def calculate_physics_generation(self, row_or_df):
        if isinstance(row_or_df, pd.DataFrame):
            irr = row_or_df['irradiance_W_m2'].astype(float).values
            cloud = row_or_df['cloud_percentage'].astype(float).values
            eff = np.where(cloud <= 30, 1.0, np.where(cloud <= 60, 0.8, np.where(cloud <= 90, 0.6, 0.2)))
            total_area = self.num_panels * PANEL_AREA_M2
            power_kw = (irr * total_area * eff) / 1000.0
            return np.maximum(0.0, power_kw)
        else:
            try:
                irr = float(row_or_df['irradiance_W_m2'])
                cloud = float(row_or_df['cloud_percentage'])
            except Exception:
                irr = 0.0
                cloud = 100.0
            if cloud <= 30:
                efficiency = 1.0
            elif cloud <= 60:
                efficiency = 0.8
            elif cloud <= 90:
                efficiency = 0.6
            else:
                efficiency = 0.2
            total_area = self.num_panels * PANEL_AREA_M2
            power_kw = (irr * total_area * efficiency) / 1000.0
            return max(0.0, power_kw)

    def train_solar_ai(self):
        training_data = self.df.copy()
        if 'datetime' not in training_data.columns:
            raise ValueError("DataFrame must contain 'datetime' column for training.")
        training_data['hour'] = training_data['datetime'].dt.hour
        training_data['power_output'] = training_data.apply(lambda r: self.calculate_physics_generation(r), axis=1)
        features = ['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']
        X = training_data[features].fillna(0.0)
        y = training_data['power_output'].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        y_pred = self.ml_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    def get_prediction(self, feature_df: pd.DataFrame):
        expected = ['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']
        for c in expected:
            if c not in feature_df.columns:
                feature_df[c] = 0.0
        feature_df = feature_df[expected].astype(float).copy()
        if self.ml_model is not None:
            X_scaled = self.scaler.transform(feature_df)
            preds = self.ml_model.predict(X_scaled)
            return np.maximum(0.0, preds)
        else:
            return self.calculate_physics_generation(feature_df)

    def get_current_row(self):
        if len(self.df) == 0:
            return pd.Series({
                'datetime': pd.Timestamp.now(),
                'irradiance_W_m2': 0.0,
                'temperature_C': 0.0,
                'cloud_percentage': 100.0
            })
        idx = min(max(0, int(self.current_index)), len(self.df) - 1)
        return self.df.iloc[idx]

    def predict_day_generation(self):
        current_dt = self.get_current_row()['datetime']
        start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = current_dt.replace(hour=23, minute=59, second=59, microsecond=0)
        mask = (self.df['datetime'] >= start) & (self.df['datetime'] <= end)
        day_data = self.df.loc[mask].copy()
        if day_data.empty:
            idx = min(self.current_index, len(self.df) - 1)
            start_idx = max(0, idx - 12)
            day_data = self.df.iloc[start_idx:start_idx + 24].copy()
        day_data['hour'] = day_data['datetime'].dt.hour
        features = day_data[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']].fillna(0.0)
        day_data['pred'] = self.get_prediction(features)
        return day_data

    def intelligent_scheduler(self):
        current_dt = self.get_current_row()['datetime']
        end_dt = current_dt + timedelta(hours=24)
        mask = (self.df['datetime'] > current_dt) & (self.df['datetime'] <= end_dt)
        future_df = self.df.loc[mask].copy()
        if future_df.empty:
            idx = min(self.current_index, len(self.df) - 1)
            future_df = self.df.iloc[idx: idx + 24].copy()
        future_df['hour'] = future_df['datetime'].dt.hour
        features = future_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']].fillna(0.0)
        future_df['pred'] = self.get_prediction(features)
        best_slots = future_df.sort_values('pred', ascending=False).head(3)
        return best_slots

    def comparative_analysis(self):
        current_dt = self.get_current_row()['datetime']
        end_dt = current_dt + timedelta(days=30)
        mask = (self.df['datetime'] >= current_dt) & (self.df['datetime'] < end_dt)
        sim_df = self.df.loc[mask].copy()
        if sim_df.empty:
            sim_df = self.df.tail(min(len(self.df), 30 * 24)).copy()
        sim_df['hour'] = sim_df['datetime'].dt.hour
        features = sim_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']].fillna(0.0)
        sim_df['solar_gen'] = self.get_prediction(features)
        bill_A_grid_only = 0.0
        bill_B_solar_unopt = 0.0
        bill_C_solar_opt = 0.0
        sim_df['date'] = sim_df['datetime'].dt.date
        for date, day_group in sim_df.groupby('date'):
            if day_group.empty:
                continue
            peak_sun_idx = day_group['solar_gen'].idxmax()
            peak_hour = int(day_group.loc[peak_sun_idx, 'hour'])
            for _, row in day_group.iterrows():
                h = int(row['hour'])
                solar = float(row['solar_gen'])
                hourly_load_base = 0.0
                hourly_load_base += LOAD_PROFILE["Fans"]["kwh"] * LOAD_PROFILE["Fans"]["qty"]
                if h >= 22 or h < 6:
                    hourly_load_base += LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] * LOAD_PROFILE["AC (1.5 Ton)"]["qty"]
                if 18 <= h <= 23:
                    hourly_load_base += LOAD_PROFILE["LEDs"]["kwh"] * LOAD_PROFILE["LEDs"]["qty"]
                wm_kwh = LOAD_PROFILE["Washing Machine"]["kwh"]
                load_A = hourly_load_base
                if h == 20:
                    load_A += wm_kwh
                bill_A_grid_only += (load_A * GRID_RATE)
                load_B = hourly_load_base
                if h == 20:
                    load_B += wm_kwh
                net_B = max(0.0, load_B - solar)
                bill_B_solar_unopt += (net_B * GRID_RATE)
                load_C = hourly_load_base
                if h == peak_hour:
                    load_C += wm_kwh
                net_C = max(0.0, load_C - solar)
                bill_C_solar_opt += (net_C * GRID_RATE)
        return {
            'grid_only': bill_A_grid_only,
            'solar_unopt': bill_B_solar_unopt,
            'solar_opt': bill_C_solar_opt
        }

# --- Session state initialization ---
if 'system' not in st.session_state:
    st.session_state.system = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'num_panels' not in st.session_state:
    st.session_state.num_panels = 10

# --- Header (SHEOS) ---
st.markdown('<div class="main-header">☀ SHEOS</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color: #0b2545; font-size: 1.0rem; margin-top: -0.35rem;">Advanced Solar Analytics & Optimization Platform</p>',
    unsafe_allow_html=True
)

# --- Sidebar configuration ---
with st.sidebar:
    st.markdown("### ⚙ System Configuration")
    uploaded_file = st.file_uploader("📁 Upload Weather Data (CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            renames = {
                'timestamp': 'datetime',
                'time': 'datetime',
                'date_time': 'datetime',
                'temp_C': 'temperature_C',
                'temperature': 'temperature_C',
                'precipitation_probability_pct': 'cloud_percentage',
                'cloud_pct': 'cloud_percentage',
                'cloud_percentage': 'cloud_percentage',
                'irradiance': 'irradiance_W_m2',
                'irradiance_W_m2': 'irradiance_W_m2',
            }
            raw_df.rename(columns={k: v for k, v in renames.items() if k in raw_df.columns}, inplace=True)
            expected_cols = ['datetime', 'irradiance_W_m2', 'temperature_C', 'cloud_percentage']
            df_copy = raw_df.copy()
            for c in expected_cols:
                if c not in df_copy.columns:
                    if c == 'datetime':
                        df_copy[c] = pd.NaT
                    elif c == 'irradiance_W_m2':
                        df_copy[c] = 0.0
                    elif c == 'temperature_C':
                        df_copy[c] = df_copy.get('temp_C', 0.0)
                    else:
                        df_copy[c] = 100.0
            st.session_state.df = df_copy[expected_cols].copy()
            try:
                st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'], format='%d/%m/%Y %H:%M')
            except Exception:
                st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'], errors='coerce')
                if st.session_state.df['datetime'].isna().all():
                    st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'].fillna(method='ffill').fillna(pd.Timestamp.now()))
            st.success("✅ Data loaded successfully!")
            st.info(f"📊 Records: {len(st.session_state.df):,}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    st.session_state.num_panels = st.number_input(
        "🔆 Number of Solar Panels",
        min_value=1,
        max_value=100,
        value=st.session_state.num_panels,
        step=1
    )

    if st.button("🚀 Initialize System", use_container_width=True):
        if st.session_state.df is not None and len(st.session_state.df) > 0:
            with st.spinner("🧠 Training AI Model..."):
                try:
                    st.session_state.system = SolarHomeSystem(st.session_state.df, st.session_state.num_panels)
                    r2 = st.session_state.system.train_solar_ai()
                    st.success(f"✅ Model Trained! R²: {r2:.3f}")
                except Exception as e:
                    st.error(f"❌ Error initializing system: {e}")
        else:
            st.warning("⚠ Please upload a valid CSV data file first!")

    st.markdown("---")
    st.markdown("### 📊 Load Profile")
    for appliance, data in LOAD_PROFILE.items():
        st.markdown(f"**{shorten_label(appliance, 40)}**")
        st.text(f"Qty: {data['qty']} | {data['kwh']} kWh")

# --- Main content ---
if st.session_state.system is not None:
    system = st.session_state.system
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📡 Live Status", "📅 AI Forecast", "🏠 Load Monitor", "⚡ Smart Scheduler", "💰 ROI Analysis"]
    )

    # TAB 1
    with tab1:
        current_row = system.get_current_row()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("📅 Date & Time")
            try:
                dt_text = current_row['datetime'].strftime('%d/%m/%Y %H:%M')
            except Exception:
                dt_text = str(current_row.get('datetime', pd.Timestamp.now()))
            st.markdown(f"<h3 style='color: #0b2545;'>{shorten_label(dt_text, 40)}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("🌡 Temperature")
            st.markdown(f"<h3 style='color: #b45309;'>{float(current_row.get('temperature_C', 0.0)):.1f}°C</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("☁ Cloud Cover")
            st.markdown(f"<h3 style='color: #4c1d95;'>{float(current_row.get('cloud_percentage', 0.0)):.0f}%</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("☀ Irradiance")
            st.markdown(f"<h3 style='color: #b45309;'>{float(current_row.get('irradiance_W_m2', 0.0)):.0f} W/m²</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        input_data = pd.DataFrame([{
            'irradiance_W_m2': current_row.get('irradiance_W_m2', 0.0),
            'temperature_C': current_row.get('temperature_C', 0.0),
            'cloud_percentage': current_row.get('cloud_percentage', 100.0),
            'hour': current_row['datetime'].hour if hasattr(current_row['datetime'], 'hour') else pd.Timestamp.now().hour
        }])
        try:
            current_gen = float(system.get_prediction(input_data).ravel()[0])
        except Exception:
            current_gen = float(system.calculate_physics_generation(input_data.iloc[0]))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center; padding: 1.5rem;">', unsafe_allow_html=True)
            st.markdown(f'<p class="big-metric">{current_gen:.3f} kW</p>', unsafe_allow_html=True)
            st.markdown("<p style='color: #0b2545; font-size: 1.05rem;'>Current Power Output</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("⚙ System Info")
            st.markdown(f"Active Panels: *{system.num_panels}*")
            st.markdown(f"Total Area: *{system.num_panels * PANEL_AREA_M2} m²*")
            st.markdown(f"Grid Rate: *₹{GRID_RATE}/kWh*")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔆 Solar Panel Array")
        cols_per_row = 5
        num_rows = (system.num_panels + cols_per_row - 1) // cols_per_row
        for r in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                panel_num = r * cols_per_row + col_idx + 1
                if panel_num <= system.num_panels:
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                                    border-radius: 8px; padding: 0.8rem; text-align: center;
                                    box-shadow: 0 4px 12px rgba(11,37,69,0.04);'>
                            <strong style='font-size: 1.0rem; color: #0b2545;'>#{panel_num}</strong>
                        </div>
                        """, unsafe_allow_html=True)

    # TAB 2: Forecast
    with tab2:
        st.markdown("### 📊 24-Hour Generation Forecast")
        day_data = system.predict_day_generation()
        current_dt = system.get_current_row()['datetime']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=day_data['datetime'],
            y=day_data['pred'],
            mode='lines+markers',
            name='Predicted Output',
            line=dict(color='#d97706', width=3),
            marker=dict(size=6, color='#b45309'),
            fill='tozeroy',
            fillcolor='rgba(217,119,6,0.12)'
        ))
        fig.update_layout(
            title=f"Solar Generation: {current_dt.strftime('%d/%m/%Y') if hasattr(current_dt, 'strftime') else str(current_dt)}",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            template="plotly_white",
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Hourly Breakdown")
        display_df = day_data[['datetime', 'irradiance_W_m2', 'pred']].copy()
        display_df['Time'] = display_df['datetime'].dt.strftime('%H:%M')
        display_df = display_df[['Time', 'irradiance_W_m2', 'pred']]
        display_df.columns = ['Time', 'Irradiance (W/m²)', 'Output (kW)']
        display_df['Output (kW)'] = display_df['Output (kW)'].round(3)
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=360)

        total_daily = float(day_data['pred'].sum()) if not day_data.empty else 0.0
        st.markdown(f"""
        <div class='success-box' style='text-align: center; font-size: 1.05rem;'>
            ✅ Total Daily Generation: <strong>{total_daily:.2f} kWh</strong>
        </div>
        """, unsafe_allow_html=True)

    # TAB 3: Load Monitor
    with tab3:
        st.markdown("### 🏠 Real-Time Load Monitoring")
        col1, col2 = st.columns(2)
        with col1:
            load = st.number_input(
                "Current House Load (kW)",
                min_value=0.0,
                max_value=50.0,
                value=2.5,
                step=0.1,
                format="%.2f"
            )
        current_row = system.get_current_row()
        input_data = pd.DataFrame([{
            'irradiance_W_m2': current_row.get('irradiance_W_m2', 0.0),
            'temperature_C': current_row.get('temperature_C', 0.0),
            'cloud_percentage': current_row.get('cloud_percentage', 100.0),
            'hour': current_row['datetime'].hour if hasattr(current_row['datetime'], 'hour') else pd.Timestamp.now().hour
        }])
        try:
            gen = float(system.get_prediction(input_data).ravel()[0])
        except Exception:
            gen = float(system.calculate_physics_generation(input_data.iloc[0]))
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("⚡ Current Generation")
            st.markdown(f"<h2 style='color: #0b6b3a;'>{gen:.3f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        net = load - gen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("☀ Solar Generation")
            st.markdown(f"<h2 style='color: #d97706;'>{gen:.2f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("🏠 House Load")
            st.markdown(f"<h2 style='color: #0b2545;'>{load:.2f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            if net > 0:
                st.markdown(f"""
                <div class='warning-box' style='text-align: center;'>
                    <h3>⚠ Grid Import</h3>
                    <h2>{net:.2f} kW</h2>
                    <p>Cost: ₹{net*GRID_RATE:.2f}/hr</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='success-box' style='text-align: center;'>
                    <h3>✅ Surplus Energy</h3>
                    <h2>{abs(net):.2f} kW</h2>
                    <p>Export Available</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        nodes = ["Solar Generation", "House Load", "Grid Import" if net > 0 else "Grid Export"]
        if net > 0:
            sources = [0, 0]
            targets = [1, 2]
            values = [min(gen, load), max(0.0, net)]
            colors = ["rgba(217,119,6,0.2)", "rgba(239,68,68,0.12)"]
        else:
            sources = [0]
            targets = [1]
            values = [min(gen, load)]
            colors = ["rgba(217,119,6,0.2)"]

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="rgba(11,37,69,0.08)", width=0.5),
                label=nodes,
                color=["#f59e0b", "#0b2545", "#ef4444" if net > 0 else "#059669"]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        ))
        fig.update_layout(
            title="Energy Flow Diagram",
            template="plotly_white",
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Smart Scheduler
    with tab4:
        st.markdown("### ⚡ Intelligent Load Scheduling")
        st.markdown("*Best times to run heavy appliances in the next 24 hours*")
        best_slots = system.intelligent_scheduler()
        rank = 1
        for _, row in best_slots.iterrows():
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='display: flex; align-items: center; gap: 0.8rem;'>
                        <div style='font-size: 1.6rem;'>{medal}</div>
                        <div>
                            <h3 style='margin: 0; color: #d97706;'>Rank #{rank}</h3>
                            <p style='margin: 0; color: #0b2545;'>🕒 {row['datetime'].strftime('%d/%m %H:%M')}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h2 style='color: #059669; margin: 0;'>{row['pred']:.2f} kW</h2>
                    <p style='color: #0b2545; margin: 0;'>Available</p>
                </div>
                """, unsafe_allow_html=True)
            rank += 1

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='metric-card'>
            <h3>💡 Smart Recommendations</h3>
            <ul style='color: #0b2545; line-height: 1.8;'>
                <li>✅ Run Washing Machine during top 3 time slots for maximum savings</li>
                <li>✅ Charge EV during peak solar hours (12:00-15:00)</li>
                <li>✅ Use dishwasher, water heater during high generation periods</li>
                <li>⚠ Avoid heavy loads after 18:00 (low solar availability)</li>
                <li>💰 Shifting 1 kW load from night to solar hours saves ₹7/hour</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # TAB 5: ROI Analysis
    with tab5:
        st.markdown("### 💰 30-Day Financial Comparison")
        with st.spinner("📊 Running financial simulation..."):
            roi_results = system.comparative_analysis()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(239, 68, 68, 0.18);'>
                <div style='text-align: center;'>
                    <h4 style='color: #7f1d1d;'>🔴 Grid Only (No Solar)</h4>
                    <h1 style='color: #7f1d1d; margin: 0.4rem 0;'>₹{roi_results['grid_only']:,.0f}</h1>
                    <p style='color: #0b2545;'>Baseline Cost</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(217,119,6,0.18);'>
                <div style='text-align: center;'>
                    <h4 style='color: #b45309;'>🟡 Solar (Unoptimized)</h4>
                    <h1 style='color: #b45309; margin: 0.4rem 0;'>₹{roi_results['solar_unopt']:,.0f}</h1>
                    <p style='color: #0b2545;'>Poor Load Timing</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(5,150,105,0.18);'>
                <div style='text-align: center;'>
                    <h4 style='color: #065f46;'>🟢 Solar + AI (Optimized)</h4>
                    <h1 style='color: #065f46; margin: 0.4rem 0;'>₹{roi_results['solar_opt']:,.0f}</h1>
                    <p style='color: #0b2545;'>Smart Scheduling</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        savings_vs_grid = roi_results['grid_only'] - roi_results['solar_opt']
        savings_vs_unopt = roi_results['solar_unopt'] - roi_results['solar_opt']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='success-box' style='text-align: center; padding: 1.2rem;'>
                <h3>💰 Savings vs Grid Only</h3>
                <h1 style='margin: 0.4rem 0;'>₹{savings_vs_grid:,.0f}</h1>
                <p>Monthly Savings</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='success-box' style='text-align: center; padding: 1.2rem;'>
                <h3>⚡ Extra AI Optimization Savings</h3>
                <h1 style='margin: 0.4rem 0;'>₹{savings_vs_unopt:,.0f}</h1>
                <p>By shifting heavy loads to peak sun hours</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        annual_savings = savings_vs_grid * 12
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(236,72,153,0.04) 100%);
                                        border: 1px solid rgba(11,37,69,0.04);
                                        text-align: center; padding: 1rem;'>
            <h2 style='color: #0b2545;'>📈 Annual Projection (12 Months)</h2>
            <h1 class='big-metric' style='font-size: 2.2rem; margin: 0.6rem 0;'>₹{annual_savings:,.0f}</h1>
            <p style='color: #0b2545; font-size: 1.0rem;'>Total Yearly Savings with AI Optimization</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Visual Comparison")
        comparison_data = {
            'Scenario': ['Grid Only', 'Solar (Unoptimized)', 'Solar + AI'],
            'Cost': [roi_results['grid_only'], roi_results['solar_unopt'], roi_results['solar_opt']],
            'Color': ['#ef4444', '#d97706', '#059669']
        }
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data['Scenario'],
                y=comparison_data['Cost'],
                marker=dict(color=comparison_data['Color'], line=dict(color='rgba(11,37,69,0.06)', width=1.2)),
                text=[f"₹{c:,.0f}" for c in comparison_data['Cost']],
                textposition='outside',
                textfont=dict(size=12, color='#0b2545', family='Inter')
            )
        ])
        fig.update_layout(
            title="30-Day Electricity Bill Comparison",
            yaxis_title="Cost (₹)",
            template="plotly_white",
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💸 Return on Investment Timeline")
        panel_cost_per_unit = 25000
        total_investment = system.num_panels * panel_cost_per_unit
        monthly_savings = max(0.0, savings_vs_grid)
        if monthly_savings > 0:
            months_to_roi = total_investment / monthly_savings
            years_to_roi = months_to_roi / 12.0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #0b2545;'>Total Investment</h4>
                    <h2 style='color: #0b2545;'>₹{total_investment:,.0f}</h2>
                    <p style='color: #0b2545;'>{system.num_panels} panels @ ₹{panel_cost_per_unit:,}/panel</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #0b2545;'>Monthly Savings</h4>
                    <h2 style='color: #059669;'>₹{monthly_savings:,.0f}</h2>
                    <p style='color: #0b2545;'>With AI optimization</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #0b2545;'>Payback Period</h4>
                    <h2 style='color: #d97706;'>{years_to_roi:.1f} years</h2>
                    <p style='color: #0b2545;'>≈ {months_to_roi:.0f} months</p>
                </div>
                """, unsafe_allow_html=True)

            months = list(range(0, int(max(1, months_to_roi)) + 24))
            cumulative_savings = [month * monthly_savings for month in months]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=cumulative_savings,
                mode='lines',
                name='Cumulative Savings',
                line=dict(color='#059669', width=3),
                fill='tozeroy',
                fillcolor='rgba(5,150,105,0.08)'
            ))
            fig.add_hline(
                y=total_investment,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Break-even: ₹{total_investment:,.0f}",
                annotation_position="right"
            )
            fig.update_layout(
                title="Cumulative Savings Over Time",
                xaxis_title="Months",
                yaxis_title="Savings (₹)",
                template="plotly_white",
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                height=360
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly savings are zero or negative; ROI cannot be calculated with current inputs.")

        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, rgba(255,243,199,0.04) 0%, rgba(255,235,205,0.02) 100%);
                                       border: 1px solid rgba(11,37,69,0.02);'>
            <h4 style='color: #0b2545;'>📝 Important Notes:</h4>
            <ul style='color: #0b2545; line-height: 1.6;'>
                <li><strong>Scenario A (Grid Only):</strong> No solar panels, all power from grid at ₹7/kWh</li>
                <li><strong>Scenario B (Solar Unoptimized):</strong> Has solar panels but runs heavy loads at night/evening (poor timing)</li>
                <li><strong>Scenario C (Solar + AI):</strong> Intelligently schedules heavy loads during peak solar hours</li>
                <li>💡 "Extra AI Savings" represents money saved by better timing of appliance usage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Welcome (if system not initialized) ---
else:
    st.markdown("""
    <div class='metric-card' style='text-align: center; padding: 2rem; margin: 1.5rem 0;'>
        <h1 style='color: #d97706; margin-bottom: 0.8rem;'>👈 Get Started</h1>
        <p style='color: #0b2545; font-size: 1.0rem;'>
            1. Upload your weather CSV file in the sidebar<br>
            2. Configure number of solar panels<br>
            3. Click "Initialize System" to begin<br>
            4. Explore AI-powered solar analytics and optimization!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0b2545;'>🎯 Key Features</h3>
            <ul style='color: #0b2545; line-height: 1.8;'>
                <li>📡 Real-time generation monitoring</li>
                <li>📅 24-hour AI-powered forecasting</li>
                <li>🏠 Dynamic load monitoring</li>
                <li>⚡ Intelligent load scheduling</li>
                <li>💰 Comprehensive ROI analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #0b2545;'>🧠 AI Capabilities</h3>
            <ul style='color: #0b2545; line-height: 1.8;'>
                <li>🤖 Random Forest ML model</li>
                <li>📊 Physics-based baseline & ML refinement</li>
                <li>🎯 Robust fallback if model not yet trained</li>
                <li>⚙ Intelligent scheduling algorithms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #0b2545;'>📋 Required CSV Format</h3>
        <p style='color: #0b2545;'>Required columns (or reasonable equivalents): <strong>datetime, irradiance_W_m2, temperature_C, cloud_percentage</strong></p>
    </div>
    """, unsafe_allow_html=True)

    sample_df = pd.DataFrame({
        'datetime': ['23/11/2025 00:00', '23/11/2025 01:00', '23/11/2025 02:00'],
        'irradiance_W_m2': [0, 0, 0],
        'temperature_C': [28, 27, 26],
        'cloud_percentage': [15, 20, 25]
    })
    st.dataframe(sample_df, use_container_width=True, height=180)

# --- Footer (SHEOS) ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #0b2545; padding: 1rem;'>
    <p>⚡ Powered by Machine Learning • 🌍 Real-time Weather Integration • 💡 Smart Energy Optimization</p>
    <p style='font-size: 0.9rem;'>SHEOS © 2025 - Revolutionizing Solar Energy Management</p>
</div>
""", unsafe_allow_html=True)
