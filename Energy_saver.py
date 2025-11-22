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

# Page Configuration
st.set_page_config(
    page_title="SuryaShakti AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #581c87 50%, #0f172a 100%);
    }
    
    /* Animated background orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: 10%;
        left: 5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(234, 179, 8, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: pulse 4s ease-in-out infinite;
        z-index: 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%);
        color: #1e293b;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(251, 191, 36, 0.4);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        border-color: rgba(251, 191, 36, 0.5);
    }
    
    /* Metric value styling */
    .big-metric {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%);
        color: #1e293b;
        border: none;
    }
    
    /* Success/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #10b981;
        font-weight: 600;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(248, 113, 113, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
PANEL_AREA_M2 = 2.0
GRID_RATE = 7.0
LOAD_PROFILE = {
    "AC (1.5 Ton)": {"qty": 1, "kwh": 1.5},
    "Fans": {"qty": 5, "kwh": 0.075},
    "LEDs": {"qty": 10, "kwh": 0.01},
    "Washing Machine": {"qty": 1, "kwh": 0.5}
}

class SolarHomeSystem:
    def __init__(self, df, num_panels):
        self.df = df
        self.num_panels = num_panels
        self.scaler = StandardScaler()
        self.ml_model = None
        self.current_index = random.randint(0, len(df) - (24*31))
        
    def calculate_physics_generation(self, row):
        irradiance = row['irradiance_W_m2']
        cloud = row['cloud_percentage']
        
        if 0 <= cloud <= 30:
            efficiency = 1.0
        elif 30 < cloud <= 60:
            efficiency = 0.8
        elif 60 < cloud <= 90:
            efficiency = 0.6
        else:
            efficiency = 0.2
            
        total_area = self.num_panels * PANEL_AREA_M2
        power_kw = (irradiance * total_area * efficiency) / 1000.0
        return max(0.0, power_kw)
    
    def train_solar_ai(self):
        training_data = self.df.copy()
        training_data['hour'] = training_data['datetime'].dt.hour
        training_data['power_output'] = training_data.apply(self.calculate_physics_generation, axis=1)
        
        features = ['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']
        X = training_data[features]
        y = training_data['power_output']
        
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        y_pred = self.ml_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return r2
    
    def get_prediction(self, feature_df):
        scaled_features = self.scaler.transform(feature_df)
        return self.ml_model.predict(scaled_features)
    
    def get_current_row(self):
        return self.df.iloc[self.current_index]
    
    def predict_day_generation(self):
        current_dt = self.get_current_row()['datetime']
        start = current_dt.replace(hour=0, minute=0)
        end = current_dt.replace(hour=23, minute=59)
        
        mask = (self.df['datetime'] >= start) & (self.df['datetime'] <= end)
        day_data = self.df.loc[mask].copy()
        day_data['hour'] = day_data['datetime'].dt.hour
        
        features = day_data[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
        day_data['pred'] = self.get_prediction(features)
        
        return day_data
    
    def intelligent_scheduler(self):
        current_dt = self.get_current_row()['datetime']
        end_dt = current_dt + timedelta(hours=24)
        
        mask = (self.df['datetime'] > current_dt) & (self.df['datetime'] <= end_dt)
        future_df = self.df.loc[mask].copy()
        future_df['hour'] = future_df['datetime'].dt.hour
        
        features = future_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
        future_df['pred'] = self.get_prediction(features)
        
        best_slots = future_df.sort_values('pred', ascending=False).head(3)
        return best_slots
    
    def comparative_analysis(self):
        current_dt = self.get_current_row()['datetime']
        end_dt = current_dt + timedelta(days=30)
        
        mask = (self.df['datetime'] >= current_dt) & (self.df['datetime'] < end_dt)
        sim_df = self.df.loc[mask].copy()
        sim_df['hour'] = sim_df['datetime'].dt.hour
        
        features = sim_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
        sim_df['solar_gen'] = self.get_prediction(features)
        
        bill_A_grid_only = 0
        bill_B_solar_unopt = 0
        bill_C_solar_opt = 0
        
        sim_df['date'] = sim_df['datetime'].dt.date
        
        for date, day_group in sim_df.groupby('date'):
            peak_sun_idx = day_group['solar_gen'].idxmax()
            peak_hour = day_group.loc[peak_sun_idx, 'hour']
            
            for _, row in day_group.iterrows():
                h = row['hour']
                solar = row['solar_gen']
                
                hourly_load_base = 0
                hourly_load_base += LOAD_PROFILE["Fans"]["kwh"] * LOAD_PROFILE["Fans"]["qty"]
                
                if h >= 22 or h < 6:
                    hourly_load_base += LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] * LOAD_PROFILE["AC (1.5 Ton)"]["qty"]
                
                if 18 <= h <= 23:
                    hourly_load_base += LOAD_PROFILE["LEDs"]["kwh"] * LOAD_PROFILE["LEDs"]["qty"]
                
                wm_kwh = LOAD_PROFILE["Washing Machine"]["kwh"]
                
                # Scenario A: Grid Only
                load_A = hourly_load_base
                if h == 20:
                    load_A += wm_kwh
                bill_A_grid_only += (load_A * GRID_RATE)
                
                # Scenario B: Solar Unoptimized
                load_B = hourly_load_base
                if h == 20:
                    load_B += wm_kwh
                net_B = max(0, load_B - solar)
                bill_B_solar_unopt += (net_B * GRID_RATE)
                
                # Scenario C: Solar Optimized
                load_C = hourly_load_base
                if h == peak_hour:
                    load_C += wm_kwh
                net_C = max(0, load_C - solar)
                bill_C_solar_opt += (net_C * GRID_RATE)
        
        return {
            'grid_only': bill_A_grid_only,
            'solar_unopt': bill_B_solar_unopt,
            'solar_opt': bill_C_solar_opt
        }

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'num_panels' not in st.session_state:
    st.session_state.num_panels = 10

# Header
st.markdown('<div class="main-header">☀️ SURYASHAKTI AI</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color: #cbd5e1; font-size: 1.2rem; margin-top: -1rem;">Advanced Solar Analytics & Optimization Platform</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    uploaded_file = st.file_uploader("📁 Upload Weather Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            
            # Standardize column names
            raw_df.rename(columns={
                'timestamp': 'datetime',
                'temp_C': 'temperature_C',
                'precipitation_probability_pct': 'cloud_percentage'
            }, inplace=True)
            
            # Filter columns
            st.session_state.df = raw_df[['datetime', 'irradiance_W_m2', 'temperature_C', 'cloud_percentage']].copy()
            
            # Parse datetime
            try:
                st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'], format='%d/%m/%Y %H:%M')
            except:
                st.session_state.df['datetime'] = pd.to_datetime(st.session_state.df['datetime'])
            
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
        if st.session_state.df is not None:
            with st.spinner("🧠 Training AI Model..."):
                st.session_state.system = SolarHomeSystem(st.session_state.df, st.session_state.num_panels)
                r2 = st.session_state.system.train_solar_ai()
                st.success(f"✅ Model Trained! Accuracy: {r2*100:.2f}%")
        else:
            st.warning("⚠️ Please upload CSV data first!")
    
    st.markdown("---")
    st.markdown("### 📊 Load Profile")
    for appliance, data in LOAD_PROFILE.items():
        st.markdown(f"**{appliance}**")
        st.text(f"Qty: {data['qty']} | {data['kwh']} kWh")

# Main Content
if st.session_state.system is not None:
    system = st.session_state.system
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Live Status",
        "📅 AI Forecast", 
        "🏠 Load Monitor",
        "⚡ Smart Scheduler",
        "💰 ROI Analysis"
    ])
    
    # TAB 1: Live Status
    with tab1:
        current_row = system.get_current_row()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**📅 Date & Time**")
            st.markdown(f"<h3 style='color: #60a5fa;'>{current_row['datetime'].strftime('%d/%m/%Y %H:%M')}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**🌡️ Temperature**")
            st.markdown(f"<h3 style='color: #fb923c;'>{current_row['temperature_C']}°C</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**☁️ Cloud Cover**")
            st.markdown(f"<h3 style='color: #a78bfa;'>{current_row['cloud_percentage']}%</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**☀️ Irradiance**")
            st.markdown(f"<h3 style='color: #fbbf24;'>{current_row['irradiance_W_m2']} W/m²</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Current Generation
        input_data = pd.DataFrame([{
            'irradiance_W_m2': current_row['irradiance_W_m2'],
            'temperature_C': current_row['temperature_C'],
            'cloud_percentage': current_row['cloud_percentage'],
            'hour': current_row['datetime'].hour
        }])
        
        current_gen = system.get_prediction(input_data).item()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
            st.markdown(f'<p class="big-metric">{current_gen:.3f} kW</p>', unsafe_allow_html=True)
            st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Current Power Output</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**⚙️ System Info**")
            st.markdown(f"Active Panels: **{system.num_panels}**")
            st.markdown(f"Total Area: **{system.num_panels * PANEL_AREA_M2} m²**")
            st.markdown(f"Grid Rate: **₹{GRID_RATE}/kWh**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 3D Panel Visualization
        st.markdown("### 🔆 Solar Panel Array")
        cols_per_row = 5
        num_rows = (system.num_panels + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                panel_num = row * cols_per_row + col_idx + 1
                if panel_num <= system.num_panels:
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
                                    border-radius: 8px; padding: 1rem; text-align: center;
                                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
                                    animation: pulse 2s ease-in-out infinite;
                                    animation-delay: {panel_num * 0.1}s;'>
                            <strong style='font-size: 1.2rem;'>#{panel_num}</strong>
                        </div>
                        """, unsafe_allow_html=True)
    
    # TAB 2: AI Forecast
    with tab2:
        st.markdown("### 📊 24-Hour Generation Forecast")
        
        day_data = system.predict_day_generation()
        current_dt = system.get_current_row()['datetime']
        
        # Create plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=day_data['datetime'],
            y=day_data['pred'],
            mode='lines+markers',
            name='Predicted Output',
            line=dict(color='#fbbf24', width=3),
            marker=dict(size=8, color='#f97316'),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.2)'
        ))
        
        fig.update_layout(
            title=f"Solar Generation: {current_dt.strftime('%d/%m/%Y')}",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Hourly Breakdown")
        
        display_df = day_data[['datetime', 'irradiance_W_m2', 'pred']].copy()
        display_df['datetime'] = display_df['datetime'].dt.strftime('%H:%M')
        display_df.columns = ['Time', 'Irradiance (W/m²)', 'Output (kW)']
        display_df['Output (kW)'] = display_df['Output (kW)'].round(3)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        total_daily = day_data['pred'].sum()
        st.markdown(f"""
        <div class='success-box' style='text-align: center; font-size: 1.2rem;'>
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
                max_value=20.0,
                value=2.5,
                step=0.1,
                format="%.2f"
            )
        
        current_row = system.get_current_row()
        input_data = pd.DataFrame([{
            'irradiance_W_m2': current_row['irradiance_W_m2'],
            'temperature_C': current_row['temperature_C'],
            'cloud_percentage': current_row['cloud_percentage'],
            'hour': current_row['datetime'].hour
        }])
        gen = system.get_prediction(input_data).item()
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**⚡ Current Generation**")
            st.markdown(f"<h2 style='color: #10b981;'>{gen:.3f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Net energy flow
        net = load - gen
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("**☀️ Solar Generation**")
            st.markdown(f"<h2 style='color: #fbbf24;'>{gen:.2f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("**🏠 House Load**")
            st.markdown(f"<h2 style='color: #60a5fa;'>{load:.2f} kW</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if net > 0:
                st.markdown(f"""
                <div class='warning-box' style='text-align: center;'>
                    <h3>⚠️ Grid Import</h3>
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
        
        # Energy flow diagram
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=0.5),
                label=["Solar Generation", "House Load", "Grid Import" if net > 0 else "Grid Export"],
                color=["#fbbf24", "#60a5fa", "#ef4444" if net > 0 else "#10b981"]
            ),
            link=dict(
                source=[0, 1] if net > 0 else [0],
                target=[1, 2] if net > 0 else [1],
                value=[min(gen, load), max(0, net)] if net > 0 else [gen],
                color=["rgba(251, 191, 36, 0.4)", "rgba(239, 68, 68, 0.4)"] if net > 0 else ["rgba(251, 191, 36, 0.4)"]
            )
        ))
        
        fig.update_layout(
            title="Energy Flow Diagram",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Smart Scheduler
    with tab4:
        st.markdown("### ⚡ Intelligent Load Scheduling")
        st.markdown("**Best times to run heavy appliances in the next 24 hours**")
        
        best_slots = system.intelligent_scheduler()
        
        rank = 1
        for _, row in best_slots.iterrows():
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <div style='font-size: 2rem;'>{medal}</div>
                        <div>
                            <h3 style='margin: 0; color: #fbbf24;'>Rank #{rank}</h3>
                            <p style='margin: 0; color: #cbd5e1;'>🕒 {row['datetime'].strftime('%d/%m %H:%M')}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h2 style='color: #10b981; margin: 0;'>{row['pred']:.2f} kW</h2>
                    <p style='color: #94a3b8; margin: 0;'>Available</p>
                </div>
                """, unsafe_allow_html=True)
            
            rank += 1
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("""
        <div class='metric-card'>
            <h3>💡 Smart Recommendations</h3>
            <ul style='color: #cbd5e1; line-height: 2;'>
                <li>✅ Run Washing Machine during top 3 time slots for maximum savings</li>
                <li>✅ Charge EV during peak solar hours (12:00-15:00)</li>
                <li>✅ Use dishwasher, water heater during high generation periods</li>
                <li>⚠️ Avoid heavy loads after 18:00 (low solar availability)</li>
                <li>💰 Shifting 1 kW load from night to solar hours saves ₹7/hour</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 5: ROI Analysis
    with tab5:
        st.markdown("### 💰 30-Day Financial Comparison")
        
        with st.spinner("📊 Running financial simulation..."):
            roi_results = system.comparative_analysis()
        
        # Main comparison cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(239, 68, 68, 0.5);'>
                <div style='text-align: center;'>
                    <h4 style='color: #ef4444;'>🔴 Grid Only (No Solar)</h4>
                    <h1 style='color: #ef4444; margin: 0.5rem 0;'>₹{roi_results['grid_only']:,.0f}</h1>
                    <p style='color: #94a3b8;'>Baseline Cost</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(251, 191, 36, 0.5);'>
                <div style='text-align: center;'>
                    <h4 style='color: #fbbf24;'>🟡 Solar (Unoptimized)</h4>
                    <h1 style='color: #fbbf24; margin: 0.5rem 0;'>₹{roi_results['solar_unopt']:,.0f}</h1>
                    <p style='color: #94a3b8;'>Poor Load Timing</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid rgba(16, 185, 129, 0.5);'>
                <div style='text-align: center;'>
                    <h4 style='color: #10b981;'>🟢 Solar + AI (Optimized)</h4>
                    <h1 style='color: #10b981; margin: 0.5rem 0;'>₹{roi_results['solar_opt']:,.0f}</h1>
                    <p style='color: #94a3b8;'>Smart Scheduling</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Savings breakdown
        savings_vs_grid = roi_results['grid_only'] - roi_results['solar_opt']
        savings_vs_unopt = roi_results['solar_unopt'] - roi_results['solar_opt']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='success-box' style='text-align: center; padding: 2rem;'>
                <h3>💰 Savings vs Grid Only</h3>
                <h1 style='margin: 0.5rem 0;'>₹{savings_vs_grid:,.0f}</h1>
                <p>Monthly Savings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='success-box' style='text-align: center; padding: 2rem;'>
                <h3>⚡ Extra AI Optimization Savings</h3>
                <h1 style='margin: 0.5rem 0;'>₹{savings_vs_unopt:,.0f}</h1>
                <p>By shifting heavy loads to peak sun hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Annual projection
        annual_savings = savings_vs_grid * 12
        
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
                                        border: 2px solid rgba(139, 92, 246, 0.5);
                                        text-align: center; padding: 2rem;'>
            <h2 style='color: #cbd5e1;'>📈 Annual Projection (12 Months)</h2>
            <h1 class='big-metric' style='font-size: 4rem; margin: 1rem 0;'>₹{annual_savings:,.0f}</h1>
            <p style='color: #94a3b8; font-size: 1.2rem;'>Total Yearly Savings with AI Optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Visual Comparison")
        
        comparison_data = {
            'Scenario': ['Grid Only', 'Solar (Unoptimized)', 'Solar + AI'],
            'Cost': [roi_results['grid_only'], roi_results['solar_unopt'], roi_results['solar_opt']],
            'Color': ['#ef4444', '#fbbf24', '#10b981']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data['Scenario'],
                y=comparison_data['Cost'],
                marker=dict(
                    color=comparison_data['Color'],
                    line=dict(color='white', width=2)
                ),
                text=[f"₹{c:,.0f}" for c in comparison_data['Cost']],
                textposition='outside',
                textfont=dict(size=16, color='white', family='Inter')
            )
        ])
        
        fig.update_layout(
            title="30-Day Electricity Bill Comparison",
            yaxis_title="Cost (₹)",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI Timeline
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💸 Return on Investment Timeline")
        
        panel_cost_per_unit = 25000  # Approximate cost per panel in INR
        total_investment = system.num_panels * panel_cost_per_unit
        monthly_savings = savings_vs_grid
        
        if monthly_savings > 0:
            months_to_roi = total_investment / monthly_savings
            years_to_roi = months_to_roi / 12
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #94a3b8;'>Total Investment</h4>
                    <h2 style='color: #60a5fa;'>₹{total_investment:,.0f}</h2>
                    <p style='color: #94a3b8;'>{system.num_panels} panels @ ₹{panel_cost_per_unit:,}/panel</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #94a3b8;'>Monthly Savings</h4>
                    <h2 style='color: #10b981;'>₹{monthly_savings:,.0f}</h2>
                    <p style='color: #94a3b8;'>With AI optimization</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color: #94a3b8;'>Payback Period</h4>
                    <h2 style='color: #fbbf24;'>{years_to_roi:.1f} years</h2>
                    <p style='color: #94a3b8;'>≈ {months_to_roi:.0f} months</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ROI Timeline Chart
            months = list(range(0, int(months_to_roi) + 24))
            cumulative_savings = [month * monthly_savings for month in months]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=cumulative_savings,
                mode='lines',
                name='Cumulative Savings',
                line=dict(color='#10b981', width=3),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)'
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
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Note
        st.markdown("""
        <div class='metric-card' style='background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%);
                                       border: 1px solid rgba(251, 191, 36, 0.3);'>
            <h4>📝 Important Notes:</h4>
            <ul style='color: #cbd5e1; line-height: 1.8;'>
                <li><strong>Scenario A (Grid Only):</strong> No solar panels, all power from grid at ₹7/kWh</li>
                <li><strong>Scenario B (Solar Unoptimized):</strong> Has solar panels but runs heavy loads at night/evening (poor timing)</li>
                <li><strong>Scenario C (Solar + AI):</strong> Intelligently schedules heavy loads during peak solar hours</li>
                <li>💡 The "Extra AI Savings" represents money saved just by better timing of appliance usage</li>
                <li>⚡ Fixed loads (AC at night, fans 24/7, LEDs evening) are same across all scenarios</li>
                <li>🔋 Variable load (Washing Machine) timing makes the difference in optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome Screen
    st.markdown("""
    <div class='metric-card' style='text-align: center; padding: 3rem; margin: 2rem 0;'>
        <h1 style='color: #fbbf24; margin-bottom: 1rem;'>👈 Get Started</h1>
        <p style='color: #cbd5e1; font-size: 1.2rem;'>
            1. Upload your weather CSV file in the sidebar<br>
            2. Configure number of solar panels<br>
            3. Click "Initialize System" to begin<br>
            4. Explore AI-powered solar analytics and optimization!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Key Features</h3>
            <ul style='color: #cbd5e1; line-height: 2;'>
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
            <h3>🧠 AI Capabilities</h3>
            <ul style='color: #cbd5e1; line-height: 2;'>
                <li>🤖 Random Forest ML model</li>
                <li>📊 3-year weather data training</li>
                <li>🎯 High accuracy predictions (90%+)</li>
                <li>⚙️ Physics-based calculations</li>
                <li>📈 Optimization algorithms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data format
    st.markdown("""
    <div class='metric-card'>
        <h3>📋 Required CSV Format</h3>
        <p style='color: #cbd5e1;'>Your CSV file should contain these columns:</p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_df = pd.DataFrame({
        'datetime': ['23/11/2025 00:00', '23/11/2025 01:00', '23/11/2025 02:00'],
        'irradiance_W_m2': [0, 0, 0],
        'temperature_C': [28, 27, 26],
        'cloud_percentage': [15, 20, 25]
    })
    
    st.dataframe(sample_df, use_container_width=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p>⚡ Powered by Machine Learning • 🌍 Real-time Weather Integration • 💡 Smart Energy Optimization</p>
    <p style='font-size: 0.9rem;'>SuryaShakti AI © 2025 - Revolutionizing Solar Energy Management</p>
</div>
""", unsafe_allow_html=True)
