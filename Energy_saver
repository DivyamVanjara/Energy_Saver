import streamlit as st
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SmartCity AI - Energy Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-style {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        color: #546e7a;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="header-style">🌱 SmartCity AI – Renewable Energy Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Monitor, analyze and predict clean energy generation for sustainable cities.</p>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("⚙ Configuration Panel")
st.sidebar.markdown("---")

# User inputs
num_panels = st.sidebar.number_input(
    "🔆 Number of Solar Panels",
    min_value=1,
    max_value=10000,
    value=100,
    step=10,
    help="Enter the total number of solar panels installed"
)

panel_power = st.sidebar.number_input(
    "⚡ Solar Panel Wattage (W)",
    min_value=100,
    max_value=500,
    value=250,
    step=50,
    help="Power rating per solar panel in watts"
)

num_turbines = st.sidebar.number_input(
    "💨 Number of Wind Turbines",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
    help="Number of wind turbines in the system"
)

household_count = st.sidebar.number_input(
    "🏠 Number of Households",
    min_value=1,
    max_value=1000,
    value=50,
    step=5,
    help="Estimated number of houses to power"
)

st.sidebar.markdown("---")
simulate_button = st.sidebar.button("🚀 Simulate Energy Generation", type="primary", use_container_width=True)

# Function to calculate energy
def calculate_energy(num_panels, panel_power, num_turbines, household_count):
    # Random environmental factors
    sunlight_hours = random.uniform(3, 8)
    efficiency = random.uniform(0.70, 0.95)
    
    weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy"]
    weather = random.choice(weather_conditions)
    
    weather_impact = {
        "Sunny": 1.0,
        "Partly Cloudy": 0.85,
        "Cloudy": 0.65
    }
    
    weather_emoji = {
        "Sunny": "☀",
        "Partly Cloudy": "⛅",
        "Cloudy": "☁"
    }
    
    effective_efficiency = efficiency * weather_impact[weather]
    
    # Wind energy simulation
    wind_speed = random.uniform(2, 12)
    turbine_power_rating = 1500  # 1.5 kW per turbine
    
    if wind_speed < 3:
        wind_energy = 0
    elif wind_speed < 10:
        wind_energy = (turbine_power_rating * (wind_speed / 10)) * 5 * num_turbines
    else:
        wind_energy = turbine_power_rating * 5 * num_turbines
    
    wind_energy_kwh = wind_energy / 1000  # Convert to kWh
    
    # Solar energy calculation
    total_solar_energy = (panel_power / 1000) * sunlight_hours * effective_efficiency * num_panels
    
    # Household consumption
    avg_house_consumption_kw = random.uniform(0.8, 4.0)
    daily_house_consumption = avg_house_consumption_kw * 24
    total_consumption = daily_house_consumption * household_count
    
    # Total energy
    total_energy = total_solar_energy + wind_energy_kwh
    houses_powered = total_energy / daily_house_consumption
    
    energy_efficiency = (total_energy / total_consumption) * 100 if total_consumption > 0 else 0
    
    return {
        "sunlight_hours": sunlight_hours,
        "efficiency": efficiency,
        "weather": weather,
        "weather_emoji": weather_emoji[weather],
        "effective_efficiency": effective_efficiency,
        "wind_speed": wind_speed,
        "solar_energy": total_solar_energy,
        "wind_energy": wind_energy_kwh,
        "total_energy": total_energy,
        "avg_consumption": avg_house_consumption_kw,
        "daily_consumption": daily_house_consumption,
        "total_consumption": total_consumption,
        "houses_powered": houses_powered,
        "energy_efficiency": energy_efficiency
    }

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Run simulation
if simulate_button:
    with st.spinner("🔄 Simulating energy generation..."):
        st.session_state.results = calculate_energy(num_panels, panel_power, num_turbines, household_count)
    st.success("✅ Simulation Complete!")
    
    # Show celebration if surplus energy
    if st.session_state.results['energy_efficiency'] >= 100:
        st.balloons()

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Environmental Conditions Section
    st.markdown("## 🌤 Environmental Conditions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Weather",
            value=results['weather'],
            delta=results['weather_emoji']
        )
    
    with col2:
        st.metric(
            label="Sunlight Hours",
            value=f"{results['sunlight_hours']:.2f} hrs"
        )
    
    with col3:
        st.metric(
            label="Solar Efficiency",
            value=f"{results['efficiency']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Wind Speed",
            value=f"{results['wind_speed']:.2f} m/s"
        )
    
    st.markdown("---")
    
    # Energy Generation Section
    st.markdown("## ⚡ Energy Generation Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔆 Solar Energy",
            value=f"{results['solar_energy']:.2f} kWh",
            delta="Daily Production"
        )
    
    with col2:
        st.metric(
            label="💨 Wind Energy",
            value=f"{results['wind_energy']:.2f} kWh",
            delta="Daily Production"
        )
    
    with col3:
        st.metric(
            label="🌱 Total Renewable",
            value=f"{results['total_energy']:.2f} kWh",
            delta=f"{results['energy_efficiency']:.1f}% of demand"
        )
    
    # Progress bar for energy utilization
    st.markdown("### 📊 Energy Utilization")
    utilization = min(results['energy_efficiency'], 100)
    
    if results['energy_efficiency'] >= 100:
        st.progress(1.0)
        st.success(f"🎉 Excellent! You're producing {results['energy_efficiency']:.1f}% of required energy - Surplus available!")
    elif results['energy_efficiency'] >= 75:
        st.progress(utilization / 100)
        st.info(f"✅ Good! You're meeting {results['energy_efficiency']:.1f}% of energy demand")
    else:
        st.progress(utilization / 100)
        st.warning(f"⚠ Energy production at {results['energy_efficiency']:.1f}% - Consider adding more renewable sources")
    
    st.markdown("---")
    
    # Key Results
    st.markdown("## 🏠 Household Power Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Houses Target",
            value=f"{household_count}",
            delta="Configured"
        )
    
    with col2:
        st.metric(
            label="Houses Powered",
            value=f"{results['houses_powered']:.1f}",
            delta=f"{results['houses_powered'] - household_count:+.1f} homes"
        )
    
    with col3:
        st.metric(
            label="Daily Consumption",
            value=f"{results['total_consumption']:.2f} kWh",
            delta=f"{results['daily_consumption']:.1f} kWh/home"
        )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 📈 Data Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Bar chart for energy contributions
        fig_bar = go.Figure(data=[
            go.Bar(
                name='Energy Source',
                x=['Solar Energy', 'Wind Energy'],
                y=[results['solar_energy'], results['wind_energy']],
                marker_color=['#ffd54f', '#4fc3f7'],
                text=[f"{results['solar_energy']:.2f} kWh", f"{results['wind_energy']:.2f} kWh"],
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title="🔋 Energy Generation by Source",
            yaxis_title="Energy (kWh/day)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_col2:
        # Pie chart for energy vs consumption
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Generated Energy', 'Remaining Demand'] if results['total_energy'] < results['total_consumption'] 
                   else ['Generated Energy', 'Surplus Energy'],
            values=[min(results['total_energy'], results['total_consumption']), 
                   abs(results['total_energy'] - results['total_consumption'])],
            marker_colors=['#66bb6a', '#ef5350'] if results['total_energy'] < results['total_consumption']
                          else ['#66bb6a', '#ffa726'],
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="⚖ Energy Supply vs Demand",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Expandable calculation details
    with st.expander("🔍 Calculation Details"):
        st.markdown(f"""
        ### Solar Energy Calculation
        - *Formula*: (Panel Power / 1000) × Sunlight Hours × Effective Efficiency × Number of Panels
        - *Panel Power*: {panel_power}W
        - *Sunlight Hours*: {results['sunlight_hours']:.2f} hours
        - *Base Efficiency*: {results['efficiency']*100:.2f}%
        - *Weather Impact*: {results['weather']} ({results['effective_efficiency']/results['efficiency']*100:.0f}%)
        - *Effective Efficiency*: {results['effective_efficiency']*100:.2f}%
        - *Number of Panels*: {num_panels}
        - *Result*: {results['solar_energy']:.2f} kWh/day
        
        ### Wind Energy Calculation
        - *Wind Speed*: {results['wind_speed']:.2f} m/s
        - *Turbine Rating*: 1500W (1.5 kW) per turbine
        - *Number of Turbines*: {num_turbines}
        - *Operating Hours*: 5 hours/day (average)
        - *Result*: {results['wind_energy']:.2f} kWh/day
        
        ### Household Consumption
        - *Average Consumption*: {results['avg_consumption']:.2f} kW/hour per home
        - *Daily Consumption*: {results['daily_consumption']:.2f} kWh/home
        - *Total Households*: {household_count}
        - *Total Daily Demand*: {results['total_consumption']:.2f} kWh
        
        ### Final Analysis
        - *Total Energy Generated*: {results['total_energy']:.2f} kWh/day
        - *Total Energy Required*: {results['total_consumption']:.2f} kWh/day
        - *Houses That Can Be Powered*: {results['houses_powered']:.2f} homes
        - *Energy Efficiency*: {results['energy_efficiency']:.1f}%
        """)
    
    # Timestamp
    st.markdown("---")
    st.caption(f"📅 Simulation generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    # Welcome screen
    st.info("👈 Configure your renewable energy system in the sidebar and click '🚀 Simulate Energy Generation' to begin!")
    
    st.markdown("""
    ### 🌟 About SmartCity AI
    
    This dashboard helps city planners, sustainability managers, and renewable energy enthusiasts:
    
    - *📊 Monitor* real-time renewable energy generation from solar and wind sources
    - *🔍 Analyze* the impact of weather conditions on energy efficiency
    - *🏘 Predict* how many households can be powered by clean energy
    - *💡 Optimize* renewable infrastructure for sustainable urban development
    
    ### 🎯 How It Works
    
    1. *Configure* your solar panels, wind turbines, and target households
    2. *Simulate* energy generation with randomized environmental conditions
    3. *Analyze* the results with interactive charts and metrics
    4. *Optimize* your renewable energy strategy for maximum impact
    
    ### 🌍 Building a Sustainable Future, One Simulation at a Time
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### ☀ Solar Energy
        - Panel efficiency modeling
        - Weather impact analysis
        - Sunlight hour variations
        """)
    
    with col2:
        st.markdown("""
        #### 💨 Wind Energy
        - Wind speed simulation
        - Turbine power curves
        - Multi-turbine support
        """)
    
    with col3:
        st.markdown("""
        #### 🏠 Smart Analytics
        - Household power tracking
        - Energy surplus detection
        - Optimization insights
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #546e7a;'>
    <small>🌱 SmartCity AI Dashboard v1.0</small><br>
    <small>Powering Sustainable Cities</small>
</div>
""", unsafe_allow_html=True)
