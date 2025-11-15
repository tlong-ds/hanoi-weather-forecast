import streamlit as st
st.set_page_config(layout="wide", page_title="Weather Forecast", page_icon="🌤️")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ========================================

weather_df = pd.read_csv("dataset/hn_daily.csv", parse_dates=["datetime"])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df = weather_df.set_index('datetime')

# Yêu cầu: DataFrame với datetime index và các cột như temp, tempmax, tempmin, humidity, etc.

# DEMO DATA - Xóa phần này khi có data thật
#def create_demo_data():
#    """Tạo demo data cho 30 ngày"""
#    dates = pd.date_range(start='2024-10-01', periods=30, freq='D')
    
#    # Tạo data ngẫu nhiên nhưng realistic cho Hà Nội tháng 10
#    np.random.seed(42)
#    data = {
#        'datetime': dates,
#        'temp': np.random.uniform(24, 30, 30),
#        'tempmax': np.random.uniform(28, 34, 30),
#        'tempmin': np.random.uniform(20, 25, 30),
#        'feelslike': np.random.uniform(25, 31, 30),
#        'feelslikemax': np.random.uniform(29, 36, 30),
#        'feelslikemin': np.random.uniform(21, 26, 30),
#        'humidity': np.random.uniform(65, 85, 30),
#        'precip': np.random.exponential(2, 30),
#        'precipprob': np.random.uniform(20, 80, 30),
#        'precipcover': np.random.uniform(0, 40, 30),
#        'preciptype': np.random.choice(['rain', 'None', None], 30),
#        'windspeed': np.random.uniform(5, 20, 30),
#        'windgust': np.random.uniform(10, 30, 30),
#        'winddir': np.random.uniform(0, 360, 30),
#        'pressure': np.random.uniform(1008, 1016, 30),
#        'cloudcover': np.random.uniform(30, 70, 30),
#        'visibility': np.random.uniform(8, 15, 30),
#        'uvindex': np.random.uniform(4, 8, 30),
#        'dew': np.random.uniform(18, 24, 30),
#        'sunrise': ['06:00:00'] * 30,
#        'sunset': ['17:45:00'] * 30,
#        'moonphase': np.linspace(0, 1, 30),
#        'conditions': np.random.choice(['Partially cloudy', 'Rain', 'Clear', 'Overcast'], 30),
#        'description': ['Partly cloudy throughout the day with occasional rain showers.'] * 30,
#        'icon': np.random.choice(['partly-cloudy-day', 'rain', 'clear-day', 'cloudy'], 30),
#        'source': ['vcw'] * 30,
#        'stations': ['VVNB'] * 30,
#        'severerisk': np.random.uniform(10, 50, 30)
#    }
    
#    df = pd.DataFrame(data)
#    df['datetime'] = pd.to_datetime(df['datetime'])
#    df = df.set_index('datetime')
#    return df

#weather_df = create_demo_data()  # Comment dòng này khi có data thật

# ========================================
# CONFIGURATION
# ========================================
DEFAULT_LOCATION = "Hanoi, Vietnam"

# ========================================
# HELPER FUNCTIONS
# ========================================
def get_weather_icon(condition: str) -> str:
    """
    Trả về emoji tương ứng với điều kiện thời tiết trong cột 'conditions'.
    """
    condition = condition.lower()

    if "rain" in condition:
        return "🌧️"
    elif "overcast" in condition:
        return "☁️"
    elif "partially cloudy" in condition or "cloudy" in condition:
        return "⛅"
    elif "clear" in condition:
        return "☀️"
    else:
        return "🌍"  # mặc định

def get_weather_icon_and_text(icon, conditions):
    """Map icon code to emoji and description"""
    icon_map = {
        'clear-day': ('☀️', 'Clear Sky'),
        'clear-night': ('🌙', 'Clear Night'),
        'rain': ('🌧️', 'Rainy'),
        'snow': ('❄️', 'Snowy'),
        'wind': ('💨', 'Windy'),
        'fog': ('🌫️', 'Foggy'),
        'cloudy': ('☁️', 'Cloudy'),
        'partly-cloudy-day': ('⛅', 'Partly Cloudy'),
        'partly-cloudy-night': ('☁️', 'Partly Cloudy'),
        'thunderstorm': ('⛈️', 'Thunderstorm'),
    }
    
    if icon in icon_map:
        return icon_map[icon]
    elif conditions and 'rain' in str(conditions).lower():
        return ('🌧️', 'Rainy')
    elif conditions and 'thunder' in str(conditions).lower():
        return ('⛈️', 'Thunderstorm')
    else:
        return ('🌤️', 'Fair Weather')

def get_moon_phase_emoji(moonphase):
    """Convert moon phase value to emoji"""
    if pd.isna(moonphase):
        return '🌑'
    phase = float(moonphase)
    if phase < 0.05 or phase > 0.95:
        return '🌑 New Moon'
    elif phase < 0.25:
        return '🌒 Waxing Crescent'
    elif phase < 0.30:
        return '🌓 First Quarter'
    elif phase < 0.45:
        return '🌔 Waxing Gibbous'
    elif phase < 0.55:
        return '🌕 Full Moon'
    elif phase < 0.70:
        return '🌖 Waning Gibbous'
    elif phase < 0.75:
        return '🌗 Last Quarter'
    else:
        return '🌘 Waning Crescent'

def create_hourly_forecast(tempmin, tempmax):
    """Create synthetic hourly temperature curve"""
    hours = np.arange(0, 24)
    peak_hour = 14
    mean = (tempmax + tempmin) / 2
    amp = (tempmax - tempmin) / 2 if tempmax > tempmin else 1.0
    temps = mean + amp * np.sin((hours - peak_hour) / 24 * 2 * np.pi)
    return hours, temps

def get_uv_level(uvindex):
    """Get UV index level description"""
    if pd.isna(uvindex):
        return "Unknown"
    uv = float(uvindex)
    if uv < 3:
        return "Low"
    elif uv < 6:
        return "Moderate"
    elif uv < 8:
        return "High"
    elif uv < 11:
        return "Very High"
    else:
        return "Extreme"

def get_severe_risk_level(risk):
    """Get severe weather risk description"""
    if pd.isna(risk):
        return "Unknown", "#gray"
    r = float(risk)
    if r < 30:
        return "Low Risk", "#4CAF50"
    elif r < 70:
        return "Moderate Risk", "#FF9800"
    else:
        return "High Risk", "#F44336"

# ========================================
# STYLING - NAVY BLUE THEME
# ========================================
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Nền trang - Navy Blue */
        .main {
            background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        
        /* Card chính - Blue Gradient */
        .main-card {
            background: linear-gradient(135deg, #2e5c8a 0%, #1e3a5f 100%);
            color: white;
            border-radius: 24px;
            padding: 35px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Forecast card */
        .forecast-card {
            background: linear-gradient(135deg, #2a3f5f 0%, #1a2838 100%);
            border-radius: 16px;
            padding: 18px;
            margin: 10px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .forecast-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .temp-large {
            font-size: 72px;
            font-weight: 700;
            margin: 0;
            color: #ffffff;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Detail box */
        .detail-box {
            background: linear-gradient(135deg, #2a3f5f 0%, #1a2838 100%);
            border-radius: 16px;
            padding: 18px;
            margin: 10px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: transform 0.2s;
        }
        
        .detail-box:hover {
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e2d3d 0%, #0f1419 100%);
        }
        
        section[data-testid="stSidebar"] .element-container {
            color: #e0e7ff;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: 'Inter', sans-serif;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 24px;
            font-weight: 600;
        }
        
        [data-testid="stMetricDelta"] {
            color: #94a3b8;
        }
        
        /* Info boxes */
        .stAlert {
            background: linear-gradient(135deg, #2a3f5f 0%, #1a2838 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e0e7ff;
        }
        
        /* Subheader styling */
        .css-10trblm {
            color: #e0e7ff;
        }
        
        /* Caption text */
        .css-1629p8f, .st-emotion-cache-1629p8f {
            color: #94a3b8 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("⚙️ Settings")

# 🌍 Global Location List
locations = [
    "Hanoi, Vietnam", "Ho Chi Minh City, Vietnam", "Bangkok, Thailand", "Singapore, Singapore",
    "Tokyo, Japan", "Seoul, South Korea", "Beijing, China", "Shanghai, China", 
    "New York, USA", "Los Angeles, USA", "London, UK", "Paris, France", "Berlin, Germany",
    "Sydney, Australia", "Melbourne, Australia", "Toronto, Canada", "Vancouver, Canada",
    "Dubai, UAE", "Mumbai, India", "Jakarta, Indonesia"
]

selected_location = st.sidebar.selectbox("🌍 Choose Location", locations, index=0)

# 🗓 Date Picker (Simplified)
if weather_df is not None and len(weather_df) > 0:
    min_date = weather_df.index.min().date()
    max_date = weather_df.index.max().date()
    selected_date = st.sidebar.date_input(
        "📅 Select Date",
        value = max_date,
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_date = datetime.now().date()

st.sidebar.markdown("---")


# ========================================
# CHECK DATA
# ========================================
if weather_df is None or len(weather_df) == 0:
    st.warning("⚠️ No weather data available")
    st.info("""
    ### Để sử dụng dashboard:
    1. Truyền DataFrame vào biến `weather_df`
    2. DataFrame cần có datetime index
    3. Các cột cần thiết: `temp`, `tempmax`, `tempmin`, `humidity`, `precip`, `windspeed`, etc.
    """)
    st.stop()

# Sort data
weather_df = weather_df.sort_index()

# ========================================
# HEADER
# ========================================
display_date = pd.to_datetime(selected_date).strftime('%A, %B %d, %Y')

st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 32px; margin: 0; color: #ffffff;">Have a nice day! 🌤️</h1>
        <p style="color: #94a3b8; font-size: 16px; margin-top: 5px;">
            {display_date}
        </p>
    </div>
""", unsafe_allow_html=True)


# ========================================
# MAIN LAYOUT
# ========================================
left_col, right_col = st.columns([2.5, 1])

# ========================================
# LEFT COLUMN
# ========================================
with left_col:
    selected_ts = pd.to_datetime(selected_date)
    # Trường hợp index có time, nên select ngày gần nhất
    if selected_ts in weather_df.index:
        current = weather_df.loc[selected_ts]
    else:
        current = weather_df.loc[weather_df.index.date == selected_ts.date()].iloc[-1]

    icon = get_weather_icon(current['conditions'])

    st.markdown(f"""
        <div class="main-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="margin:0; font-size:24px;">Weather Overview</h2>
                <div style="background: rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 20px; font-size: 14px;">
                    Forecast Summary
                </div>
            </div>
            <div style="font-size:70px;">{icon}</div> 
            <div style="font-size:36px; font-weight:700;">{current['temp']:.1f}°C</div>
            <div style="font-size:18px; color:#e0e7ff;">{current['conditions']}</div>
        </div>
    """, unsafe_allow_html=True)
    

    # 🧭 Weekly Forecast Chart
    st.markdown('<h3 style="margin-top: 30px;">📈 Weekly Temperature Trend</h3>', unsafe_allow_html=True)
    selected_ts = pd.to_datetime(selected_date)
    weekly_df = weather_df.loc[weather_df.index >= selected_ts].head(7)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1a2332')
    ax.set_facecolor('#1a2332')

    ax.plot(weekly_df.index, weekly_df['tempmax'], label="Max Temp", linewidth=3, marker='o')
    ax.plot(weekly_df.index, weekly_df['tempmin'], label="Min Temp", linewidth=3, marker='o', linestyle="--")
    ax.fill_between(weekly_df.index, weekly_df['tempmin'], weekly_df['tempmax'], alpha=0.2)

    ax.set_title("7-Day Temperature Range", color="#e0e7ff")
    ax.set_ylabel("Temperature (°C)", color="#94a3b8")
    ax.tick_params(axis='x', colors='#94a3b8', rotation=15)
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.legend(facecolor="#1a2332", labelcolor="#e0e7ff")
    for spine in ax.spines.values():
        spine.set_color('#2a3f5f')
    st.pyplot(fig)
    plt.close()

    
    
    # 7-Day Forecast
    st.markdown('<h3 style="color: #ffffff; font-size: 20px; margin: 30px 0 15px 0;">📅 7-Day Forecast</h3>', unsafe_allow_html=True)
    forecast_days = weather_df.loc[weather_df.index >= selected_ts].head(7)

    
    cols = st.columns(min(7, len(forecast_days)))
    for i, (idx, row) in enumerate(forecast_days.iterrows()):
        if i < len(cols):
            with cols[i]:
                icon_emoji, _ = get_weather_icon_and_text(row.get('icon', ''), row.get('conditions', ''))
                precip_prob = row.get('precipprob', 0)
                
                st.markdown(f"""
                    <div class="forecast-card">
                        <div style="font-weight:600; color:#e0e7ff; font-size: 14px; margin-bottom: 10px;">
                            {idx.strftime('%a, %b %d')}
                        </div>
                        <div style="font-size:42px; margin:12px 0; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">
                            {icon_emoji}
                        </div>
                        <div style="color:#ffffff; font-size:22px; font-weight:700; margin: 8px 0;">
                            {row['temp']:.0f}°C
                        </div>
                        <div style="color:#94a3b8; font-size:13px; margin-bottom: 8px;">
                            {row['tempmin']:.0f}° / {row['tempmax']:.0f}°
                        </div>
                        <div style="color:#60a5fa; font-size:12px; background: rgba(96,165,250,0.1); 
                                    padding: 4px 8px; border-radius: 8px; display: inline-block;">
                            💧 {precip_prob:.0f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# ========================================
# RIGHT COLUMN
# ========================================
with right_col:
    # Sun & Moon
    st.markdown('<h3 style="color: #ffffff; font-size: 18px; margin: 0 0 15px 0;">🌅 Sun & Moon</h3>', unsafe_allow_html=True)
    sunrise = current.get('sunrise', 'N/A')
    sunset = current.get('sunset', 'N/A')
    moon_phase = get_moon_phase_emoji(current.get('moonphase', 0))
    
    st.markdown(f"""
        <div class="detail-box">
            <div style="display:flex; justify-content:space-between; margin:12px 0; color: #e0e7ff;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 20px;">🌅</span>
                    <span>Sunrise</span>
                </div>
                <div style="font-weight:600; color: #ffffff;">{sunrise}</div>
            </div>
            <div style="display:flex; justify-content:space-between; margin:12px 0; color: #e0e7ff;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 20px;">🌇</span>
                    <span>Sunset</span>
                </div>
                <div style="font-weight:600; color: #ffffff;">{sunset}</div>
            </div>
            <div style="display:flex; justify-content:space-between; margin:12px 0; color: #e0e7ff;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 20px;">🌙</span>
                    <span>Moon</span>
                </div>
                <div style="font-weight:600; color: #ffffff; font-size: 13px;">{moon_phase}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Weather Details
    st.markdown('<h3 style="color: #ffffff; font-size: 18px; margin: 25px 0 15px 0;">📍 Weather Details</h3>', unsafe_allow_html=True)
    
    # Wind
    windgust = current.get('windgust', current['windspeed'])
    winddir = current.get('winddir', 0)
    wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_dir_text = wind_directions[int((winddir + 22.5) / 45) % 8]
    
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">💨</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">Wind</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff; margin-bottom:5px;">
                {current['windspeed']:.0f} km/h
            </div>
            <div style="font-size:13px; color:#94a3b8;">
                Direction: {wind_dir_text} • Gusts: {windgust:.0f} km/h
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Humidity & Dew Point
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">💧</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">Humidity & Dew</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff; margin-bottom:5px;">
                {current['humidity']:.0f}%
            </div>
            <div style="font-size:13px; color:#94a3b8;">Dew point: {current['dew']:.0f}°C</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Precipitation
    precip_cover = current.get('precipcover', 0)
    precip_type = current.get('preciptype', 'None')
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">🌧️</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">Precipitation</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff; margin-bottom:5px;">
                {current['precip']:.1f} mm
            </div>
            <div style="font-size:13px; color:#94a3b8;">
                Type: {precip_type if pd.notna(precip_type) else 'None'} • Coverage: {precip_cover:.0f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # UV Index
    uv_level = get_uv_level(current.get('uvindex', 0))
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">☀️</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">UV Index</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff;">
                {current.get('uvindex', 0):.0f}
            </div>
            <div style="font-size:13px; color:#94a3b8;">{uv_level}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Visibility
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">👁️</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">Visibility</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff; margin-bottom:5px;">
                {current.get('visibility', 0):.1f} km
            </div>
            <div style="font-size:13px; color:#94a3b8;">Cloud cover: {current.get('cloudcover', 0):.0f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Pressure
    st.markdown(f"""
        <div class="detail-box">
            <div style="font-size:28px; margin-bottom: 8px;">🌡️</div>
            <div style="color:#94a3b8; font-size:13px; margin-bottom:8px;">Pressure</div>
            <div style="font-size:24px; font-weight:700; color:#ffffff;">
                {current.get('pressure', 0):.0f} mb
            </div>
        </div>
    """, unsafe_allow_html=True)

# ========================================