import streamlit as st
import base64
st.set_page_config(layout="wide", page_title="Weather Forecast", page_icon="🌤️")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

# ML forecaster (optional)
try:
    from src.daily_forecast_model.infer import WeatherForecaster
except Exception:
    WeatherForecaster = None

# HTTP client for remote API
try:
    import requests
except Exception:
    requests = None

# ========================================

weather_df = pd.read_csv("dataset/hn_daily.csv", parse_dates=["datetime"])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df = weather_df.set_index('datetime')


# ========================================

hourly_df = pd.read_csv("dataset/hn_hourly.csv", parse_dates=["datetime"])
hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
hourly_df = hourly_df.set_index('datetime')

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

# Option to enable ML predictions
use_ml = st.sidebar.checkbox("Enable ML 5-day predictions (WeatherForecaster)", value=True)

if use_ml:
    # Lazy-load forecaster and keep in session state to avoid reloading repeatedly
    if 'forecaster' not in st.session_state:
        if WeatherForecaster is None:
            st.sidebar.error("WeatherForecaster import failed. Ensure src.daily_forecast_model is available.")
            st.session_state['forecaster'] = None
        else:
            try:
                st.sidebar.info("Loading ML models (this may take a few seconds)...")
                st.session_state['forecaster'] = WeatherForecaster()
                st.sidebar.success("ML models loaded")
            except Exception as e:
                st.session_state['forecaster'] = None
                st.sidebar.error(f"Failed to load models: {e}")
else:
    st.session_state['forecaster'] = None

# Remote API option (from API_DESIGN.md)
use_remote_api = st.sidebar.checkbox("Use remote API (FastAPI)", value=False)
api_base = st.sidebar.text_input("API base URL", value="http://localhost:8000/api/v1")
api_key = st.sidebar.text_input("API key (optional)", value="", type="password")

# Options matching API_DESIGN.md
include_metadata = st.sidebar.checkbox("Include metadata in API response", value=False)
include_confidence = st.sidebar.checkbox("Include confidence intervals in API response", value=False)


@st.cache_data(ttl=120)
def fetch_daily_from_api(date_str, location=DEFAULT_LOCATION, include_metadata=False, include_confidence=False, hours_ahead=24):
    """Call remote /forecast/daily endpoint. Returns parsed JSON or raises."""
    if requests is None:
        raise RuntimeError("requests library not available")
    url = f"{api_base.rstrip('/')}/forecast/daily"
    payload = {
        "location": location,
        "date": date_str,
        "include_metadata": include_metadata,
        "include_confidence": include_confidence
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_hourly_from_api(reference_datetime, location=DEFAULT_LOCATION, include_confidence=False, hours_ahead=24):
    """Call remote /forecast/hourly endpoint. Returns parsed JSON or raises."""
    if requests is None:
        raise RuntimeError("requests library not available")
    url = f"{api_base.rstrip('/')}/forecast/hourly"
    payload = {
        "location": location,
        "datetime": reference_datetime,
        "include_confidence": include_confidence,
        "hours_ahead": hours_ahead
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def get_api_health():
    if requests is None:
        return None
    try:
        url = f"{api_base.rstrip('/')}/health"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


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


    #======================
    # HOURLY TEMP

    # Icon map
    icon_map = {
        'clear-day': '☀️',
        'clear-night': '🌙',
        'partly-cloudy-day': '🌤️',
        'partly-cloudy-night': '⛅',
        'cloudy': '☁️',
        'rain': '🌧️',
        'fog': '🌫️',
        'wind': '💨'
    }
    def icon_from_code(code):
        return icon_map.get(str(code), "❓")

    # Prepare hourly source: remote API (if enabled) or local dataframe
    hourly_api_ok = False
    predicted_by_hour = {}
    temp_fallback = 20
    current_hour = pd.Timestamp.now().hour

    if use_remote_api and requests is not None:
        try:
            with st.spinner("Fetching remote hourly forecast..."):
                ref_dt_str = selected_ts.strftime('%Y-%m-%dT%H:00:00')
                api_resp = fetch_hourly_from_api(
                    reference_datetime=ref_dt_str,
                    location=DEFAULT_LOCATION,
                    include_confidence=include_confidence,
                    hours_ahead=24
                )

            if api_resp and api_resp.get('status') == 'success' and 'predictions' in api_resp:
                preds = api_resp['predictions']
                base_hour = pd.to_datetime(api_resp.get('reference_datetime', ref_dt_str)).hour
                for p in preds:
                    try:
                        tgt = p.get('target', '')
                        offset = int(tgt.split('+')[1]) if '+' in tgt else None
                    except Exception:
                        offset = None
                    if offset is None:
                        # fallback sequential mapping
                        offset = preds.index(p) + 1
                    hour_index = (base_hour + offset) % 24
                    predicted_by_hour[hour_index] = p.get('temperature')

                temp_fallback = float(list(predicted_by_hour.values())[0]) if predicted_by_hour else 20
                hourly_api_ok = True
        except Exception as e:
            st.error(f"Failed to fetch remote hourly forecast: {e}")
            hourly_api_ok = False

    if not hourly_api_ok:
        # fallback to local hourly_df for the selected date
        hourly_today = hourly_df[hourly_df.index.date == selected_date]
        temp_fallback = hourly_today['temp'].mean() if not hourly_today.empty else temp_fallback

    # --- Build scroll ngang ---
    html = '''
    <div style="
        display:flex; 
        overflow-x:auto; 
        overflow-y:hidden;
        gap:10px; 
        padding:10px; 
        white-space:nowrap;
        height:120px;">
    '''

    for hour in range(24):
        if hourly_api_ok and hour in predicted_by_hour:
            temp = f"{predicted_by_hour[hour]:.0f}°"
            icon = '🌤️'
        else:
            # local fallback if available
            if not hourly_api_ok and not hourly_today.empty:
                row = hourly_today[hourly_today.index.hour == hour]
                if row is not None and not row.empty:
                    temp = f"{row.iloc[0]['temp']:.0f}°"
                    icon = icon_from_code(row.iloc[0]['icon'])
                else:
                    temp = f"{temp_fallback:.0f}°"
                    icon = "❓"
            else:
                temp = f"{temp_fallback:.0f}°"
                icon = "❓"

        label = "Now" if hour == current_hour else f"{hour}:00"
        border = "2px solid #3b82f6" if hour == current_hour else "none"

        html += f"""
        <div style="
            flex:0 0 auto;
            width:70px; 
            background:#1e293b; 
            color:#e2e8f0; 
            border-radius:12px; 
            padding:10px; 
            text-align:center; 
            border:{border};
            height:100px;">
            <div style="font-size:18px; font-weight:bold; color:#3b82f6;">{temp}</div>
            <div style="font-size:26px; margin:4px 0;">{icon}</div>
            <div style="font-size:12px; color:#cbd5e1;">{label}</div>
        </div>
        """

    html += "</div>"

    st.components.v1.html(html, height=140, scrolling=False)


    # 5-Day Forecast
    st.markdown("""<h3 style="color: #ffffff; font-size: 30px; margin: 30px 0 15px 0;">📅 5-Day Forecast</h3>""", unsafe_allow_html=True)

    # If remote API is enabled, call it according to API_DESIGN.md
    used_remote = False
    if use_remote_api:
        try:
            with st.spinner("Fetching remote 5-day forecast..."):
                api_resp = fetch_daily_from_api(
                    date_str=selected_ts.strftime('%Y-%m-%d'),
                    location=DEFAULT_LOCATION,
                    include_metadata=include_metadata,
                    include_confidence=include_confidence
                )

            if api_resp and api_resp.get('status') == 'success' and 'predictions' in api_resp:
                preds = api_resp['predictions']
                cols = st.columns(5)
                # Render each prediction card per API schema
                for i in range(5):
                    with cols[i]:
                        if i < len(preds):
                            p = preds[i]
                            target = p.get('target', '')
                            fdate = p.get('forecast_date', '')
                            temp = p.get('temperature', None)
                            unit = p.get('unit', 'celsius')
                            ci = p.get('confidence_interval')
                            perf = p.get('model_performance')

                            display_temp = 'N/A' if temp is None else f"{temp:.1f}°C"
                            ci_html = ''
                            if ci and include_confidence:
                                try:
                                    lower = ci.get('lower')
                                    upper = ci.get('upper')
                                    clevel = ci.get('confidence_level', '')
                                    ci_html = f"<div style='color:#94a3b8; font-size:12px;'>CI({clevel}): {lower:.1f}° / {upper:.1f}°</div>"
                                except Exception:
                                    ci_html = ''

                            perf_html = ''
                            if perf and include_metadata:
                                try:
                                    rmse = perf.get('test_rmse')
                                    mae = perf.get('test_mae')
                                    r2 = perf.get('test_r2')
                                    perf_html = f"<div style='color:#94a3b8; font-size:12px;'>RMSE: {rmse:.2f} • MAE: {mae:.2f} • R2: {r2:.3f}</div>"
                                except Exception:
                                    perf_html = ''

                            st.markdown(f"""
                                <div class="forecast-card">
                                    <div style="font-weight:600; color:#e0e7ff; font-size: 14px; margin-bottom: 10px;">
                                        {pd.to_datetime(fdate).strftime('%a, %b %d') if fdate else 'N/A'}
                                    </div>
                                    <div style="font-size:42px; margin:12px 0;">
                                        🌤️
                                    </div>
                                    <div style="color:#ffffff; font-size:22px; font-weight:700; margin: 8px 0;">
                                        {display_temp}
                                    </div>
                                    {ci_html}
                                    {perf_html}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Empty placeholder if API returned fewer than 5
                            st.markdown("""
                                <div class="forecast-card">
                                    <div style="font-weight:600; color:#e0e7ff; font-size: 14px; margin-bottom: 10px;">-</div>
                                    <div style="font-size:42px; margin:12px 0;">-</div>
                                    <div style="color:#ffffff; font-size:22px; font-weight:700; margin: 8px 0;">-</div>
                                </div>
                            """, unsafe_allow_html=True)

                used_remote = True
            else:
                err = api_resp.get('error') if isinstance(api_resp, dict) else 'Unexpected response'
                st.error(f"Remote API returned an error: {err}")
        except Exception as e:
            # Show a clear error and fallback to raw-data display
            msg = str(e)
            # If requests HTTPError with response JSON, try to extract message
            if hasattr(e, 'response') and e.response is not None:
                try:
                    err_json = e.response.json()
                    msg = err_json.get('error', err_json.get('detail', msg))
                except Exception:
                    pass
            st.error(f"Failed to fetch remote forecast: {msg}")

    # If remote not used or failed, fallback to raw-data display (no ML calls)
    if not used_remote:
        forecast_days = weather_df.loc[weather_df.index >= selected_ts].head(5)

        cols = st.columns(min(5, len(forecast_days)))
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


    # 🧭 Weekly Forecast Chart
    st.markdown("""<h3 style="color: #ffffff; font-size: 30px; margin: 30px 0 15px 0;">📈 Weekly Temperature Trend</h3>""", unsafe_allow_html=True)
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
                {current.get('sealevelpressure', 0):.0f} mb
            </div>
        </div>
    """, unsafe_allow_html=True)



# SETUP BACKGROUND WEB
bg_path = r"bg.png"  
# --- Đọc ảnh và chuyển sang base64 ---
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_local(bg_file):
    bin_str = get_base64(bg_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Thêm background ---
set_bg_from_local(bg_path)




