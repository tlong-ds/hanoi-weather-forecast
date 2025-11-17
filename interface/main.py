from dotenv import load_dotenv
load_dotenv()
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")  # Default fallback

import streamlit as st
import base64
st.set_page_config(layout="wide", page_title="Weather Forecast", page_icon="🌤️")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# ========================================
# FETCH DATA FROM API
# ========================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_data_from_api():
    """
    Fetch weather data and forecast from API endpoint.
    Returns a DataFrame with combined historical and forecast data and the raw API response.
    """
    try:
        api_base = API_URL.rstrip('/')
        # Fetch historical data from API
        historical_df = pd.DataFrame()
        historical_endpoint = f"{api_base}/api/v1/data/historical"
        try:
            # Request full historical dataset (no days limit) so server can decide; this
            # allows the API to return all records when available.
            hist_response = requests.get(historical_endpoint, timeout=30)
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if hist_data.get('status') == 'success' and hist_data.get('records'):
                    records = hist_data['records']
                    historical_df = pd.DataFrame(records)
                    historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])
                    historical_df['is_forecast'] = False
                    historical_df.set_index('datetime', inplace=True)
        except requests.exceptions.RequestException:
            pass

        # Fetch forecast from API
        endpoint = f"{api_base}/api/v1/forecast/daily"
        response = requests.post(
            endpoint,
            json={"location": "Hanoi, Vietnam", "include_confidence": True},
            timeout=10
        )

        if response.status_code != 200:
            # Return historical data only if API fails
            if not historical_df.empty:
                return historical_df, None
            return pd.DataFrame(), None

        data = response.json()
        predictions = data.get('predictions', [])
        if not predictions:
            return (historical_df if not historical_df.empty else pd.DataFrame()), data

        forecast_records = []
        for pred in predictions:
            temp = pred.get('temperature', np.nan)
            confidence = pred.get('confidence_interval', {})
            forecast_records.append({
                'datetime': pd.to_datetime(pred.get('forecast_date')),
                'temp': temp,
                'tempmax': confidence.get('upper', temp + 2 if not pd.isna(temp) else np.nan),
                'tempmin': confidence.get('lower', temp - 2 if not pd.isna(temp) else np.nan),
                'conditions': pred.get('conditions', 'AI Forecast'),
                'icon': pred.get('icon', 'partly-cloudy-day'),
                'humidity': pred.get('humidity', 70),
                'precip': pred.get('precip', 0),
                'precipprob': pred.get('precipprob', 20),
                'windspeed': pred.get('windspeed', 10),
                'winddir': pred.get('winddir', 90),
                'windgust': pred.get('windgust', 15),
                'sunrise': pred.get('sunrise', '06:00:00'),
                'sunset': pred.get('sunset', '18:00:00'),
                'moonphase': pred.get('moonphase', 0.5),
                'dew': pred.get('dew', temp - 5 if not pd.isna(temp) else np.nan),
                'precipcover': pred.get('precipcover', 0),
                'preciptype': pred.get('preciptype', None),
                'uvindex': pred.get('uvindex', 5),
                'visibility': pred.get('visibility', 10),
                'cloudcover': pred.get('cloudcover', 50),
                'sealevelpressure': pred.get('sealevelpressure', 1013),
                'is_forecast': True
            })

        forecast_df = pd.DataFrame(forecast_records)
        if not forecast_df.empty:
            forecast_df.set_index('datetime', inplace=True)

        if not historical_df.empty:
            combined_df = pd.concat([historical_df, forecast_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        else:
            combined_df = forecast_df

        return combined_df, data
    except Exception:
        return pd.DataFrame(), None


@st.cache_data(ttl=3600)
def fetch_hourly_data_from_api(reference_datetime=None):
    """
    Fetch hourly weather forecast from API endpoint.
    Returns a DataFrame with hourly forecast data.
    """
    try:
        api_base = API_URL.rstrip('/')
        payload = {"location": "Hanoi, Vietnam", "include_confidence": False}
        if reference_datetime:
            payload["reference_datetime"] = reference_datetime
        endpoint = f"{api_base}/api/v1/forecast/hourly"
        response = requests.post(endpoint, json=payload, timeout=60)
        if response.status_code != 200:
            return pd.DataFrame()
        data = response.json()
        predictions = data.get('predictions', [])
        if not predictions:
            return pd.DataFrame()
        hourly_records = []
        for pred in predictions:
            hourly_records.append({
                'datetime': pd.to_datetime(pred.get('forecast_datetime')),
                'temp': pred.get('temperature', np.nan),
                'icon': pred.get('icon', 'partly-cloudy-day'),
                'conditions': pred.get('conditions', 'Forecast'),
                'humidity': pred.get('humidity', 70),
                'windspeed': pred.get('windspeed', 10),
            })
        hourly_df = pd.DataFrame(hourly_records)
        if not hourly_df.empty:
            hourly_df.set_index('datetime', inplace=True)
            hourly_df = hourly_df.sort_index()
        return hourly_df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_historical_hourly_data():
    """
    Fetch historical hourly data from API.
    Returns the last hour of historical data as a DataFrame, or empty DataFrame on failure.
    """
    try:
        api_base = API_URL.rstrip('/')
        endpoint = f"{api_base}/api/v1/data/historical/hourly"
        
        response = requests.get(
            endpoint,
            params={"hours": 1},  # Get only the last hour
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and data.get('records'):
                records = data['records']
                df = pd.DataFrame(records)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Could not fetch historical hourly data: {e}")
        return pd.DataFrame()


# Fetch data from API
weather_df, api_response = fetch_weather_data_from_api()

# Fetch hourly forecast (next 24 hours)
hourly_forecast = fetch_hourly_data_from_api(reference_datetime=None)

# Fetch last historical hour
last_historical_hour = fetch_historical_hourly_data()

# Combine last historical hour with forecast (full 24-hour predictions)
if not last_historical_hour.empty and not hourly_forecast.empty:
    # Add is_forecast flag
    last_historical_hour['is_forecast'] = False
    hourly_forecast['is_forecast'] = True
    
    # Combine them: 1 historical + 24 forecast = 25 hours total (hour t to hour t next day)
    hourly_df = pd.concat([last_historical_hour, hourly_forecast])
    hourly_df = hourly_df.sort_index()
    print(f"✅ Combined hourly data: 1 historical + 24 forecast = {len(hourly_df)} total (hour t to hour t+24)")
elif not hourly_forecast.empty:
    # If no historical data, just show 24 hours of forecast
    hourly_df = hourly_forecast.head(24)
    hourly_df['is_forecast'] = True
else:
    hourly_df = pd.DataFrame()

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

def get_weather_background(icon, conditions):
    """Get background gradient and overlay based on weather condition"""
    icon_str = str(icon).lower()
    conditions_str = str(conditions).lower() if conditions else ''
    
    # Rain backgrounds
    if 'rain' in icon_str or 'rain' in conditions_str:
        return 'linear-gradient(180deg, rgba(74,95,122,0.95) 0%, rgba(45,62,80,0.95) 50%, rgba(26,35,50,0.95) 100%)'
    
    # Thunderstorm backgrounds
    elif 'thunder' in icon_str or 'thunder' in conditions_str:
        return 'linear-gradient(180deg, rgba(44,26,77,0.95) 0%, rgba(26,16,53,0.95) 50%, rgba(10,5,18,0.95) 100%)'
    
    # Clear day backgrounds
    elif 'clear-day' in icon_str or 'clear' in icon_str:
        return 'linear-gradient(180deg, rgba(86,204,242,0.95) 0%, rgba(47,128,237,0.95) 50%, rgba(30,91,168,0.95) 100%)'
    
    # Clear night backgrounds
    elif 'clear-night' in icon_str or 'night' in conditions_str:
        return 'linear-gradient(180deg, rgba(26,31,58,0.95) 0%, rgba(15,20,25,0.95) 50%, rgba(0,0,0,0.95) 100%)'
    
    # Cloudy backgrounds
    elif 'cloudy' in icon_str or 'overcast' in conditions_str:
        return 'linear-gradient(180deg, rgba(95,109,126,0.95) 0%, rgba(61,74,92,0.95) 50%, rgba(42,52,66,0.95) 100%)'
    
    # Partly cloudy backgrounds
    elif 'partly-cloudy' in icon_str:
        return 'linear-gradient(180deg, rgba(111,163,216,0.95) 0%, rgba(61,125,181,0.95) 50%, rgba(46,92,138,0.95) 100%)'
    
    # Fog backgrounds
    elif 'fog' in icon_str or 'fog' in conditions_str:
        return 'linear-gradient(180deg, rgba(138,154,168,0.95) 0%, rgba(95,111,126,0.95) 50%, rgba(61,74,92,0.95) 100%)'
    
    # Snow backgrounds
    elif 'snow' in icon_str or 'snow' in conditions_str:
        return 'linear-gradient(180deg, rgba(224,242,247,0.95) 0%, rgba(179,217,230,0.95) 50%, rgba(127,184,212,0.95) 100%)'
    
    # Wind backgrounds
    elif 'wind' in icon_str or 'wind' in conditions_str:
        return 'linear-gradient(180deg, rgba(123,168,196,0.95) 0%, rgba(74,117,145,0.95) 50%, rgba(46,92,138,0.95) 100%)'
    
    # Default background
    else:
        return 'linear-gradient(180deg, rgba(46,92,138,0.95) 0%, rgba(30,58,95,0.95) 100%)'

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
# STYLING - LOAD CUSTOM CSS
# ========================================
def load_css():
    """Load custom CSS from external file"""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ style.css not found")

load_css()

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

# Auto-select the last historical date (not forecast date)
if weather_df is not None and len(weather_df) > 0:
    # Filter for historical data only (where is_forecast is False)
    historical_data = weather_df[weather_df['is_forecast'] == False]
    if not historical_data.empty:
        selected_date = historical_data.index.max().date()
    else:
        # Fallback to latest date if no historical data
        selected_date = weather_df.index.max().date()
else:
    selected_date = datetime.now().date()

# Get current data for background
selected_ts = pd.to_datetime(selected_date)
if selected_ts in weather_df.index:
    current_for_bg = weather_df.loc[selected_ts]
else:
    current_for_bg = weather_df.loc[weather_df.index.date == selected_ts.date()].iloc[-1]

# Get weather background for entire page
page_bg = get_weather_background(current_for_bg.get('icon', ''), current_for_bg['conditions'])

# Apply dynamic background to entire page
st.markdown(f"""
    <style>
        .main {{
            background: {page_bg} !important;
        }}
        .stApp {{
            background: {page_bg} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# ========================================
# HEADER
# ========================================
display_date = pd.to_datetime(selected_date).strftime('%A, %B %d, %Y')

# Show API status
api_status = "✅ Connected" if api_response else "❌ Disconnected"
model_version = api_response.get('model_version', 'N/A') if api_response else 'N/A'

st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <p style="color: #60a5fa; font-size: 13px; margin-top: 5px;">
            API Status: {api_status} | Model: {model_version}
        </p>
    </div>
""", unsafe_allow_html=True)


# ========================================
# MAIN LAYOUT
# ========================================

# Ensure we have a session_state key to track the current view
if 'view' not in st.session_state:
    st.session_state['view'] = "Weather Forecast"

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Weather Forecast", "Historical Data Analysis", "Model Performance"])

# ========================================
# WEATHER FORECAST TAB
# ========================================
with tab1:
    st.session_state['view'] = "Weather Forecast"

    # Get current data for the selected date
    selected_ts = pd.to_datetime(selected_date)
    if selected_ts in weather_df.index:
        current = weather_df.loc[selected_ts]
    else:
        current = weather_df.loc[weather_df.index.date == selected_ts.date()].iloc[-1]

    # Weather Overview
    st.markdown(f"""
    <div style="
        text-align: center; 
        max-width: 600px; 
        margin: 0 auto 25px auto; 
        padding: 35px;
        position: relative;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        line-height: 1.35;
    ">
        <div style="font-size:20px; color:#ffffff; margin-bottom: 15px; text-shadow: 1px 1px 3px rgba(0,0,0,0.5); font-weight: 600; letter-spacing: 0.6px; line-height:1.4;">Hanoi, {display_date}</div>
        <div style="font-size:72px; font-weight:700; margin-bottom: 10px; text-shadow: 2px 2px 8px rgba(0,0,0,0.5); letter-spacing: 0.8px; line-height:1.05;">{round(current['temp'])}°</div>
        <div style="font-size:24px; color:#ffffff; margin-bottom: 20px; text-shadow: 1px 1px 4px rgba(0,0,0,0.5); font-weight:500; letter-spacing:0.4px; line-height:1.4;">{current['conditions']}</div>
        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 20px;">
            <div style="text-align: center;">
                <div style="color:#ffffff; font-size:14px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-weight:400; line-height:1.3;">High</div>
                <div style="color:#ffffff; font-size:20px; font-weight:500; text-shadow: 1px 1px 4px rgba(0,0,0,0.5); line-height:1.2;">{current['tempmax']:.0f}°</div>
            </div>
            <div style="text-align: center;">
                <div style="color:#ffffff; font-size:14px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); font-weight:400; line-height:1.3;">Low</div>
                <div style="color:#ffffff; font-size:20px; font-weight:500; text-shadow: 1px 1px 4px rgba(0,0,0,0.5); line-height:1.2;">{current['tempmin']:.0f}°</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

    # Hourly Forecast
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

    # Build unified glass morphism container with horizontal scroll
    html = '''
    <div style="
        background: rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 25px;
        margin-right: 0px; /* Adjust this value to modify the right margin */
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        backdrop-filter: blur(10px) saturate(140%);
        -webkit-backdrop-filter: blur(10px) saturate(140%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        overflow: hidden;
    ">
        <p style="
            color: #ffffff; 
            font-size: 15px; 
            margin: 0 20px 0 0; 
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            letter-spacing: -0.02em;
        ">24-Hour Forecast</p>
        <div style="
            display: flex; 
            overflow-x: auto; 
            overflow-y: hidden;
            gap: 0;
            padding: 5px 0;
            white-space: nowrap;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.3) rgba(255, 255, 255, 0.1);
        ">
    '''

    # Display all hourly data (1 historical + 24 forecast = 25 hours)
    if not hourly_df.empty:
        for i, (idx, row) in enumerate(hourly_df.iterrows()):
            if i >= 25:  # Allow 25 hours to show hour t to hour t next day
                break
            
            temp = f"{row['temp']:.0f}°"
            icon = icon_from_code(row.get('icon', ''))
            hour_label = idx.strftime('%H:%M')
            
            # Add separator between hours except for the first one
            border_left = "1px solid rgba(255, 255, 255, 0.1)" if i > 0 else "none"
            
            html += f"""
            <div style="
                flex: 1 1 auto;
                min-width: 70px;
                max-width: 120px;
                padding: 15px 10px;
                text-align: center;
                border-left: {border_left};
                transition: all 0.3s ease;
                font-family: 'Inter', sans-serif;
            ">
                <div style="
                    font-size: 12px; 
                    color: rgba(255, 255, 255, 0.7);
                    margin-bottom: 8px;
                    font-weight: 500;
                    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
                ">{hour_label}</div>
                <div style="
                    font-size: 32px; 
                    margin: 8px 0;
                    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
                ">{icon}</div>
                <div style="
                    font-size: 18px; 
                    font-weight: 600; 
                    color: #ffffff;
                    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
                ">{temp}</div>
            </div>
            """
    else:
        # Show placeholder if no hourly data
        for hour in range(24):
            border_left = "1px solid rgba(255, 255, 255, 0.1)" if hour > 0 else "none"
            html += f"""
            <div style="
                flex: 1 1 auto;
                min-width: 70px;
                max-width: 120px;
                padding: 15px 10px;
                text-align: center;
                border-left: {border_left};
                font-family: 'Inter', sans-serif;
            ">
                <div style="font-size: 12px; color: rgba(255, 255, 255, 0.5); margin-bottom: 8px;">{hour:02d}:00</div>
                <div style="font-size: 32px; margin: 8px 0; opacity: 0.3;">❓</div>
                <div style="font-size: 18px; font-weight: 600; color: rgba(255, 255, 255, 0.5);">--°</div>
            </div>
            """

    html += """
        </div>
    </div>
    <style>
        /* Custom scrollbar for webkit browsers */
        div::-webkit-scrollbar {
            height: 8px;
        }
        div::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin: 0 10px;
        }
        div::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
        }
        div::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.4);
        }
    </style>
    """
    st.components.v1.html(html, height=200, scrolling=False)

    # TWO-COLUMN LAYOUT
    left_col, right_col = st.columns([1, 1.5])

    # LEFT COLUMN - 5-Day Forecast
    with left_col:
        forecast_days = weather_df.loc[weather_df.index >= selected_ts].head(5)
        
        # Build unified container for 5-day forecast
        forecast_html = '''
        <div style="
            background: rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 0; /* Adjusted bottom margin to align with Weather Details */
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            backdrop-filter: blur(10px) saturate(140%);
            -webkit-backdrop-filter: blur(10px) saturate(140%);
            border: 1px solid rgba(255, 255, 255, 0.12);
        ">
            <p style="
                color: #ffffff; 
                font-size: 15px; 
                margin: 0 20px 0 0; 
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
                letter-spacing: -0.02em;
            ">5-Day Forecast</p>
        '''
        
        for i, (idx, row) in enumerate(forecast_days.iterrows()):
            icon_emoji, _ = get_weather_icon_and_text(row.get('icon', ''), row.get('conditions', ''))
            precip_prob = row.get('precipprob', 0)
            
            border_top = "1px solid rgba(255, 255, 255, 0.1)" if i > 0 else "none"
            
            forecast_html += f"""
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 15px 0;
                border-top: {border_top};
                font-family: 'Inter', sans-serif;
            ">
                <div style="flex: 1; min-width: 80px;">
                    <div style="
                        font-weight: 600; 
                        color: #ffffff; 
                        font-size: 14px; 
                        margin-bottom: 4px;
                        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
                    ">{idx.strftime('%a, %b %d')}</div>
                    <div style="
                        color: rgba(255, 255, 255, 0.7); 
                        font-size: 12px;
                    ">💧 {precip_prob:.0f}%</div>
                </div>
                <div style="
                    font-size: 42px; 
                    margin: 0 15px;
                    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
                ">{icon_emoji}</div>
                <div style="text-align: right; flex: 1; min-width: 80px;">
                    <div style="
                        color: #ffffff; 
                        font-size: 24px; 
                        font-weight: 700;
                        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
                    ">{row['temp']:.0f}°C</div>
                    <div style="
                        color: rgba(255, 255, 255, 0.7); 
                        font-size: 13px;
                    ">{row['tempmin']:.0f}° / {row['tempmax']:.0f}°</div>
                </div>
            </div>
            """
        
        forecast_html += "</div>"
        st.components.v1.html(forecast_html, height=600, scrolling=False)

    # RIGHT COLUMN - Temperature Chart & Details
    with right_col:
        # Temperature Trend Chart
        selected_ts = pd.to_datetime(selected_date)
        weekly_df = weather_df.loc[weather_df.index >= selected_ts].head(7)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        ax.plot(weekly_df.index, weekly_df['tempmax'], label="Max Temp", linewidth=3, marker='o', color='#ff6b6b')
        ax.plot(weekly_df.index, weekly_df['tempmin'], label="Min Temp", linewidth=3, marker='o', linestyle="--", color='#4ecdc4')
        
        # Add predicted temperature line if available
        if 'temp' in weekly_df.columns:
            ax.plot(weekly_df.index, weekly_df['temp'], label="Predicted Temp", linewidth=3, marker='s', linestyle=':', color='#ffd93d')
        
        ax.fill_between(weekly_df.index, weekly_df['tempmin'], weekly_df['tempmax'], alpha=0.2, color='#ffffff')
        
        ax.set_title("", color="#ffffff")
        ax.set_ylabel("Temperature (°C)", color="#ffffff", fontsize=10)
        ax.tick_params(axis='x', colors='#ffffff', rotation=15, labelsize=9)
        ax.tick_params(axis='y', colors='#ffffff', labelsize=9)
        ax.legend(facecolor='none', labelcolor="#ffffff", frameon=False, fontsize=9)
        ax.grid(True, alpha=0.2, color='#ffffff', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#ffffff')
            spine.set_alpha(0.2)
        
        # Convert plot to base64 image
        import io
        import base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', transparent=True, dpi=100)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        # Render container with embedded image
        st.markdown(f'''
        <div style="
            background: rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            padding: 20px;
            margin: 7px 10px 20px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
            backdrop-filter: blur(10px) saturate(140%);
            -webkit-backdrop-filter: blur(10px) saturate(140%);
            border: 1px solid rgba(255, 255, 255, 0.12);
        ">
            <p style="
                color: #ffffff; 
                font-size: 15px; 
                margin: 0 0 15px 0; 
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
                letter-spacing: -0.02em;
            ">Temperature Trend</p>
            <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; max-width: 100%;">
        </div>
        ''', unsafe_allow_html=True)
        
        # Weather Details - compact single-row (4 columns)
        detail_cols = st.columns(4)

        # 1) Sun & Moon (combined) - show only sunrise & sunset
        with detail_cols[0]:
            sunrise = current.get('sunrise', 'N/A')
            sunset = current.get('sunset', 'N/A')
            # Format times to HH:MM if possible
            try:
                sunrise_time = pd.to_datetime(sunrise).strftime('%H:%M')
            except Exception:
                sunrise_time = str(sunrise)
            try:
                sunset_time = pd.to_datetime(sunset).strftime('%H:%M')
            except Exception:
                sunset_time = str(sunset)

            st.markdown(f"""
                <div class="detail-box" style="padding:10px; min-height:98px; text-align:left;">
                    <p style="color: #ffffff; font-size:15px; margin:0 0 8px 0; font-weight:600; font-family: 'Inter', sans-serif; text-shadow: 0 2px 6px rgba(0,0,0,0.4);">Sunrise</p>
                    <div style="font-size:20px; margin-bottom:6px;">🌅</div>
                    <div style="font-size:18px; font-weight:700; color:#ffffff; margin-bottom:6px;">{sunrise_time}</div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.6);">Sunset: {sunset_time}</div>
                </div>
            """, unsafe_allow_html=True)

        # 2) Humidity & Dew
        with detail_cols[1]:
            st.markdown(f"""
                <div class="detail-box" style="padding:10px; min-height:98px; text-align:left;">
                    <p style="color: #ffffff; font-size:15px; margin:0 0 8px 0; font-weight:600; font-family: 'Inter', sans-serif; text-shadow: 0 2px 6px rgba(0,0,0,0.4);">Humidity & Dew</p>
                    <div style="font-size:20px; margin-bottom:6px;">💧</div>
                    <div style="font-size:20px; font-weight:700; color:#ffffff; margin-bottom:4px;">{current['humidity']:.0f}%</div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.7);">Dew point: {current.get('dew', 0):.0f}°C</div>
                </div>
            """, unsafe_allow_html=True)

        # 3) Wind
        with detail_cols[2]:
            windgust = current.get('windgust', current['windspeed'])
            winddir = current.get('winddir', 0)
            wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            wind_dir_text = wind_directions[int((winddir + 22.5) / 45) % 8]
            st.markdown(f"""
                <div class="detail-box" style="padding:10px; min-height:98px; text-align:left;">
                    <p style="color: #ffffff; font-size:15px; margin:0 0 8px 0; font-weight:600; font-family: 'Inter', sans-serif; text-shadow: 0 2px 6px rgba(0,0,0,0.4);">Wind</p>
                    <div style="font-size:20px; margin-bottom:6px;">💨</div>
                    <div style="font-size:20px; font-weight:700; color:#ffffff; margin-bottom:4px;">{current['windspeed']:.0f} km/h</div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.7);">{wind_dir_text} • Gusts: {windgust:.0f} km/h</div>
                </div>
            """, unsafe_allow_html=True)

        # 4) UV Index
        with detail_cols[3]:
            uv_level = get_uv_level(current.get('uvindex', 0))
            st.markdown(f"""
                <div class="detail-box" style="padding:10px; min-height:98px; text-align:left;">
                    <p style="color: #ffffff; font-size:15px; margin:0 0 8px 0; font-weight:600; font-family: 'Inter', sans-serif; text-shadow: 0 2px 6px rgba(0,0,0,0.4);">UV Index</p>
                    <div style="font-size:20px; margin-bottom:6px;">☀️</div>
                    <div style="font-size:20px; font-weight:700; color:#ffffff; margin-bottom:4px;">{current.get('uvindex', 0):.0f}</div>
                    <div style="font-size:12px; color:rgba(255,255,255,0.7);">{uv_level}</div>
                </div>
            """, unsafe_allow_html=True)


    # Adjust the top margin of the Weather Details section
        st.markdown(
            """
            <style>
                .detail-box {
                    margin-top: 0px; /* Adjust this value to reduce or increase the gap */
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Adjust the right margin of the Weather Details section
        st.markdown(
            """
            <style>
                .detail-box {
                    margin-right: 9px; /* Adjust this value to modify the right margin */
                }
            </style>
            """,
            unsafe_allow_html=True
        )

# ========================================
# HISTORICAL DATA TAB
# ========================================
with tab2:
    st.session_state['view'] = "Historical Data Analysis"
    st.title("Historical Data Analysis")
    st.markdown("This page shows historical temperature series fetched directly from the API (no file reads).")

    # Fetch combined data from API (cached) and use only historical rows
    with st.spinner("Loading historical data from API..."):
        df_api, api_info = fetch_weather_data_from_api()

    if df_api is None or df_api.empty:
        st.warning("No data returned from API.")
    else:
        # Select historical rows only (where available)
        if 'is_forecast' in df_api.columns:
            hist = df_api[df_api['is_forecast'] == False]
        else:
            # If API didn't tag forecasts, assume all rows before the first forecast date are historical
            hist = df_api.copy()

        if hist.empty:
            st.info("No historical records available from the API.")
        else:
            # Ensure datetime index
            if not isinstance(hist.index, pd.DatetimeIndex):
                try:
                    hist.index = pd.to_datetime(hist.index)
                except Exception:
                    pass

            # Add filtering options inside an expander (widgets only).
            # Use explicit session_state keys so changes reliably trigger reruns.
            month_options = ["All"] + list(hist.index.month_name().unique())
            year_options = ["All"] + list(hist.index.year.unique())

            # Read widget values from session_state and apply filters (outside the expander)
            start_date = st.session_state.get('hist_start_date', hist.index.min().date())
            end_date = st.session_state.get('hist_end_date', hist.index.max().date())
            month = st.session_state.get('hist_month', 'All')
            year = st.session_state.get('hist_year', 'All')

            # Validate start_date and end_date
            if start_date > end_date:
                st.error("Start Date must be less than or equal to End Date.")
            else:
                # Apply filters
                filtered_hist = hist.copy()
                if start_date:
                    filtered_hist = filtered_hist[filtered_hist.index.date >= start_date]
                if end_date:
                    filtered_hist = filtered_hist[filtered_hist.index.date <= end_date]
                if month != "All":
                    filtered_hist = filtered_hist[filtered_hist.index.month_name() == month]
                if year != "All":
                    filtered_hist = filtered_hist[filtered_hist.index.year == year]

                # Plot historical `temp` series without zoom or table (rendered outside the expander)
                if 'temp' in filtered_hist.columns and not filtered_hist['temp'].dropna().empty:
                    # Time Series Decomposition
                    temp_series = filtered_hist['temp'].dropna()
                    
                    # Decomposition requires at least 2 full periods of data
                    period = 7 # Weekly seasonality
                    if len(temp_series) >= 2 * period:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        from plotly.subplots import make_subplots
                        import plotly.graph_objects as go
                        
                        decomposition = seasonal_decompose(temp_series, model='additive', period=period)
                        
                        # Create subplot figure
                        fig_decomp = make_subplots(rows=4, cols=1, 
                                                   subplot_titles=("Observed", "Trend", "Seasonality", "Residuals"),
                                                   vertical_spacing=0.08,
                                                   shared_xaxes=True)

                        # Add Observed
                        fig_decomp.add_trace(go.Scatter(x=temp_series.index, y=temp_series, 
                                                        mode='lines', name='Observed', line=dict(color='#636EFA')), 
                                             row=1, col=1)
                        fig_decomp.update_yaxes(title_text="Observed", row=1, col=1)

                        # Add Trend
                        fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, 
                                                        mode='lines', name='Trend', line=dict(color='#FF7F50')), 
                                             row=2, col=1)
                        fig_decomp.update_yaxes(title_text="Trend", row=2, col=1)

                        # Add Seasonality
                        fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                                                        mode='lines', name='Seasonality', line=dict(color='#00CC96')), 
                                             row=3, col=1)
                        fig_decomp.update_yaxes(title_text="Seasonality", row=3, col=1)

                        # Add Residuals
                        fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, 
                                                        mode='markers', name='Residuals', marker=dict(color='#EF553B', size=3)), 
                                             row=4, col=1)
                        fig_decomp.update_yaxes(title_text="Residuals", row=4, col=1)

                        # Update layout for the combined figure
                        fig_decomp.update_layout(
                            showlegend=False,
                            margin=dict(l=20, r=20, t=30, b=20),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=800 # Adjust height as needed
                        )
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                    else:
                        st.info(f"Not enough data to perform time series decomposition. At least {2*period} data points are required.")

                else:
                    st.info("Filtered data does not contain a 'temp' column or contains no values.")

# ========================================
# MODEL PERFORMANCE TAB
# ========================================
with tab3:
    st.session_state['view'] = "Model Performance"
    st.title("Model Performance")

    api_base_url = API_URL.rstrip('/')
    daily_perf_url = f"{api_base_url}/api/v1/performance/daily"
    hourly_perf_url = f"{api_base_url}/api/v1/performance/hourly"
    daily_plots_url = f"{api_base_url}/api/v1/performance/daily/plots"
    hourly_plots_url = f"{api_base_url}/api/v1/performance/hourly/plots"

    # Create sub-tabs for daily and hourly models
    daily_tab, hourly_tab = st.tabs(["Daily Forecast Model", "Hourly Forecast Model"])

    # -------------------------
    # DAILY FORECAST MODEL TAB
    # -------------------------
    with daily_tab:
        st.header("Daily Forecast Model Evaluation")
        try:
            # --- Display Metrics ---
            st.subheader("Evaluation Metrics")
            response = requests.get(daily_perf_url, timeout=10)
            response.raise_for_status()
            daily_metrics_data = response.json()
            daily_metrics_df = pd.DataFrame(daily_metrics_data)
            st.table(daily_metrics_df.set_index('target'))

            # --- Display Plots ---
            st.subheader("Evaluation Plots")
            st.image(f"{daily_plots_url}/scatter_all_targets.png", caption="Scatter Plot for All Targets")
            st.image(f"{daily_plots_url}/timeseries_all_targets.png", caption="Time Series for All Targets")

            # Per-target plots
            target_options = [f"t+{i}" for i in range(1, 6)]
            selected_target = st.selectbox("Select a target to view detailed plots:", options=target_options, key="daily_target_select")

            if selected_target:
                st.image(f"{daily_plots_url}/scatter_{selected_target}.png", caption=f"Scatter Plot for {selected_target}")
                st.image(f"{daily_plots_url}/timeseries_{selected_target}.png", caption=f"Time Series for {selected_target}")
                st.image(f"{daily_plots_url}/timeseries_{selected_target}_zoom.png", caption=f"Zoomed Time Series for {selected_target}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not fetch daily performance data from API: {e}")
        except Exception as e:
            st.error(f"An error occurred while displaying daily performance data: {e}")

    # -------------------------
    # HOURLY FORECAST MODEL TAB
    # -------------------------
    with hourly_tab:
        st.header("Hourly Forecast Model Evaluation")
        try:
            response = requests.get(hourly_perf_url, timeout=10)
            response.raise_for_status()
            hourly_results = response.json()

            # --- Display Average Metrics ---
            st.subheader("Average Metrics (across all hours)")
            avg_metrics = hourly_results.get("average_metrics", {})
            if avg_metrics:
                cols = st.columns(4)
                cols[0].metric("MAE", f"{avg_metrics.get('MAE', 0):.3f}")
                cols[1].metric("RMSE", f"{avg_metrics.get('RMSE', 0):.3f}")
                cols[2].metric("MAPE", f"{avg_metrics.get('MAPE', 0):.3f}%")
                cols[3].metric("R²", f"{avg_metrics.get('R2', 0):.3f}")

            # --- Display Per-Hour Metrics ---
            st.subheader("Metrics Per Hour")
            per_hour_metrics = hourly_results.get("per_hour_metrics", {})
            if per_hour_metrics:
                per_hour_df = pd.DataFrame.from_dict(per_hour_metrics, orient='index')
                metrics_df = per_hour_df['metrics'].apply(pd.Series)
                per_hour_df = pd.concat([per_hour_df[['hour']], metrics_df], axis=1)
                st.table(per_hour_df)

            # --- Display Plots ---
            st.subheader("Evaluation Plots")
            st.image(f"{hourly_plots_url}/metrics_comparison.png", caption="Metrics Comparison Across Horizons")
            st.image(f"{hourly_plots_url}/time_series_predictions.png", caption="Time Series Predictions")

            # Per-hour prediction plots
            hour_options = [f"t+{h}h" for h in [1, 6, 12, 18, 24]]
            selected_hour = st.selectbox("Select an hour to view prediction vs. actuals:", options=hour_options, key="hourly_target_select")

            if selected_hour:
                st.image(f"{hourly_plots_url}/predictions_vs_actuals_{selected_hour}.png", caption=f"Predictions vs. Actuals for {selected_hour}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not fetch hourly performance data from API: {e}")
        except Exception as e:
            st.error(f"An error occurred while displaying hourly performance data: {e}")

# ========================================
# BACKGROUND IMAGE
# ========================================
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
# set_bg_from_local(bg_path)