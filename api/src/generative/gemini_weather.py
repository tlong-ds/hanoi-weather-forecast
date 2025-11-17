from __future__ import annotations
import os
import json
import google.generativeai as genai
import pandas as pd
from typing import Dict, Any, Optional

# Configure the generative AI model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def _build_prompt_for_record(date: str, record: Dict[str, Any]) -> str:
    """Create a prompt that asks the LLM to return a JSON with weather details
    (exclude any temperature fields) for the given date. The model must reply
    with a clean JSON object only.
    """
    keys = [
        "conditions", "humidity", "precip", "precipprob", "preciptype",
        "windspeed", "winddir", "windgust", "sunrise", "sunset", "uvindex",
        "visibility", "cloudcover", "moonphase", "dew", "sealevelpressure"
    ]
    sample = {k: record.get(k) for k in keys}
    prompt = (
        f"You are a concise weather data formatter. Given the following raw record for {date},"
        " return a JSON object (no surrounding text) with the exact keys listed below."
        " If a value is missing, use null. Do NOT include any temperature fields."
        f"\n\nRequired keys: {keys}\n\nRecord:\n{json.dumps(sample, default=str)}\n\n"
    )
    return prompt

def call_gemini(prompt: str, timeout: int = 20) -> Optional[str]:
    """Call the configured Gemini model and return the raw text response."""
    if not GEMINI_API_KEY:
        return None
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return None

def get_weather_details_via_gemini(date: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured weather details (no temperature fields) for the
    specified date. This will call the Gemini endpoint when configured; when
    not available it returns the `record` with temp keys removed.
    """
    prompt = _build_prompt_for_record(date, record)
    raw = call_gemini(prompt)

    if raw:
        try:
            text = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"raw": text}
        except Exception:
            return {"raw": raw}

    fallback_keys = {k: v for k, v in record.items() if k not in ("temp", "tempmin", "tempmax")}
    for k, v in list(fallback_keys.items()):
        if isinstance(v, (pd.Timestamp if 'pd' in globals() else object,)):
            try:
                fallback_keys[k] = str(v)
            except Exception:
                fallback_keys[k] = None
    return fallback_keys

def _build_prompt_for_hourly(dt: str, record: Dict[str, Any]) -> str:
    """Create a prompt for hourly record metadata (exclude temperature fields)."""
    keys = [
        "conditions", "humidity", "precip", "precipprob", "preciptype",
        "windspeed", "winddir", "windgust", "dew", "cloudcover", "uvindex",
        "visibility", "confidence", "icon"
    ]
    sample = {k: record.get(k) for k in keys}
    prompt = (
        f"You are a concise weather data formatter. Given this hourly record for {dt},"
        " return a JSON object (no surrounding text) with the exact keys listed below."
        " If a value is missing, use null. Do NOT include any temperature fields."
        f"\n\nRequired keys: {keys}\n\nRecord:\n{json.dumps(sample, default=str)}\n\n"
    )
    return prompt

def get_hourly_metadata_via_gemini(dt: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured hourly metadata (no temperature) for the given datetime."""
    prompt = _build_prompt_for_hourly(dt, record)
    raw = call_gemini(prompt)

    if raw:
        try:
            text = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"raw": raw}
        except Exception:
            return {"raw": raw}

    fallback = {k: v for k, v in record.items() if k not in ("temp", "tempmin", "tempmax", "temperature")}
    for k, v in list(fallback.items()):
        try:
            if hasattr(v, 'isoformat'):
                fallback[k] = v.isoformat()
        except Exception:
            fallback[k] = None
    return fallback