import ccxt
import pandas as pd
import joblib
import os
import sys
import requests
import asyncio
from scipy.stats import linregress 
from datetime import datetime, timedelta, timezone 
import numpy as np
import json 

# Ensure correct path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Import Utility Functions and Constants ---
from utils.feature_calculator import (
    calculate_bb_position,
    calculate_base_features, 
    calculate_rejection_power,
    ATR_PERIOD,
    SLOPE_WINDOW
)

# --- Configuration and Constants ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') 
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TELEGRAM_PROXY_URL = os.environ.get('TELEGRAM_PROXY_URL') 

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_PATH, 'models') 

TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']
MINIMUM_ADVANTAGE_ALERT = 70.0 # Minimum Net Advantage (%) to trigger an alert.
VENEZUELA_TZ_OFFSET = timedelta(hours=-4) # UTC-4 for local time adjustment

# --- Helper Functions (Loaded Models and Break Even) ---

# Global variable for models (loaded once at the beginning of the process)
LOADED_MODELS = None

def load_all_models():
    """
    Loads all 7 trained Random Forest models from the MODELS_DIR directory.
    
    Returns:
        dict: A dictionary mapping target names to their loaded model objects.
    """
    global LOADED_MODELS
    if LOADED_MODELS is not None:
        return LOADED_MODELS
        
    models = {}
    print("Loading all 7 multi-target models...")
    for target in TARGET_COLUMNS:
        model_path = os.path.join(MODELS_DIR, f'random_forest_multi_target_{target}.pkl')
        try:
            # Note: We need to use 'rb' mode for joblib load
            models[target] = joblib.load(model_path)
        except FileNotFoundError:
            print(f"⚠️ ERROR: Model not found at {model_path}. Ensure tune_model.py has been executed.")
            return None
    LOADED_MODELS = models
    return models

def calculate_break_even(target_value):
    """
    Calculates the minimum required success probability (Break Even) based on 
    the Risk-to-Reward (R:R) ratio for the given target value.
    """
    sl_multiplier = 1.0
    if target_value in [10, 15, 20]:
        sl_multiplier = 3.0
    
    # Break Even Formula: Risk / (Risk + Reward)
    be_needed = (sl_multiplier / (sl_multiplier + target_value)) * 100 
    return round(be_needed, 2)

# --- Feature Processing and Data Fetching (Modularized) ---

def fetch_and_process_features(symbol: str, timeframe: str = '1h', limit: int = 100):
    """
    Fetches OHLCV data for a single symbol and calculates the 5 required features.
    
    Returns:
        pd.DataFrame: A one-row DataFrame containing the 5 features, or empty DataFrame.
    """
    exchange_name = 'kucoin' 

    print(f"📡 Fetching data for {symbol} from {exchange_name}...")
    
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        
        # ccxt is blocking I/O, which is fine as this function will be run in a separate thread.
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print(f"❌ Error: ccxt did not return OHLCV data for {symbol}.")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df[~df.index.duplicated(keep='first')] 
        df_ohlcv_context = df.tail(limit)

    except Exception as e:
        print(f"❌ Error connecting or obtaining data from {exchange_name} for {symbol}: {e}")
        return pd.DataFrame()

    # Feature Calculation Logic (copied from original process_live_features)
    if len(df_ohlcv_context) < SLOPE_WINDOW:
        print(f"❌ Insufficient historical data ({len(df_ohlcv_context)} candles) to calculate Slope 50 for {symbol}.")
        return pd.DataFrame()

    df = df_ohlcv_context.copy()
    df = calculate_bb_position(df)
    df = calculate_base_features(df, atr_period=ATR_PERIOD)
    
    def apply_slope(series):
        if len(series) < SLOPE_WINDOW:
            return np.nan
        x = range(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
        
    df['Slope_50'] = df['close'].rolling(window=SLOPE_WINDOW).apply(apply_slope, raw=False)
    last_candle = df_ohlcv_context.iloc[-1]
    rejection_power = calculate_rejection_power(
        open_price=last_candle['open'],
        close_price=last_candle['close'],
        high_price=last_candle['high'],
        low_price=last_candle['low']
    )
    last_features = df.iloc[-1].copy() 
    last_features['Rejection_Power'] = rejection_power
    
    feature_cols = ['ATR_14', 'RSI_14', 'Slope_50', 'Rejection_Power', 'BB_Position']
    
    return last_features[feature_cols].to_frame().T.dropna()

def format_prediction_result(features, prediction_metrics, timestamp_local, symbol: str, source: str = 'SCHEDULED') -> str:
    """Formats the prediction result into a Telegram message string."""
    
    # Calculate R:R for the best target for the alert message
    target_val = int(prediction_metrics['best_target'].split('_')[1].replace('x', ''))
    sl_mult = 1.0 if target_val <= 5 else 3.0
    r_r_ratio = target_val / sl_mult
    
    # Determine header and source tag
    if source in ['MANUAL_API', 'TELEGRAM_WEBHOOK', 'SINGLE_SYMBOL']:
        header_emoji = "🕵️"
        source_tag = "(Manual Scan)"
    else:
        header_emoji = "🔥"
        source_tag = "(Scheduled Scan)"

    # Build the message
    message = f"""
{header_emoji} ML SIGNAL ALERT! {source_tag} {header_emoji}
------------------------------------------
⌚ Time (Local): {timestamp_local.strftime('%Y-%m-%d %H:%M')}
📈 Symbol: {symbol}
🚀 Direction: {prediction_metrics['direction']}
🎯 Best Target: {prediction_metrics['best_target']} (R:R {r_r_ratio:.1f}:1)
📊 P. Prediction: {prediction_metrics['best_prob']:.2f}%
🟢 Net Advantage: +{prediction_metrics['max_advantage']:.2f}%
------------------------------------------
RSI: {features['RSI_14']:.2f} | Slope 50: {features['Slope_50']:.2f} | ATR 14: {features['ATR_14']:.2f}
"""
    return message


# --- Telegram Alert Functions ---
# (Using the same functions from the previous implementation)
def _send_telegram_alert_internal(message: str) -> bool:
    """Internal function to send a message via Proxy or Direct Telegram API."""
    # Logic remains the same: attempts to send via proxy or direct API
    if TELEGRAM_PROXY_URL:
        # ... (Proxy logic) ...
        url = TELEGRAM_PROXY_URL
        payload = {"message": message} 
    elif TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    else:
        print("❌ ERROR: No Telegram credentials/Proxy URL found.")
        return False
        
    try:
        if TELEGRAM_PROXY_URL:
            response = requests.post(url, json=payload, timeout=15)
        else:
            response = requests.post(url, data=payload, timeout=15)
                    
            response.raise_for_status() 
            print("✅ Telegram: Alert successfully sent.")
            return True
    except requests.exceptions.RequestException as e:
        print(f"❌ FATAL NETWORK/API ERROR: Telegram alert FAILED. {e}")
        return False

def send_ping_message(message: str) -> bool:
    """Public function to send a non-critical ping message. Used by the scheduler."""
    return _send_telegram_alert_internal(message)

# --- Core ML Prediction Logic ---

def _get_prediction_metrics(df_current_features, models):
    """Calculates all 7 prediction probabilities and returns the best metrics."""
    if df_current_features.empty:
        return None

    features = df_current_features.iloc[0]
    rsi_value = features['RSI_14']
    
    # Heuristic for Direction
    if rsi_value < 40:
        direction = 'LONG'
    elif rsi_value > 60:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'
    
    max_advantage = -np.inf
    best_target = None
    best_prob = 0.0

    for target, model in models.items():
        data_point = df_current_features.copy()

        # Prediction is a blocking operation, which is fine as this will be run in a thread
        prob = model.predict_proba(data_point)[:, 1][0]
        prob_percent = round(prob * 100, 2)
        
        target_value = int(target.split('_')[1].replace('x', ''))
        be_needed = calculate_break_even(target_value)
        advantage = round(prob_percent - be_needed, 2)

        if advantage > max_advantage:
            max_advantage = advantage
            best_target = target
            best_prob = prob_percent
            
    return {
        'direction': direction,
        'max_advantage': max_advantage,
        'best_target': best_target,
        'best_prob': best_prob,
        'rsi_value': rsi_value
    }

# --- New Interactive Prediction Function (NOW SYNCHRONOUS) ---
def run_single_symbol_prediction(symbol: str) -> dict | None:
    """
    Executes the full prediction for a single symbol and returns the result dictionary.
    
    This is a synchronous (blocking) function designed to be executed in a separate thread.
    
    Returns:
        dict | None: Dictionary with prediction details if successful, otherwise None.
    """
    
    print(f"--- 🧠 Running single symbol prediction for {symbol} ---")
    
    models = load_all_models()
    if not models:
        return None
        
    # Fetching data and processing features are blocking I/O/CPU operations
    df_current_features = fetch_and_process_features(symbol, limit=100)
    if df_current_features.empty:
        return None
        
    # Getting prediction metrics is a CPU-bound operation
    prediction_metrics = _get_prediction_metrics(df_current_features, models)
    if not prediction_metrics:
        return None
        
    # Add features and timestamp to the result for formatting
    timestamp_utc = df_current_features.index[0].tz_localize(timezone.utc)
    timestamp_local = timestamp_utc + VENEZUELA_TZ_OFFSET
    
    result = {
        'symbol': symbol,
        'timestamp_local': timestamp_local.strftime('%Y-%m-%d %H:%M:%S'),
        'features': df_current_features.iloc[0].to_dict(),
        'metrics': prediction_metrics,
        'alert_found': prediction_metrics['max_advantage'] >= MINIMUM_ADVANTAGE_ALERT
    }
    
    return result

# --- Main Scan and Alert Logic (Updated to use modular functions) ---

async def predict_and_alert(df_current_features, models, symbol: str, source: str = 'SCHEDULED'):
    """
    Predicts the probability and sends an alert if the minimum advantage is met.
    (This function is used by the continuous scheduler).
    """
    
    if df_current_features.empty:
        return False
        
    prediction_metrics = _get_prediction_metrics(df_current_features, models)
    if not prediction_metrics or prediction_metrics['direction'] == 'NEUTRAL':
        return False
        
    max_advantage = prediction_metrics['max_advantage']

    # 2. Generate Alert (if threshold is met)
    if max_advantage >= MINIMUM_ADVANTAGE_ALERT:
        
        timestamp_utc = df_current_features.index[0].tz_localize(timezone.utc)
        timestamp_local = timestamp_utc + VENEZUELA_TZ_OFFSET
        
        # Use formatting function for alert message
        alert_message = format_prediction_result(df_current_features.iloc[0], prediction_metrics, timestamp_local, symbol, source)

        print(f"--- ✅ ALERTA ENVIADA: Net Advantage {max_advantage:.2f}% ---")
        _send_telegram_alert_internal(alert_message)
        return True # <-- Signal found and sent

    else:
        # Signal rejected - logs internally but remains SILENT on Telegram
        print(f"🟡 Prediction found ({prediction_metrics['best_target']}), but Net Advantage ({max_advantage:.2f}%) is below the alert threshold ({MINIMUM_ADVANTAGE_ALERT:.2f}%).")
        return False # <-- Signal rejected

async def run_scanner_main(source: str = 'SCHEDULED'):
    """
    Orchestrates the live scanner process (used by the scheduler).
    Currently runs for ETH/USDT only.
    """
    print("--- ⏳ Initializing Live Market Scanner ---")
    
    models = load_all_models()
    if not models:
        return False
        
    symbol_to_scan = 'ETH/USDT'
    
    df_current_features = fetch_and_process_features(symbol_to_scan, limit=100)
    
    alert_sent = await predict_and_alert(df_current_features, models, symbol_to_scan, source)
    
    print("--- ✅ Scan Complete. ---")
    
    return alert_sent