"""This script functions as the live market scanner and inference engine. It simulates fetching real-time OHLCV data, processes the features for the current candle, loads the trained multi-target models, and predicts the statistical advantage. It generates a clear alert if the net advantage exceeds a predefined minimum threshold."""
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Import Utility Functions and Constants ---
from utils.feature_calculator import (
    calculate_bb_position,
    calculate_base_features, 
    calculate_rejection_power,
    ATR_PERIOD,
    SLOPE_WINDOW
)

# --- Telegram Configuration ---
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN') 
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TELEGRAM_PROXY_URL_FULL = os.environ.get('TELEGRAM_PROXY_URL') 

# --- Global Configuration ---
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_PATH, 'models') 
TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']
MINIMUM_ADVANTAGE_ALERT = 70.0 # Minimum Net Advantage (%) to trigger an alert.
VENEZUELA_TZ_OFFSET = timedelta(hours=-4) # UTC-4 for local time adjustment
NETWORK_TIMEOUT_SECONDS = 60 

# --- Proxy URL Correction (CRITICAL FIX for Render 404) ---
# We must use the root URL for a Keep-Alive ping, not the /send_alert endpoint.
try:
    # Example: If TELEGRAM_PROXY_URL_FULL is 'https://model-breakup1-1.onrender.com/send_alert',
    # PROXY_BASE_URL becomes 'https://model-breakup1-1.onrender.com'
    if TELEGRAM_PROXY_URL_FULL and '/' in TELEGRAM_PROXY_URL_FULL:
        PROXY_BASE_URL = TELEGRAM_PROXY_URL_FULL.rsplit('/', 1)[0]
    else:
        PROXY_BASE_URL = TELEGRAM_PROXY_URL_FULL
except:
    PROXY_BASE_URL = TELEGRAM_PROXY_URL_FULL

# --- Helper Functions (Borrowed from predict_probability.py) ---
def load_all_models():
    """
    Loads all 7 trained Random Forest models from the MODELS_DIR directory.
            
    Returns:
        dict: A dictionary mapping target names to their loaded model objects.
    """
    models = {}
    print("Loading all 7 multi-target models...")
    for target in TARGET_COLUMNS:
        model_path = os.path.join(MODELS_DIR, f'random_forest_multi_target_{target}.pkl')
        try:
            models[target] = joblib.load(model_path)
        except FileNotFoundError:
            print(f"⚠️ ERROR: Model not found at {model_path}. Ensure tune_model.py has been executed.")
            return None
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

# --- Live Data and Feature Processing Functions (Updated) ---
def fetch_live_data_simulation(symbol: str) -> pd.DataFrame:
    """
    Fetches the last 100 OHLCV candles from the KuCoin exchange using ccxt for the given symbol.
    
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    exchange_name = 'kucoin' 
    timeframe = '1h'
    limit = 100
    print(f"📡 Fetching live data for {symbol} from {exchange_name}...")
    
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        
        # --- MODIFICADO: USA EL ARGUMENTO 'symbol' ---
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print(f"❌ Error: ccxt did not return OHLCV data for {symbol} from {exchange_name}. Symbol might be invalid.")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        df = df[~df.index.duplicated(keep='first')] 
        return df.tail(limit)
    except Exception as e:
        print(f"❌ Error connecting or obtaining data for {symbol} from {exchange_name}: {e}")
        return pd.DataFrame()

def process_live_features(df_ohlcv_context):
    """
    Calculates the 5 required features for the last candle in the OHLCV context.
    
    Returns:
        pd.DataFrame: A one-row DataFrame containing the 5 features.
    """
    
    # Ensure the context has enough data
    if len(df_ohlcv_context) < SLOPE_WINDOW:
        print(f"❌ Insufficient historical data ({len(df_ohlcv_context)} candles) to calculate Slope 50.")
        return pd.DataFrame()

    df = df_ohlcv_context.copy()
    
    # 1. BB_Position 
    df = calculate_bb_position(df)
    
    # 2. ATR and RSI
    df = calculate_base_features(df, atr_period=ATR_PERIOD)
    
    # 3. Slope 50 
    def apply_slope(series):
        if len(series) < SLOPE_WINDOW:
            return np.nan
        x = range(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
        
    df['Slope_50'] = df['close'].rolling(window=SLOPE_WINDOW).apply(apply_slope, raw=False)
    
    # 4. Rejection Power (Calculated only on the latest candle)
    last_candle = df_ohlcv_context.iloc[-1]
    rejection_power = calculate_rejection_power(
        open_price=last_candle['open'],
        close_price=last_candle['close'],
        high_price=last_candle['high'],
        low_price=last_candle['low']
    )
    
    # Select the features from the last row (the most recent one)
    last_features = df.iloc[-1].copy() 
    
    # Overwrite Rejection Power
    last_features['Rejection_Power'] = rejection_power
    
    feature_cols = ['ATR_14', 'RSI_14', 'Slope_50', 'Rejection_Power', 'BB_Position']
    
    # Return a one-row DataFrame for prediction
    return last_features[feature_cols].to_frame().T.dropna()

# --- Telegram Alert Functions (UPDATED TIMEOUT) ---
def _send_telegram_alert_internal(message: str):
    """
    Internal function to send a message via Proxy or Direct Telegram API.
    """
    global NETWORK_TIMEOUT_SECONDS 

    if TELEGRAM_PROXY_URL_FULL:
        if not TELEGRAM_PROXY_URL_FULL.startswith('http'):
            print("❌ ERROR: TELEGRAM_PROXY_URL is set but is invalid (must start with http).")
            return False
        print(f"📡 Using Proxy URL: {TELEGRAM_PROXY_URL_FULL}")
        url = TELEGRAM_PROXY_URL_FULL
        payload = {"message": message}
            
    elif TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("🔗 Using Direct Telegram API access.")
        base_url = "https://api.telegram.org/bot" 
        url = f"{base_url}{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
    else:
        print("❌ ERROR: No Telegram credentials (BOT/CHAT ID) or Proxy URL found. Logging alert locally.")
        return False
        
    try:
        # HTTP Request using the selected URL (Proxy or Direct)
        if TELEGRAM_PROXY_URL_FULL:
            # Proxy expects JSON
            response = requests.post(url, json=payload, timeout=NETWORK_TIMEOUT_SECONDS) 
        else:
            # Telegram API expects form-data
            response = requests.post(url, data=payload, timeout=NETWORK_TIMEOUT_SECONDS) 
                
        response.raise_for_status() 
        
        print("✅ Telegram: Alert successfully sent.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ FATAL NETWORK/API ERROR: Telegram alert FAILED. (Endpoint: {'PROXY' if TELEGRAM_PROXY_URL_FULL else 'DIRECT'})")
        print(f"Failure Reason: {type(e).__name__} - {e}")
        return False

# Public wrapper for simple status/wake-up messages
def send_ping_message(message: str):
    """
    Public function to send a non-critical ping message. Used by the scheduler.
    """
    return _send_telegram_alert_internal(message)

# --- FUNCIÓN PARA EL KEEP-ALIVE DEL PROXY (CORRECCIÓN DE URL) ---
def send_proxy_keep_alive():
    """
    Sends a GET request to the Proxy Service's BASE URL 
    to prevent the Render server from sleeping (fixes the 404 error).
    """
    global NETWORK_TIMEOUT_SECONDS
    global PROXY_BASE_URL
    
    if not PROXY_BASE_URL:
        print("🟡 Keep-Alive: TELEGRAM_PROXY_URL no configurado. Saltando ping.")
        return False
        
    # The URL is now the base URL, which should respond with 200 or 400 (if it's not a GET request)
    # The important part is that the server is woken up.
    url = PROXY_BASE_URL 
    
    try:
        # The GET request now uses the unified timeout (60s)
        response = requests.get(url, timeout=NETWORK_TIMEOUT_SECONDS) 
        response.raise_for_status() 
        print(f"✅ Keep-Alive: Ping exitoso al proxy en {url}. Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Keep-Alive ERROR: Falló el ping al proxy en {url}.")
        print(f"Razón de la falla: {type(e).__name__} - {e}")
        return False

# --- Prediction and Alert Logic (Updated to use internal alert function) ---
def predict_and_calculate_metrics(df_current_features, models, symbol: str) -> dict | None:
    """
    Predicts the probability for all targets and calculates the metrics,
    returning them as a dictionary, rather than sending an alert immediately.
    
    Returns:
        dict: Containing 'metrics', 'timestamp_local', and 'alert_found'.
    """
    if df_current_features.empty:
        print(f"❌ Insufficient data or feature calculation failed for {symbol}.")
        return None
    
    # Ensure it's timezone aware (UTC), then convert to local time for display
    timestamp_utc = df_current_features.index[0].tz_localize(timezone.utc) 
    timestamp_local = timestamp_utc + VENEZUELA_TZ_OFFSET 
    
    features = df_current_features.iloc[0]
    if 'ATR_14' not in features:
        print(f"❌ ATR_14 feature missing from current features for {symbol}.")
        return None
        
    atr_value = features['ATR_14']
    rsi_value = features['RSI_14']
    
    # Heuristic for Direction based on RSI
    if rsi_value < 40:
        direction = 'LONG'
    elif rsi_value > 60:
        direction = 'SHORT'
    else:
        direction = 'NEUTRAL'

    max_advantage = -np.inf
    best_target = None
    best_prob = 0.0

    # 1. Execute Predictions and find the best Net Advantage
    for target, model in models.items():
        data_point = df_current_features.copy()
        prob = model.predict_proba(data_point)[:, 1][0]
        prob_percent = round(prob * 100, 2)
        
        target_value = int(target.split('_')[1].replace('x', ''))
        be_needed = calculate_break_even(target_value)
        advantage = round(prob_percent - be_needed, 2)
        
        if advantage > max_advantage:
            max_advantage = advantage
            best_target = target
            best_prob = prob_percent

    # Calculate R:R for the best target
    if best_target:
        target_val = int(best_target.split('_')[1].replace('x', ''))
        sl_mult = 1.0 if target_val <= 5 else 3.0
        r_r_ratio = round(target_val / sl_mult, 1)
    else:
        r_r_ratio = 0.0

    return {
        'metrics': {
            'direction': direction,
            'best_target': best_target,
            'best_prob': best_prob,
            'max_advantage': max_advantage,
            'rsi_value': rsi_value,
            'slope_50': features['Slope_50'],
            'atr_value': atr_value,
            'r_r_ratio': r_r_ratio
        },
        'timestamp_local': timestamp_local.strftime('%Y-%m-%d %H:%M'),
        'alert_found': max_advantage >= MINIMUM_ADVANTAGE_ALERT
    }

def generate_alert_message(symbol: str, metrics: dict, timestamp_local: str) -> str:
    """Helper function to format the Telegram alert message."""
    
    message = f"""
    🚨 ML SIGNAL ALERT! 🚨
    ------------------------------------------
    ⌚ Time (Local): {timestamp_local} VET
    📈 Asset: {symbol}
    🚀 Direction: {metrics['direction']}
    🎯 Best Target: {metrics['best_target']} (R:R {metrics['r_r_ratio']}:1)
    📊 P. Prediction: {metrics['best_prob']:.2f}%
    🟢 Net Advantage: +{metrics['max_advantage']:.2f}%
    ------------------------------------------
    RSI: {metrics['rsi_value']:.2f} | Slope 50: {metrics['slope_50']:.2f} | ATR 14: {metrics['atr_value']:.2f}
    """
    return message

async def run_scanner_main(source: str, symbol: str = 'ETH/USDT') -> bool | None:
    """
    Orchestrates the scheduled scanner process.
    
    Returns:
        bool: True if an alert was sent, False otherwise. None if execution failed.
    """
    print(f"--- ⏳ Initializing Scanner Run ({source}) for {symbol} ---")
    
    # 1. Load the "Brain"
    models = load_all_models()
    if not models:
        return None
        
    # 2. Fetch Data Context (API Simulation)
    df_ohlcv_context = fetch_live_data_simulation(symbol=symbol)
    if df_ohlcv_context.empty:
        return False
        
    # 3. Process the 5 Features of the last candle
    df_current_features = process_live_features(df_ohlcv_context)
    if df_current_features.empty:
        return False
        
    # 4. Predict and Calculate Metrics
    result = predict_and_calculate_metrics(df_current_features, models, symbol)
    if not result:
        return False

    # 5. Generate Alert (if threshold is met)
    if result['alert_found']:
        message = generate_alert_message(symbol, result['metrics'], result['timestamp_local'])
        print(message)
        _send_telegram_alert_internal(message)
        print("--- ✅ Alert Sent. ---")
        return True
    else:
        print(f"🟡 Prediction found for {symbol}, but Net Advantage ({result['metrics']['max_advantage']:.2f}%) is below the alert threshold.")
        
        # --- HEARTBEAT LOGIC (ONLY if no signal was sent AND in scheduled mode) ---
        if source not in ['CLI_TEST', 'REPORT_0800', 'REPORT_1200', 'REPORT_1700', 'REPORT_2100']:
            current_time_utc = datetime.now(timezone.utc)
            current_time_local = current_time_utc + VENEZUELA_TZ_OFFSET
            is_daytime_window = 8 <= current_time_local.hour < 22
                                
            if is_daytime_window:
                # This message is now only sent if no signal was found and it's daytime.
                heartbeat_message = (
                    f"🟢 **Scanner Status: OK** (Local Time)\n"
                    f"No high-probability signal (>{MINIMUM_ADVANTAGE_ALERT}%) found for {symbol} at {current_time_local.strftime('%Y-%m-%d %H:%M:%S')} VET."
                )
                _send_telegram_alert_internal(heartbeat_message)
            else:
                print(" Silent mode active: No signal found and outside 8 AM - 10 PM window.")
        # --------------------------------------------------
        
        return False

# --- NUEVA FUNCIÓN PARA EL WEBHOOK /ANALYZE ---
def run_single_symbol_prediction(symbol: str) -> dict | None:
    """
    Executes the prediction pipeline for a single symbol and returns the structured
    metrics result dictionary for the API/Webhook to format and send.
    
    Returns:
        dict | None: The result dictionary containing metrics, timestamp, and alert status.
    """
    # 1. Load the "Brain"
    models = load_all_models()
    if not models:
        return None
        
    # 2. Fetch Data Context
    df_ohlcv_context = fetch_live_data_simulation(symbol=symbol)
    if df_ohlcv_context.empty:
        return None

    # 3. Process the 5 Features
    df_current_features = process_live_features(df_ohlcv_context)
    if df_current_features.empty:
        return None
        
    # 4. Predict and Calculate Metrics (No Alert Sent Here)
    result = predict_and_calculate_metrics(df_current_features, models, symbol)
    return result

if __name__ == '__main__':
    # This block is now updated to call run_scanner_main with default symbol
    async def main():
        await run_scanner_main(source='CLI_TEST', symbol='ETH/USDT')
    asyncio.run(main())