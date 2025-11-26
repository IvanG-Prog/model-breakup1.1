"""
This script functions as the live market scanner and inference engine. It simulates 
fetching real-time OHLCV data, processes the features for the current candle, 
loads the trained multi-target models, and predicts the statistical advantage. 
It generates a clear alert if the net advantage exceeds a predefined minimum threshold.
"""
import ccxt
import pandas as pd
import joblib
import os
import sys
from scipy.stats import linregress 
from datetime import datetime
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

# --- Global Configuration ---
# Adjusted path for src/model_pipeline/
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_PATH, 'models') 

TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']
MINIMUM_ADVANTAGE_ALERT = 70.0 # Minimum Net Advantage (%) to trigger an alert.

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


# --- Live Data and Feature Processing Functions ---

def fetch_live_data_simulation():
    """
    Fetches the last 100 OHLCV candles from the Binance exchange using ccxt.
    This replaces the data simulation with a real-time API call for the context 
    needed to calculate features (e.g., Slope 50).
    
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    exchange_name = 'binance'
    symbol = 'ETH/USDT'
    timeframe = '1h'
    limit = 100

    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True})
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            print("❌ Error: ccxt no devolvió datos OHLCV.")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        df = df[~df.index.duplicated(keep='first')] 
        
        return df.tail(limit)

    except Exception as e:
        print(f"❌ Error connecting or obtaining data from  {exchange_name}: {e}")
        return pd.DataFrame()


def process_live_features(df_ohlcv_context):
    """
    Calculates the 5 required features for the last candle in the OHLCV context.
    
    Returns:
        pd.DataFrame: A one-row DataFrame containing the 5 features.
    """
    
    # Ensure the context has enough data
    if len(df_ohlcv_context) < SLOPE_WINDOW:
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


def predict_and_alert(df_current_features, models):
    """
    Predicts the probability for all targets and generates a clear alert 
    message if the maximum Net Advantage exceeds the predefined minimum.
    """
    if df_current_features.empty:
        print("❌ Insufficient data or feature calculation failed for the current candle.")
        return

    timestamp = df_current_features.index[0]
    features = df_current_features.iloc[0]
    
    print(f"\n--- 🧠 LIVE PREDICTION DIAGNOSTICS: {timestamp} ---")
    
    rsi_value = features['RSI_14']
    
    # Heuristic for Direction based on RSI
    if rsi_value < 40:
        direction = 'LONG'
    elif rsi_value > 60:
        direction = 'SHORT'
    else:
        print(f"🟡 Neutral State: RSI ({rsi_value:.2f}). No strong entry predicted.")
        return
    
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

    # 2. Generate Alert (if threshold is met)
    
    if max_advantage >= MINIMUM_ADVANTAGE_ALERT:
        
        # Calculate R:R for the best target for the alert message
        target_val = int(best_target.split('_')[1].replace('x', ''))
        sl_mult = 1.0 if target_val <= 5 else 3.0
        r_r_ratio = target_val / sl_mult
        
        message = f"""
        🚨 ML SIGNAL ALERT! 🚨
        ------------------------------------------
        ⌚ Time: {timestamp.strftime('%Y-%m-%d %H:%M')}
        🚀 Direction: {direction}
        🎯 Best Target: {best_target} (R:R {r_r_ratio:.1f}:1)
        📊 P. Prediction: {best_prob:.2f}%
        🟢 Net Advantage: +{max_advantage:.2f}%
        ------------------------------------------
        RSI: {rsi_value:.2f} | Slope 50: {features['Slope_50']:.2f}
        """
        print(message)
    else:
        print(f"🟡 Prediction found ({best_target}), but Net Advantage ({max_advantage:.2f}%) is below the alert threshold ({MINIMUM_ADVANTAGE_ALERT:.2f}%).")


def main():
    """
    Orchestrates the live scanner process: loads models, fetches data, 
    processes features, and triggers predictions/alerts.
    """
    print("--- ⏳ Initializing Live Market Scanner ---")
    
    # 1. Load the "Brain"
    models = load_all_models()
    if not models:
        return
        
    # 2. Fetch Data Context (API Simulation)
    df_ohlcv_context = fetch_live_data_simulation()
    if df_ohlcv_context.empty:
        return

    # 3. Process the 5 Features of the last candle
    df_current_features = process_live_features(df_ohlcv_context)
    
    # 4. Predict and Alert
    predict_and_alert(df_current_features, models)
    
    print("--- ✅ Scan Complete. ---")


if __name__ == '__main__':
    main()