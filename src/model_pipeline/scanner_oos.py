"""
This script performs Out-of-Sample (OOS) backtesting to validate the robustness 
of the multi-target models. It loads historical OHLCV data, calculates features 
for the full history, and runs predictions on the last 'N' candles that the model 
has not seen during training. It generates a report showing high-conviction 
signals (Net Advantage > Historical Precision) for visual inspection.
"""
import pandas as pd
import joblib
import os
import sys
from scipy.stats import linregress 

# Adjust path to import from utils/ and src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Import Utility Functions and Constants ---
from utils.feature_calculator import (
    calculate_bb_position,
    calculate_base_features, # Renamed for consistency
    calculate_rejection_power,
    ATR_PERIOD,
    SLOPE_WINDOW
)
from utils.data_loader import load_data # Use the clean loader from utils

# --- Global Configuration ---
OHLCV_PATH = '../../data/raw/eth_usdt_1h_raw.csv'
MODELS_DIR = '../../models/'
TARGET_COLUMNS = ['Target_1x', 'Target_2x', 'Target_3x', 'Target_5x', 'Target_10x', 'Target_15x', 'Target_20x']
LOOKBACK_CANDLES = 100 # Number of latest candles to scan for OOS validation

# Historical precision thresholds (used for decision logic)
HISTORICAL_PRECISION = {
    'Target_1x': 0.98, 'Target_2x': 0.96, 'Target_3x': 0.94, 
    'Target_5x': 0.89, 'Target_10x': 0.86, 'Target_15x': 0.84, 
    'Target_20x': 0.79
}

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
    
    be_needed = (sl_multiplier / (sl_multiplier + target_value)) * 100 
    return round(be_needed, 2)


def process_ohlcv_for_scanning(df_ohlcv):
    """
    Calculates all 5 necessary features across the entire OHLCV history.
    """
    df = df_ohlcv.copy()

    # 1. BB_Position 
    df = calculate_bb_position(df)
    
    # 2. ATR and RSI
    df = calculate_base_features(df, atr_period=ATR_PERIOD)
    
    # 3. Slope 50 
    def apply_slope(series):
        if len(series) < SLOPE_WINDOW:
            return np.nan
        # Calculates the slope of the linear regression for the last SLOPE_WINDOW candles
        x = range(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
        
    df['Slope_50'] = df['close'].rolling(window=SLOPE_WINDOW).apply(apply_slope, raw=False)


    # 4. Rejection Power
    df['Rejection_Power'] = df.apply(
        lambda row: calculate_rejection_power(
            open_price=row['open'],
            close_price=row['close'],
            high_price=row['high'],
            low_price=row['low']
        ), axis=1
    )
    
    # Select only the 5 features and drop rows with NaN values (start of history)
    feature_cols = ['ATR_14', 'RSI_14', 'Slope_50', 'Rejection_Power', 'BB_Position']
    return df[feature_cols].dropna()


def scan_and_predict(df_features, models, lookback_candles=LOOKBACK_CANDLES):
    """
    Performs prediction on the last 'lookback_candles' (OOS data) and generates a report.
    """
    
    df_scan = df_features.tail(lookback_candles)
    
    print(f"\n--- 🧠 OOS SCANNER: Diagnostics on the last {len(df_scan)} candles ---")
    
    results = []
    
    for timestamp, features in df_scan.iterrows():
        # Data point must be a 2D array
        data_point = features.to_frame().T
        
        # 1. Determine direction based on RSI heuristic
        rsi_value = features['RSI_14']
        
        # Heuristic: If RSI < 40, look for LONG; If RSI > 60, look for SHORT
        if rsi_value < 40:
            direction = 'LONG'
        elif rsi_value > 60:
            direction = 'SHORT'
        else:
            continue # Ignore neutral RSI signals (40-60)
            
        
        # 2. Predict probabilities for all targets
        probabilities = {}
        for target, model in models.items():
            prob = model.predict_proba(data_point)[:, 1][0]
            probabilities[target] = round(prob * 100, 2)
        
        
        # 3. Analysis of Profitability and Decision
        for target, prob in probabilities.items():
            target_value = int(target.split('_')[1].replace('x', ''))
            
            be_needed = calculate_break_even(target_value)
            advantage = round(prob - be_needed, 2)
            
            # Decision: Check if the prediction exceeds the model's historical threshold
            required_precision_hist = HISTORICAL_PRECISION.get(target, 0.0) * 100
            
            status = '🟢 ABRIR' if prob > required_precision_hist else '🟡 EVITAR'
            
            # Calculate R:R
            rr_value = round(target_value / (1.0 if target_value <= 5 else 3.0), 2)
            
            # Save results
            results.append({
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                'Target': target,
                'Direction': direction,
                'P_Prediction': prob,
                'R:R': rr_value,
                'BE_Minimo': be_needed,
                'Ventaja_Neta': advantage,
                'Decision': status,
                'RSI': rsi_value
            })
            
    return pd.DataFrame(results)


def main():
    """
    Orchestrates the OOS scanning process to validate model robustness.
    """
    print("--- ⏳ Initializing Out-of-Sample (OOS) Scanner ---")
    
    # 1. Load the "Brain"
    models = load_all_models()
    if not models:
        return
        
    # 2. Load the "Body" (Full Historical OHLCV Data)
    df_ohlcv = load_data(OHLCV_PATH, is_ohlcv=True)
    if df_ohlcv.empty:
        return
        
    # 3. Process Full History for Features
    df_features = process_ohlcv_for_scanning(df_ohlcv.iloc[:-5])
    
    # 4. Scan and Predict the last N signals (OOS)
    df_results = scan_and_predict(df_features, models, lookback_candles=LOOKBACK_CANDLES)
    
    if df_results.empty:    
        print(f"\n✅ The OOS Scanner did not find any strong signals (RSI outside 40-60) in the last {LOOKBACK_CANDLES} candles.")
        return

    # 5. Generate Consolidated Report
    
    df_open_trades = df_results[df_results['Decision'] == '🟢 OPEN']
    
    if df_open_trades.empty:
        print(f"\n✅ OOS Scan complete. No high-conviction trades found in the last {LOOKBACK_CANDLES} candles (P_Prediction < Historical Precision).")
        return
        
    # Consolidate the best target (highest Net Advantage) for each unique timestamp
    df_best_trades = df_open_trades.loc[df_open_trades.groupby('Timestamp')['Ventaja_Neta'].idxmax()]
    
    print(f"\n--- ✅ HIGH CONVICTION SIGNALS (Net Advantage > Historical Precision) ---")
    
    # Final table format
    report_cols = ['Timestamp', 'Direction', 'Target', 'R:R', 'P_Prediction', 'Ventaja_Neta', 'RSI']
    
    print(df_best_trades[report_cols].to_markdown(index=False, floatfmt=".2f"))
    
    print("\n--- OOS VALIDATION COMPLETE ---")
    print("The signals above represent the highest advantage trades from the OOS period.")
    print("Visually verify these timestamps on your chart to confirm model robustness (non-overfitting).")


if __name__ == '__main__':
    main()