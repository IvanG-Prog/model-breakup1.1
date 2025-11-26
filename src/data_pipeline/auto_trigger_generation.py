import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_data_from_csv 
from utils.feature_calculator import calculate_all_indicators, calculate_divergence 


# --- Global Configuration ---
OHLCV_PATH = '../data/raw/eth_usdt_1h_raw.csv' 
OUTPUT_EVENTS_PATH = '../data/processed/auto_generated_events.csv'
SLOPE_WINDOW = 50 
ATR_PERIOD = 14
ATR_WINDOW_AVG = 50 
RSI_PERIOD = 14


def generate_triggers(df_indicators):
    """
    Applies strict rule-based filters (Slope, RSI, Divergence, Volatility) 
    to identify high-quality "exhaustion" entry points (triggers).
    """
    
    # --- Thresholds (Defines "Extreme") ---
    SLOPE_THRESHOLD = 3.5
    RSI_OVERBOUGHT = 65  
    RSI_OVERSOLD = 35    
    ATR_FACTOR = 1.05 

    
    # --- Long Condition: Bullish Exhaustion (Selling failed) ---
    long_conditions = (df_indicators[f'Slope_{SLOPE_WINDOW}'] < -SLOPE_THRESHOLD) & \
                      (df_indicators[f'RSI_{RSI_PERIOD}'] < RSI_OVERSOLD) & \
                      (df_indicators['RSI_Divergence'] == 1) & \
                      (df_indicators[f'ATR_{ATR_PERIOD}'] > df_indicators[f'ATR_Avg_{ATR_WINDOW_AVG}'] * ATR_FACTOR)

    # --- Short Condition: Bearish Exhaustion (Buying failed) ---
    short_conditions = (df_indicators[f'Slope_{SLOPE_WINDOW}'] > SLOPE_THRESHOLD) & \
                       (df_indicators[f'RSI_{RSI_PERIOD}'] > RSI_OVERBOUGHT) & \
                       (df_indicators['RSI_Divergence'] == -1) & \
                       (df_indicators[f'ATR_{ATR_PERIOD}'] > df_indicators[f'ATR_Avg_{ATR_WINDOW_AVG}'] * ATR_FACTOR)
                       
    
    # --- Consolidate and Output ---
    long_events = df_indicators[long_conditions].copy()
    long_events['event_type'] = 'long'
    
    short_events = df_indicators[short_conditions].copy()
    short_events['event_type'] = 'short'
    
    all_events = pd.concat([long_events, short_events])
    all_events.reset_index(inplace=True)
    all_events.rename(columns={'timestamp': 'timestamp_gatillo'}, inplace=True)
    
    final_events = all_events[['timestamp_gatillo', 'event_type']]

    return final_events

if __name__ == '__main__':
    """
    Main execution block: Loads data, calculates technical indicators, 
    applies strict rules to find high-quality triggers, and saves the result.
    """
    print("--- Starting Automated Trigger Generation ---")
    
    # Load data using the utility function
    df_ohlcv = load_data_from_csv(OHLCV_PATH) 
    
    if df_ohlcv.empty:
        print("ERROR: Failed to load OHLCV data. Terminating.")
    else:
        # Calculate all necessary indicators using the utility function
        df_full_indicators = calculate_all_indicators(df_ohlcv)
        
        # Apply the rule-based filter
        df_new_events = generate_triggers(df_full_indicators)
        
        # Save the triggers
        df_new_events.to_csv(OUTPUT_EVENTS_PATH, index=False)
        
        print("\n--- Automated Trigger Generation Completed ---")
        print(f"✅ were generated {len(df_new_events)} new potential events.")
        print(f"Output saved to {OUTPUT_EVENTS_PATH}")