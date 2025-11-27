"""
This script coordinates the entire Feature Engineering process. It is responsible 
for loading raw price data and manual triggers, calculating all necessary features (X) 
using utility functions, assigning multi-target labels (Y), and saving the final 
ready-to-train dataset.
"""
import pandas as pd
import numpy as np 
import os 
import sys

# Append the parent directory to the path for utility imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- UTILITIES IMPORT ---
# Import loading and feature calculation functions
from utils.data_loader import load_data_from_csv
from utils.feature_calculator import (
    calculate_base_features, 
    calculate_bb_position, 
    calculate_relative_volume, 
    calculate_slope, 
    calculate_rejection_power
)


# --- Global Configuration ---
ATR_PERIOD = 14
OHLCV_PATH = '/home/ivang/Ivan/git/model-breakup1.1/data/raw/eth_usdt_1h_raw.csv' 
EVENTS_PATH = '/home/ivang/Ivan/git/model-breakup1.1/data/processed/all_468_raw_events.csv' 
SLOPE_WINDOW = 50 


def label_trade_outcome_multiple(df_ohlcv, timestamp_trigger, atr_value, event_type):
    """
    Assigns binary labels (1=Hit, 0=Miss) for 7 Take Profit (TP) targets 
    (1x, 2x, 3x, 5x, 10x, 15x, and 20x ATR) using a dynamic Stop Loss (SL).
    
    SL is set to 1x ATR for small targets (1x-5x) and 3x ATR for large targets (10x-20x).

    Args:
        df_ohlcv (pd.DataFrame): Price data (OHLCV).
        timestamp_trigger (datetime): Time of the entry signal.
        atr_value (float): ATR value at the time of entry, used for sizing TP/SL.
        event_type (str): 'long' or 'short'.
        
    Returns:
        dict: Dictionary with target names (e.g., 'Target_1x') and binary outcomes (0 or 1).
    """
    
    SL_MULTIPLIERS = {
        1: 1.0, 2: 1.0, 3: 1.0, 5: 1.0, 
        10: 3.0, 15: 3.0, 20: 3.0 
    }
    
    df_future = df_ohlcv.loc[df_ohlcv.index > timestamp_trigger]
    if df_future.empty:
        return {f'Target_{m}x': 0 for m in SL_MULTIPLIERS.keys()}
        
    price_entry = df_ohlcv.loc[df_ohlcv.index <= timestamp_trigger]['close'].iloc[-1]
    
    TP_LEVELS = {f'Target_{m}x': atr_value * m for m in SL_MULTIPLIERS.keys()}
    
    outcomes = {k: 0 for k in TP_LEVELS.keys()}
    
    for index, candle in df_future.iterrows():
        
        for target_key, tp_value in TP_LEVELS.items():
            if outcomes[target_key] == 1:
                continue
                
            tp_multiplier = int(target_key.split('_')[1].replace('x', ''))
            
            sl_multiplier = SL_MULTIPLIERS[tp_multiplier]
            STOP_LOSS = atr_value * sl_multiplier 
            
            sl_level = price_entry - STOP_LOSS if event_type == 'long' else price_entry + STOP_LOSS
            tp_level = price_entry + tp_value if event_type == 'long' else price_entry - tp_value
            
            tp_hit = (event_type == 'long' and candle['high'] >= tp_level) or \
                     (event_type == 'short' and candle['low'] <= tp_level)
            
            sl_hit = (event_type == 'long' and candle['low'] <= sl_level) or \
                     (event_type == 'short' and candle['high'] >= sl_level)
            
            if tp_hit:
                outcomes[target_key] = 1
            elif sl_hit:
                pass 
                
    return outcomes


def extract_features_for_events(df_ohlcv_indicators, df_events, df_ohlcv_raw):
    """
    Extracts all calculated features (X) and assigns the multiple target labels (Y) 
    for each event timestamp, creating the final training dataset.
    """
    
    if 'timestamp_gatillo' not in df_events.columns:
        raise KeyError("The column 'timestamp_gatillo' was not found in the event file.")

    df_events['timestamp_gatillo'] = pd.to_datetime(df_events['timestamp_gatillo'])
    
    event_features = []
    
    atr_col = f'ATR_{ATR_PERIOD}'
    rsi_col = f'RSI_{ATR_PERIOD}'
    
    print(f"Processing {len(df_events)} events...")

    for index, row in df_events.iterrows():
        trigger_time = row['timestamp_gatillo']
        
        try:
            context_candle_indicators = df_ohlcv_indicators.loc[df_ohlcv_indicators.index <= trigger_time].iloc[-1]
            context_candle_raw = df_ohlcv_raw.loc[df_ohlcv_raw.index <= trigger_time].iloc[-1]
            
            atr_value = context_candle_indicators[atr_col]
            event_type = row['event_type'].lower() 
            
            # 1. Labeling the outcome (Y)
            outcome_labels = label_trade_outcome_multiple(
                df_ohlcv=df_ohlcv_raw, 
                timestamp_trigger=trigger_time, 
                atr_value=atr_value, 
                event_type=event_type
            )
            
            # 2. Calculate Contextual Features (X)
            
            # Slope
            slope_50 = calculate_slope(df_ohlcv=df_ohlcv_raw, 
                                       timestamp_trigger=trigger_time, 
                                       window=SLOPE_WINDOW)
            
            # Rejection Power
            rejection_power = calculate_rejection_power(
                open_price=context_candle_raw['open'],
                close_price=context_candle_raw['close'],
                high_price=context_candle_raw['high'],
                low_price=context_candle_raw['low']
            )

            # --- FEATURE EXTRACTION ---
            bb_position = context_candle_indicators['BB_Position']
            
            features = {
                'timestamp_gatillo': trigger_time,
                'event_type': event_type,
                'ATR_14': atr_value, 
                'RSI_14': context_candle_indicators[rsi_col], 
                'Slope_50': slope_50, 
                'Rejection_Power': rejection_power, 
                'BB_Position': bb_position, 
                # --- The 7 Targets (Y) ---
                **outcome_labels
            }
            event_features.append(features)
        
        except IndexError:
            print(f"Warning: Insufficient historical data (start of history) for timestamp {trigger_time}. Skipping.")
            pass

    return pd.DataFrame(event_features)


if __name__ == '__main__':
    """
    Orchestrates the feature engineering process: loads data, calculates all 
    features and indicators, assigns multi-target labels, and saves the 
    final training dataset.
    """
    print("--- Starting Feature Engineering and Labeling Process ---")
    
    # 1. Load Data
    df_ohlcv_raw = load_data_from_csv(OHLCV_PATH)
    df_events = load_data_from_csv(EVENTS_PATH)
    
    if df_ohlcv_raw.empty or df_events.empty:
        print("ERROR: Failed to load OHLCV or event data. Terminating.")
    else:
        # 2. Calculate and combine all features
        df_ohlcv_raw = calculate_bb_position(df_ohlcv_raw)
        df_ohlcv_raw = calculate_relative_volume(df_ohlcv_raw) 
        
        # Calculate base indicators (RSI, ATR)
        df_indicators = calculate_base_features(df_ohlcv_raw.copy(), ATR_PERIOD)
        
        # 3. Extract Features (X) and Assign Labels (Y)
        df_final_dataset = extract_features_for_events(df_indicators, df_events, df_ohlcv_raw)
        
        dataset_original_len = len(df_final_dataset)
        df_final_dataset.dropna(inplace=True) 
        
        # 4. Save the Final Dataset
        OUTPUT_FILE = '../data/processed/training_data.csv'
        df_final_dataset.to_csv(OUTPUT_FILE, index=False)
        
        print("\n--- Labeling and Feature Extraction Completed ---")
        print(f"Total manual triggers processed: {dataset_original_len}")
        print(f"Total events ready for ML (after NaN cleanup): {len(df_final_dataset)}")
        print(f"Final training dataset saved to {OUTPUT_FILE}")