"""
This utility script centralizes all technical analysis calculations and feature 
engineering logic used across the project (for both model training and live triggers). 
It contains specialized functions for indicators (ATR, RSI, BB Position), 
trend metrics (Slope), candle pattern detection (Rejection Power), 
and momentum signals (Divergence).
"""
import pandas as pd
import pandas_ta as ta 
import talib as ta_lib
from scipy.stats import linregress 
import numpy as np 

# --- Global Constants ---
ATR_PERIOD = 14
SLOPE_WINDOW = 50 
ATR_WINDOW_AVG = 50 
RSI_PERIOD = 14     


def calculate_bb_position(df_ohlcv):
    """
    Calculates the Bollinger Bands Position (BBP), which normalizes the 
    closing price position within the Bollinger Bands (0=Lower Band, 1=Upper Band).
    """
    df = df_ohlcv.copy()
    
    upper, middle, lower = ta_lib.BBANDS(
        df['close'].values, 
        timeperiod=20, 
        nbdevup=2, 
        nbdevdn=2, 
        matype=ta_lib.MA_Type.SMA
    )

    df['BB_Upper'] = upper
    df['BB_Lower'] = lower

    range_bb = df['BB_Upper'] - df['BB_Lower']
    
    # Formula BBP: (Close Price - Lower Band) / (Upper Band - Lower Band)
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / range_bb
    
    df['BB_Position'] = df['BB_Position'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    df.drop(columns=['BB_Upper', 'BB_Lower'], inplace=True)
    return df


def calculate_relative_volume(df_ohlcv):
    """
    Calculates Relative Volume (Current Volume / Average Volume (20 periods)).
    Note: Results will be limited if the source data lacks accurate volume.
    """
    df = df_ohlcv.copy()
    
    df['Avg_Volume_20'] = df['volume'].rolling(window=20).mean()
    
    # Calculate Relative Volume, adding a small epsilon to avoid division by zero
    df['Relative_Volume'] = df['volume'] / (df['Avg_Volume_20'] + 1e-9) 
    
    df.drop(columns=['Avg_Volume_20'], inplace=True)
    
    return df


def calculate_base_features(df_ohlcv, atr_period=ATR_PERIOD):
    """
    Calculates the Average True Range (ATR) and Relative Strength Index (RSI).
    This function is used in feature_engineering.py.
    """
    df = df_ohlcv.copy() 
    
    atr_col_name = f'ATR_{atr_period}'
    rsi_col_name = f'RSI_{atr_period}'

    try:
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        atr_series.name = atr_col_name 
        df = pd.concat([df, atr_series], axis=1)

        rsi_series = ta.rsi(df['close'], length=atr_period)
        rsi_series.name = rsi_col_name 
        df = pd.concat([df, rsi_series], axis=1)
    
    except Exception as e:
        print(f"ERROR: Failed to calculate ATR/RSI. Error: {e}")
        return df

    return df


def calculate_slope(df_ohlcv, timestamp_trigger, window=SLOPE_WINDOW):
    """
    Calculates the linear regression slope over a historical window, typically 
    used for a specific time point (timestamp_trigger) in feature extraction.
    
    Note: This is an adaptation of the rolling slope calculation for specific time points.
    """
    # Selects the last 'window' candles up to or at the trigger time
    df_window = df_ohlcv.loc[df_ohlcv.index <= timestamp_trigger].iloc[-window:]
    
    if len(df_window) < window:
        return np.nan 
        
    y = df_window['close'].values
    x = range(len(y))
    
    slope, _, _, _, _ = linregress(x, y)
    
    return slope


def calculate_rejection_power(open_price, close_price, high_price, low_price):
    """
    Calculates a score for candle rejection (Pinbar/Hammer), measuring the 
    relative size of the wick against the total range and body size.
    """
    body_size = abs(close_price - open_price)
    full_range = high_price - low_price
    
    if full_range == 0:
        return 0.0
        
    body_ratio = body_size / full_range 
    
    if close_price >= open_price: # Bullish candle (rejection at the bottom)
        lower_wick = open_price - low_price
        rejection_power = lower_wick / full_range 
    else: # Bearish candle (rejection at the top)
        upper_wick = high_price - close_price
        rejection_power = upper_wick / full_range

    score = rejection_power * (1 - body_ratio)
    return score


def calculate_divergence(series_a, series_b, window=5):
    """
    Calculates RSI divergence by comparing price peaks/troughs (series_a) 
    against indicator peaks/troughs (series_b) over a defined window.
    Returns 1 for bullish divergence and -1 for bearish divergence.
    """
    # Simple detection of peaks/troughs (extrema)
    price_peaks = series_a.rolling(window=window).max()
    price_troughs = series_a.rolling(window=window).min()
    rsi_peaks = series_b.rolling(window=window).max()
    rsi_troughs = series_b.rolling(window=window).min()

    divergence = pd.Series(0, index=series_a.index)

    # Bullish: Price dropped, but momentum (RSI) went up.
    bullish_condition = (series_a.shift(window) > series_a) & \
                        (series_b.shift(window) < series_b)

    # Bearish: Price rose, but momentum (RSI) went down.
    bearish_condition = (series_a.shift(window) < series_a) & \
                        (series_b.shift(window) > series_b)

    divergence[bullish_condition.fillna(False)] = 1
    divergence[bearish_condition.fillna(False)] = -1
    
    return divergence


def calculate_all_indicators(df_ohlcv):
    """
    Calculates all necessary indicators (Slope, RSI, ATR, Divergence) 
    for the rule-based trigger system and feature engineering.
    """
    df = df_ohlcv.copy() 
    
    # 1. Slope (Trend Strength/Direction)
    df[f'Slope_{SLOPE_WINDOW}'] = df['close'].rolling(window=SLOPE_WINDOW).apply(
        lambda x: linregress(range(SLOPE_WINDOW), x)[0], raw=True
    )

    # 2. RSI (Momentum)
    df[f'RSI_{RSI_PERIOD}'] = ta.rsi(df['close'], length=RSI_PERIOD)
    
    # 3. ATR and ATR Average (Volatility Context)
    df[f'ATR_{ATR_PERIOD}'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    df[f'ATR_Avg_{ATR_WINDOW_AVG}'] = df[f'ATR_{ATR_PERIOD}'].rolling(window=ATR_WINDOW_AVG).mean()
    
    # 4. Divergence (Signal of Reversal/Exhaustion)
    df['RSI_Divergence'] = calculate_divergence(df['close'], df[f'RSI_{RSI_PERIOD}'], window=5)

    df.dropna(inplace=True)
    return df