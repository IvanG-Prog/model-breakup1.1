import pandas as pd
import numpy as np

def clean_data_complete(df, timeframe):
    """
    Apply the complete Treatment and Cleaning to an OHLCV DataFrame.
    """
    
    # 1. Type Conversion (Ensure data is numeric for calculations)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        # 'coerce' converts any non-numeric value to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 2. Handling Missing Values (A, B, and C)
    if timeframe == '1h':
        freq = 'H' 
    elif timeframe == '5m':
        freq = '5min'
    else:
        return df 

    # A. Create Perfect Index
    complete_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

    # B. Insert Empty Rows (Reindex to include all timestamps)
    df_with_gaps = df.reindex(complete_range)

    # C. Fill Values (Use the last known price)
    df_cleaned = df_with_gaps.fillna(method='ffill')

    # 3. Final Cleaning (Remove duplicates and remaining NaNs)
    # Although this was already handled, we ensure no index duplicates remain
    df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')] 

    # Remove any NaNs that may have remained at the start (if no prior price existed)
    df_cleaned = df_cleaned.dropna()

    return df_cleaned