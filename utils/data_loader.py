import ccxt
import pandas as pd
import os
import time
import numpy as np


# ==== Parameters ====
EXCHANGE_NAME = 'binance'
SYMBOL = 'ETH/USDT'
LIMIT_1H = 5000  
LIMIT_5M = 100000
DATA_PATH = '/home/ivang/Ivan/git/model-breakup1.1/data/raw'

def fetch_ohlcv_data(exchange_name, symbol, timeframe, since=None, limit=1000):
    """
    Fetches OHLCV data from a specified cryptocurrency exchange (e.g., Binance) 
    using the ccxt library. It handles pagination to retrieve data beyond the 
    single request limit.
    
    Args:
        exchange_name (str): Name of the exchange (e.g., 'binance').
        symbol (str): Trading pair (e.g., 'ETH/USDT').
        timeframe (str): Time interval (e.g., '1h', '5m').
        since (int, optional): Timestamp in milliseconds to start fetching from.
        limit (int): Total number of candles to retrieve.
        
    Returns:
        pd.DataFrame: DataFrame indexed by Timestamp, containing OHLCV and volume.
    """
    try:
        # Dynamically get the exchange class from ccxt
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True}) 
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' not supported by ccxt.")
        return None

    all_ohlcv = []
    fetch_count = 0
    
    # Loop to fetch data until the desired limit is reached
    while fetch_count < limit:
        # Fetches up to 1000 candles per API call
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        # Update 'since' timestamp to the next candle's start time (+1ms)
        since = ohlcv[-1][0] + 1 
        fetch_count += len(ohlcv)

        # Respect the exchange's rate limit to avoid being banned
        time.sleep(exchange.rateLimit / 1000)

    # Convert the list of OHLCV data into a clean DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')] 
    return df.tail(limit)


def load_data_from_csv(path):
    """
    Loads OHLCV data from a local CSV file, cleans the format (especially 
    for columns and index conversion), and ensures proper sorting. 
    This is typically used for loading training data.
    
    Args:
        path (str): The file path to the CSV data.
        
    Returns:
        pd.DataFrame: Cleaned and sorted DataFrame.
    """
    try:
        # Corrección de Emergencia: Leer sin encabezado forzando 5 columnas
        # El archivo tiene 5 columnas (time;open;high;low;close)
        df = pd.read_csv(path, 
                         sep=';', 
                         skipinitialspace=True, 
                         header=None, # Leer sin encabezado
                         names=['timestamp_raw', 'open', 'high', 'low', 'close'], # Forzar 5 nombres
                         skiprows=1, # Saltar la fila de encabezado que ya no usaremos
                         quotechar='"') 
        
        # 1. Limpieza y Establecimiento del Índice
        df['timestamp'] = pd.to_datetime(df['timestamp_raw'].astype(str).str.strip())
        df = df.set_index('timestamp')
        df.index.name = 'timestamp'
        df = df.drop(columns=['timestamp_raw'])

        # 2. Conversión de Tipos (Solo 4 columnas de datos restantes)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        return df.sort_index()
    except Exception as e:
        print(f"Error loading OHLCV from CSV: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    """
    Main execution block: Downloads 1H and 5M data for ETH/USDT and saves them 
    as raw CSV files.
    """
    
    if not os.path.exists(DATA_PATH):   
        os.makedirs(DATA_PATH)
        print(f"Directory created: {DATA_PATH}")

    print(f"\n--- Downloading data for {SYMBOL} ---")

    # Download and save 1H data
    df_1h = fetch_ohlcv_data(EXCHANGE_NAME, SYMBOL, '1h', limit=LIMIT_1H)
    if df_1h is not None and not df_1h.empty:
        file_path_1h = os.path.join(DATA_PATH, 'eth_usdt_1h_raw.csv')
        # We ensure the index (timestamp) is included in the CSV
        df_1h.to_csv(file_path_1h)
        print(f"✅ 1H: {len(df_1h)} candles downloaded and saved to {file_path_1h}")

    # Download and save 5M data
    df_5m = fetch_ohlcv_data(EXCHANGE_NAME, SYMBOL, '5m', limit=LIMIT_5M)
    if df_5m is not None and not df_5m.empty:
        file_path_5m = os.path.join(DATA_PATH, 'eth_usdt_5m_raw.csv')
        df_5m.to_csv(file_path_5m)
        print(f"✅ 5M: {len(df_5m)} candles downloaded and saved to {file_path_5m}")