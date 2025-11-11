import ccxt
import pandas as pd
import os
import time


# ==== paramethers ====
EXCHANGE_NAME = 'binance'
SYMBOL = 'ETH/USDT'
LIMIT_1H = 5000  
LIMIT_5M = 100000
DATA_PATH = '/home/ivang/Ivan/git/model-breakup1.1/data/raw'

def fetch_ohlcv_data(exchange_name, symbol, timeframe, since=None, limit=1000):
    """
    Fetch OHLCV data from a specified exchange using ccxt.
    Returns a pandas DataFrame indexed by Timestamp.
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({'enableRateLimit': True}) 
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' not supported by ccxt.")
        return None

    all_ohlcv = []
    
    fetch_count = 0
    
    while fetch_count < limit:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        
        since = ohlcv[-1][0] + 1 
        fetch_count += len(ohlcv)

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')] 
    return df.tail(limit) 

if __name__ == "__main__":
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory created: {DATA_PATH}")

    print(f"\n--- Downloading data for {SYMBOL} ---")

    # Download and save 1H data
    df_1h = fetch_ohlcv_data(EXCHANGE_NAME, SYMBOL, '1h', limit=LIMIT_1H)
    if df_1h is not None:
        file_path_1h = os.path.join(DATA_PATH, 'eth_usdt_1h_raw.csv')
        df_1h.to_csv(file_path_1h)
        print(f"✅ 1H: {len(df_1h)} candles downloaded and saved to {file_path_1h}")

    # Download and save 5M data
    df_5m = fetch_ohlcv_data(EXCHANGE_NAME, SYMBOL, '5m', limit=LIMIT_5M)
    if df_5m is not None:
        file_path_5m = os.path.join(DATA_PATH, 'eth_usdt_5m_raw.csv')
        df_5m.to_csv(file_path_5m)
        print(f"✅ 5M: {len(df_5m)} candles downloaded and saved to {file_path_5m}")

