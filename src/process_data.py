import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from clean_data import clean_data_complete 

DATA_PATH_RAW = '/home/ivang/Ivan/git/model-breakup1.1/data/raw'
DATA_PATH_PROCESSED = '/home/ivang/Ivan/git/model-breakup1.1/data/processed'

def process_existing_data():
    
    if not os.path.exists(DATA_PATH_PROCESSED):
        os.makedirs(DATA_PATH_PROCESSED)
        print(f"Directory created: {DATA_PATH_PROCESSED}")

    print(f"\n--- Starting Data Processing and Cleaning ---")

    # --- LIST OF FILES TO PROCESS ---
    files_to_process = [
        {'name': 'eth_usdt_1h_raw.csv', 'timeframe': '1h'},
        {'name': 'eth_usdt_5m_raw.csv', 'timeframe': '5m'},
    ]
    
    for item in files_to_process:
        timeframe = item['timeframe']
        file_name_raw = item['name']
        
        file_path_raw = os.path.join(DATA_PATH_RAW, file_name_raw)
        
        try:
            # Read the RAW data file
            df_raw = pd.read_csv(file_path_raw, index_col='timestamp', parse_dates=True)
            print(f"Reading RAW data for {timeframe.upper()} ({len(df_raw)} candles)...")

            # Apply the cleaning function (your utils tool)
            df_clean = clean_data_complete(df_raw, timeframe)
            
            # Save the cleaned DataFrame
            file_name_clean = f'eth_usdt_{timeframe}_clean.csv'
            file_path_clean = os.path.join(DATA_PATH_PROCESSED, file_name_clean)
            df_clean.to_csv(file_path_clean)

            print(f"✅ {timeframe.upper()}: Cleaning completed. Saved to {DATA_PATH_PROCESSED}. Final candles: {len(df_clean)}")
            
        except FileNotFoundError:
            print(f"⚠️ {timeframe.upper()}: RAW file not found at {file_path_raw}. Skipping processing.")
        except Exception as e:
            print(f"❌ Error processing {timeframe.upper()}: {e}")

    print("\n--- Phase 1 (Data Preparation) SUCCESSFULLY COMPLETED. ---")

if __name__ == "__main__":
    process_existing_data()