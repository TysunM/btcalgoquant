import pandas as pd
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = r"C:\btc_quant\data"
OUTPUT_FILE = r"C:\btc_quant\opt\BTC_MASTER_TRAINING_SET.csv"

# Columns based on the standard Binance/crypto minute data format you uploaded
COLUMN_NAMES = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def weld_data():
    print(f"--- ALCHEMICAL ENGINE: MASTER DATA WELD ---")
    print(f"Scanning {DATA_DIR} for yearly minute data...")
    
    # Find all the yearly files
    file_pattern = os.path.join(DATA_DIR, "BTCUSDT_*_minute.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("CRITICAL ERROR: No CSV files found. Check your data folder.")
        return

    print(f"Found {len(all_files)} files. Initiating fusion...")
    
    df_list = []
    
    for file in sorted(all_files):
        print(f"  -> Processing: {os.path.basename(file)}...")
        try:
            # Read the CSV. 
            # We assume no header based on standard raw crypto dumps, so we assign them.
            df = pd.read_csv(file, header=None, names=COLUMN_NAMES, usecols=[0, 2, 3, 4, 5, 6], on_bad_lines='skip')
            df_list.append(df)
        except Exception as e:
            # If the file actually has a header, we catch it and read it normally
            try:
                df = pd.read_csv(file, usecols=[0, 2, 3, 4, 5, 6])
                df.columns = COLUMN_NAMES
                df_list.append(df)
            except Exception as e2:
                print(f"     [!] Failed to read {file}: {e2}")

    if df_list:
        print("\nWelding dataframes together...")
        master_df = pd.concat(df_list, ignore_index=True)
        
        # Convert timestamp to standard datetime
        print("Normalizing time dimensions...")
        master_df['timestamp'] = pd.to_datetime(master_df['timestamp'], errors='coerce')
        
        # Drop any rows that failed conversion or have NaN values
        master_df.dropna(inplace=True)
        
        # Sort chronologically (Crucial for LSTM training)
        master_df.sort_values('timestamp', inplace=True)
        
        # Save the master tensor
        print(f"Saving Master Training Set to {OUTPUT_FILE}...")
        master_df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n--- FUSION COMPLETE ---")
        print(f"Total Rows: {len(master_df):,}")
        print(f"Date Range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
    else:
        print("Fusion failed. No data extracted.")

if __name__ == "__main__":
    weld_data()