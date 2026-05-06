import pandas as pd
from sqlalchemy import create_engine, text
import os

# Hardcoded database connection
DB_URL = 'postgresql://postgres:775238xpro@127.0.0.1:5432/btc_quant_db'
engine = create_engine(DB_URL)

def ingest_all():
    data_dir = r"C:\btc_quant\data"
    # Scans for all minute-level CSVs
    files = [f for f in os.listdir(data_dir) if f.endswith('_1min.csv')]
    
    if not files:
        print("No 1min CSV files found. Please ensure fetch_big_data.py ran successfully.")
        return

    for file in files:
        # Extract symbol from the filename (e.g., "BTCUSD_1min.csv" -> "BTCUSD")
        symbol = file.split('_')[0]
        print(f"Ingesting {symbol} into PostgreSQL...")
        
        # Load the CSV
        df = pd.read_csv(os.path.join(data_dir, file))
        
        # Inject the symbol identifier so the LSTM knows which row belongs to which asset
        df['symbol'] = symbol 
        
        # Append to the master minute-resolution table
        df.to_sql('market_data_min', engine, if_exists='append', index=False)
        print(f"SUCCESS: {symbol} loaded. ({len(df)} rows)")

if __name__ == "__main__":
    print("Initializing Multivariate Schema: Wiping old matrix...")
    with engine.connect() as conn:
        # Drop the table if it exists to ensure a clean slate for the new 4-asset matrix
        conn.execute(text("DROP TABLE IF EXISTS market_data_min;"))
        conn.commit()
    ingest_all()
    print("\nAll assets successfully ingested. Database is ready for the Quad-Input LSTM.")