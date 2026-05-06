import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load Windows Local Environment
load_dotenv('C:/Users/Tysun/btc_quant/opt/.env')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_NAME = os.getenv('DB_NAME', 'btc_quant_db')

engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}')

def ingest_data():
    # Updated Windows data path
    path = r'C:\Users\Tysun\btc_quant\data'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not all_files:
        print(f"No CSV files found in {path}")
        return

    for file in all_files:
        print(f"Processing {os.path.basename(file)}...")
        df = pd.read_csv(file)
        df.to_sql('btc_historical', engine, if_exists='append', index=False)
        print(f"Successfully inserted {len(df)} rows into database.")

if __name__ == "__main__":
    ingest_data()