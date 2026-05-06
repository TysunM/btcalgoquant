import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def generate_alpha_signals(timeframe='1d'):
    # Load Windows Local Environment
    load_dotenv('C:/btc_quant/.env')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_NAME = os.getenv('DB_NAME', 'btc_quant_db')

    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}')
    
    print(f"Extracting {timeframe} data from local {DB_NAME}...")
    query = "SELECT snapped_at as timestamp, price as close_price FROM btc_historical ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"Database error: {e}")
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Technical Indicators
    df['EMA_10'] = df['close_price'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close_price'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close_price'].ewm(span=200, adjust=False).mean()

    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Runaway_Alignment'] = (df['close_price'] > df['EMA_10']) & \
                              (df['EMA_10'] > df['EMA_50']) & \
                              (df['EMA_50'] > df['EMA_200']) & \
                              (df['RSI'] > 50)
    
    df.dropna(inplace=True)
    print(f"Signals generated for {len(df)} candles.")
    return df

if __name__ == "__main__":
    df = generate_alpha_signals('1d')
    if df is not None:
        print("\n--- RUNAWAY SYSTEM STATUS ---")
        print(df[['close_price', 'EMA_10', 'EMA_50', 'EMA_200', 'Runaway_Alignment']].tail())