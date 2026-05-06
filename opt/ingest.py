import ccxt
import time
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Force load the .env from the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from db_schema import MarketData

# Database Setup Fallback
DB_USER = os.getenv("DB_USER", "quant_admin")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "btc_quant_db")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Initialize Kraken
exchange = ccxt.kraken({'enableRateLimit': True})
symbol = 'BTC/USD'

def fetch_historical_data(symbol, timeframe, days_back):
    print(f"Fetching {days_back} days of {timeframe} data for {symbol}...")
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days_back)).isoformat())
    
    all_ohlcv = []
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            since = ohlcv[-1][0] + 1 
            all_ohlcv.extend(ohlcv)
            print(f"Batch received. Total {timeframe} candles: {len(all_ohlcv)}")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    records = []
    for row in all_ohlcv:
        dt = datetime.utcfromtimestamp(row[0] / 1000.0)
        existing = session.query(MarketData).filter_by(timestamp=dt, timeframe=timeframe).first()
        if not existing:
            records.append(MarketData(
                timestamp=dt, timeframe=timeframe,
                open_price=row[1], high_price=row[2],
                low_price=row[3], close_price=row[4], volume=row[5]
            ))
        
    if records:
        session.bulk_save_objects(records)
        session.commit()
        print(f"Saved {len(records)} new {timeframe} records.")
    else:
        print(f"No new {timeframe} records to save.")

if __name__ == "__main__":
    # Fetching 1000 days of Daily data for the 200 EMA baseline
    fetch_historical_data(symbol, '1d', days_back=1000)
    # Fetching 100 days of Hourly data for precision
    fetch_historical_data(symbol, '1h', days_back=100)
