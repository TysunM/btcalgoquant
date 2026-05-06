import requests
import pandas as pd

def download_fmp_data():
    print("Downloading pristine daily BTC data from Financial Modeling Prep (Stable Endpoint)...")
    
    api_key = "qDWmzXTltVzGrnQiNV1CsxoTESU934Gb"
    # Using the exact stable endpoint from your list
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=BTCUSD&apikey={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    # Handle FMP's varied JSON structures
    if isinstance(data, dict):
        if 'Error Message' in data:
            print("API Error:", data['Error Message'])
            return
        elif 'historical' in data:
            historical_data = data['historical']
        else:
            print("Unexpected dict format:", data)
            return
    elif isinstance(data, list):
        historical_data = data
    else:
        print("Unrecognized API response format.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Isolate and format core price action columns
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # Fallback check if the endpoint uses slightly different naming (like 'adjclose')
    if not all(col in df.columns for col in required_cols):
        print(f"API returned different columns than expected. Found: {list(df.columns)}")
        return
        
    df = df[required_cols]
    
    # FMP returns data newest-to-oldest. We need chronological order for the LSTM.
    df = df.sort_values('date', ascending=True)
    
    file_path = r"C:\btc_quant\data\BTCUSDT_daily_clean.csv"
    df.to_csv(file_path, index=False)
    print(f"SUCCESS: Saved {len(df)} days of FMP data to {file_path}")

if __name__ == "__main__":
    download_fmp_data()