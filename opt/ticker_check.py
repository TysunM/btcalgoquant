import requests

# --- CONFIGURATION ---
FMP_API_KEY = "M0vKOeg7eFEfvWjkwJwNxreLHRSID3um"
SEARCH_QUERY = "BTC"

def check_ticker():
    print(f"Checking FMP Stable Search for: {SEARCH_QUERY}")
    url = f"https://financialmodelingprep.com/stable/search-symbol?query={SEARCH_QUERY}&apikey={FMP_API_KEY}"
    
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            if data:
                print("\n--- FOUND TICKERS ---")
                for item in data[:5]:
                    print(f"SYMBOL: {item['symbol']} | NAME: {item['name']} | EXCH: {item['exchange']}")
                print("---------------------\n")
                print(f"Use the SYMBOL above that exactly matches your needs in your mass scraper.")
            else:
                print("No symbols found. Check your API key or query.")
        else:
            print(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Fault: {e}")

if __name__ == "__main__":
    check_ticker()
