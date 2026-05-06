import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Alpaca-py SDK
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from lstm_model import BTCPredictor

# Load Environment
load_dotenv('C:/btc_quant/.env')
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "dummy_key")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "dummy_secret")
SIMULATION_MODE = ALPACA_API_KEY == "dummy_key"

# Initialize Alpaca SDK Clients
data_client = CryptoHistoricalDataClient()
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def get_live_data_and_predict():
    print("Fetching live market data via Alpaca SDK...")
    
    # Fetch 250 days to accurately calculate the 200 EMA buffer
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=250)
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD"],
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt
    )
    
    btc_bars = data_client.get_crypto_bars(request_params)
    df = btc_bars.df.reset_index()
    
    if df.empty or len(df) < 200:
        print("Insufficient live data to calculate technicals. Exiting.")
        return
        
    df.rename(columns={'close': 'close_price'}, inplace=True)
    
    # Calculate live signals natively from Alpaca feed
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
    
    features = ['close_price', 'EMA_10', 'EMA_50', 'EMA_200', 'RSI', 'Runaway_Alignment']
    data = df[features].copy()
    data['Runaway_Alignment'] = data['Runaway_Alignment'].astype(int)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(data[features])
    scaler_y.fit(data[['close_price']])
    
    recent_14_days = data[features].tail(14)
    scaled_recent = scaler_X.transform(recent_14_days)
    
    tensor_input = torch.tensor(np.array([scaled_recent]), dtype=torch.float32)
    
    print("Loading LSTM neural network weights...")
    model = BTCPredictor(input_size=6)
    model_path = os.path.join(os.path.dirname(__file__), "lstm_btc_weights.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        scaled_prediction = model(tensor_input)
        
    predicted_price = scaler_y.inverse_transform(scaled_prediction.numpy())[0][0]
    current_price = recent_14_days['close_price'].iloc[-1]
    
    print("\n" + "="*40)
    print("       QUANT PREDICTION ENGINE       ")
    print("="*40)
    print(f"Current BTC Price:   ${current_price:,.2f}")
    print(f"Predicted Price:     ${predicted_price:,.2f}")
    
    price_diff = predicted_price - current_price
    threshold = current_price * 0.005
    
    if price_diff > threshold:
        signal = "BUY"
    elif price_diff < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    print(f"System Action:       {signal}")
    print("="*40)
    
    if not SIMULATION_MODE:
        execute_trade(signal)
    else:
        print("\n[SIMULATION MODE] - Live Alpaca keys not found in .env.")

def execute_trade(signal):
    print(f"\nRouting {signal} order via Alpaca SDK...")
    try:
        positions = trading_client.get_all_positions()
        btc_position = next((p for p in positions if p.symbol == 'BTCUSD' or p.symbol == 'BTC/USD'), None)
        
        if signal == "BUY" and not btc_position:
            order_data = MarketOrderRequest(
                symbol="BTC/USD",
                qty=0.1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            print("SUCCESS: Buy order executed.")
        elif signal == "SELL" and btc_position:
            trading_client.close_position('BTC/USD')
            print("SUCCESS: Position closed.")
        else:
            print("Trade ignored (Already in position or no position to sell).")
    except Exception as e:
        print(f"Order failed: {e}")

if __name__ == "__main__":
    get_live_data_and_predict()