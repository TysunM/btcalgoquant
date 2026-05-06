import pandas as pd
import numpy as np
import torch
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from quad_lstm import Quad_LSTM, DB_URL, SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS

def predict_next_candle():
    print("Initializing Inference Engine...")
    
    # HARDWARE LOCK: Forcing CPU. Pip wheels lack sm_120 support. 
    # Must use CPU until a successful CUDA 13.2 source compile is completed.
    device = torch.device("cpu")
    print(f"Executing on: {device}")

    # 1. Pull the latest data to recreate the environment and scaler
    engine = create_engine(DB_URL)
    query = "SELECT date, symbol, open, high, low, close, volume FROM market_data_min ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    
    pivot_df = df.pivot(index='date', columns='symbol')
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df.ffill(inplace=True)
    pivot_df.dropna(inplace=True)

    features = pivot_df.columns.tolist()
    target_idx = features.index('BTCUSD_close')
    input_size = len(features)

    # Re-fit scaler 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(pivot_df.values)

    # 2. Isolate the most recent 60 minutes (The Current State)
    recent_60 = scaled_data[-SEQ_LENGTH:]
    tensor_input = torch.tensor(recent_60, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Load the Trained Neural Network
    model = Quad_LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(r"C:\btc_quant\opt\quad_lstm_model.pth", map_location=device, weights_only=True))
    model.eval()

    # 4. Execute Prediction
    with torch.no_grad():
        raw_prediction = model(tensor_input).numpy()

    # 5. Inverse transform the prediction back to actual dollar value
    dummy_array = np.zeros((1, input_size))
    dummy_array[0, target_idx] = raw_prediction[0][0]
    predicted_price = scaler.inverse_transform(dummy_array)[0, target_idx]

    current_price = pivot_df.iloc[-1]['BTCUSD_close']
    delta = predicted_price - current_price

    print("\n" + "="*40)
    print(f"CURRENT BTC PRICE:   ${current_price:,.2f}")
    print(f"PREDICTED NEXT MIN:  ${predicted_price:,.2f}")
    print("="*40)
    
    if delta > 0:
        print(f"SIGNAL: [ BUY ] (Expected Move: +${delta:.2f})")
    else:
        print(f"SIGNAL: [ SELL ] (Expected Move: -${abs(delta):.2f})")

if __name__ == "__main__":
    predict_next_candle()