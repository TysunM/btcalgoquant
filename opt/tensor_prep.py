import pandas as pd
import numpy as np
import torch
import json
import os

# --- CONFIGURATION ---
INPUT_FILE = r"C:\btc_quant\opt\BTC_MASTER_TRAINING_SET.csv"
OUTPUT_X = r"C:\btc_quant\opt\X_train.pt"
OUTPUT_Y = r"C:\btc_quant\opt\y_train.pt"
SCALER_FILE = r"C:\btc_quant\opt\scaler_params.json"

# The Model's "Memory" - Predicting the next minute based on the last 60 minutes
SEQ_LEN = 60 

def build_tensors():
    print("--- ALCHEMICAL ENGINE: TENSOR PIPELINE ---")
    print("Loading Master Dataset...")
    
    # We only load the 'close' price to keep RAM usage highly efficient
    df = pd.read_csv(INPUT_FILE, usecols=['close'])
    data_array = df['close'].values
    
    print("Normalizing data ranges...")
    min_val = float(np.min(data_array))
    max_val = float(np.max(data_array))
    
    # We must save these exact values to reverse the math during live trading
    with open(SCALER_FILE, 'w') as f:
        json.dump({'min': min_val, 'max': max_val}, f)
    print(f"Scaler logic locked and saved to {SCALER_FILE}")
        
    # Scale data between 0 and 1
    scaled_data = (data_array - min_val) / (max_val - min_val)
    
    print(f"Structuring sequences (Window: {SEQ_LEN} minutes)...")
    # High-speed vectorized windowing (bypasses slow Python loops)
    windows = np.lib.stride_tricks.sliding_window_view(scaled_data, SEQ_LEN + 1)
    
    X_np = windows[:, :-1]
    y_np = windows[:, -1]
    
    # PyTorch LSTMs require a 3D tensor: [Batch Size, Sequence Length, Features]
    X_np = np.expand_dims(X_np, axis=2)
    
    print("Allocating memory and converting to PyTorch Tensors...")
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    
    print("Writing binary tensor files to disk (This may take a moment)...")
    torch.save(X_tensor, OUTPUT_X)
    torch.save(y_tensor, OUTPUT_Y)
    
    print("\n--- TENSOR PIPELINE COMPLETE ---")
    print(f"Input Tensor Shape (X): {X_tensor.shape}")
    print(f"Target Tensor Shape (Y): {y_tensor.shape}")

if __name__ == "__main__":
    build_tensors()