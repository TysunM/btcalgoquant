import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

class QuantClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        # Output layer now has 3 nodes: [0: Timeout, 1: Profit, 2: Stop Loss]
        self.linear = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # CrossEntropyLoss expects raw logits, no Softmax needed here
        return self.linear(lstm_out[:, -1, :])

class QuantDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def build_triple_barrier_matrix():
    print('\n[SYSTEM] Ingesting Matrix & Calculating Rolling Volatility...')
    # Forward fill to patch time-series, calculate rolling 100-period standard deviation for Volatility
    df = (pl.read_parquet(INPUT)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0)
          .with_columns(
              pl.col('Change').rolling_std(window_size=100).fill_null(0.0).alias('Volatility')
          ))
    
    np_data = df.to_numpy().astype(np.float32)
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print('[INFO] Vectorizing Triple Barrier Labels (PT, SL, Expiry)...')
    seq_len = 60
    horizon = 30  # Vertical Barrier: look 30 periods ahead
    pt_mult = 2.0 # Profit Take multiplier
    sl_mult = 2.0 # Stop Loss multiplier
    
    valid_rows = len(np_data) - seq_len - horizon
    X_all = np.zeros((valid_rows, seq_len, 7), dtype=np.float32) # 7 features (added Volatility)
    y_all = np.zeros((valid_rows,), dtype=np.int64)
    
    # Vectorized loop for labeling. 0: Timeout, 1: Profit, 2: Stop Loss
    for i in range(valid_rows):
        X_all[i] = np_data[i : i+seq_len]
        
        current_price = np_data[i+seq_len - 1, 0] # Index 0 is 'Price'
        current_vol = np_data[i+seq_len - 1, 6]   # Index 6 is 'Volatility'
        
        # Guard against zero volatility zero-division bugs
        if current_vol == 0:
            current_vol = 0.0001
            
        pt_price = current_price * (1 + (current_vol * pt_mult))
        sl_price = current_price * (1 - (current_vol * sl_mult))
        
        future_window = np_data[i+seq_len : i+seq_len+horizon, 0]
        
        # Find first instances crossing barriers
        hit_pt = np.argmax(future_window >= pt_price)
        hit_sl = np.argmax(future_window <= sl_price)
        
        # np.argmax returns 0 if condition is never met. 
        # We check if the actual condition is True at index 0 or elsewhere.
        pt_valid = future_window[hit_pt] >= pt_price
        sl_valid = future_window[hit_sl] <= sl_price
        
        if pt_valid and sl_valid:
            y_all[i] = 1 if hit_pt < hit_sl else 2
        elif pt_valid:
            y_all[i] = 1
        elif sl_valid:
            y_all[i] = 2
        else:
            y_all[i] = 0 # Vertical Barrier Hit (Time expired before PT/SL)
            
    print('[INFO] Executing Purged Split & Embargoing Leakage...')
    # Split 80/20, but insert a 5000-row deadzone (Embargo) to prevent serial correlation leakage
    split_idx = int(valid_rows * 0.8)
    embargo = 5000
    
    X_train_raw = X_all[:split_idx]
    y_train = y_all[:split_idx]
    
    X_val_raw = X_all[split_idx + embargo:]
    y_val = y_all[split_idx + embargo:]
    
    # Strict Training-only Normalization
    _min = X_train_raw.min(axis=(0, 1), keepdims=True)
    _max = X_train_raw.max(axis=(0, 1), keepdims=True)
    
    X_train = (X_train_raw - _min) / (_max - _min + 1e-7)
    X_val = (X_val_raw - _min) / (_max - _min + 1e-7)
    
    print(f'[INFO] Matrix Locked. Training: {len(X_train):,} | Validation: {len(X_val):,} | Embargoed: {embargo:,}')
    return X_train, y_train, X_val, y_val

def execute_institutional_training():
    device = torch.device('cpu')
    X_train, y_train, X_val, y_val = build_triple_barrier_matrix()
    
    train_loader = DataLoader(QuantDataset(X_train, y_train), batch_size=1024, shuffle=False, num_workers=0)
    val_loader = DataLoader(QuantDataset(X_val, y_val), batch_size=1024, shuffle=False, num_workers=0)
    
    model = QuantClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Classification Loss replacing MSE
    loss_fn = nn.CrossEntropyLoss() 
    
    epochs = 10
    print('\n[SYSTEM] Igniting Classification Engine (P-Cores)...')
    
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Check
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = loss_fn(outputs, y_val)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += y_val.size(0)
                correct += (predicted == y_val).sum().item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = (correct / total_predictions) * 100
        epoch_time = time.time() - start_epoch
        
        print(f'[EPOCH {epoch+1}/{epochs}] Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Accuracy: {accuracy:.2f}%')
        
        if epoch == 0:
            print('   -> [STATUS] Structural overhaul complete. Matrix is classifying.')
            break # Halt after 1 epoch for system check

if __name__ == '__main__':
    execute_institutional_training()
