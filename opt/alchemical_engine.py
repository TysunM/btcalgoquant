import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import sys
import multiprocessing

# Maximize CPU utilization
torch.set_num_threads(multiprocessing.cpu_count())

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

class QuantClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

class QuantDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def build_triple_barrier_matrix():
    print('\n[SYSTEM] Ingesting Full 7-Year Matrix...')
    df = (pl.read_parquet(INPUT)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0)
          .with_columns(
              ((pl.col('Price') / pl.col('Price').shift(1)) - 1.0)
              .rolling_std(window_size=100)
              .fill_null(0.0)
              .alias('Volatility')
          ))
    
    np_data = df.to_numpy().astype(np.float32)
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print('[INFO] Vectorizing Triple Barrier Labels for 3.8 Million Rows...')
    seq_len = 60
    horizon = 30
    pt_mult = 2.0
    sl_mult = 2.0
    
    valid_rows = len(np_data) - seq_len - horizon
    X_all = np.zeros((valid_rows, seq_len, 7), dtype=np.float32)
    y_all = np.zeros((valid_rows,), dtype=np.int64)
    
    for i in range(valid_rows):
        X_all[i] = np_data[i : i+seq_len]
        current_price = np_data[i+seq_len - 1, 0]
        current_vol = np_data[i+seq_len - 1, 6]
        if current_vol == 0: current_vol = 0.0001
            
        pt_price = current_price * (1 + (current_vol * pt_mult))
        sl_price = current_price * (1 - (current_vol * sl_mult))
        future_window = np_data[i+seq_len : i+seq_len+horizon, 0]
        
        hit_pt = np.argmax(future_window >= pt_price)
        hit_sl = np.argmax(future_window <= sl_price)
        
        pt_valid = future_window[hit_pt] >= pt_price
        sl_valid = future_window[hit_sl] <= sl_price
        
        if pt_valid and sl_valid: y_all[i] = 1 if hit_pt < hit_sl else 2
        elif pt_valid: y_all[i] = 1
        elif sl_valid: y_all[i] = 2
        else: y_all[i] = 0
            
    print('[INFO] Executing Purged Split & Embargoing Leakage...')
    split_idx = int(valid_rows * 0.8)
    embargo = 5000
    
    X_train_raw = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val_raw = X_all[split_idx + embargo:]
    y_val = y_all[split_idx + embargo:]
    
    _mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
    _std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-7
    X_train = (X_train_raw - _mean) / _std
    X_val = (X_val_raw - _mean) / _std
    
    print(f'[INFO] Matrix Locked. Training: {len(X_train):,} | Validation: {len(X_val):,}')
    return X_train, y_train, X_val, y_val

def execute_institutional_training():
    device = torch.device('cpu')
    print(f'\n[SYSTEM] Hardware Target: CPU (Max Threads: {multiprocessing.cpu_count()})')
        
    X_train, y_train, X_val, y_val = build_triple_barrier_matrix()
    
    train_loader = DataLoader(QuantDataset(X_train, y_train), batch_size=2048, shuffle=False, num_workers=0)
    val_loader = DataLoader(QuantDataset(X_val, y_val), batch_size=2048, shuffle=False, num_workers=0)
    
    model = QuantClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 3
    print(f'\n[SYSTEM] Igniting P-Cores - Batch Size 2048...')
    
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 25 == 0 and batch_idx > 0:
                sys.stdout.write(f'\r   -> [EPOCH {epoch+1}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
                sys.stdout.flush()
                
        print('\n[INFO] Validating against future unseen data...')
        model.eval()
        total_val_loss = 0.0
        correct, total_predictions = 0, 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = loss_fn(outputs, y_val)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += y_val.size(0)
                correct += (predicted == y_val).sum().item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = (correct / total_predictions) * 100
        epoch_time = time.time() - start_epoch
        
        print(f'\n[SUCCESS] Epoch {epoch+1} Completed in {epoch_time:.2f}s')
        print(f'   -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        print(f'   -> Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    execute_institutional_training()
