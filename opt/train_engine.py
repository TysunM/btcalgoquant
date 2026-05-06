import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

class QuantLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

class QuantDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]  # Target is 'Price'
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_strict_data():
    print('[INFO] Ingesting and forward-filling time-series...')
    df = (pl.read_parquet(INPUT)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0))
    
    np_data = df.to_numpy().astype(np.float32)
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # STEP 1: SPLIT FIRST to prevent Data Leakage
    split_idx = int(len(np_data) * 0.8)
    raw_train = np_data[:split_idx]
    raw_val = np_data[split_idx:]
    
    # STEP 2: FIT ONLY ON TRAINING DATA
    _min = raw_train.min(axis=0, keepdims=True)
    _max = raw_train.max(axis=0, keepdims=True)
    
    # STEP 3: APPLY TO BOTH
    train_data = (raw_train - _min) / (_max - _min + 1e-7)
    val_data = (raw_val - _min) / (_max - _min + 1e-7)
    
    print(f'[INFO] Clean Split: {len(train_data):,} Training Rows | {len(val_data):,} Validation Rows')
    return train_data, val_data

def execute_stable_training():
    print('\n[SYSTEM] Initializing Stabilized Architecture...')
    device = torch.device('cpu')
    
    train_data, val_data = prepare_strict_data()
    
    train_loader = DataLoader(QuantDataset(train_data), batch_size=1024, shuffle=False, num_workers=0)
    val_loader = DataLoader(QuantDataset(val_data), batch_size=1024, shuffle=False, num_workers=0)
    
    model = QuantLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    epochs = 10
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    
    print('\n[SYSTEM] Igniting Early-Stopping Training Loop...')
    
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Pass (No learning here, just testing)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_loss = loss_fn(model(X_val), y_val.unsqueeze(1))
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_epoch
        
        print(f'[EPOCH {epoch+1}/{epochs}] Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'   -> [WARNING] Validation loss degraded. Strike {patience_counter}/{patience}')
            if patience_counter >= patience:
                print('\n[SYSTEM] EARLY STOPPING TRIGGERED. Matrix stabilized to prevent overfitting.')
                break

if __name__ == '__main__':
    execute_stable_training()
