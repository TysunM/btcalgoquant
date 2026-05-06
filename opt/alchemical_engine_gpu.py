import torch
import torch.nn as nn
import polars as pl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ta import add_all_ta_features
from sklearn.utils.class_weight import compute_class_weight
import time
import sys

INPUT = '/btc_quant/data/processed/quant_data_clean.parquet'

class QuantClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

class DynamicQuantDataset(Dataset):
    def __init__(self, data_2d, labels, indices, seq_len=60):
        self.data = data_2d
        self.labels = labels
        self.indices = indices
        self.seq_len = seq_len
        
    def __len__(self): return len(self.indices)
    
    def __getitem__(self, i):
        idx = self.indices[i]
        x_seq = self.data[idx : idx + self.seq_len]
        y_label = self.labels[i]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

def build_triple_barrier_matrix():
    print('\n[SYSTEM] Ingesting Full 7-Year Matrix (Linux/Docker)...')
    df = (pl.read_parquet(INPUT)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0)
          .with_columns(
              ((pl.col('Price') / pl.col('Price').shift(1)) - 1.0)
              .rolling_std(window_size=100)
              .fill_null(0.0)
              .alias('Volatility')
          ))
    
    print('[INFO] Injecting Technical Indicators (TA Library)...')
    df_pd = df.to_pandas()
    df_pd = add_all_ta_features(
        df_pd, open="Open", high="High", low="Low", close="Price", volume="Vol", fillna=True
    )
    df_pd.replace([np.inf, -np.inf], 0.0, inplace=True)
    df_pd.fillna(0.0, inplace=True)
    
    np_data = df_pd.to_numpy().astype(np.float32)
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    seq_len = 60
    horizon = 30
    pt_mult = 2.0
    sl_mult = 2.0
    
    valid_rows = len(np_data) - seq_len - horizon
    input_features = np_data.shape[1]
    
    print(f'[INFO] Matrix expanded to {input_features} features. Engaging Dynamic RAM Allocation...')
    print(f'[INFO] Vectorizing Triple Barrier Labels for {valid_rows:,} Rows...')
    
    y_all = np.zeros((valid_rows,), dtype=np.int64)
    price_col = np_data[:, 0]
    vol_col = np_data[:, 6]
    
    for i in range(valid_rows):
        current_price = price_col[i+seq_len - 1]
        current_vol = vol_col[i+seq_len - 1]
        if current_vol == 0: current_vol = 0.0001
            
        pt_price = current_price * (1 + (current_vol * pt_mult))
        sl_price = current_price * (1 - (current_vol * sl_mult))
        future_window = price_col[i+seq_len : i+seq_len+horizon]
        
        hit_pt = np.argmax(future_window >= pt_price)
        hit_sl = np.argmax(future_window <= sl_price)
        
        pt_valid = future_window[hit_pt] >= pt_price
        sl_valid = future_window[hit_sl] <= sl_price
        
        if pt_valid and sl_valid: y_all[i] = 1 if hit_pt < hit_sl else 2
        elif pt_valid: y_all[i] = 1
        elif sl_valid: y_all[i] = 2
        else: y_all[i] = 0
            
    split_idx = int(valid_rows * 0.8)
    embargo = 5000
    
    train_indices = np.arange(0, split_idx)
    val_indices = np.arange(split_idx + embargo, valid_rows)
    
    y_train = y_all[train_indices]
    y_val = y_all[val_indices]
    
    print('[INFO] Normalizing massive dataset...')
    train_data_raw = np_data[0 : split_idx + seq_len]
    _mean = train_data_raw.mean(axis=0, keepdims=True)
    _std = train_data_raw.std(axis=0, keepdims=True) + 1e-7
    
    np_data_normalized = (np_data - _mean) / _std
    
    print(f'[INFO] Matrix Locked. Training Sequences: {len(y_train):,} | Validation Sequences: {len(y_val):,}')
    return np_data_normalized, y_train, y_val, train_indices, val_indices, input_features

def execute_institutional_training():
    device = torch.device('cuda')
    print(f'\n[SYSTEM] Hardware Target: {torch.cuda.get_device_name(0)}')
        
    data_norm, y_train, y_val, train_idx, val_idx, input_features = build_triple_barrier_matrix()
    
    train_dataset = DynamicQuantDataset(data_norm, y_train, train_idx)
    val_dataset = DynamicQuantDataset(data_norm, y_val, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=True)
    
    model = QuantClassifier(input_size=input_features).to(device)
    
    # THE FIX: Calculate and apply dynamic Class Weights
    print('\n[INFO] Calculating dynamic Class Weights to penalize missed trades...')
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f'   -> Target Multipliers applied: Class 0: {weights[0]:.2f}x | Class 1: {weights[1]:.2f}x | Class 2: {weights[2]:.2f}x')
    
    # THE FIX: Reduced learning rate to 0.0001 to stop the epoch-to-epoch thrashing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 3
    print(f'\n[SYSTEM] Igniting RTX 5060 (AMP Active) - Batch Size 4096...')
    
    for epoch in range(epochs):
        start_epoch = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(X)
                loss = loss_fn(outputs, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                sys.stdout.write(f'\r   -> [EPOCH {epoch+1}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
                sys.stdout.flush()
                
        print('\n[INFO] Validating against future unseen data...')
        model.eval()
        total_val_loss = 0.0
        correct, total_predictions = 0, 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
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
        print(f'   -> Validation Accuracy: {accuracy:.2f}%\n')

if __name__ == '__main__':
    execute_institutional_training()
