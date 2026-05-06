import torch
import torch.nn as nn
import polars as pl
import numpy as np
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

def verify_training_loop():
    print('\n[SYSTEM] Igniting Neural Architecture on P-Cores...')
    device = torch.device('cpu')
    
    start_prep = time.time()
    print('[INFO] Executing Time-Series Forward Fill and Min-Max Scaling...')
    
    # PATCH: Forward-fill missing ticks to preserve time sequence, backward-fill the rest, failsafe to 0.0
    df = (pl.read_parquet(INPUT)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward')
          .fill_null(strategy='backward')
          .fill_null(0.0))
    
    np_data = df.to_numpy().astype(np.float32)
    
    # Final NumPy failsafe for any remaining inf/NaNs from division by zero
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    _min = np_data.min(axis=0, keepdims=True)
    _max = np_data.max(axis=0, keepdims=True)
    norm_data = (np_data - _min) / (_max - _min + 1e-7)
    
    print('[INFO] Slicing Sequential Batches (Sequence: 60 steps, Batch: 1024)...')
    batch_size, seq_len = 1024, 60
    
    X_np = np.zeros((batch_size, seq_len, 6), dtype=np.float32)
    y_np = np.zeros((batch_size, 1), dtype=np.float32)
    
    for i in range(batch_size):
        X_np[i] = norm_data[i:i+seq_len]
        y_np[i] = norm_data[i+seq_len, 0]
    
    print('[INFO] Injecting pure batches into Main RAM...')
    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)
    
    prep_time = time.time() - start_prep
    print(f'[INFO] CPU Data prep completed in {prep_time:.2f} seconds.')
    
    print('\n[INFO] Initializing Weights and Optimizers...')
    model = QuantLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print('[SYSTEM] Engaging P-Cores (Forward & Backward Pass)...')
    start_train = time.time()
    
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
    
    train_time = time.time() - start_train
    print(f'\n[SUCCESS] Epoch 001 Complete. Math executed in {train_time:.4f} seconds.')
    print(f'[METRIC] Initial Loss: {loss.item():.6f}')
    print('[STATUS] Architecture verified. Ready for the next phase.')

if __name__ == '__main__':
    verify_training_loop()
