import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

# -----------------
# 1. Configuration
# -----------------
DB_URL = 'postgresql://postgres:775238xpro@127.0.0.1:5432/btc_quant_db'

SEQ_LENGTH = 60      # Look back 60 minutes
BATCH_SIZE = 64      # Increased batch size for 5060 VRAM
HIDDEN_SIZE = 128    # Increased complexity for multivariate data
NUM_LAYERS = 3       # Deeper network
EPOCHS = 100
LEARNING_RATE = 0.0005

def check_hardware():
    # Locked to GPU for production
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: Hardware initialization returned CPU. Check Nightly Wheel installation.")
    else:
        print(f"Hardware initialization: Utilizing CUDA ({torch.cuda.get_device_name(0)})")
    return device

# -----------------
# 2. Data Pipeline & Matrix Alignment
# -----------------
def load_and_prep_data():
    print("Connecting to PostgreSQL and building the Multivariate Matrix...")
    engine = create_engine(DB_URL)
    
    query = "SELECT date, symbol, open, high, low, close, volume FROM market_data_min ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    
    # Pivot the data: Rows = Date, Columns = Asset Features
    pivot_df = df.pivot(index='date', columns='symbol')
    
    # Flatten the MultiIndex columns (e.g., 'close', 'BTCUSD' -> 'BTCUSD_close')
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    
    # Forward-fill missing equity data during off-hours, then drop remaining NaNs at the start
    pivot_df.ffill(inplace=True)
    pivot_df.dropna(inplace=True)
    
    print(f"Matrix aligned. Total valid multi-asset minutes: {len(pivot_df)}")

    features = pivot_df.columns.tolist()
    data = pivot_df.values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    # Identify the exact column index we are trying to predict
    target_col_index = features.index('BTCUSD_close')
    
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y.append(scaled_data[i, target_col_index]) # Target: Next minute BTC Close

    X, y = np.array(X), np.array(y)
    
    # Train/Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Convert to Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler, len(features)

# -----------------
# 3. Multivariate LSTM Architecture
# -----------------
class Quad_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Quad_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input_size is now dynamically set to 20 (4 assets * 5 features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

# -----------------
# 4. Training Engine
# -----------------
def train_model():
    device = check_hardware()
    
    X_train, y_train, X_test, y_test, scaler, input_size = load_and_prep_data()
    print(f"Neural Network Input Vector Size: {input_size} features per minute.")
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Quad_LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Igniting Training Loop...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}')

    print("Training Complete.")
    
    model_path = r"C:\btc_quant\opt\quad_lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Multivariate Model saved to {model_path}")

if __name__ == "__main__":
    train_model()