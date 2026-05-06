import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from alpha_gen import generate_alpha_signals

# Hardware Detection: Force CUDA for RTX 5060
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BTCPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(BTCPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  
        return out

def prepare_data(df, seq_length=14):
    features = ['close_price', 'EMA_10', 'EMA_50', 'EMA_200', 'RSI', 'Runaway_Alignment']
    data = df[features].copy()
    data['Runaway_Alignment'] = data['Runaway_Alignment'].astype(int)
    data['target'] = data['close_price'].shift(-1)
    data.dropna(inplace=True)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_X = scaler_X.fit_transform(data[features])
    scaled_y = scaler_y.fit_transform(data[['target']])

    X, y = [], []
    for i in range(len(scaled_X) - seq_length):
        X.append(scaled_X[i:(i + seq_length)])
        y.append(scaled_y[i + seq_length])

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32), scaler_X, scaler_y

if __name__ == "__main__":
    print(f"Hardware initialization: Utilizing {device.type.upper()}")
    
    df = generate_alpha_signals('1d')
    if df is None or df.empty:
        print("Signal generation failed. Exiting.")
        exit()
        
    SEQ_LENGTH = 14
    X, y, scaler_X, scaler_y = prepare_data(df, seq_length=SEQ_LENGTH)
    
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
    
    # Push model to RTX 5060
    model = BTCPredictor(input_size=6).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 30
    print(f"Executing training sequence ({epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {total_loss/len(train_loader):.6f}")

    # Save relative to execution directory
    model_path = os.path.join(os.path.dirname(__file__), "lstm_btc_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved to: {model_path}")