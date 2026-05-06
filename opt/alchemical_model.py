import torch
import torch.nn as nn
import time

class AlchemicalEngine(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(AlchemicalEngine, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Core LSTM Layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully Connected Block for price prediction
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # x shape: (batch_size, sequence_length=60, input_size=1)
        out, _ = self.lstm(x, (h0, c0))
        
        # Extract the output of the last time step (the 60th minute)
        out = out[:, -1, :]
        
        # Push through the fully connected block to get the final price prediction
        prediction = self.fc_block(out)
        return prediction

if __name__ == "__main__":
    print("--- MODEL ARCHITECTURE VERIFICATION ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    
    # Initialize the model
    model = AlchemicalEngine().to(device)
    
    # Create a dummy tensor matching your exact X_train shape for a single batch of 2048
    dummy_input = torch.randn(2048, 60, 1).to(device)
    
    print("Running forward pass simulation...")
    start_time = time.time()
    dummy_output = model(dummy_input)
    exec_time = time.time() - start_time
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {dummy_output.shape}")
    print(f"Forward Pass Time: {exec_time:.5f} seconds")
    print("Status: READY FOR TRAINING")
