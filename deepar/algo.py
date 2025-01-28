import torch
import torch.nn as nn
import torch.optim as optim

class DeepAR(nn.Module):
    def __init__(self, window_len, num_stocks, num_features, hidden_size=64, num_layers=1):
        super(DeepAR, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, window_len, num_features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_deepar(model, train_loader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

def predict_deepar(model, x):
    with torch.no_grad():
        return model(x)
