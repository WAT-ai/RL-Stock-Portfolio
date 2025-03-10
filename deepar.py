import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt, timedelta

##############################
# DeepAR Model and Loss
##############################

class DeepARModel(nn.Module):
    """
    Deep AutoRegressive (DeepAR) model for time series forecasting.
    """
    def __init__(self, input_size: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super(DeepARModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.post_lstm_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Output layers for probabilistic forecasting
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hidden: tuple = None) -> tuple:
        """
        Forward pass of the model.
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
        fc_out = self.post_lstm_fc(lstm_out)
        mu = self.mu_layer(fc_out)
        sigma = torch.exp(self.sigma_layer(fc_out))  # Ensure positive variance
        return mu, sigma, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Generate a point prediction for the next day.
        """
        self.eval()
        with torch.no_grad():
            mu, sigma, hidden = self(x)
            # Use the prediction from the last time step
            prediction_mean = mu[:, -1:, :]
            prediction_std = sigma[:, -1:, :]
        return prediction_mean, prediction_std, hidden

def nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative log likelihood loss.
    """
    return 0.5 * torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (2 * sigma**2)

##############################
# Data Processing & Dataset
##############################

class StockDataset:
    """
    Dataset class for stock data processing.
    """
    def __init__(self, data: dict, window_len: int, train_split: float = 0.8):
        self.window_len = window_len
        self.stocks = {}
        self.norm_params = {}
        self.mode = None  # 'train' or 'val'

        for symbol, df in data.items():
            features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            split_idx = int(len(features) * train_split)
            train_features = features[:split_idx]
            mean = train_features.mean(0)
            std = train_features.std(0)
            self.norm_params[symbol] = {'mean': mean, 'std': std}
            features = (features - mean) / std
            self.stocks[symbol] = features

        self.train_indices = {}
        self.val_indices = {}
        for symbol in self.stocks:
            data_len = len(self.stocks[symbol])
            split_idx = int(data_len * train_split)
            self.train_indices[symbol] = list(range(0, split_idx - window_len))
            self.val_indices[symbol] = list(range(split_idx - window_len, data_len - window_len))

    def set_mode(self, mode: str):
        assert mode in ['train', 'val'], "Mode must be either 'train' or 'val'"
        self.mode = mode

    def denormalize(self, value: float, symbol: str, feature_idx: int = 3) -> float:
        mean = self.norm_params[symbol]['mean'][feature_idx]
        std = self.norm_params[symbol]['std'][feature_idx]
        return value * std + mean

    def __getitem__(self, idx: int) -> tuple:
        if self.mode is None:
            raise ValueError("Dataset mode not set. Call set_mode('train') or set_mode('val').")
        for symbol in self.stocks:
            indices = self.train_indices[symbol] if self.mode == 'train' else self.val_indices[symbol]
            if idx < len(indices):
                data_idx = indices[idx]
                sequence = self.stocks[symbol][data_idx:data_idx + self.window_len]
                x = torch.FloatTensor(sequence[:-1])
                y = torch.FloatTensor(sequence[1:])
                return x, y
            idx -= len(indices)
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        if self.mode is None:
            raise ValueError("Dataset mode not set. Call set_mode('train') or set_mode('val').")
        total_len = 0
        for symbol in self.stocks:
            indices = self.train_indices[symbol] if self.mode == 'train' else self.val_indices[symbol]
            total_len += len(indices)
        return total_len

##############################
# Data Fetching and Training
##############################

def fetch_stock_data(symbols: list, start_date: dt, end_date: dt, debug: bool = False) -> dict:
    """
    Fetch stock data from Yahoo Finance.
    """
    stock_data = {}
    for symbol in symbols:
        stock = yf.download(symbol, start=start_date, end=end_date)
        if stock.empty:
            print(f"WARNING: No data found for {symbol}")
            continue
        stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock = stock.reset_index()
        if debug:
            print(f"\nStock {symbol}:")
            print(stock.head())
        stock_data[symbol] = stock
    return stock_data

def train_deepar(stock_data: dict, window_len: int = 7, epochs: int = 20,
                 lr: float = 1e-4, train_split: float = 0.8,
                 batch_size: int = 10, debug: bool = False) -> DeepARModel:
    """
    Train the DeepAR model using stock data.
    """
    dataset = StockDataset(stock_data, window_len, train_split=train_split)
    dataset.set_mode('train')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset.set_mode('val')
    val_loader = DataLoader(dataset, batch_size=batch_size)

    model = DeepARModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        dataset.set_mode('train')
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            mu, sigma, _ = model(batch_x)
            # Use column index 3 (Close price) as target for prediction
            loss = nll_loss(mu[:, :-1, 0], sigma[:, :-1, 0], batch_y[:, 1:, 3]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        dataset.set_mode('val')
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in val_loader:
                mu, sigma, _ = model(batch_x)
                val_loss += nll_loss(mu[:, :-1, 0], sigma[:, :-1, 0], batch_y[:, 1:, 3]).mean().item()
                num_val_batches += 1
            avg_val_loss = val_loss / num_val_batches

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()

            model.train()

        if debug:
            print(f"Epoch {epoch+1}/{epochs}, Train NLL: {total_loss/num_batches:.4f}, Val NLL: {avg_val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model

def make_prediction_windows(data: pd.DataFrame, window_size: int) -> list:
    """
    Create sliding windows for prediction.
    """
    windows = []
    total_rows = len(data)
    for i in range(total_rows - window_size):
        window_data = data.iloc[i:i+window_size]
        if i + window_size < total_rows:
            target_date = data.iloc[i+window_size]['Date']
            target_value = float(data.iloc[i+window_size]['Close'])
            windows.append((window_data, target_date, target_value))
    final_window = data.iloc[-window_size:]
    tomorrow = pd.Timestamp.now() + pd.Timedelta(days=1)
    windows.append((final_window, tomorrow, None))
    return windows

def test_model(model: DeepARModel, testing_data: dict, window_size: int = 7, debug: bool = False) -> dict:
    """
    Test the DeepAR model using sliding windows.
    """
    results = {}
    model.eval()

    for symbol, data in testing_data.items():
        if debug:
            print(f"\nGenerating predictions for {symbol}")
        dataset = StockDataset({symbol: data}, window_size)
        prediction_windows = make_prediction_windows(data, window_size)
        predictions = []

        for window_data, target_date, target_value in prediction_windows:
            features = window_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            normalized_features = (features - dataset.norm_params[symbol]['mean']) / dataset.norm_params[symbol]['std']
            x = torch.FloatTensor(normalized_features).unsqueeze(0)
            with torch.no_grad():
                pred_mean, _, _ = model.predict(x)
            predicted_price = float(dataset.denormalize(pred_mean[0, 0, 0].item(), symbol))
            prediction_entry = {
                'date': target_date,
                'predicted': predicted_price,
                'actual': target_value
            }
            predictions.append(prediction_entry)

            if debug:
                actual_str = f"Actual: {target_value:.2f}" if target_value is not None else "Actual: N/A"
                print(f"Date: {target_date}, Predicted: {predicted_price:.2f}, {actual_str}")

        results[symbol] = {
            'predictions': predictions,
            'last_prediction': predictions[-1]
        }
    return results
