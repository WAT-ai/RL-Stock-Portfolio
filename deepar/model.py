import torch
import torch.nn as nn
import numpy as np

class DeepARModel(nn.Module):
    """
    Deep AutoRegressive (DeepAR) model for time series forecasting.
    
    Attributes:
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of LSTM layers
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the DeepAR model.

        Args:
            input_size (int, optional): Number of input features. Defaults to 6.
            hidden_size (int, optional): Size of hidden layers. Defaults to 64.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(DeepARModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
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

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_len, input_size)
            hidden (tuple, optional): Initial hidden state. Defaults to None.

        Returns:
            tuple: (mu, sigma, hidden)
                - mu: Mean predictions
                - sigma: Standard deviation predictions
                - hidden: Updated hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Pass through new FC sub-network before mu and sigma
        fc_out = self.post_lstm_fc(lstm_out)
        mu = self.mu_layer(fc_out)
        sigma = torch.exp(self.sigma_layer(fc_out))  # Ensure positive variance
        
        return mu, sigma, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction for the next day.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_len, input_size)

        Returns:
            torch.Tensor: Prediction tensor of shape (batch_size, 1, 1)
        """
        self.eval()
        with torch.no_grad():
            mu, sigma, hidden = self(x)
            prediction = mu[:, -1:, :]  # Just take the last prediction
            
        return prediction

def nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log likelihood loss.

    Args:
        mu (torch.Tensor): Predicted mean values
        sigma (torch.Tensor): Predicted standard deviations
        target (torch.Tensor): Actual target values

    Returns:
        torch.Tensor: Computed NLL loss
    """
    return 0.5 * torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (2 * sigma**2)

class StockDataset:
    """
    Dataset class for stock data processing.
    
    Attributes:
        window_len (int): Length of the sliding window
        stocks (dict): Processed stock data
        norm_params (dict): Normalization parameters
    """
    
    def __init__(self, data: dict, window_len: int):
        """
        Initialize the dataset.

        Args:
            data (dict): Dictionary with stock symbols as keys and DataFrames as values
            window_len (int): Length of the sliding window
        """
        self.window_len = window_len
        self.stocks = {}
        self.norm_params = {}
        
        # Process each stock in the dictionary
        for symbol, df in data.items():
            features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            # Store normalization parameters
            mean = features.mean(0)
            std = features.std(0)
            self.norm_params[symbol] = {'mean': mean, 'std': std}
            # Normalize features
            features = (features - mean) / std
            self.stocks[symbol] = features
    
    def denormalize(self, value: float, symbol: str, feature_idx: int = 3) -> float:
        """
        Denormalize a value for a specific symbol and feature.

        Args:
            value (float): Value to denormalize
            symbol (str): Stock symbol
            feature_idx (int, optional): Feature index (0=Open, 1=High, 2=Low, 3=Close, 4=Volume).
                                       Defaults to 3 (Close).

        Returns:
            float: Denormalized value
        """
        mean = self.norm_params[symbol]['mean'][feature_idx]
        std = self.norm_params[symbol]['std'][feature_idx]
        return value * std + mean
        
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a data sample.

        Args:
            idx (int): Index of the stock

        Returns:
            tuple: (x, y)
                - x: Input sequence tensor
                - y: Target sequence tensor
        """
        stock_id = list(self.stocks.keys())[idx]
        data = self.stocks[stock_id]
        
        # Random starting point
        start_idx = np.random.randint(0, len(data) - self.window_len)
        sequence = data[start_idx:start_idx + self.window_len]
        
        x = torch.FloatTensor(sequence[:-1])  # All features for input
        y = torch.FloatTensor(sequence[1:])   # All features for target
        
        return x, y
    
    def __len__(self) -> int:
        """
        Get the number of stocks in the dataset.

        Returns:
            int: Number of stocks
        """
        return len(self.stocks)