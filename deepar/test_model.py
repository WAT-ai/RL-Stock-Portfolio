import yfinance as yf
import torch
import pandas as pd
import numpy as np
from model import DeepARModel, StockDataset, nll_loss
from datetime import datetime, timedelta
import torch.optim as optim
from torch.utils.data import DataLoader

def fetch_stock_data(symbols: list, start_date: datetime, end_date: datetime, debug: bool = False) -> dict:
    """
    Fetch stock data from Yahoo Finance for multiple symbols.

    Args:
        symbols (list): List of stock symbols to fetch data for
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        debug (bool, optional): Enable debug printing. Defaults to False.

    Returns:
        dict: Dictionary with stock symbols as keys and pandas DataFrames as values.
              Each DataFrame contains OHLCV data.
    """
    stock_data = {}
    for symbol in symbols:
        # Download Stock Data from YFinance
        stock = yf.download(symbol, start=start_date, end=end_date)
        
        # Validate Data
        if stock.empty:
            print(f"WARNING: No data found for {symbol}")
            continue
            
        # We only need the OHLCV columns
        stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']]
        stock = stock.reset_index()
        
        if debug: # print out other details for debugging
            print(f"\nStock {symbol}:")
            print(f"Shape: {stock.shape}")
            print(f"Sample:\n{stock.head()}")
            print(f"NaN values:\n{stock.isna().sum()}")
        
        # add the data to the dictionary for the symbol
        stock_data[symbol] = stock
    
    return stock_data

def train_deepar(stock_data: dict, window_len: int = 10, epochs: int = 10, 
                lr: float = 1e-4, val_split: float = 0.2, 
                batch_size: int = 32, debug: bool = False) -> DeepARModel:
    """
    Train the DeepAR model using stock data.

    Args:
        stock_data (dict): Dictionary containing stock data for multiple symbols
        window_len (int, optional): Length of the sliding window. Defaults to 10.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        val_split (float, optional): Validation split ratio. Defaults to 0.2.
        batch_size (int, optional): Training batch size. Defaults to 32.
        debug (bool, optional): Enable debug printing. Defaults to False.

    Returns:
        DeepARModel: Trained model instance
    """
    # Split data into train and validation
    split_index = int((1 - val_split) * len(stock_data))
    train_symbols = list(stock_data.keys())[:split_index]
    val_symbols = list(stock_data.keys())[split_index:]
    
    train_dataset = StockDataset({k: stock_data[k] for k in train_symbols}, window_len)
    val_dataset = StockDataset({k: stock_data[k] for k in val_symbols}, window_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = DeepARModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Training loop with batches
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            mu, sigma, _ = model(batch_x)
            loss = nll_loss(mu[:, :-1, 0], sigma[:, :-1, 0], batch_y[:, 1:, 3]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        # Validation loop with batches
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            model.eval()
            for batch_x, batch_y in val_loader:
                mu, sigma, _ = model(batch_x)
                val_loss += nll_loss(mu[:, :-1, 0], sigma[:, :-1, 0], batch_y[:, 1:, 3]).mean().item()
                num_val_batches += 1
            model.train()
        
        if debug:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train NLL: {total_loss/num_batches:.4f}, "
                  f"Val NLL: {val_loss/num_val_batches:.4f}")
    
    return model

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calculate prediction accuracy metrics.

    Args:
        predictions (np.ndarray): Array of predicted values
        actuals (np.ndarray): Array of actual values

    Returns:
        dict: Dictionary containing metrics:
            - 'RMSE': Root Mean Square Error
            - 'MAPE': Mean Absolute Percentage Error
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Mean Squared Error
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        'RMSE': rmse,
        'MAPE': mape
    }

def test_model(model: DeepARModel, symbols: list, window_len: int = 10, 
               debug: bool = False) -> dict:
    """
    Test the model and generate next-day predictions.

    Args:
        model (DeepARModel): Trained DeepAR model instance
        symbols (list): List of stock symbols to test
        window_len (int, optional): Length of the sliding window. Defaults to 10.
        debug (bool, optional): Enable debug printing. Defaults to False.

    Returns:
        dict: Dictionary containing results for each symbol:
            - 'Last_Known_Price': Last available price
            - 'Last_Known_Date': Date of last available price
            - 'Next_Day_Prediction': Tuple of (date, predicted_price)
    """
    # Fetch recent data for predictions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=window_len*2)  # Get enough historical data for the window
    current_data = fetch_stock_data(symbols, start_date, end_date)
    
    future_predictions = {}
    next_day = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    for symbol, data in current_data.items():
        if debug:
            print(f"\nGenerating prediction for {symbol}")
        
        dataset = StockDataset({symbol: data}, window_len)
        x, _ = dataset[0]
        x = x.unsqueeze(0)
        
        # Generate next day prediction
        prediction = model.predict(x)
        
        # Denormalize prediction
        normalized_pred = prediction[0, 0, 0].item()
        future_pred = dataset.denormalize(normalized_pred, symbol)
        
        future_predictions[symbol] = {
            'Last_Known_Price': float(data['Close'].iloc[-1].item()),
            'Last_Known_Date': end_date.strftime('%Y-%m-%d'),
            'Next_Day_Prediction': (next_day, future_pred)
        }
        
        # Print results (regardless of debug flag)
        print(f"\nResults for {symbol}:")
        print(f"Last Known Price: ${future_predictions[symbol]['Last_Known_Price']:.2f}")
        print(f"Next Day Prediction ({next_day}): ${future_pred:.2f}")
        
    return future_predictions

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'SHOP', "MSFT", "NVDA", "AMZN"]
    stock_data = fetch_stock_data(symbols, datetime.now() - timedelta(days=365), datetime.now())
    model = train_deepar(stock_data, debug=True)
    results = test_model(model, symbols, debug=True)
