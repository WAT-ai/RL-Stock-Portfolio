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

def train_deepar(stock_data: dict, window_len: int = 7, epochs: int = 20, 
                lr: float = 1e-4, train_split: float = 0.8, 
                batch_size: int = 10, debug: bool = False) -> DeepARModel:
    """
    Train the DeepAR model using stock data.

    Args:
        stock_data (dict): Dictionary containing stock data for multiple symbols
        window_len (int, optional): Length of the sliding window. Defaults to 10.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        train_split (float, optional): Training split ratio. Defaults to 0.8.
        batch_size (int, optional): Training batch size. Defaults to 7.
        debug (bool, optional): Enable debug printing. Defaults to False.

    Returns:
        DeepARModel: Trained model instance
    """
    # Create dataset with time-based splitting
    dataset = StockDataset(stock_data, window_len, train_split=train_split)
    
    # Create data loaders for train and validation
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
        # Training phase
        dataset.set_mode('train')
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            mu, sigma, _ = model(batch_x)
            loss = nll_loss(mu[:, :-1, 0], sigma[:, :-1, 0], batch_y[:, 1:, 3]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        # Validation phase
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
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train NLL: {total_loss/num_batches:.4f}, "
                  f"Val NLL: {avg_val_loss:.4f}")
    
    # Load best model state
    model.load_state_dict(best_model_state)
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

def make_prediction_windows(data: pd.DataFrame, window_size: int) -> list:
    """
    Create sliding windows for prediction.
    """
    windows = []
    total_rows = len(data)
    
    # Create windows up to the second-to-last day (last day is for final prediction)
    for i in range(total_rows - window_size):
        window_data = data.iloc[i:i+window_size]
        if i + window_size < total_rows:
            target_date = data.iloc[i+window_size]['Date']
            target_value = float(data.iloc[i+window_size]['Close'])  # Convert to float
            windows.append((window_data, target_date, target_value))
    
    # Add final window for tomorrow's prediction
    final_window = data.iloc[-window_size:]
    tomorrow = pd.Timestamp.now() + pd.Timedelta(days=1)
    windows.append((final_window, tomorrow, None))
    
    return windows

def test_model(model: DeepARModel, symbols: list, window_size: int = 7, 
               test_days: int = 50, debug: bool = False) -> dict:
    """
    Test the model using sliding windows of real data.

    Args:
        model (DeepARModel): Trained DeepAR model instance
        symbols (list): List of stock symbols to test
        window_size (int): Size of sliding window
        test_days (int): Number of past days to test on
        debug (bool): Enable debug printing

    Returns:
        dict: Dictionary containing results for each symbol
    """
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=test_days)
    current_data = fetch_stock_data(symbols, start_date, end_date)
    
    results = {}
    model.eval()
    
    for symbol, data in current_data.items():
        if debug:
            print(f"\nGenerating predictions for {symbol}")
        
        # Create dataset for normalization
        dataset = StockDataset({symbol: data}, window_size)
        
        # Get prediction windows
        prediction_windows = make_prediction_windows(data, window_size)
        predictions = []
        
        for window_data, target_date, target_value in prediction_windows:
            features = window_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            normalized_features = (features - dataset.norm_params[symbol]['mean']) / dataset.norm_params[symbol]['std']
            x = torch.FloatTensor(normalized_features).unsqueeze(0)
            
            # Make prediction - only use the mean value
            with torch.no_grad():
                pred_mean, _, _ = model.predict(x)
            
            # Denormalize prediction
            predicted_price = float(dataset.denormalize(pred_mean[0, 0, 0].item(), symbol))
            
            # Store results - simplified to just prediction and actual
            prediction_entry = {
                'date': target_date,
                'predicted': predicted_price,
                'actual': target_value
            }
            predictions.append(prediction_entry)
            
            if debug:
                actual_str = f"Actual: {target_value:.2f}" if target_value is not None else "Actual: N/A"
                print(f"Date: {target_date}, Predicted: {predicted_price:.2f}, {actual_str}")

        # Calculate metrics excluding the last prediction
        valid_predictions = predictions[:-1]
        actual_values = [p['actual'] for p in valid_predictions]
        predicted_values = [p['predicted'] for p in valid_predictions]
        metrics = calculate_metrics(predicted_values, actual_values)
        
        results[symbol] = {
            'predictions': predictions,
            'metrics': metrics
        }
        
        if debug:
            print(f"\nMetrics for {symbol}:")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")

    return results

if __name__ == "__main__":
    # Modified example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    window_size = 7
    
    # Training data (one year ending 31 days ago to avoid overlap with test data)
    train_end = datetime.now() - timedelta(days=50)
    train_start = train_end - timedelta(days=150)
    train_data = fetch_stock_data(symbols, train_start, train_end)
    
    # Train model
    model = train_deepar(train_data, window_len=window_size, debug=True)
    
    # Test model on recent data
    results = test_model(model, symbols, window_size=window_size, debug=True)
