import os
import torch
from datetime import datetime as dt
from deepar import fetch_stock_data, train_deepar

def main():
    # Define the stock symbols for training DeepAR.
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Define the training date range for DeepAR.
    train_start = dt(2012, 1, 1)
    train_end = dt(2024, 1, 1)
    
    print("Fetching stock data for DeepAR training...")
    train_data = fetch_stock_data(symbols, train_start, train_end, debug=True)
    
    print("Training DeepAR model...")
    # Adjust window_len, epochs, lr, train_split, and batch_size as needed.
    model = train_deepar(train_data, window_len=7, epochs=20, lr=1e-4, train_split=0.8, batch_size=10, debug=True)
    
    # Save the trained model to disk.
    save_path = "deepar_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"DeepAR model saved to {save_path}")

if __name__ == '__main__':
    main()
