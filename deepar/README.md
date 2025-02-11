# DeepAR Stock Price Prediction

A PyTorch implementation of DeepAR for stock price prediction using LSTM-based probabilistic forecasting.

## Technical Implementation

### Model Architecture

- **LSTM Layers**: 2-layer LSTM with 64 hidden units
- **Input Features**: 5 dimensions (OHLCV data)
- **Output**: Probabilistic forecast (μ and σ parameters)
- **Dropout**: 0.1 between LSTM layers
- **Post-LSTM Network**: Two fully connected layers with ReLU activation

### Data Processing

1. **Normalization**:
   - Feature-wise standardization (μ=0, σ=1)
   - Separate normalization parameters per stock
   - Parameters computed only on training data

2. **Windowing**:
   - Sliding window approach
   - Default window size: 7 days
   - Sequential splits for train/validation

### Training Details

```python
# Model initialization
model = DeepARModel(
    input_size=5,      # OHLCV features
    hidden_size=64,    # LSTM hidden units
    num_layers=2,      # LSTM layers
    dropout=0.1
)

# Training parameters
batch_size = 10
learning_rate = 1e-4
epochs = 20
train_split = 0.8

# Loss function
loss = nll_loss(mu, sigma, targets)  # Negative log likelihood
optimizer = Adam(lr=learning_rate)
```

### Prediction Pipeline

1. **Data Fetch**:
   ```python
   data = fetch_stock_data(symbols, start_date, end_date)
   ```

2. **Model Training**:
   ```python
   model = train_deepar(
       stock_data=data,
       window_len=7,
       epochs=20,
       lr=1e-4
   )
   ```

3. **Testing/Prediction**:
   ```python
   results = test_model(
       model=model,
       symbols=symbols,
       window_size=7,
       test_days=50
   )
   ```

### Dependencies

- PyTorch
- yfinance
- pandas
- numpy
- matplotlib