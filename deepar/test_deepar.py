import pytest
import torch
import yfinance as yf
from algo import DeepAR, train_deepar, predict_deepar
from sklearn.preprocessing import StandardScaler

@pytest.mark.parametrize("symbol", ["MSFT"])
def test_deepar_training(symbol):
    # Fetch data from YFinance
    df = yf.download(symbol, period="1mo", interval="1d")
    df = df.dropna().reset_index()
    df = df[["Open","High","Low","Close","Volume"]]

    print(df.head()) # testing purposes

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    data = torch.tensor(scaled_data, dtype=torch.float)

    def create_windows(tensor_data, window_len=10):
        xs, ys = [], []
        for i in range(len(tensor_data) - window_len):
            xs.append(tensor_data[i:i+window_len])
            ys.append(tensor_data[i+window_len, 3])      # predict "Close"
        return torch.stack(xs), torch.stack(ys).unsqueeze(1)

    window_len = 10
    x, y = create_windows(data, window_len)

    model = DeepAR(window_len=window_len, num_stocks=1, num_features=5)

    dataset = list(zip(x, y))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    train_deepar(model, train_loader, epochs=10)

    x_single = x[0:1]
    pred = predict_deepar(model, x_single)
    
    unscaled_pred = scaler.inverse_transform(
        torch.zeros((1, df.shape[1])).numpy()
    )
    unscaled_pred[0, 3] = pred.item()  # 3 is the index for Close price
    print("Next day price (scaled):", pred.item()) # testing purposes
    print("Next day price (unscaled):", unscaled_pred[0, 3]) # testing purposes
    
    assert pred.shape[0] == 1, "Prediction shape mismatch"
    assert not torch.isnan(pred).any(), "Prediction contains NaN"

def test_deepar_prediction():
    model = DeepAR(window_len=10, num_stocks=1, num_features=1)
    x = torch.zeros((1, 10, 1))
    pred = predict_deepar(model, x)
    assert pred.shape == (1, 1)