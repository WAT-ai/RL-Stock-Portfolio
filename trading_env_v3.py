import gym
from gym import spaces
from gym.spaces.box import Box
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import timedelta
from deepar import test_model  # Import DeepAR testing function

class PortfolioEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date, initial_balance=100000, window_len=20, deepar_model=None):
        """
        Initialize the Portfolio Environment.

        Parameters:
          tickers (list): List of stock tickers.
          start_date, end_date (str): Date range for historical data.
          initial_balance (float): Starting portfolio value.
          window_len (int): Number of days in the observation window.
          deepar_model: Pretrained DeepAR model (optional). If provided, its predictions
                        (one per ticker) will be appended to the observation.
        """
        super(PortfolioEnv, self).__init__()

        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.window_len = window_len
        self.deepar_model = deepar_model  # Save the DeepAR model

        # Total assets include cash plus each stock.
        self.num_assets = self.num_stocks + 1

        self.data = self._load_data(tickers, start_date, end_date)
        self.dates = self.data.index.get_level_values("Date").unique()

        # Portfolio properties.
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0  # Start with 100% cash
        self.portfolio_value = initial_balance
        self.current_step = 0

        # Action space: portfolio weights.
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        # Observation space: historical features (6 per ticker per day), current weights,
        # and (if provided) one extra prediction per ticker.
        obs_len_per_day_per_ticker = 6
        extra_dim = self.num_stocks if self.deepar_model is not None else 0
        self.obs_shape = (self.window_len * self.num_stocks * obs_len_per_day_per_ticker) + self.num_assets + extra_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

    def _load_data(self, tickers, start_date, end_date):
        start_dt = pd.to_datetime(start_date)
        pull_dt = start_dt - timedelta(days=30)  # Warm-up period for indicators

        data = yf.download(tickers, start=pull_dt.strftime("%Y-%m-%d"), end=end_date, interval="1d")
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        print("Data after selecting OHLCV columns:")
        print(data.head())
        data = data.stack(level=1).reset_index()
        print("Data after stacking and resetting index:")
        print(data.head())
        data = data.rename(columns={"level_0": "Date", "level_1": "Ticker"})
        data = data.set_index(["Date", "Ticker"]).sort_index()
        print("Data after renaming columns:")
        print(data.head())

        def compute_indicators(df_):
            df_ = df_.sort_values("Date").copy()
            df_["RSI"] = ta.rsi(df_["Close"], length=14)
            return df_

        data = data.groupby("Ticker", group_keys=False).apply(compute_indicators)
        data = data.groupby(level="Ticker").ffill().fillna(0.0)
        data = data[["Open", "High", "Low", "Close", "Volume", "RSI"]]
        data = data.loc[data.index.get_level_values("Date") >= start_dt]
        return data

    def reset(self):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0
        self.portfolio_value = self.initial_balance
        self.current_step = self.window_len
        # self.current_step = np.random.randint(self.window_len, len(self.dates) - 253)

        print("Environment reset:")
        print(f"Initial balance: {self.balance}")
        print(f"Initial portfolio value: {self.portfolio_value}")
        print(f"Initial weights: {self.weights}")

        info = {}
        return self._get_observation(), info

    def step(self, action):
        """Execute one time step within the environment."""
        print(f"Step: {self.current_step}")
        print(f"Action: {action}")

        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action[0] = 1.0
        action /= np.sum(action)
        self.weights = action
        print(f"Weights after normalization: {self.weights}")

        try:
            current_prices = self.data.loc[self.dates[self.current_step], :]["Close"].values
            next_prices = self.data.loc[self.dates[self.current_step + 1], :]["Close"].values
            print(f"Current prices: {current_prices}")
            print(f"Next prices: {next_prices}")
        except KeyError as e:
            print(f"KeyError during price extraction: {e}")
            raise

        if len(current_prices) == 0 or len(next_prices) == 0:
            raise ValueError("Empty price arrays encountered.")
        returns = (next_prices - current_prices) / current_prices
        returns = np.insert(returns, 0, 0.0)
        print(f"Returns: {returns}")

        portfolio_return = np.dot(self.weights, returns)
        print(f"Portfolio return: {portfolio_return}")
        self.portfolio_value *= (1 + portfolio_return)
        print(f"Updated portfolio value: {self.portfolio_value}")

        risk = np.std(returns) if np.std(returns) > 0 else 1e-8
        val = portfolio_return / risk + 1
        log_val = np.log(abs(val))
        reward = log_val if val >= 0 else -log_val

        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}
    
    def _get_observation(self):
        """
        Build the observation by:
        1. Extracting a window of raw historical data,
        2. Saving a copy for DeepAR predictions,
        3. Applying row-based normalization on one copy for RL,
        4. Flattening and appending current portfolio weights,
        5. Using the raw copy to obtain DeepAR predictions (which will internally be z-score normalized),
        6. Normalizing the predictions by dividing by the most recent day's Close,
        7. Concatenating the predictions to the observation,
        8. Printing intermediate details.
        """
        print(f"\n===== GET_OBSERVATION =====")
        print(f"Current step index: {self.current_step}")

        # 1) Determine the date window.
        start_idx = self.current_step - self.window_len + 1
        if start_idx < 0:
            start_idx = 0
        end_idx = self.current_step
        dates_window = self.dates[start_idx: end_idx + 1]
        print(f"Dates from index {start_idx} to {end_idx}: {list(dates_window)}")

        # 2) Extract raw sub-data for those dates and all tickers.
        raw_sub_data = self.data.loc[(dates_window, slice(None)), :].reset_index().sort_values(["Date", "Ticker"])
        print("Raw sub_data HEAD:\n", raw_sub_data.head(15))

        # (Optional) If you want to ensure every ticker is present for each date,
        # reindex the data. (Uncomment if needed.)
        # all_index = pd.MultiIndex.from_product([dates_window, self.tickers], names=["Date", "Ticker"])
        # raw_sub_data = raw_sub_data.set_index(["Date", "Ticker"]).reindex(all_index).fillna(method="ffill").reset_index()

        # 3) Create a copy and apply row-based normalization for RL observation.
        norm_sub_data = raw_sub_data.copy()
        # For OHLC, divide by the row's Close value.
        norm_sub_data[["Open", "High", "Low", "Close"]] = norm_sub_data[["Open", "High", "Low", "Close"]].div(norm_sub_data["Close"], axis=0)
        # For Volume and RSI, scale using the max in the entire dataset.
        global_max_volume = self.data["Volume"].max()
        global_max_RSI = self.data["RSI"].max()
        norm_sub_data["Volume"] = norm_sub_data["Volume"] / global_max_volume
        norm_sub_data["RSI"] = norm_sub_data["RSI"] / global_max_RSI
        print("Normalized sub_data (row-based) HEAD:\n", norm_sub_data.head(15))

        # 4) Flatten the normalized data.
        col_order = ["Open", "High", "Low", "Close", "Volume", "RSI"]
        arr_2d = norm_sub_data[col_order].to_numpy()  # shape: (#rows, len(col_order))
        observation = arr_2d.flatten()               # shape: (#rows * len(col_order),)

        # 5) Append current portfolio weights.
        observation = np.concatenate([observation, self.weights], axis=0)

        # 6) If a DeepAR model is provided, get predictions using the raw (un-normalized) data.
        if self.deepar_model is not None:
            # Use the raw sub_data (which has the original scale) for DeepAR predictions.
            prediction_dataframe = raw_sub_data.copy()
            # Group the raw data by ticker.
            grouped_data = {ticker: prediction_dataframe[prediction_dataframe['Ticker'] == ticker].reset_index(drop=True)
                            for ticker in prediction_dataframe['Ticker'].unique()}
            # Call test_model, which will normalize its inputs using z-score based on its own StockDataset.
            results = test_model(self.deepar_model, grouped_data)
            
            # Print raw DeepAR predictions.
            print("Raw DeepAR predictions:")
            for ticker, result in results.items():
                raw_pred = result["last_prediction"]['predicted']
                print(f"Ticker {ticker}: {raw_pred}")
            
            # 7) Normalize each prediction by dividing by the most recent day's Close from the raw data.
            normalized_predictions = {}
            for ticker, data in grouped_data.items():
                last_close = data.iloc[-1]['Close']
                raw_prediction = results[ticker]["last_prediction"]['predicted']
                normalized_predictions[ticker] = raw_prediction / (last_close)
            
            # Sort tickers for consistent order.
            sorted_tickers = sorted(normalized_predictions.keys())
            numpy_predictions = np.array([normalized_predictions[ticker] for ticker in sorted_tickers])
            print("Normalized DeepAR predictions appended:", numpy_predictions)
        else:
            numpy_predictions = np.array([])

        # 8) Concatenate DeepAR predictions to the observation.
        observation = np.concatenate([observation, numpy_predictions], axis=0)

        # 9) Print final observation details.
        print("\nFinal observation shape:", observation.shape)
        print("Final observation (first ~30 vals):", observation[:30], "...")
        print("===== END GET_OBSERVATION =====\n")

        if np.isnan(observation).any() or np.isinf(observation).any():
            print("NaN/Inf found in observation!")
        return observation


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value}, Weights: {self.weights}")
