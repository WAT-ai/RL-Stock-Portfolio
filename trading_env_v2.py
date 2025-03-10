import gym
from gym import spaces
from gym.spaces.box import Box
import numpy as np
import pandas as pd
import yfinance as yf
from twelvedata import TDClient
import pandas_ta as ta
import datetime 
from datetime import timedelta
from yahoo_fin import stock_info as si

'''
Burning question: How should we scale the stock data. -> Normalize along lookbad window (rolling window normalization), divide row by close price and global max for certain cols, etc. 
Infuse with prediction module
train an existing model more (specify actor and critic path)
experiment with reward function, data normalization, etc. 
More Complex Neural Network!!!!!

# Rolling window normalization
# Get rid of MACD
# GO back to simpler 64 neuron architecture
# Go back to sharpe ratio?
# I'm concerned about how we're normalizing portfolio weights
# Go through research papers to find techniques for improving PPO performance
'''


class PortfolioEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date, initial_balance=100000,window_len=20):
        """
        Initialize the Portfolio Environment.
        :param tickers: List of stock tickers to include in the portfolio.
        :param start_date: Start date for historical data.
        :param end_date: End date for historical data.
        :param initial_balance: Initial portfolio balance.
        """
        super(PortfolioEnv, self).__init__()

        # Load data
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.window_len = window_len

        # We'll add "cash" as an extra "asset"
        # So total assets = num_stocks + 1 (cash).
        self.num_assets = self.num_stocks + 1

        self.data = self._load_data(tickers, start_date, end_date)

        # -------------------------------------------------
        # 1) We'll step by date only (not date+ticker).
        #    So we get unique dates from level='Date'.
        # -------------------------------------------------
        self.dates = self.data.index.get_level_values("Date").unique()

        # Portfolio properties
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Now our weights vector includes [weight_cash, weight_ticker1, ..., weight_tickerN]
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0  # 100% in cash initially

        self.portfolio_value = initial_balance
        self.current_step = 0

        # Define action and observation spaces
        # Action = weights for each ticker
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        # OBSERVATION SPACE
        # We'll produce a 1D array that includes:
        # [ past window_len days * (Open, High, Low, Close, Volume, RSI, MACD, etc) for each ticker, + current weights ]
        # For example, if for each ticker we store 7 columns (5 OHLCV + 2 indicators), that’s 7*N tickers per day,
        # times window_len days => 7*N * window_len. Then we add self.num_assets more for current weights.
        # We'll guess a shape, but we'll confirm in _get_observation()

        # Let's say we have 2 indicators (rsi, macd) => 5(OHLCV) +2 =7 columns per ticker
        obs_len_per_day_per_ticker = 6
        self.obs_shape = (self.window_len * self.num_stocks * obs_len_per_day_per_ticker) + self.num_assets
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

    def _load_data(self, tickers, start_date, end_date):
        """
        1) Pull from (start_date - 30 days) to end_date
        2) Compute RSI/MACD
        3) Forward-fill or fillna(0)
        4) Finally drop all rows before the actual user start_date
        """

        # 1) Build the date range offset
        start_dt = pd.to_datetime(start_date)
        pull_dt  = start_dt - timedelta(days=30)  # to warm up indicators

        # # Initialize the TDClient with your API key
        # td = TDClient(apikey="69ec617153f34e86ba95c9ce301f0391")

        # # We'll fetch each ticker's daily data, store in a list
        # dfs = []

        # for ticker in tickers:
        #     print(f"Pulling data for {ticker} from {pull_dt.date()} to {end_date} via TwelveData...")
        #     try:
        #         # 2) Query daily time series
        #         ts = td.time_series(
        #             symbol=ticker,
        #             interval="1day",
        #             start_date=pull_dt.strftime("%Y-%m-%d"),
        #             end_date=end_date,
        #             outputsize=5000  # or more if needed
        #         )
        #         df_t = ts.as_pandas()

        #         # Some checks
        #         if df_t is None or df_t.empty:
        #             print(f"No data returned for {ticker}")
        #             continue

        #         # Convert index to a column if needed
        #         df_t.reset_index(inplace=True)

        #         # Rename columns to standard OHLCV
        #         df_t.rename(columns={
        #             "datetime": "Date",
        #             "open":     "Open",
        #             "high":     "High",
        #             "low":      "Low",
        #             "close":    "Close",
        #             "volume":   "Volume"
        #         }, inplace=True, errors="ignore")

        #         df_t["Ticker"] = ticker

        #         # Convert date str to datetime, sort
        #         df_t["Date"] = pd.to_datetime(df_t["Date"])
        #         df_t.sort_values(by="Date", inplace=True)

        #         dfs.append(df_t)
        #     except Exception as e:
        #         print(f"Error fetching data for {ticker}: {e}")

        # # Combine
        # if len(dfs) == 0:
        #     raise ValueError("No data returned from TwelveData for these tickers/date range.")

        # data = pd.concat(dfs, axis=0)
        # data.set_index(["Date","Ticker"], inplace=True)
        # data.sort_index(inplace=True)

        # print("Combined raw data head:")
        # print(data.head(10))

        # # Build a daily date range from pull_dt to end_date
        # # Here, we use freq="D", so we'll get weekends/holidays. 
        # # If you want to skip them, see "Alternative" below.
        # all_dates = pd.date_range(
        #     start=pull_dt.strftime("%Y-%m-%d"),
        #     end=end_date,
        #     freq="D"
        # )
        # # Reindex to ensure each date & ticker has a row
        # # -> forward fill from last known day
        # ticker_index = pd.Index(tickers, name="Ticker")
        # full_index   = pd.MultiIndex.from_product(
        #     [all_dates, ticker_index],
        #     names=["Date","Ticker"]
        # )

        # # Reindex, forward fill missing
        # data = data.reindex(full_index, method="ffill")
        # data = data.fillna(0.0)  # if anything is left as NaN

        # print("After reindex + ffill head:")
        # print(data.head(15))

        # # Now compute indicators
        # def compute_indicators(df_):
        #     df_ = df_.copy()
        #     df_.sort_values("Date", inplace=True)
        #     # RSI
        #     df_["RSI"] = ta.rsi(df_["Close"], length=14)
        #     # MACD
        #     macd_df = ta.macd(df_["Close"], fast=12, slow=26, signal=9)
        #     df_["MACD"] = macd_df["MACD_12_26_9"]
        #     # forward fill again
        #     df_ = df_.ffill().fillna(0.0)
        #     return df_

        # data = data.reset_index()
        # data = data.groupby("Ticker", group_keys=False).apply(compute_indicators)

        # # Make sure we have columns
        # keep_cols = ["Date","Ticker","Open","High","Low","Close","Volume","RSI","MACD"]
        # data = data[keep_cols]

        # # set index => (Date,Ticker) + sort
        # data.set_index(["Date","Ticker"], inplace=True)
        # data.sort_index(inplace=True)

        # # Finally drop rows strictly before the user’s actual start_date
        # data = data.loc[data.index.get_level_values("Date") >= start_dt]

        # print("After computing RSI/MACD & dropping older than start_date:")
        # print(data.head(25))

        # return data
        # yf.download()
        data = yf.download(
            tickers,
            start=pull_dt.strftime("%Y-%m-%d"),
            end=end_date,
            interval="1d"
        )

        #2. Select relevant OHLCV columns
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        print("Data after selecting OHLCV columns:")
        print(data.head())

        # 3. Stack on level=1 (the TICKER level), then reset index
        data = data.stack(level=1).reset_index()
        print("Data after stacking and resetting index:")
        print(data.head())

        # 4. Rename columns for clarity
        data = data.rename(columns={"level_0": "Date", "level_1": "Ticker"})
        data = data.set_index(["Date","Ticker"]).sort_index()

        print("Data after renaming columns:")
        print(data.head())

        # Compute indicators per Ticker
        # We'll group by Ticker, then apply pandas_ta
        def compute_indicators(df_):
            df_ = df_.sort_values("Date").copy()
            df_["RSI"] = ta.rsi(df_["Close"], length=14)
            return df_

        data = data.groupby("Ticker", group_keys=False).apply(compute_indicators)

        # Now pivot: index=(Date,Ticker), columns=(Open,High,Low,Close,Volume,RSI,MACD)
         # 5) Fill missing values
        data = data.groupby(level="Ticker").ffill()  # forward-fill missing RSI/MACD for each ticker
        data = data.fillna(0.0)  # if you prefer 0 for any leftover

        data = data[["Open","High","Low","Close","Volume","RSI"]]

        # 6) Drop all rows strictly before official start_date
        data = data.loc[data.index.get_level_values("Date") >= start_dt]

        # # 5. Melt to go from wide to long
        # data = data.melt(id_vars=["Date", "Ticker"],
        #                  var_name="Price_Type",
        #                  value_name="Value")
        # print("Data after melting:")
        # print(data.head())

        # # 6. Pivot to get columns = Price_Type, index = (Date, Ticker)
        # data = data.pivot(index=["Date", "Ticker"], columns="Price_Type", values="Value")
        # print("Data after pivoting:")
        # print(data.head())

        # 7. Return the pivoted DataFrame
        return data
    
    def reset(self):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0  # 100% cash
        self.portfolio_value = self.initial_balance
       
        # Choose a random start index so each episode doesn't always begin at day 0
        # We subtract 1 to ensure there's at least one more step after we start.
        # self.current_step = np.random.randint(self.window_len, len(self.dates) - 253)
        self.current_step = self.window_len

        print("Environment reset:")
        print(f"Initial balance: {self.balance}")
        print(f"Initial portfolio value: {self.portfolio_value}")
        print(f"Initial weights: {self.weights}")

        info = {}

        return self._get_observation(), info

    def step(self, action):
        """Execute one time step within the environment.
        :param action: Vector of weights for the portfolio (must sum to 1).
        """
        print(f"Step: {self.current_step}")
        print(f"Action: {action}")

        # Normalize action to ensure it sums to 1 -> could this be redundant?
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action[0] = 1.0
        action /= np.sum(action)
        self.weights = action
        print(f"Weights after normalization: {self.weights}")

        # Extract prices
        try:
            current_prices = self.data.loc[self.dates[self.current_step], :]["Close"].values
            next_prices = self.data.loc[self.dates[self.current_step + 1], :]["Close"].values
            print(f"Current prices: {current_prices}")
            print(f"Next prices: {next_prices}")
        except KeyError as e:
            print(f"KeyError during price extraction: {e}")
            raise

        # Calculate returns
        if len(current_prices) == 0 or len(next_prices) == 0:
            raise ValueError(f"Empty price arrays. Current prices: {current_prices}, Next prices: {next_prices}")
        returns = (next_prices - current_prices) / current_prices
        returns = np.insert(returns, 0, 0.0)
        print(f"Returns: {returns}")

        # Calculate portfolio return
        portfolio_return = np.dot(self.weights, returns)
        print(f"Portfolio return: {portfolio_return}")
        self.portfolio_value *= (1 + portfolio_return)
        print(f"Updated portfolio value: {self.portfolio_value}")

        # Calculate reward (log of risk-adjusted return)
        risk = np.std(returns) if np.std(returns) > 0 else 1e-8
        val = portfolio_return / risk + 1
        val_abs = abs(val)
        # val = max(val, 1e-8)  # or some positive epsilon
        log_Val = np.log(val_abs)

        # sortino ratio
        #pr_abs = abs(portfolio_return)
        # # shift = 1 + pr_abs (avoid log(0))
        # val = max(pr_abs,1e-8) 
        # raw_log = np.log(val)

        if val >= 0:
            reward = log_Val  # log(1 + return)
        else:
            reward = -log_Val # negative log(1 + abs(return))
            
        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}

    def _get_observation(self):
        """
        1) Identify the date range for the last `window_len` days up to self.current_step.
        2) Extract sub-data for each day & ticker.
        3) Row-based normalization for [Open,High,Low,Close], volume by max volume, 
        optionally scale RSI, MACD by some factor.
        4) Flatten final columns => 1D array => shape: (window_len * num_tickers * #features,).
        5) Print intermediate steps.
        """
        print(f"\n===== GET_OBSERVATION =====")
        print(f"Current step index: {self.current_step}")

        # 1) Figure out date window
        start_idx = self.current_step - self.window_len + 1
        if start_idx < 0:
            start_idx = 0
        end_idx = self.current_step
        dates_window = self.dates[start_idx : end_idx + 1]
        print(f"Dates from index {start_idx} to {end_idx}: {list(dates_window)}")

        # 2) sub_data => rows for those days, all tickers
        sub_data = self.data.loc[(dates_window, slice(None)), :].reset_index().sort_values(["Date","Ticker"])
        print("Sub_data (before normalization) HEAD:\n", sub_data.head(15))

       # 3) Row-based normalization:
        # For OHLC, divide by the row's Close value.
        sub_data[["Open", "High", "Low", "Close"]] = sub_data[["Open", "High", "Low", "Close"]].div(sub_data["Close"], axis=0)

        # For Volume and RSI, scale using the max in the entire dataset.
        global_max_volume = self.data["Volume"].max()
        global_max_RSI = self.data["RSI"].max()
        sub_data["Volume"] = sub_data["Volume"] / global_max_volume
        sub_data["RSI"] = sub_data["RSI"] / global_max_RSI

        print("Sub_data AFTER row-based normalization HEAD:\n", sub_data.head(15))


        # 3) Flatten => shape: (window_len * num_tickers * #features,)
        # We'll define the final columns in col_order
        # This might be: [Open,High,Low,Close,Volume,RSI,MACD]
        col_order = ["Open","High","Low","Close","Volume","RSI"]
        arr_2d = sub_data[col_order].to_numpy()  # shape=(#rows, len(col_order))

        observation = arr_2d.flatten()  # shape => (#rows * len(col_order),)
        observation = np.concatenate([observation, self.weights], axis=0)

        print("\nFinal observation shape:", observation.shape)
        print("Final observation (first ~30 vals):", observation[:30], "...")

        print("===== END GET_OBSERVATION =====\n")
        if np.isnan(observation).any() or np.isinf(observation).any():
            print("NaN/Inf found in observation!")
        return observation
    
    def render(self, mode="human"):
        """Render the environment."""
        print(f"Step: {self.current_step}, "
              f"Portfolio Value: {self.portfolio_value}, "
              f"Weights: {self.weights}")
