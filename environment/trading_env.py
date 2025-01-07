from collections.abc import Callable
import gymnasium as gym
import pandas as pd
import numpy as np
from typing import Any, NewType, Tuple
from functools import cache

# TODO: implement seeded rng
import random
import math

# The actions will be a list of floats specifying the new weights of the portfolio
ActType = NewType("ActType", list[float])

# TODO: change this if a more specific ObsType is needed
ObsType = pd.DataFrame


class TradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for simulating a trading system.

    Attributes:
        COL_ID (str): Column name for asset identifiers.
        COL_OPEN (str): Column name for open prices.
        COL_CLOSE (str): Column name for close prices.
        COL_ADJ_CLOSE (str): Column name for adjusted close prices.
        COL_HIGH (str): Column name for high prices.
        COL_LOW (str): Column name for low prices.
        COL_VOLUME (str): Column name for trading volume.
        COL_TIME (str): Column name for time index.

    Methods:
        __init__: Initializes the environment with configuration parameters.
        _get_observation: Retrieves the current window of OHCLV data.
        step: Takes an action, computes the reward, and advances the environment.
        _get_capital: Calculates the current capital based on positions and market changes.
        reset: Resets the environment to its initial state.
    """

    COL_ID = "Id"
    COL_OPEN = "Open"
    COL_CLOSE = "Close"
    COL_ADJ_CLOSE = "AdjClose"
    COL_HIGH = "High"
    COL_LOW = "Low"
    COL_VOLUME = "Volume"
    COL_TIME = "Time"

    def __init__(
        self,
        ohclv_data: pd.DataFrame,
        num_risky_assets: int,
        window_len: int,
        initial_capital: float,
        transaction_cost: float,
        reward_function: Callable,
        batch_len: int,
        index_to_id: list[str],
        seed: int,
    ) -> None:
        """
        Initializes the trading environment.

        The ohclv dataframe must contain a timeseries column for the following features:
        - Date, Id, Open, High, Low, Close, Volume,
        and optionally an AdjClose column.

        An example dataframe looks like:
                 Date     Id        Open        High         Low       Close    Volume
        0  2024-11-20   AAPL  228.059998  229.929993  225.889999  229.000000  35169600
        1  2024-11-20   GOOG  178.627344  178.907025  175.131310  177.129044  15729800
        2  2024-11-20   MSFT  416.037221  416.456396  409.759778  414.659973  19191700
        3  2024-11-21   AAPL  228.880005  230.160004  225.710007  228.520004  42108300
        4  2024-11-21   GOOG  175.256171  175.381029  165.122663  169.048218  38839400
        ...

        Args:
            ohclv_data (pd.DataFrame): Input OHCLV data.
            num_risky_assets (int): Number of risky assets (e.g., stocks).
            window_len (int): Length of the observation window.
            capital (float): Initial capital in the portfolio.
            transaction_cost (float): Transaction cost as a fraction of capital.
            reward_function (Callable): Function to compute the reward.
            index_to_id (list[str]): List mapping indices to asset identifiers.
        """
        np.random.seed(seed)
        random.seed(seed)

        self._ohclv_data = ohclv_data
        self._format_ohclv_data()
        self._num_risky_assets = num_risky_assets
        self._positions = np.zeros(num_risky_assets + 1)  # +1 for cash
        self._positions[0] = 1
        self._window_len = window_len
        self._initial_capital = initial_capital
        self._portfolio_value = initial_capital
        self._tcost = transaction_cost
        self._reward_function = reward_function
        self._index_to_id = index_to_id
        self._batch_len = batch_len
        self.reward = 0

        self._data_len = self._ohclv_data[self.COL_TIME].max()

        # initially starts as the "start_time"
        self._cur_end_time = random.randint(
            self._window_len - 1, self._data_len - self._batch_len + 1
        )
        self._episode_end_time = self._cur_end_time = self._batch_len

        # using the "start_time" to get the episode_end_timeself._episode_end_time = self._cur_end_time + self._batch_len

        super().__init__()

    def _format_ohclv_data(self):
        for col in ("Date", "Id", "Open", "High", "Low", "Close", "Volume"):
            assert (
                col in self._ohclv_data
            ), f"Column={col} not found in columns of ohclv data"

        self._ohclv_data["Time"] = pd.factorize(self._ohclv_data["Date"])[0]
        self._ohclv_data.drop(columns=["Date"], inplace=True)

    def _get_observation(self) -> ObsType:
        """
        Retrieves the current observation window of OHCLV data.

        Returns:
            ObsType: A subset of the OHCLV data representing the current window.
        """
        observation = self._ohclv_data[
            (
                self._cur_end_time - self._window_len + 1
                <= self._ohclv_data[self.COL_TIME]
            )
            & (self._ohclv_data[self.COL_TIME] <= self._cur_end_time)
        ]
        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional reset options. Defaults to None.

        Returns:
            tuple: Initial observation and additional info.
        """
        np.random.seed(seed)
        random.seed(seed)

        self._cur_end_time = random.randint(
            self._window_len - 1, self._data_len - self._batch_len + 1
        )
        self._episode_end_time = self._cur_end_time = self._batch_len
        self._capital = self._original_capital
        self._positions = np.zeros(self._num_risky_assets)
        self._positions[0] = 1  # First index represents cash value

        return self._get_observation(), {}

    def step(
        self, action: list[float]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """
        Executes a step in the environment by taking the specified action.

        Args:
            action (list[float]): Portfolio weights to allocate to each asset.

        Returns:
            tuple: Observation, reward, termination flag, truncation flag, and additional info.
        """
        assert len(action) == len(self._positions)

        action = np.array(action)

        # reward = self._reward_function(self._positions, action, self._trf)
        reward = 0

        ## Update positions and portfolio value
        self._positions = action
        self._update_portfolio_value(action)

        ## Get observations of upcoming day to input to the model
        observation = self._get_observation()

        self._cur_end_time += 1

        terminated = self._cur_end_time > self._episode_end_time
        truncated = self._capital <= 0

        info = {}
        return observation, reward, terminated, truncated, info

    def get_relevant_close_prices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the close prices for the current and previous days."""
        close_column = self.COL_ADJ_CLOSE
        if self.COL_ADJ_CLOSE not in self._ohclv_data.columns:
            close_column = self.COL_CLOSE

        yesterday_close = self._ohclv_data[
            (self._ohclv_data[self.COL_TIME] == self._cur_end_time - 1)
        ][close_column].to_numpy()
        today_close = self._ohclv_data[
            (self._ohclv_data[self.COL_TIME] == self._cur_end_time)
        ][close_column].to_numpy()

        return yesterday_close, today_close

    def _get_relative_close_prices(self):
        """
        If an asset changes price from $100 to $120, this would be a 1.2 relative price change.
        First element is the relative change in cash, which is generally 1.
        y_t = p_t / p_{t-1}
        """
        yesterday_close, today_close = self.get_relevant_close_prices()
        rel = today_close / yesterday_close

        CASH_CHANGE = 0
        rel = np.append([1 + CASH_CHANGE], rel)

        return rel

    def _get_market_adjusted_positions(self) -> np.array:
        """Gets the updated position weights after close due to market movements during the day."""
        rel_prices = self._get_relative_close_prices()
        return self._positions * rel_prices / self._positions.dot(rel_prices)

    def _trf(self, action: np.array) -> float:
        """
        Calculates the transaction remainder factor (TRF) for the given action.
        """
        adj_positions = self._get_market_adjusted_positions()

        def f(mu):
            a = 1 / (1 - self._tcost * action[0])
            b = (
                1
                - self._tcost * adj_positions[0]
                - (2 * self._tcost - self._tcost**2)
                * np.maximum(adj_positions - mu * action, 0)[1:].sum()
            )
            return a * b

        EPS = 1e-6
        mu = self._tcost * np.abs((adj_positions - action)[1:]).sum()
        next_mu = f(mu)
        while abs(mu - next_mu) > EPS:
            mu, next_mu = next_mu, f(next_mu)

        return next_mu

    def _update_portfolio_value(self, action: np.array) -> float:
        """
        Updates the portfolio value based on the action of the agent using the transaction remainder factor.
        """

        mu = self._trf(action)
        self._portfolio_value = (
            self._portfolio_value
            * mu
            * (self._get_relative_close_prices().dot(self._positions))
        )

    def reward_function(self):

        close_prices = self._get_relative_close_prices()
        reward = np.log(close_prices.dot(self._positions))
        return reward


# def reward_function(env: TradingEnv) -> float:
#     u_t = np.ones(num_risky_assets + 1)
#     u_t[1:] = today_close / yesterday_close

#     momemtum_weights = (u_t * prev_positions) / (u_t.dot(prev_positions))
#     transaction_cost = (
#         today_close.dot(np.abs(momemtum_weights[1:] - curr_positions[1:]))
#         * transaction_percentage
#     )

#     reward = math.log((u_t * transaction_cost).dot(prev_positions) - transaction_cost)

#     return reward


# if __name__ == "__main__":
#     import yfinance as yf

#     def get_random_positions(num_assets) -> np.array:
#         positions = np.random.random(num_assets + 1)
#         positions /= positions.sum()
#         return positions

#     tickers = yf.Tickers("MSFT AAPL GOOG")
#     data = tickers.download()
#     data = data[["Open", "High", "Low", "Close", "Volume"]]
#     flattened_data = (
#         data.stack(level=1)  # Move tickers to rows
#         .reset_index()  # Reset index to convert to a flat DataFrame
#         .rename(columns={"level_1": "Ticker"})  # Rename the column for tickers
#     )
#     flattened_data["Time"] = pd.factorize(flattened_data["Date"])[0]

#     my_env = TradingEnv(
#         flattened_data, 3, 5, 1000, 0.03, lambda: -999, ["MSFT", "AAPL", "GOOG"]
#     )

#     while True:
#         positions = list(get_random_positions(3))
#         _, _, terminated, truncated, _ = my_env.step(positions)
#         if terminated:
#             break
