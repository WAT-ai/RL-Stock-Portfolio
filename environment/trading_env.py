from collections.abc import Callable
import gymnasium as gym
import pandas as pd
import numpy as np
from typing import Any, NewType

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
        capital: float,
        transaction_cost: float,
        reward_function: Callable,
        index_to_id: list[str],
    ) -> None:
        """
        Initializes the trading environment.

        Args:
            ohclv_data (pd.DataFrame): Input OHCLV data.
            num_risky_assets (int): Number of risky assets (e.g., stocks).
            window_len (int): Length of the observation window.
            capital (float): Initial capital in the portfolio.
            transaction_cost (float): Transaction cost as a fraction of capital.
            reward_function (Callable): Function to compute the reward.
            index_to_id (list[str]): List mapping indices to asset identifiers.
        """
        self._ohclv_data = ohclv_data
        self._num_risky_assets = num_risky_assets
        self._positions = np.zeros(num_risky_assets + 1)  # +1 for cash
        self._positions[0] = 1
        self._window_len = window_len
        self._original_capital = capital
        self._capital = capital
        self._tcost = transaction_cost
        self._reward_function = reward_function
        self._index_to_id = index_to_id

        self._cur_end_time = window_len - 1
        super().__init__()

    def _get_observation(self) -> ObsType:
        """
        Retrieves the current observation window of OHCLV data.

        Returns:
            ObsType: A subset of the OHCLV data representing the current window.
        """
        observation = self._ohclv_data[
            (self._cur_end_time - self._window_len + 1 <= self._ohclv_data[self.COL_TIME])
            & (self._ohclv_data[self.COL_TIME] <= self._cur_end_time)
        ]
        return observation

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

        reward = self._reward_function()
        self._positions = action
        self._capital = self._get_capital(self._positions)
        observation = self._get_observation()
        self._cur_end_time += 1

        terminated = self._cur_end_time > self._ohclv_data[self.COL_TIME].max()
        truncated = self._capital <= 0

        info = {}
        return observation, reward, terminated, truncated, info

    def _get_capital(self, positions):
        """
        Calculates the current capital based on positions and market returns.

        Args:
            positions (np.ndarray): Current portfolio allocation.

        Returns:
            float: Updated capital after applying returns and transaction costs.
        """
        close_column = self.COL_ADJ_CLOSE
        if self.COL_ADJ_CLOSE not in self._ohclv_data.columns:
            close_column = self.COL_CLOSE

        if self._cur_end_time == 0:
            return self._capital

        yesterday_close = self._ohclv_data[(self._ohclv_data[self.COL_TIME] == self._cur_end_time - 1)][close_column].to_numpy()
        today_close = self._ohclv_data[(self._ohclv_data[self.COL_TIME] == self._cur_end_time)][close_column].to_numpy()
        returns = np.zeros(self._num_risky_assets + 1)
        returns[1:] = (today_close - yesterday_close) / yesterday_close

        capital = (self._capital * (1 + returns) * positions * (1 - self._tcost)).sum()
        return capital

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
        self._cur_end_time = 0
        self._capital = self._original_capital
        self._positions = [0] * self._num_risky_assets
        self._positions[0] = 1  # First index represents cash value

        return self._get_observation(), {}


if __name__ == "__main__":
    import yfinance as yf

    def get_random_positions(num_assets) -> np.array:
        positions = np.random.random(num_assets + 1)
        positions /= positions.sum()
        return positions

    tickers = yf.Tickers("MSFT AAPL GOOG")
    data = tickers.download()
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    flattened_data = (
        data.stack(level=1)  # Move tickers to rows
        .reset_index()  # Reset index to convert to a flat DataFrame
        .rename(columns={"level_1": "Ticker"})  # Rename the column for tickers
    )
    flattened_data["Time"] = pd.factorize(flattened_data["Date"])[0]

    my_env = TradingEnv(
        flattened_data, 3, 5, 1000, 0, lambda: -999, ["MSFT", "AAPL", "GOOG"]
    )

    while True:
        positions = list(get_random_positions(3))
        _, _, terminated, truncated, _ = my_env.step(positions)
        if terminated:
            break
