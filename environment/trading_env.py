from collections.abc import Callable
import gymnasium as gym
import pandas as pd
from typing import NewType

ActType = NewType("ActType", list[int])


class TradingEnv(gym.Env):
    def __init__(
        self,
        ohclv_data: pd.DataFrame,
        num_assets: int,
        window_len: int,
        capital: float,
        transaction_cost: float,
        reward_function: Callable,
        index_to_id: list[str],
    ) -> None:
        """
        ohclv_data
        num_assets
        window_len
        capital
        index_to_id
        """

        self._ohclv_data = ohclv_data
        self._positions = [0] * num_assets
        self._positions[0] = 1
        self._window_len = window_len
        self._capital = capital
        self._tcost = transaction_cost
        self._reward_function = reward_function
        self._index_to_id = index_to_id

        self._cur_row = 0
        super().__init__()

    def step(
        self, action: list[float]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert len(action) == len(self._positions)

        reward = self._reward_function()

        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def close(self):
        return super().close()
