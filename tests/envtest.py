import unittest
from environment import TradingEnv, load_data


class TradingEnvTests(unittest.TestCase):
    """
    Objectives for further test cases (more to be added later):
    - do datatype checks on every column to ensure every column contains only data type
    - must return float reward
    - env.step must return tuple of not NaN values (tuple must be valid size >= 4)
        - each value in tuple must not be None
    - observation must be dataframe of floats
    - trf must be float
    - ensure no NaN values (can either preprocessed beforehand or keep this as an edge case - to be discussed further)
    - reset must reset values to defaults (0, None, etc. depending on the value)
    - ensure capital does not fall below min threshold
    """
    environment = None

    def test_initialization(self):
        # a demonstration of usage if anything

        symbols = ["AAPL", "VOD"]
        ohclv = load_data(
            symbols,
            start_date="2020-10-01",
            end_date="2020-11-05",
        )
        TradingEnvTests.environment = TradingEnv(
            ohclv,
            num_risky_assets=len(symbols),
            window_len=5,
            initial_capital=1000,
            transaction_cost=0,
            reward_function=lambda x: 1,
            episode_len=10,
            index_to_id={x: i for i, x in enumerate(symbols)},
            seed=123,
        )
        self.assertIsInstance(TradingEnvTests.environment, TradingEnv)

    def test_public_methods(self):
        assert TradingEnvTests.environment.reset()
        assert TradingEnvTests.environment.get_relevant_close_prices()
        assert TradingEnvTests.environment.reward_function(0.1, [0.1]*50)



if __name__ == "__main__":
    unittest.main()
