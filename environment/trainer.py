import torch
from torch.utils.data import DataLoader
from dataset import StockOHLCVDataset
from trading_env import TradingEnv

class Trainer:
    def __init__(self, model):
        ...