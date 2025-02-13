# test model
import torch.nn as nn
import torch.nn.functional as F

class FFNeuralNetwork(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


