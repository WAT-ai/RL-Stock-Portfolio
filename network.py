import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network with softmax for the actor.
    """
    def __init__(self, in_dim, out_dim, is_actor=False):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            out_dim - output dimensions as an int (number of assets for actor or scalar for critic)
            is_actor - whether the network is an actor (True) or a critic (False)

        Return:
            None
        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.is_actor = is_actor  # Flag to differentiate actor and critic

    def forward(self, obs):
        """
        Runs a forward pass on the neural network.

        Parameters:
            obs - observation to pass as input

        Return:
            output - the output of our forward pass (softmax for actor, linear for critic)
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        # if self.is_actor:
        #     # Apply softmax for actor network to ensure weights sum to 1
        #     output = F.softmax(output, dim=-1)

        return output
