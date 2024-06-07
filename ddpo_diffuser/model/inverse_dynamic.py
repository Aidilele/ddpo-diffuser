import torch
from torch import nn


class LinearInverseDynamic(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, x):
        return self.model(x)
