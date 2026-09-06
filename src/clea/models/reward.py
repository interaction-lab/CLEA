"""
Reward learning models.

These lightweight MLP heads take an embedding and output a scalar reward value,
used to evaluate the quality of learned representations via preference queries.
"""

import torch.nn as nn


class RewardLearner(nn.Module):
    """
    MLP that maps an embedding vector to a scalar reward.
    Used to evaluate representation quality: a better representation should allow
    this model to learn human preferences from fewer queries.
    """

    def __init__(self, input_dim=1024, hidden_dim=1024, device='cuda'):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.encoder(x)
