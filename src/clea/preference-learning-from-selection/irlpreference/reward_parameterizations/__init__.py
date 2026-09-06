"""Reward Parameterizations describe how the human's reward function is modeled.
They implement a mapping from omega to a reward

.. autosummary::
    :toctree:

    src.reward_parameterizations.MonteCarloLinearReward
"""
from irlpreference.reward_parameterizations._linear_reward import MonteCarloLinearReward


__all__ = [
    "MonteCarloLinearReward",
]
