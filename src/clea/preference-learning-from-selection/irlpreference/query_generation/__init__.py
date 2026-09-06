"""Query generations create queries to ask a user.

.. autosummary::
    :toctree:

    src.query_generation.RandomQueryGenerator
    src.query_generation.VolumeRemovalQueryGenerator
    src.query_generation.InfoGainQueryGenerator
"""
from irlpreference.query_generation._random_query import RandomQueryGenerator
from irlpreference.query_generation._information_gain_query import InfoGainQueryGenerator
from irlpreference.query_generation._volume_removal_query import VolumeRemovalQueryGenerator


__all__ = [
    "RandomQueryGenerator",
    "InfoGainQueryGenerator",
    "VolumeRemovalQueryGenerator"
]
