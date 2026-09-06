"""input models describe how probable inputs are for given modalities of input.

.. autosummary::
    :toctree:
    
    src.input_models.LuceShepardChoice
"""
from irlpreference.input_models._luce_shepard_choice import LuceShepardChoice
from irlpreference.input_models._weak_preference_choice import WeakPreferenceChoice

__all__ = [
    "LuceShepardChoice",
    "WeakPreferenceChoice",
]