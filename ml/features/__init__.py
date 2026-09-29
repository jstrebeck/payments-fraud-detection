"""Feature engineering shared by training and online inference.

Pure functions only: no I/O, no database, no network (see README.md).
"""

from ml.features.build import FEATURE_NAMES, FEATURE_VERSION, build_features
from ml.features.context import LOOKBACK, MAX_HISTORY, CardContext, PastTransaction

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "LOOKBACK",
    "MAX_HISTORY",
    "CardContext",
    "PastTransaction",
    "build_features",
]
