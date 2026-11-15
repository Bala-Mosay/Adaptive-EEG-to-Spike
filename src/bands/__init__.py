"""Adaptive frequency-band selection (Phase 3, Contribution 1)."""

from .bank import SEVEN_BANDS
from .knn_mi import band_mi_scores
from .selector import AdaptiveBandSelector

__all__ = ["SEVEN_BANDS", "band_mi_scores", "AdaptiveBandSelector"]
