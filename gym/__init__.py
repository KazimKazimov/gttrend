"""Trend-detection gym: real-time evaluation of 10Y trend detectors that use other variables."""
from .data import Panel
from .env import TrendGym, TrendEnv, DetectorAgent, Result
from .detectors import (RuleDetector, CompositeRule, LogitDetector, GBMDetector, MSDetector, Climatology)
from .metrics import score, compare
from . import labels

__all__ = ["Panel", "TrendGym", "TrendEnv", "DetectorAgent", "Result", "RuleDetector", "CompositeRule",
           "LogitDetector", "GBMDetector", "MSDetector", "Climatology", "score", "compare", "labels"]
