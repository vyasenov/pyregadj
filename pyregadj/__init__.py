"""
PyRegAdj - Treatment effects estimation for randomized controlled trials.

A Python package for estimating average treatment effects (ATE) in randomized
experiments using various adjustment methods including regression adjustment
and machine learning approaches.
"""

from .regadjrct import RegAdjustRCT
from .constants import *

__all__ = ["RegAdjustRCT"] 