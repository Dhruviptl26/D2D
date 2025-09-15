"""
D2D Machine Learning Package

A comprehensive machine learning framework for data-driven decision making.
"""

__version__ = "0.1.0"
__author__ = "D2D Team"
__email__ = "team@d2d.com"

from d2d.data import DataLoader, DataPreprocessor
from d2d.models import MLModel, ModelFactory
from d2d.training import Trainer
from d2d.evaluation import Evaluator
from d2d.inference import Predictor

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "MLModel",
    "ModelFactory",
    "Trainer",
    "Evaluator",
    "Predictor"
]