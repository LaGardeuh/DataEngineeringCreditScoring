"""
Credit Scoring Package

Modules:
- data_prep: Data loading, cleaning, and preprocessing functions
- model_utils: Model training and evaluation utilities
- metrics: Business and technical metrics calculation
- explainability: SHAP-based model interpretability
"""

__version__ = "1.0.0"
__author__ = "Noé G"

from . import data_prep
from . import model_utils
from . import metrics
from . import explainability

__all__ = [
    'data_prep',
    'model_utils',
    'metrics',
    'explainability'
]
