"""
imperfekt: A framework to analyze imperfections in time-series datasets.

This library provides tools for:
- Intravariable analysis of missing data patterns
- Intervariable analysis of co-occurring missingness
- Feature generation based on missingness patterns
"""

from imperfekt.analysis.imperfekt import Imperfekt
from imperfekt.analysis.intervariable.intervariable import IntervariableImperfection
from imperfekt.analysis.intravariable.intravariable import IntravariableImperfection
from imperfekt.features.core import FeatureGenerator

__version__ = "0.2.0"
__all__ = [
    "Imperfekt",
    "IntravariableImperfection",
    "IntervariableImperfection",
    "FeatureGenerator",
    "__version__",
]
