"""
Experimental models — subject to change without notice.

Import these explicitly to signal that you accept the instability:

    from deeptab.models.experimental import ModernNCA, Tangos, Trompt
"""

from .modern_nca import ModernNCAClassifier, ModernNCALSS, ModernNCARegressor
from .tangos import TangosClassifier, TangosLSS, TangosRegressor
from .trompt import TromptClassifier, TromptLSS, TromptRegressor

__all__ = [
    "ModernNCAClassifier",
    "ModernNCALSS",
    "ModernNCARegressor",
    "TangosClassifier",
    "TangosLSS",
    "TangosRegressor",
    "TromptClassifier",
    "TromptLSS",
    "TromptRegressor",
]
