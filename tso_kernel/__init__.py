"""
TSO Kernel — Pure math/neuro kernel with no external NLP dependencies.

Builds on NumPy only. Designed for validation, testability, and future
C++/CUDA porting.
"""
from .neurons import LIFCluster
from .plasticity import RSTDPPlasticity, EligibilityTrace
from .friction import FrictionCalculator
from .operators import TopographicOperator
from .core import TSOCore
