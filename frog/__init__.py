"""
FROG: Frequency-Resolved Optical Gating Algorithm Package

This package provides implementations of FROG algorithms for pulse characterization:
- PCGP (Principal Component Generalized Projections) algorithm
- Vanilla FROG algorithm
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

from .core.pcgp import PCGPAlgorithm
from .core.vanilla import VanillaFROG
from .utils.trace_operations import padded_trace
from .utils.plotting import plot_frog_results, calculate_fwhm

__all__ = ['PCGPAlgorithm', 'VanillaFROG', 'padded_trace', 'plot_frog_results', 'calculate_fwhm']