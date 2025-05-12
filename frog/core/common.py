"""Common utilities shared between FROG algorithms."""

import numpy as np


def padded_trace(trace, l, w):
    """Pad a trace with zeros.
    
    Parameters
    ----------
    trace : array
        Input trace
    l : int
        Number of rows to add
    w : int
        Number of columns to add
        
    Returns
    -------
    array
        Padded trace
    """
    n = trace.shape[0]
    new_rows = n + l 
    new_cols = n + w
    padded = np.zeros((new_rows, new_cols))

    start_row = (new_rows - n) // 2
    start_col = (new_cols - n) // 2

    padded[start_row:start_row+n, start_col:start_col+n] = trace
    return padded


def generate_simulated_pulse(N, chirp=0.5, duration=10, third_order=0.0):
    """Generate a simulated pulse with specified parameters.
    
    Parameters
    ----------
    N : int
        Number of time points
    chirp : float
        Chirp parameter (phase quadratic term coefficient)
    duration : float
        Pulse duration (relative to N)
    third_order : float
        Third-order dispersion term
        
    Returns
    -------
    array
        Complex pulse array
    """
    t = np.linspace(-N//2, N//2-1, N)
    # Gaussian envelope with quadratic phase (chirp) and optional third-order phase
    amplitude = np.exp(-t**2/(N/duration)**2)
    phase = chirp * t**2/(N/duration)**2 + third_order * t**3/(N/duration)**3
    pulse = amplitude * np.exp(1j * phase)
    return pulse