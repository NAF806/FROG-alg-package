"""Utility functions for trace operations."""

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


def preprocess_trace(trace, noise_floor=None):
    """Preprocess experimental trace data.
    
    Parameters
    ----------
    trace : array
        Raw experimental trace
    noise_floor : float, optional
        Noise floor value to subtract
        
    Returns
    -------
    array
        Preprocessed trace
    """
    # Remove baseline noise
    if noise_floor is None:
        trace = trace - np.min(trace)
    else:
        trace = trace - noise_floor
    
    # Set negative values to zero
    trace[trace < 0] = 0
    
    # Normalize
    trace = trace / np.max(trace)
    
    return trace