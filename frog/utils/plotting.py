"""Plotting utilities for FROG results."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calculate_fwhm(x, y):
    """Calculate the FWHM of a pulse.
    
    Parameters
    ----------
    x : array
        The x-axis data (time or frequency)
    y : array
        The intensity profile
        
    Returns
    -------
    float or None
        The FWHM value
    """
    # Normalize the data
    y_norm = y / np.max(y)
    
    # Find the indices where the intensity is closest to 0.5 (half maximum)
    half_max_indices = np.where(np.diff(np.signbit(y_norm - 0.5)))[0]
    
    # If we have at least two crossings
    if len(half_max_indices) >= 2:
        # Use linear interpolation to find more precise positions
        x1, x2 = x[half_max_indices[0]], x[half_max_indices[-1]]
        fwhm = abs(x2 - x1)
        return fwhm
    else:
        print("Could not find two crossings at half maximum.")
        return None


def gaussian(x, A, mu, sigma, offset):
    """Gaussian function for fitting."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


def fit_gaussian(x, y):
    """Fit a Gaussian to data.
    
    Parameters
    ----------
    x : array
        X-axis data
    y : array
        Y-axis data
        
    Returns
    -------
    popt : array
        Optimal parameters (A, mu, sigma, offset)
    pcov : array
        Parameter covariance matrix
    fwhm : float
        FWHM calculated from Gaussian fit
    """
    # Initial guesses
    A0 = np.max(y)
    mu0 = x[np.argmax(y)]
    sigma0 = (x[-1] - x[0]) / 10
    offset0 = np.min(y)
    p0 = [A0, mu0, sigma0, offset0]
    
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0)
        # Calculate FWHM from Gaussian parameters
        fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
        return popt, pcov, fwhm
    except:
        return None, None, None


def plot_frog_results(retrieved_pulse, recovered_trace, exp_trace, t, freq=None):
    """Plot FROG retrieval results.
    
    Parameters
    ----------
    retrieved_pulse : array
        Retrieved pulse
    recovered_trace : array
        Recovered FROG trace
    exp_trace : array
        Experimental FROG trace
    t : array
        Time axis
    freq : array, optional
        Frequency axis for spectral plots
    """
    # Temporal domain plot
    plt.figure(figsize=(15, 10))
    
    # FROG traces comparison
    plt.subplot(2, 3, 1)
    plt.imshow(exp_trace, aspect='auto', cmap='viridis', 
               origin='lower', extent=[t[0], t[-1], t[0], t[-1]])
    plt.title('Experimental FROG Trace', fontsize=14)
    plt.xlabel('Time (fs)', fontsize=12)
    plt.ylabel('Delay (fs)', fontsize=12)
    
    plt.subplot(2, 3, 2)
    plt.imshow(recovered_trace.T, aspect='auto', cmap='inferno',
               origin='lower', extent=[t[0], t[-1], t[0], t[-1]])
    plt.title('Recovered FROG Trace', fontsize=14)
    plt.xlabel('Time (fs)', fontsize=12)
    plt.ylabel('Delay (fs)', fontsize=12)
    
    # Temporal pulse
    plt.subplot(2, 3, 3)
    intensity = np.abs(retrieved_pulse)**2
    phase = np.unwrap(np.angle(retrieved_pulse))
    
    ax1 = plt.gca()
    ax1.plot(t, intensity, 'r-', label='Temporal Intensity')
    ax1.set_xlabel('Time (fs)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    ax2 = ax1.twinx()
    ax2.plot(t, phase, 'g-', label='Phase')
    ax2.set_ylabel('Phase (rad)', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Retrieved Temporal Pulse', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Fit and FWHM for temporal pulse
    plt.subplot(2, 3, 4)
    popt_temp, _, fwhm_temp = fit_gaussian(t, intensity)
    plt.plot(t, intensity, 'r-', label='Temporal Intensity')
    if popt_temp is not None:
        plt.plot(t, gaussian(t, *popt_temp), 'b--', 
                label=f'Gaussian Fit (FWHM = {fwhm_temp:.2f} fs)')
    plt.xlabel('Time (fs)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Temporal Intensity with Fit', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Spectral analysis
    if freq is not None:
        plt.subplot(2, 3, 5)
        spectrum = np.fft.fftshift(np.fft.fft(retrieved_pulse))
        spectral_intensity = np.abs(spectrum)**2
        spectral_intensity /= np.max(spectral_intensity)
        
        plt.plot(freq, spectral_intensity, 'b-', label='Spectral Intensity')
        plt.xlabel('Frequency (THz)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Spectral Intensity', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to wavelength
        c = 3e8  # Speed of light (m/s)
        lambda0 = 1030e-9  # Central wavelength (m)
        f0 = c / lambda0
        
        # Create wavelength axis
        valid_freq = freq[freq > 0]
        wavelength = 1e9 * c / (f0 + valid_freq * 1e12)
        spectral_int_wavelength = spectral_intensity[freq > 0]
        
        plt.subplot(2, 3, 6)
        plt.plot(wavelength, spectral_int_wavelength, 'r-', label='Spectral Intensity')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Spectral Intensity (Wavelength)', fontsize=14)
        plt.xlim(800, 1200)  # Typical range around 1030 nm
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.show()