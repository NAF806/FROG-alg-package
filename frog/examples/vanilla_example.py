"""Example usage of the Vanilla FROG algorithm."""

import numpy as np
import matplotlib.pyplot as plt
from frog.core.vanilla import VanillaFROG
from frog.utils.trace_operations import padded_trace, preprocess_trace
from frog.utils.plotting import plot_frog_results, calculate_fwhm


def main():
    # Load experimental data
    exp_trace = np.loadtxt('/Users/nihalfaiz/Documents/4th Year Project/CODE/preprocessed trace (2).txt')
    
    # Preprocess the trace
    exp_trace = preprocess_trace(exp_trace, noise_floor=26)
    
    # Create computational grid
    vanilla_frog = VanillaFROG()
    time, taus, freq, dt = vanilla_frog.grid(1000, 1000, 5679)
    
    # Pad the trace if necessary
    if exp_trace.shape[0] < 1000:
        pad_size = 1000 - exp_trace.shape[0]
        exp_trace = padded_trace(exp_trace, pad_size, pad_size)
    
    # Create initial guess
    # Option 1: Gaussian pulse with chirp
    fwhm = 150  # fs
    wavelength = 1030  # nm
    c = 300  # speed of light (nm/fs)
    omega = 2 * np.pi * c / wavelength
    
    pulse = np.exp(-2 * np.log(2) * (time/fwhm)**2)
    pulse = pulse * np.exp(1j * omega * time)
    initial_pulse = pulse
    
    # Option 2: Random initial guess (uncomment to use)
    # initial_pulse = np.random.rand(1000) + 1j * np.random.rand(1000)
    
    # Run the algorithm
    print("Running Vanilla FROG algorithm...")
    retrieved_pulse, error_history = vanilla_frog.retrieve_pulse(
        initial_pulse,
        exp_trace,
        taus,
        time,
        freq,
        iterations=100,
        plot_every=20
    )
    
    # Calculate pulse properties
    temporal_intensity = np.abs(retrieved_pulse)**2
    temporal_fwhm = calculate_fwhm(time, temporal_intensity)
    print(f"Temporal FWHM: {temporal_fwhm:.2f} fs")
    
    # Calculate spectral properties
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(retrieved_pulse)))
    spectral_intensity = np.abs(spectrum)**2
    spectral_fwhm = calculate_fwhm(freq, spectral_intensity)
    print(f"Spectral FWHM: {spectral_fwhm:.2f} THz")
    
    # Convert spectral FWHM to nm
    central_wavelength = 1030  # nm
    c_ms = 3e8  # speed of light in m/s
    central_freq = c_ms / (central_wavelength * 1e-9)  # Hz
    lambda1 = c_ms / ((central_freq + spectral_fwhm * 1e12/2)) * 1e9  # nm
    lambda2 = c_ms / ((central_freq - spectral_fwhm * 1e12/2)) * 1e9  # nm
    spectral_fwhm_nm = abs(lambda2 - lambda1)
    print(f"Spectral FWHM: {spectral_fwhm_nm:.2f} nm")
    
    # Final visualization
    plot_frog_results(
        retrieved_pulse,
        exp_trace,  # Using exp_trace as "recovered" for visualization
        exp_trace,
        time,
        freq
    )
    
    # Plot error history
    plt.figure(figsize=(8, 6))
    plt.plot(error_history)
    plt.xlabel('Iteration')
    plt.ylabel('FROG Error')
    plt.title('Convergence History')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Save results
    np.savetxt('retrieved_pulse_vanilla.txt', 
               np.column_stack((time, np.abs(retrieved_pulse), np.angle(retrieved_pulse))))


if __name__ == "__main__":
    main()