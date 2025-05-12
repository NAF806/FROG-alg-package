"""Example usage of the PCGP algorithm."""

import numpy as np
import matplotlib.pyplot as plt
from frog.core.pcgp import PCGPAlgorithm
from frog.utils.trace_operations import padded_trace, preprocess_trace
from frog.utils.plotting import plot_frog_results


def main():
    # Load experimental data
    exp_trace = np.loadtxt('/Users/nihalfaiz/Documents/4th Year Project/CODE/preprocessed trace (2).txt')
    
    # Preprocess the trace
    exp_trace = preprocess_trace(exp_trace, noise_floor=20)
    
    # Pad the trace if necessary
    N = 256 * 5.734  # Calibration factor
    N = int(np.round(N/2) * 2)
    if exp_trace.shape[0] < N:
        pad_size = N - exp_trace.shape[0]
        exp_trace = padded_trace(exp_trace, pad_size, pad_size)
    
    # Create time axis
    t = np.linspace(-N//2, N//2-1, N)
    
    # Initial guess: Gaussian pulse with a small quadratic phase (chirp)
    initial_pulse = np.exp(-t**2/(N/10)**2).astype(np.complex128)
    initial_pulse *= np.exp(1j * 0.2 * t**2/(N/10)**2)
    
    # Initialize the PCGP algorithm
    pcgp = PCGPAlgorithm()
    
    # Progress callback function
    def progress_callback(iteration, max_iter, error):
        print(f"Progress: {iteration}/{max_iter}, Error: {error:.6f}")
    
    # Run the algorithm
    print("Running PCGP algorithm...")
    recovered_pulse, recovered_trace, t, error = pcgp.retrieve_pulse(
        initial_pulse,
        exp_trace.T,  # Transpose for correct orientation
        max_iter=50,
        tolerance=1e-8,
        progress_callback=progress_callback
    )
    
    # Normalize the recovered trace for display
    recovered_trace = recovered_trace / np.max(recovered_trace)
    
    # Plot results
    plot_frog_results(
        recovered_pulse,
        recovered_trace,
        exp_trace,
        t
    )
    
    print(f"Final error: {error:.6f}")
    
    # Save results
    np.savetxt('recovered_pulse.txt', np.column_stack((t, np.abs(recovered_pulse), 
                                                      np.angle(recovered_pulse))))
    np.savetxt('recovered_trace.txt', recovered_trace)


if __name__ == "__main__":
    main()