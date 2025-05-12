"""PCGP (Principal Component Generalized Projections) algorithm for FROG."""

import numpy as np


class PCGPAlgorithm:
    """Principal Component Generalized Projections algorithm for FROG pulse retrieval."""
    
    def __init__(self):
        self.pulse = None
        self.trace = None
        self.error_history = []
    
    @staticmethod
    def row_shift_minus(matrix):
        """Shift each row to the left by its row index."""
        shifted = np.zeros_like(matrix, dtype=complex)
        N = matrix.shape[0]
        for i in range(N):
            shifted[i, :] = np.roll(matrix[i, :], -i)
        return shifted
    
    @staticmethod
    def reverse_row_shift(matrix):
        """Reverse the row shifting operation."""
        shifted = np.zeros_like(matrix, dtype=complex)
        N = matrix.shape[0]
        for i in range(N):
            shifted[i, :] = np.roll(matrix[i, :], i)
        return shifted
    
    def FROG(self, pulse):
        """Compute the FROG trace from a pulse.
        
        Parameters
        ----------
        pulse : array
            Complex electric field pulse
            
        Returns
        -------
        intensity : array
            FROG trace intensity
        frequency_domain : array
            Frequency domain representation
        """
        pulse = pulse / np.max(np.abs(pulse))  # Normalize pulse amplitude
        
        # Create the outer product (signal field for each time delay)
        # For SHG-FROG, we use E(t)E(t-τ)
        outer_product = np.outer(pulse, pulse)
        
        # Apply the row shift to implement the time delay
        row_shifted = self.row_shift_minus(outer_product)
        
        # FFT along time axis (columns) to get the frequency axis
        frequency_domain = np.fft.fftshift(np.fft.fft(row_shifted, axis=1), axes=1)
        
        # FROG trace is the squared magnitude of the frequency domain
        intensity = np.abs(frequency_domain)**2
        
        return intensity, frequency_domain
    
    def data_constraint(self, freq_domain, exp_trace):
        """Apply the data constraint to the frequency domain.
        
        Parameters
        ----------
        freq_domain : array
            Current frequency domain estimate
        exp_trace : array
            Experimental FROG trace
            
        Returns
        -------
        array
            Updated frequency domain
        """
        exp_trace_normalized = exp_trace / np.max(exp_trace)
        
        # Add small value to avoid division by zero
        magnitude = np.sqrt(exp_trace_normalized + 1e-10)
        
        # Get the phase from the current estimate
        phase = np.angle(freq_domain)
        
        # Create new frequency domain with experimental magnitude and estimated phase
        new_freq = magnitude * np.exp(1j * phase)
        
        return new_freq
    
    def invert_FROG_transform(self, freq_domain):
        """Convert from frequency domain back to time domain."""
        time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain, axes=1), axis=1)
        return time_domain
    
    def pcgp_iteration(self, pulse, exp_trace):
        """Perform one iteration of the PCGP algorithm."""
        # Step 1: Generate outer product and FROG trace from the current pulse guess
        intensity, freq_domain = self.FROG(pulse)
        
        # Step 2: Apply the data constraint in frequency domain
        constrained_freq = self.data_constraint(freq_domain, exp_trace)
        
        # Step 3: Transform back to time domain
        time_domain = self.invert_FROG_transform(constrained_freq)
        
        # Step 4: Convert back to outer product form by reversing the row shift
        outer_product_rec = self.reverse_row_shift(time_domain)
        
        # Step 5: Apply SVD to get the rank-1 approximation (principal component)
        U, S, Vh = np.linalg.svd(outer_product_rec, full_matrices=True)
        new_pulse = U[:, 0] * np.sqrt(S[0])
        
        # Normalize the new pulse
        new_pulse = new_pulse / np.max(np.abs(new_pulse))
        
        return new_pulse
    
    def retrieve_pulse(self, initial_pulse, exp_trace, max_iter=100, tolerance=1e-6, 
                      progress_callback=None):
        """Run the PCGP algorithm until convergence or max iterations.
        
        Parameters
        ----------
        initial_pulse : array
            Initial guess for the pulse
        exp_trace : array
            Experimental FROG trace
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        progress_callback : callable
            Callback function for progress updates
            
        Returns
        -------
        pulse : array
            Retrieved pulse
        final_trace : array
            Final FROG trace
        t : array
            Time axis
        error : float
            Final error
        """
        pulse = initial_pulse.copy()
        self.error_history = []
        
        N = exp_trace.shape[0]
        t = np.linspace(-N//2, N//2-1, N)
        
        for i in range(max_iter):
            new_pulse = self.pcgp_iteration(pulse, exp_trace)
            error = np.linalg.norm(np.abs(new_pulse) - np.abs(pulse)) / np.linalg.norm(np.abs(pulse))
            
            current_trace, _ = self.FROG(new_pulse)
            trace_error = np.linalg.norm(current_trace - exp_trace) / np.linalg.norm(exp_trace)
            self.error_history.append(trace_error)
            
            pulse = new_pulse
            
            # Apply phase correction to avoid drift
            max_idx = np.argmax(np.abs(pulse)**2)
            phase_correction = np.angle(pulse[max_idx])
            pulse *= np.exp(-1j * phase_correction)
            
            # Update progress
            if progress_callback:
                progress_callback(i+1, max_iter, trace_error)
            
            if error < tolerance and trace_error < np.sqrt(tolerance):
                break
            if i % 10 == 0:
                print(f"Iteration {i}, Error: {error:.6f}")
                
        final_trace, _ = self.FROG(pulse)
        return pulse, final_trace, t, error