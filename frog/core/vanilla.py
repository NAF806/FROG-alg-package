"""Vanilla FROG algorithm implementation."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift


class VanillaFROG:
    """Vanilla FROG algorithm for pulse retrieval."""
    
    def __init__(self):
        self.error_history = []
    
    @staticmethod
    def grid(N, N_tau, range_val):
        """Create a computational grid.
        
        Parameters
        ----------
        N : int
            Number of time points
        N_tau : int
            Number of delay points
        range_val : float
            Time range
            
        Returns
        -------
        time : array
            Time axis
        taus : array
            Delay axis
        freq : array
            Frequency axis
        dt : float
            Time step
        """
        time = np.linspace(-range_val, range_val, N)
        taus = np.linspace(-range_val * 0.08, range_val * 0.08, N_tau)
        dt = time[1] - time[0]
        freq = np.fft.fftshift(np.fft.fftfreq(time.shape[-1], 
                              d=np.mean(np.diff(time*1e-15))))*1e-12
        return time, taus, freq, dt 
    
    def compute_frog_trace(self, E, taus, time, freq, plot=False, 
                          signal_trace=False, check=False, trace=False):
        """Compute the FROG trace for a given electric field.
        
        Parameters
        ----------
        E : array
            Electric field pulse
        taus : array
            Delay values
        time : array
            Time axis
        freq : array
            Frequency axis
        plot : bool
            Whether to plot intermediate results
        signal_trace : bool
            Whether to plot the signal trace
        check : bool
            Whether to perform checks
        trace : bool
            Whether to plot the FROG trace
            
        Returns
        -------
        s : array
            Signal field
        spectrum : array
            Spectral field
        intensity : array
            FROG trace intensity
        """
        spectrum = []
        s = []
        
        for i, tau in enumerate(taus):
            roll_index = np.nanargmin(np.abs(time)) - np.nanargmin(np.abs(time + tau))
            rolled_pulse = np.roll(E, roll_index)
            
            if plot:
                plt.plot(time, E, color='b')
                plt.plot(time, rolled_pulse, color='r')
                plt.show()
            
            Sig = E * rolled_pulse
            s.append(Sig)
            
            spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Sig)))
            
            if check:
                plt.plot(freq, spec)
                plt.xlim(0, 1000)
                plt.ylim(-100, 600000)
                plt.show()
                
            spectrum.append(spec)
        
        spectrum = np.array(spectrum)
        intensity = np.abs(spectrum)**2
        intensity = intensity / np.max(intensity)
        
        s = np.array(s)
        
        if signal_trace:
            xx, yy = np.meshgrid(time, taus)
            plt.pcolormesh(xx, yy, s.real)
            plt.xlim(-30, 30)
            plt.ylabel('Delay (fs)')
            plt.xlabel('Time (fs)')
            plt.title('SIGNAL TRACE')
            plt.show()
        
        if trace:
            XX, YY = np.meshgrid(taus, freq)
            plt.pcolormesh(XX, YY, intensity.T)
            plt.ylabel('Delay (fs)')
            plt.xlabel('Frequency (THz)')
            plt.title('FROG TRACE')
            plt.show()
        
        return s, spectrum, intensity
    
    def retrieve_pulse(self, initial_pulse, exp_data, taus, time, freq, 
                      iterations=100, geometry="SHG", plot_every=20):
        """Run the vanilla FROG algorithm.
        
        Parameters
        ----------
        initial_pulse : array
            Initial guess for the pulse
        exp_data : array
            Experimental FROG trace
        taus : array
            Delay values
        time : array
            Time axis
        freq : array
            Frequency axis
        iterations : int
            Number of iterations
        geometry : str
            FROG geometry type
        plot_every : int
            Plot results every N iterations
            
        Returns
        -------
        pulse : array
            Retrieved pulse
        error_history : list
            Error history over iterations
        """
        pulse = initial_pulse.copy()
        error_history = []
        
        for i in range(iterations):
            # Step 1: Generate signal field based on current E(t)
            signal, spectrum, intensity = self.compute_frog_trace(pulse, taus, time, freq)
            
            # Calculate FROG error
            mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
            error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
            error_history.append(error)
            
            if i % plot_every == 0:
                print(f"Iteration {i}, FROG error: {error:.6f}")
                
                plt.figure(figsize=(12, 4))
                
                plt.subplot(131)
                plt.title(f"Retrieved E(t), iter {i}")
                plt.plot(time, np.abs(pulse), label='Intensity')
                plt.plot(time, np.angle(pulse), label='Phase')
                plt.legend()
                
                plt.subplot(132)
                plt.title("Experimental FROG trace")
                plt.pcolormesh(taus, freq, exp_data.T, shading='auto')
                
                plt.subplot(133)
                plt.title(f"Retrieved FROG trace, error={error:.6f}")
                plt.pcolormesh(taus, freq, intensity.T, shading='auto', cmap='inferno')
                
                plt.tight_layout()
                plt.show()
            
            # Step 2: Apply data constraint
            amp = np.abs(spectrum)
            phase = np.angle(spectrum)
            epsilon = 1e-10  # Avoid division by zero
            new_spectrum = np.sqrt(exp_data + epsilon) * np.exp(1j * phase)
            
            # Step 3: Inverse Fourier transform to get E'sig(t, τ)
            new_signal = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(new_spectrum, axes=1), axis=1), axes=1)
            
            # Step 4: Generate new E(t) by integration over τ
            pulse = np.sum(new_signal, axis=0)
            
            # Normalize the pulse
            pulse = pulse / np.max(np.abs(pulse))
        
        # Final result
        signal, spectrum, intensity = self.compute_frog_trace(pulse, taus, time, freq)
        mu = np.sum(exp_data * intensity) / np.sum(intensity**2)
        final_error = np.sqrt(np.mean((exp_data - mu * intensity)**2))
        
        print(f"Final FROG error after {iterations} iterations: {final_error:.6f}")
        
        # Final visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title("Experimental FROG trace")
        plt.pcolormesh(taus, freq, exp_data.T, shading='auto')
        
        plt.subplot(122)
        plt.title(f"Retrieved FROG trace, error={final_error:.6f}")
        plt.pcolormesh(taus, freq, intensity.T, shading='auto', cmap='inferno')
        
        plt.tight_layout()
        plt.show()
        
        self.error_history = error_history
        return pulse, error_history