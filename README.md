# FROG: Frequency-Resolved Optical Gating

A Python package for pulse characterization using FROG (Frequency-Resolved Optical Gating) algorithms.

## Features

- **PCGP Algorithm**: Principal Component Generalized Projections algorithm for FROG pulse retrieval
- **Vanilla FROG Algorithm**: Standard FROG algorithm implementation
- **Utility Functions**: Trace preprocessing, padding, and plotting utilities
- **Examples**: Ready-to-use example scripts for both algorithms

## Installation

### From source

```bash
git clone https://github.com/yourusername/frog.git
cd frog
pip install -e .
```

### Using pip

```bash
pip install .
```

## Quick Start

### Using the PCGP Algorithm

```python
from frog import PCGPAlgorithm
import numpy as np

# Load your experimental FROG trace
exp_trace = np.loadtxt('your_frog_trace.txt')

# Create initial pulse guess
N = exp_trace.shape[0]
t = np.linspace(-N//2, N//2-1, N)
initial_pulse = np.exp(-t**2/(N/10)**2).astype(np.complex128)

# Initialize and run PCGP
pcgp = PCGPAlgorithm()
recovered_pulse, recovered_trace, t, error = pcgp.retrieve_pulse(
    initial_pulse,
    exp_trace,
    max_iter=50,
    tolerance=1e-8
)
```

### Using the Vanilla FROG Algorithm

```python
from frog import VanillaFROG
import numpy as np

# Initialize vanilla FROG
vanilla_frog = VanillaFROG()

# Create computational grid
time, taus, freq, dt = vanilla_frog.grid(1000, 1000, 5679)

# Create initial pulse guess
initial_pulse = np.exp(-2*np.log(2) * (time/150)**2)

# Run the algorithm
retrieved_pulse, error_history = vanilla_frog.retrieve_pulse(
    initial_pulse,
    exp_trace,
    taus,
    time,
    freq,
    iterations=100
)
```

### Using the Examples

The package includes example scripts in the form of a Jupyter Notebook that can be saved and run: 

```
Example.ipynb
```

Note: There is an example experimental trace called 'preprocessed trace.txt'.

## Package Structure

```
frog/
├── core/           # Core algorithm implementations
│   ├── pcgp.py     # PCGP algorithm
│   └── vanilla.py  # Vanilla FROG algorithm
├── utils/          # Utility functions
│   ├── trace_operations.py  # Trace preprocessing and padding
│   └── plotting.py          # Plotting and analysis functions
├── Example.ipynb
└──preprocessed trace.txt
```

## API Reference

### PCGPAlgorithm

The main class for the PCGP algorithm:

```python
pcgp = PCGPAlgorithm()
pulse, trace, t, error = pcgp.retrieve_pulse(initial_pulse, exp_trace)
```

### VanillaFROG

The main class for the Vanilla FROG algorithm:

```python
vanilla = VanillaFROG()
pulse, error_history = vanilla.retrieve_pulse(initial_pulse, exp_trace, taus, time, freq)
```

### Utility Functions

- `padded_trace(trace, l, w)`: Pad a trace with zeros
- `preprocess_trace(trace, noise_floor=None)`: Preprocess experimental trace data
- `calculate_fwhm(x, y)`: Calculate the FWHM of a pulse
- `plot_frog_results(...)`: Plot comprehensive FROG results


## Citation

If you use this package in your research, please cite:

```
@software{frog_package,
  author = {Nihal Faiz},
  title = {FROG ALGORITHM IMPLEMENTATIONS},
  year = {2024},
  url = {https://github.com/NAF806/frog}
}
```

## References

1. Trebino, R. (2000). Frequency-Resolved Optical Gating: The Measurement of Ultrashort Laser Pulses. Springer.
2. DeLong, K. W., et al. (1994). "Practical issues in ultrashort-laser-pulse measurement using frequency-resolved optical gating." IEEE Journal of Quantum Electronics.