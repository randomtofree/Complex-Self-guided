# CSPSA Projects: Complex Self-guided Methods for Quantum Optimization

A Python implementation of Complex Self-guided Parameter Space Analysis (CSPSA) and Self-guided Algorithm (SGA) methods for quantum optimization, specifically designed for instrumental inequality violations in quantum systems.

## Overview

This project implements and compares two optimization algorithms for quantum parameter estimation:

- **CSPSA (Complex Self-guided Parameter Space Analysis)**: A complex-valued optimization method
- **SGA (Self-guided Algorithm)**: A real-valued optimization approach
- **CCSPSA (Continuous CSPSA)**: A continuous version of CSPSA

The algorithms are applied to quantum Bell inequality violations and instrumental inequality tests, with comprehensive analysis of their performance under different conditions.

## Features

- **Quantum State Optimization**: Support for various quantum states (Bell states, random mixed states)
- **Multiple Test Scenarios**: CHSH inequality and instrumental inequality violations
- **Performance Analysis**: Comprehensive comparison of algorithm efficiency
- **Visualization**: Detailed plotting capabilities for analysis results
- **Noise Modeling**: Support for Poisson noise, shot noise, and measurement uncertainties

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `qutip>=5.0.0` - Quantum toolbox for Python
- `numpy>=1.20.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Plotting and visualization

## Project Structure

```
CSPSAProjects/
├── algorithms/           # Core algorithm implementations
│   ├── cspsa2.py        # CSPSA algorithm
│   ├── sga2.py          # SGA algorithm
│   └── simulation_utils2.py  # Utility functions
├── data_plots/          # Jupyter notebooks for analysis
│   ├── fig_2.ipynb     # Figure 2 generation
│   ├── fig_3.ipynb     # Figure 3 generation
│   └── fig_4.ipynb     # Figure 4 generation
├── requirements.txt     # Project dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## Usage

### Running Simulations

1. **Basic Algorithm Comparison**:
   ```python
   from algorithms.cspsa2 import run_instrumental_cspsa_simulation
   from algorithms.sga2 import run_instrumental_sga_simulation
   
   # Configure your parameters
   config = {
       'state': your_quantum_state,
       'iterations': 300,
       'hparams': {'a': 1.0, 's': 1.0, 'b': 0.25, 'r': 1/6.0},
       'photon_num': 1000,
       'state_variation': 0.0,
       'uncertainty': 0.0
   }
   
   # Run simulation
   results = run_instrumental_cspsa_simulation(config)
   ```

2. **Generate Analysis Figures**:
   - Open `data_plots/fig_2.ipynb` for efficiency comparison
   - Open `data_plots/fig_3.ipynb` for hyperparameter analysis
   - Open `data_plots/fig_4.ipynb` for additional analysis

### Key Parameters

- **a, s, b, r**: Hyperparameters controlling algorithm behavior
- **photon_num**: Number of photons for measurement simulation
- **state_variation**: Quantum state preparation uncertainty
- **uncertainty**: Measurement uncertainty

## Algorithm Details

### CSPSA (Complex Self-guided Parameter Space Analysis)
- Complex-valued parameter optimization
- Suitable for quantum state parameter estimation
- Handles complex measurement operators

### SGA (Self-guided Algorithm)
- Real-valued parameter optimization
- Traditional gradient-based approach
- Efficient for real parameter spaces

### CCSPSA (Continuous CSPSA)
- Continuous version of CSPSA
- Enhanced exploration of parameter space
- Improved convergence properties

## Research Applications

This project is designed for:
- Quantum parameter estimation
- Bell inequality violation studies
- Instrumental inequality analysis
- Quantum optimization algorithm comparison
- Quantum state preparation optimization

## Citation

If you use this software in your research, please cite the relevant papers and acknowledge this implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or collaboration, please contact the project maintainers.