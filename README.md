# Quantum Computing Simulator

A powerful and flexible quantum computing simulator implemented in Python, designed for educational purposes and quantum algorithm experimentation.

## Overview

This simulator provides a comprehensive environment for simulating quantum circuits and algorithms, with support for:

- Pure and mixed quantum states
- Common quantum gates (Hadamard, Pauli gates, CNOT, etc.)
- Noise modeling (decoherence and gate errors)
- Popular quantum algorithms (Grover's, Shor's, etc.)
- Visualization tools for quantum states and circuits

## Features

- **Efficient Quantum Circuit Simulation**: Optimized for both state vector and density matrix representations
- **Comprehensive Gate Library**: Includes all standard quantum gates and support for custom gates
- **Realistic Noise Modeling**: Simulates various quantum noise channels (depolarizing, amplitude damping, phase damping)
- **Quantum Algorithm Implementations**:
  - Grover's search algorithm
  - Shor's factoring algorithm
  - Simon's algorithm
  - Bernstein-Vazirani algorithm
  - Variational Quantum Eigensolver (VQE)
  - Quantum Approximate Optimization Algorithm (QAOA)
- **Visualization Tools**: Plot quantum states, density matrices, Bloch spheres, and circuit diagrams
- **Command-Line Interface**: Easy experimentation with quantum algorithms and circuits
- **Benchmarking Tools**: Evaluate performance with increasing qubit count and gate operations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-computing-simulator.git
cd quantum-computing-simulator

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- (Optional) QuTiP for advanced visualizations

## Quick Start

### Basic Circuit Operations

```python
from quantum_circuit import QuantumCircuit
from gates import QuantumGates
from visualization import QuantumVisualizer

# Create a 2-qubit circuit
circuit = QuantumCircuit(2)

# Apply Hadamard gate to the first qubit
circuit.apply_gate(QuantumGates.hadamard(), [0])

# Apply CNOT gate with control on qubit 0 and target on qubit 1
circuit.apply_gate(QuantumGates.cnot(), [0, 1])

# Measure the circuit
outcome = circuit.measure()
print(f"Measurement outcome: {outcome}")

# Visualize the quantum state
visualizer = QuantumVisualizer(circuit)
visualizer.plot_quantum_state()
```

### Running Grover's Algorithm

```python
from algorithms import grover_algorithm
from quantum_circuit import QuantumCircuit

# Initialize a 3-qubit circuit
circuit = QuantumCircuit(3)

# Specify the target state (in decimal)
oracle_target = 5  # Binary: 101

# Run Grover's algorithm
result = grover_algorithm(circuit, oracle_target)
print(f"Grover's algorithm found: {result} (binary: {format(result, '03b')})")
```

### Running with Noise Simulation

```python
from quantum_circuit import QuantumCircuit
from noise_model import NoiseModel, DepolarizingChannel, AmplitudeDampingChannel

# Create a circuit
circuit = QuantumCircuit(2)

# Create a noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_channel(DepolarizingChannel(0.01))
noise_model.add_qubit_channel(0, AmplitudeDampingChannel(0.05))

# Apply gates
circuit.apply_gate(QuantumGates.hadamard(), [0])
circuit.apply_gate(QuantumGates.cnot(), [0, 1])

# Apply noise to the quantum state
noisy_state = noise_model.apply_qubit_channels(circuit.state)
circuit.state = noisy_state

# Measure the noisy state
outcome = circuit.measure()
print(f"Noisy measurement outcome: {outcome}")
```

### Using the Command-Line Interface

```bash
# Run Grover's algorithm with 3 qubits and oracle target 5
python cli.py --qubits 3 --algorithm grover --oracle 5 --visualize

# Apply a Hadamard gate to qubit 0 and visualize the circuit
python cli.py --qubits 2 --apply-gate hadamard --qubits-gate 0 --circuit

# Run with noise simulation
python cli.py --qubits 2 --apply-gate hadamard --qubits-gate 0 --noise --depolarizing 0.02 --visualize
```

## Module Overview

- `quantum_circuit.py`: Core quantum circuit simulator
- `gates.py`: Quantum gate definitions
- `noise_model.py`: Quantum noise channels and models
- `algorithms.py`: Implementations of quantum algorithms
- `visualization.py`: Tools for visualizing quantum states and circuits
- `benchmark.py`: Performance benchmarking utilities
- `cli.py`: Command-line interface
- `test_quantum_circuit.py`: Unit tests

## Examples

The repository includes several example scripts:

- `grover_example.py`: Demonstrates Grover's search algorithm
- `shor_example.py`: Demonstrates Shor's factoring algorithm

## Running Tests

```bash
python -m unittest test_quantum_circuit.py
```

## Benchmarking

You can benchmark the simulator's performance with:

```bash
python benchmark.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- This simulator was inspired by various quantum computing frameworks and educational resources
- Special thanks to the quantum computing community for their valuable insights and research
