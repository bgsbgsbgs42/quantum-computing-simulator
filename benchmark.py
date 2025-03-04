# benchmark.py
import time
import numpy as np
from quantum_circuit import QuantumCircuit
from gates import QuantumGates

def benchmark_quantum_circuit(num_qubits: int, num_gates: int):
    """
    Benchmark the performance of the quantum circuit.

    Parameters:
    num_qubits (int): Number of qubits in the circuit.
    num_gates (int): Number of gates to apply.
    """
    print(f"Benchmarking quantum circuit with {num_qubits} qubits and {num_gates} gates...")

    # Initialize the quantum circuit
    circuit = QuantumCircuit(num_qubits)

    # Apply random gates
    start_time = time.time()
    for _ in range(num_gates):
        gate = QuantumGates.hadamard()
        qubit = np.random.randint(0, num_qubits)
        circuit.apply_gate(gate, [qubit])
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def benchmark_gate_application(num_qubits: int, num_trials: int):
    """
    Benchmark the performance of gate application.

    Parameters:
    num_qubits (int): Number of qubits in the circuit.
    num_trials (int): Number of trials to run.
    """
    print(f"Benchmarking gate application with {num_qubits} qubits and {num_trials} trials...")

    # Initialize the quantum circuit
    circuit = QuantumCircuit(num_qubits)

    # Benchmark gate application
    start_time = time.time()
    for _ in range(num_trials):
        gate = QuantumGates.hadamard()
        qubit = np.random.randint(0, num_qubits)
        circuit.apply_gate(gate, [qubit])
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def benchmark_measurement(num_qubits: int, num_trials: int):
    """
    Benchmark the performance of quantum measurement.

    Parameters:
    num_qubits (int): Number of qubits in the circuit.
    num_trials (int): Number of trials to run.
    """
    print(f"Benchmarking measurement with {num_qubits} qubits and {num_trials} trials...")

    # Initialize the quantum circuit
    circuit = QuantumCircuit(num_qubits)

    # Benchmark measurement
    start_time = time.time()
    for _ in range(num_trials):
        circuit.measure()
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    # Benchmark quantum circuit with 10 qubits and 1000 gates
    benchmark_quantum_circuit(num_qubits=10, num_gates=1000)

    # Benchmark gate application with 10 qubits and 1000 trials
    benchmark_gate_application(num_qubits=10, num_trials=1000)

    # Benchmark measurement with 10 qubits and 1000 trials
    benchmark_measurement(num_qubits=10, num_trials=1000)