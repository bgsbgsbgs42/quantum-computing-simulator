# grover_example.py
from quantum_circuit import QuantumCircuit
from gates import QuantumGates
from visualization import QuantumVisualizer
from algorithms import grover_algorithm

def run_grover_example():
    """
    Example script to demonstrate Grover's algorithm.
    """
    # Number of qubits
    num_qubits = 3

    # Oracle target (the state we want to find)
    oracle_target = 5  # Binary: 101

    # Initialize the quantum circuit
    circuit = QuantumCircuit(num_qubits)

    # Run Grover's algorithm
    print(f"Running Grover's algorithm with {num_qubits} qubits and oracle target {oracle_target}...")
    result = grover_algorithm(circuit, oracle_target)

    # Print the result
    print(f"Grover's algorithm result: {result} (binary: {format(result, f'0{num_qubits}b')})")

    # Visualize the final quantum state
    visualizer = QuantumVisualizer(circuit)
    visualizer.plot_quantum_state(title="Quantum State After Grover's Algorithm")

    # Display the circuit diagram
    visualizer.plot_quantum_circuit()

if __name__ == "__main__":
    run_grover_example()