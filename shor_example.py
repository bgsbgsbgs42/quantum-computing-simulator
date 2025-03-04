# shor_example.py
from quantum_circuit import QuantumCircuit
from gates import QuantumGates
from visualization import QuantumVisualizer
from algorithms import shor_algorithm

def run_shor_example():
    """
    Example script to demonstrate Shor's algorithm.
    """
    # Number to factorize
    n = 15  # Example: Factorize 15

    # Run Shor's algorithm
    print(f"Running Shor's algorithm to factorize {n}...")
    factors = shor_algorithm(n)

    # Print the result
    if factors[0] == 1:
        print(f"Failed to factorize {n}.")
    else:
        print(f"Shor's algorithm result: {n} = {factors[0]} × {factors[1]}")

    # Initialize a quantum circuit for visualization (optional)
    circuit = QuantumCircuit(4)  # Example circuit with 4 qubits
    visualizer = QuantumVisualizer(circuit)

    # Visualize a simple quantum state (for demonstration)
    visualizer.plot_quantum_state(title="Quantum State After Shor's Algorithm")

if __name__ == "__main__":
    run_shor_example()