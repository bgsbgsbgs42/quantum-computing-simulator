# cli.py
import argparse
from typing import List, Optional
from quantum_circuit import QuantumCircuit
from gates import QuantumGates
from visualization import QuantumVisualizer
from algorithms import grover_algorithm, quantum_fourier_transform, shor_algorithm

def main():
    """
    Main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Quantum Computing Simulator CLI")

    # General options
    parser.add_argument('--qubits', type=int, default=2, help="Number of qubits in the circuit")
    parser.add_argument('--algorithm', choices=['grover', 'qft', 'shor'], help="Quantum algorithm to run")
    parser.add_argument('--oracle', type=int, help="Oracle target for Grover's algorithm")
    parser.add_argument('--visualize', action='store_true', help="Visualize the quantum state after execution")
    parser.add_argument('--circuit', action='store_true', help="Display the quantum circuit diagram")

    # Gate application options
    parser.add_argument('--apply-gate', type=str, help="Apply a quantum gate (e.g., hadamard, cnot)")
    parser.add_argument('--qubits-gate', type=int, nargs='+', help="Qubits to apply the gate to")

    # Noise options
    parser.add_argument('--noise', action='store_true', help="Enable noise simulation")
    parser.add_argument('--depolarizing', type=float, default=0.01, help="Depolarizing noise probability")
    parser.add_argument('--amplitude-damping', type=float, default=0.01, help="Amplitude damping parameter")
    parser.add_argument('--phase-damping', type=float, default=0.01, help="Phase damping parameter")

    args = parser.parse_args()

    # Initialize the quantum circuit
    circuit = QuantumCircuit(args.qubits)

    # Apply noise if enabled
    if args.noise:
        from noise_model import NoiseModel
        noise_model = NoiseModel()
        noise_model.add_all_qubit_channel(DepolarizingChannel(args.depolarizing))
        noise_model.add_qubit_channel(0, AmplitudeDampingChannel(args.amplitude_damping))
        noise_model.add_qubit_channel(1, PhaseDampingChannel(args.phase_damping))
        circuit.noise_model = noise_model

    # Apply a gate if specified
    if args.apply_gate:
        gate_name = args.apply_gate.lower()
        qubits = args.qubits_gate if args.qubits_gate else [0]  # Default to qubit 0 if not specified

        # Get the gate from QuantumGates
        gate = getattr(QuantumGates, gate_name)()
        circuit.apply_gate(gate, qubits)

        print(f"Applied {gate_name} gate to qubits {qubits}")

    # Run the specified algorithm
    if args.algorithm:
        if args.algorithm == 'grover':
            if args.oracle is None:
                print("Error: Oracle target is required for Grover's algorithm")
                return
            result = grover_algorithm(circuit, args.oracle)
            print(f"Grover's algorithm result: {result} (binary: {format(result, f'0{args.qubits}b')})")

        elif args.algorithm == 'qft':
            quantum_fourier_transform(circuit, list(range(args.qubits)))
            result = circuit.measure()
            print(f"Quantum Fourier Transform result: {result} (binary: {format(result, f'0{args.qubits}b')})")

        elif args.algorithm == 'shor':
            factors = shor_algorithm(15)  # Example: Factorize 15
            print(f"Shor's algorithm result: 15 = {factors[0]} × {factors[1]}")

    # Visualize the quantum state if requested
    if args.visualize:
        visualizer = QuantumVisualizer(circuit)
        visualizer.plot_quantum_state()

    # Display the quantum circuit diagram if requested
    if args.circuit:
        visualizer = QuantumVisualizer(circuit)
        visualizer.plot_quantum_circuit()

if __name__ == "__main__":
    main()