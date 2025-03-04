# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from typing import List, Optional

class QuantumVisualizer:
    """
    A class for visualizing quantum states, density matrices, and quantum circuits.
    """

    def __init__(self, circuit):
        """
        Initialize the QuantumVisualizer with a quantum circuit.

        Parameters:
        circuit: The quantum circuit to visualize.
        """
        self.circuit = circuit

    def plot_quantum_state(self, title: str = "Quantum State Probabilities") -> None:
        """
        Plot the probabilities of each basis state in the quantum state.

        Parameters:
        title (str): The title of the plot.
        """
        state_flat = self.circuit.state.flatten()
        
        # Handle sparse matrices
        if isinstance(state_flat, csr_matrix):
            state_flat = state_flat.toarray().flatten()
        
        probabilities = np.abs(state_flat) ** 2
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(state_flat)), probabilities)
        
        # Label x-axis with binary representations
        num_qubits = int(np.log2(len(state_flat)))
        plt.xticks(range(len(state_flat)), [format(i, f'0{num_qubits}b') for i in range(len(state_flat))], rotation=45)
        
        plt.xlabel('Basis State')
        plt.ylabel('Probability')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_density_matrix(self, title: str = "Density Matrix") -> None:
        """
        Plot the real and imaginary parts of a density matrix.

        Parameters:
        title (str): The title of the plot.
        """
        density_matrix = self.circuit.to_density_matrix()
        
        # Handle sparse matrices
        if isinstance(density_matrix, csr_matrix):
            density_matrix = density_matrix.toarray()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot real part
        im1 = ax1.imshow(np.real(density_matrix), cmap='RdBu_r')
        ax1.set_title(f"{title} (Real Part)")
        plt.colorbar(im1, ax=ax1)
        
        # Plot imaginary part
        im2 = ax2.imshow(np.imag(density_matrix), cmap='RdBu_r')
        ax2.set_title(f"{title} (Imaginary Part)")
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

    def plot_quantum_circuit(self) -> None:
        """
        Visualize the quantum circuit as a text diagram.
        """
        gate_symbols = {
            'hadamard': 'H',
            'pauli_x': 'X',
            'pauli_y': 'Y',
            'pauli_z': 'Z',
            'cnot': 'CNOT',
            'toffoli': 'TOF',
            'swap': 'SWAP',
            'controlled_phase': 'CP',
            'rx': 'RX',
            'ry': 'RY',
            'rz': 'RZ'
        }
        
        # Create a list to represent each qubit's wire
        wires = [['|'] for _ in range(self.circuit.num_qubits)]
        
        # Add gate symbols to each wire
        for gate, qubits in self.circuit.gates:
            gate_name = gate.__class__.__name__.lower()
            symbol = gate_symbols.get(gate_name, 'G')  # Default to 'G' for generic gates
            
            max_len = max(len(wire) for wire in wires)
            for wire in wires:
                while len(wire) < max_len:
                    wire.append('-')
                    
            for qubit in qubits:
                wires[qubit].append(symbol)
                
            for i in range(len(wires)):
                if i not in qubits:
                    wires[i].append('-')
        
        # Add final wire segments
        max_len = max(len(wire) for wire in wires)
        for wire in wires:
            while len(wire) < max_len:
                wire.append('-')
            wire.append('|')
            
        # Print the circuit diagram
        for i, wire in enumerate(wires):
            print(f'Qubit {i}: {"".join(wire)}')

    def plot_bloch_sphere(self, qubit: int, title: str = "Bloch Sphere") -> None:
        """
        Visualize the state of a single qubit on the Bloch sphere.

        Parameters:
        qubit (int): The index of the qubit to visualize.
        title (str): The title of the plot.
        """
        from qutip import Bloch, Qobj

        # Extract the state of the specified qubit
        state = self.circuit.state
        num_qubits = self.circuit.num_qubits
        
        # Trace out all other qubits
        if num_qubits > 1:
            from qutip import partial_trace
            state_qobj = Qobj(state.reshape([2] * num_qubits))
            rho = partial_trace(state_qobj, list(range(num_qubits)) if qubit != 0 else partial_trace(state_qobj, list(range(1, num_qubits)))
        else:
            rho = Qobj(state)
        
        # Plot on the Bloch sphere
        b = Bloch()
        b.add_states(rho)
        b.show(title=title)

    def plot_histogram(self, measurements: List[int], title: str = "Measurement Histogram") -> None:
        """
        Plot a histogram of measurement outcomes.

        Parameters:
        measurements (List[int]): A list of measurement outcomes.
        title (str): The title of the plot.
        """
        unique, counts = np.unique(measurements, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts)
        plt.xlabel('Measurement Outcome')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.xticks(unique, [format(i, f'0{self.circuit.num_qubits}b') for i in unique], rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_gate_application(self, gate_name: str, qubits: List[int], title: str = "Gate Application") -> None:
        """
        Visualize the effect of applying a gate to specific qubits.

        Parameters:
        gate_name (str): The name of the gate.
        qubits (List[int]): The qubits to which the gate is applied.
        title (str): The title of the plot.
        """
        # Apply the gate to the circuit
        gate = getattr(self.circuit, gate_name)()
        self.circuit.apply_gate(gate, qubits)
        
        # Plot the resulting state
        self.plot_quantum_state(title=f"{title} - {gate_name} on Qubits {qubits}")

    def plot_noise_effects(self, noise_model, title: str = "Noise Effects") -> None:
        """
        Visualize the effects of noise on the quantum state.

        Parameters:
        noise_model: The noise model to apply.
        title (str): The title of the plot.
        """
        noisy_circuit = deepcopy(self.circuit)
        noisy_circuit.noise_model = noise_model
        
        # Apply noise and plot the resulting state
        noisy_circuit.apply_noise()
        self.plot_quantum_state(title=f"{title} - With Noise")