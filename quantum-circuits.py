import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from typing import List, Tuple, Optional
import gc

class QuantumCircuit:
    """
    A quantum circuit simulator with optimized state and gate operations.
    Supports both pure states (state vectors) and mixed states (density matrices).
    """

    def __init__(self, num_qubits: int, use_sparse: bool = False):
        """
        Initialize a quantum circuit with the specified number of qubits.

        Parameters:
            num_qubits (int): Number of qubits in the circuit.
            use_sparse (bool): If True, use sparse matrices for state representation.
        """
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")

        self.num_qubits = num_qubits
        self.use_sparse = use_sparse
        self.state = self.initialize_state()
        self.gates = []  # Stores applied gates for visualization or debugging

    def initialize_state(self) -> np.ndarray:
        """
        Initialize the quantum state to |0...0⟩.

        Returns:
            np.ndarray: Initial quantum state vector or density matrix.
        """
        if self.use_sparse:
            state = csr_matrix((2 ** self.num_qubits, 1), dtype=np.complex128)
            state[0, 0] = 1.0
        else:
            state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
            state[0] = 1.0
            state = state.reshape(-1, 1)
        return state

    def apply_gate(self, gate: np.ndarray, qubits: List[int]) -> None:
        """
        Apply a quantum gate to specific qubits.

        Parameters:
            gate (np.ndarray): Gate matrix.
            qubits (List[int]): List of qubits to apply the gate to.
        """
        if not all(0 <= qubit < self.num_qubits for qubit in qubits):
            raise ValueError("Qubit index out of range.")

        full_gate = self._create_full_gate(gate, qubits)
        if self.use_sparse:
            self.state = full_gate @ self.state
        else:
            self.state = np.dot(full_gate, self.state)

        # Free memory for large matrices
        del full_gate
        gc.collect()

    def _create_full_gate(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """
        Create the full gate matrix by tensoring with identity matrices.

        Parameters:
            gate (np.ndarray): Gate matrix.
            qubits (List[int]): List of qubits to apply the gate to.

        Returns:
            np.ndarray: Full gate matrix.
        """
        if len(qubits) == 1:
            return self._create_single_qubit_gate(gate, qubits[0])
        elif len(qubits) == 2:
            return self._create_two_qubit_gate(gate, qubits[0], qubits[1])
        else:
            raise NotImplementedError("Multi-qubit gates with >2 qubits are not supported.")

    def _create_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """
        Create a full gate matrix for a single-qubit gate.

        Parameters:
            gate (np.ndarray): Single-qubit gate matrix.
            qubit (int): Qubit to apply the gate to.

        Returns:
            np.ndarray: Full gate matrix.
        """
        if self.use_sparse:
            identity = eye(2, dtype=np.complex128, format='csr')
            result = eye(1, dtype=np.complex128, format='csr')
        else:
            identity = np.eye(2, dtype=np.complex128)
            result = np.array([[1.0]], dtype=np.complex128)

        for i in range(self.num_qubits):
            if i == qubit:
                result = kron(result, gate)
            else:
                result = kron(result, identity)

        return result

    def _create_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """
        Create a full gate matrix for a two-qubit gate.

        Parameters:
            gate (np.ndarray): Two-qubit gate matrix.
            qubit1 (int): First qubit.
            qubit2 (int): Second qubit.

        Returns:
            np.ndarray: Full gate matrix.
        """
        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1

        if self.use_sparse:
            identity = eye(2, dtype=np.complex128, format='csr')
            result = eye(1, dtype=np.complex128, format='csr')
        else:
            identity = np.eye(2, dtype=np.complex128)
            result = np.array([[1.0]], dtype=np.complex128)

        for i in range(self.num_qubits):
            if i == qubit1:
                result = kron(result, gate)
            elif i == qubit2:
                continue  # Skip second qubit as it's already included in the gate
            else:
                result = kron(result, identity)

        return result

    def measure(self) -> int:
        """
        Measure the quantum state and return the outcome.

        Returns:
            int: Measurement outcome.
        """
        if self.use_sparse:
            probabilities = np.abs(self.state.toarray().flatten()) ** 2
        else:
            probabilities = np.abs(self.state.flatten()) ** 2

        outcome = np.random.choice(range(2 ** self.num_qubits), p=probabilities)

        # Update the state to the measured state
        if self.use_sparse:
            new_state = csr_matrix((2 ** self.num_qubits, 1), dtype=np.complex128)
            new_state[outcome, 0] = 1.0
        else:
            new_state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
            new_state[outcome] = 1.0
            new_state = new_state.reshape(-1, 1)

        self.state = new_state
        return outcome

    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a specific qubit and update the quantum state.

        Parameters:
            qubit (int): Qubit to measure.

        Returns:
            int: Measurement outcome (0 or 1).
        """
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range.")

        # Calculate probabilities for the qubit to be |0⟩ or |1⟩
        prob_0 = 0.0
        prob_1 = 0.0

        if self.use_sparse:
            state_array = self.state.toarray().flatten()
        else:
            state_array = self.state.flatten()

        for i in range(2 ** self.num_qubits):
            binary = format(i, f'0{self.num_qubits}b')
            if binary[self.num_qubits - 1 - qubit] == '0':
                prob_0 += abs(state_array[i]) ** 2
            else:
                prob_1 += abs(state_array[i]) ** 2

        # Normalize probabilities
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob

        # Measure the qubit
        outcome = np.random.choice([0, 1], p=[prob_0, prob_1])

        # Update the state
        if self.use_sparse:
            new_state = csr_matrix((2 ** self.num_qubits, 1), dtype=np.complex128)
        else:
            new_state = np.zeros((2 ** self.num_qubits, 1), dtype=np.complex128)

        for i in range(2 ** self.num_qubits):
            binary = format(i, f'0{self.num_qubits}b')
            if binary[self.num_qubits - 1 - qubit] == str(outcome):
                if outcome == 0:
                    new_state[i, 0] = state_array[i] / np.sqrt(prob_0) if prob_0 > 0 else 0
                else:
                    new_state[i, 0] = state_array[i] / np.sqrt(prob_1) if prob_1 > 0 else 0

        self.state = new_state
        return outcome

    def to_density_matrix(self) -> np.ndarray:
        """
        Convert the pure state to a density matrix representation.

        Returns:
            np.ndarray: Density matrix.
        """
        if self.use_sparse:
            state = self.state.toarray()
        else:
            state = self.state
        return np.outer(state, np.conj(state))