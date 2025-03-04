import numpy as np
from scipy.sparse import kron, eye, csr_matrix
from typing import List, Dict, Optional
import random

class QuantumChannel:
    """
    Base class for quantum channels.
    """
    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the quantum channel to a density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix after applying the channel.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class DepolarizingChannel(QuantumChannel):
    """
    Depolarizing channel: Replaces the quantum state with the maximally mixed state
    with probability p.
    """
    def __init__(self, probability: float):
        """
        Initialize the depolarizing channel.

        Parameters:
            probability (float): Probability of depolarizing (0 to 1).
        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the depolarizing channel to a density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix.
        """
        dim = density_matrix.shape[0]
        identity = np.eye(dim, dtype=np.complex128) / dim  # Maximally mixed state
        return (1 - self.probability) * density_matrix + self.probability * identity

class AmplitudeDampingChannel(QuantumChannel):
    """
    Amplitude damping channel: Models energy dissipation/relaxation (T1 decay).
    """
    def __init__(self, gamma: float):
        """
        Initialize the amplitude damping channel.

        Parameters:
            gamma (float): Damping parameter (0 to 1).
        """
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1.")
        self.gamma = gamma

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the amplitude damping channel to a density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix.
        """
        # Kraus operators for amplitude damping
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]], dtype=np.complex128)
        E1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]], dtype=np.complex128)

        # Apply Kraus operators
        result = E0 @ density_matrix @ E0.conj().T + E1 @ density_matrix @ E1.conj().T
        return result

class PhaseDampingChannel(QuantumChannel):
    """
    Phase damping channel: Models dephasing (T2 decay).
    """
    def __init__(self, lambda_param: float):
        """
        Initialize the phase damping channel.

        Parameters:
            lambda_param (float): Phase damping parameter (0 to 1).
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError("Lambda must be between 0 and 1.")
        self.lambda_param = lambda_param

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the phase damping channel to a density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix.
        """
        # Kraus operators for phase damping
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.lambda_param)]], dtype=np.complex128)
        E1 = np.array([[0, 0], [0, np.sqrt(self.lambda_param)]], dtype=np.complex128)

        # Apply Kraus operators
        result = E0 @ density_matrix @ E0.conj().T + E1 @ density_matrix @ E1.conj().T
        return result

class BitFlipChannel(QuantumChannel):
    """
    Bit flip channel: Flips a qubit with probability p.
    """
    def __init__(self, probability: float):
        """
        Initialize the bit flip channel.

        Parameters:
            probability (float): Bit flip probability (0 to 1).
        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.probability = probability

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the bit flip channel to a density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix.
        """
        # Kraus operators for bit flip
        E0 = np.sqrt(1 - self.probability) * np.eye(2, dtype=np.complex128)
        E1 = np.sqrt(self.probability) * np.array([[0, 1], [1, 0]], dtype=np.complex128)

        # Apply Kraus operators
        result = E0 @ density_matrix @ E0.conj().T + E1 @ density_matrix @ E1.conj().T
        return result

class NoiseModel:
    """
    Noise model for a quantum computer, combining various error channels.
    """
    def __init__(self):
        """
        Initialize the noise model.
        """
        self.qubit_channels: Dict[int, List[QuantumChannel]] = {}
        self.gate_errors: Dict[str, float] = {}
        self.all_qubit_channel: Optional[QuantumChannel] = None

    def add_qubit_channel(self, qubit: int, channel: QuantumChannel) -> None:
        """
        Add a quantum channel to a specific qubit.

        Parameters:
            qubit (int): Target qubit.
            channel (QuantumChannel): Quantum channel to add.
        """
        if qubit not in self.qubit_channels:
            self.qubit_channels[qubit] = []
        self.qubit_channels[qubit].append(channel)

    def add_all_qubit_channel(self, channel: QuantumChannel) -> None:
        """
        Add a quantum channel to all qubits.

        Parameters:
            channel (QuantumChannel): Quantum channel to add.
        """
        self.all_qubit_channel = channel

    def add_gate_error(self, gate_name: str, error_rate: float) -> None:
        """
        Add an error rate to a specific gate type.

        Parameters:
            gate_name (str): Name of the gate ('h', 'x', 'cnot', etc.).
            error_rate (float): Probability of gate error (0 to 1).
        """
        if not 0 <= error_rate <= 1:
            raise ValueError("Error rate must be between 0 and 1.")
        self.gate_errors[gate_name] = error_rate

    def apply_qubit_channels(self, density_matrix: np.ndarray) -> np.ndarray:
        """
        Apply all channels to the density matrix.

        Parameters:
            density_matrix (np.ndarray): Input density matrix.

        Returns:
            np.ndarray: Output density matrix with noise applied.
        """
        result = density_matrix.copy()
        n_qubits = int(np.log2(density_matrix.shape[0]))

        # Apply qubit-specific channels
        for qubit in range(n_qubits):
            if qubit in self.qubit_channels:
                for channel in self.qubit_channels[qubit]:
                    result = channel.apply(result)

        # Apply all-qubit channel if defined
        if self.all_qubit_channel:
            result = self.all_qubit_channel.apply(result)

        return result

    def apply_gate_error(self, gate_name: str) -> bool:
        """
        Determine if a gate error occurs based on the error rate.

        Parameters:
            gate_name (str): Name of the gate.

        Returns:
            bool: True if an error occurs, False otherwise.
        """
        if gate_name in self.gate_errors:
            return random.random() < self.gate_errors[gate_name]
        return False