# gates.py
import numpy as np
from typing import List, Optional

class QuantumGates:
    """
    A class containing definitions of standard quantum gates.
    """

    @staticmethod
    def hadamard() -> np.ndarray:
        """
        Create a Hadamard gate.

        Returns:
        np.ndarray: The Hadamard gate matrix.
        """
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2)

    @staticmethod
    def pauli_x() -> np.ndarray:
        """
        Create a Pauli-X gate (NOT gate).

        Returns:
        np.ndarray: The Pauli-X gate matrix.
        """
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    @staticmethod
    def pauli_y() -> np.ndarray:
        """
        Create a Pauli-Y gate.

        Returns:
        np.ndarray: The Pauli-Y gate matrix.
        """
        return np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)

    @staticmethod
    def pauli_z() -> np.ndarray:
        """
        Create a Pauli-Z gate.

        Returns:
        np.ndarray: The Pauli-Z gate matrix.
        """
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    @staticmethod
    def identity() -> np.ndarray:
        """
        Create an identity gate.

        Returns:
        np.ndarray: The identity gate matrix.
        """
        return np.eye(2, dtype=np.complex128)

    @staticmethod
    def cnot() -> np.ndarray:
        """
        Create a CNOT gate (Controlled-NOT).

        Returns:
        np.ndarray: The CNOT gate matrix.
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.complex128)

    @staticmethod
    def swap() -> np.ndarray:
        """
        Create a SWAP gate.

        Returns:
        np.ndarray: The SWAP gate matrix.
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.complex128)

    @staticmethod
    def toffoli() -> np.ndarray:
        """
        Create a Toffoli gate (Controlled-Controlled-NOT).

        Returns:
        np.ndarray: The Toffoli gate matrix.
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ], dtype=np.complex128)

    @staticmethod
    def controlled_phase(phi: float) -> np.ndarray:
        """
        Create a Controlled-Phase gate.

        Parameters:
        phi (float): The phase angle in radians.

        Returns:
        np.ndarray: The Controlled-Phase gate matrix.
        """
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, np.exp(1j * phi)]
        ], dtype=np.complex128)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        """
        Create a rotation around the X-axis.

        Parameters:
        theta (float): The rotation angle in radians.

        Returns:
        np.ndarray: The RX gate matrix.
        """
        return np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=np.complex128)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        """
        Create a rotation around the Y-axis.

        Parameters:
        theta (float): The rotation angle in radians.

        Returns:
        np.ndarray: The RY gate matrix.
        """
        return np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=np.complex128)

    @staticmethod
    def rz(theta: float) -> np.ndarray:
        """
        Create a rotation around the Z-axis.

        Parameters:
        theta (float): The rotation angle in radians.

        Returns:
        np.ndarray: The RZ gate matrix.
        """
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)

    @staticmethod
    def custom_gate(matrix: np.ndarray) -> np.ndarray:
        """
        Create a custom gate from a given matrix.

        Parameters:
        matrix (np.ndarray): The matrix representing the custom gate.

        Returns:
        np.ndarray: The custom gate matrix.
        """
        if matrix.shape != (2, 2):
            raise ValueError("Custom gate matrix must be a 2x2 matrix.")
        return matrix.astype(np.complex128)