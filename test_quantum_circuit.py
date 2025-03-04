# test_quantum_circuit.py
import unittest
import numpy as np
from quantum_circuit import QuantumCircuit
from gates import QuantumGates

class TestQuantumCircuit(unittest.TestCase):
    """
    Unit tests for the QuantumCircuit class.
    """

    def test_initialization(self):
        """
        Test the initialization of the quantum circuit.
        """
        num_qubits = 2
        circuit = QuantumCircuit(num_qubits)
        self.assertEqual(circuit.num_qubits, num_qubits)
        self.assertTrue(np.allclose(circuit.state, np.array([[1], [0], [0], [0]])))

    def test_hadamard_gate(self):
        """
        Test the application of the Hadamard gate.
        """
        circuit = QuantumCircuit(1)
        circuit.apply_gate(QuantumGates.hadamard(), [0])
        expected_state = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
        self.assertTrue(np.allclose(circuit.state, expected_state))

    def test_cnot_gate(self):
        """
        Test the application of the CNOT gate.
        """
        circuit = QuantumCircuit(2)
        circuit.apply_gate(QuantumGates.hadamard(), [0])
        circuit.apply_gate(QuantumGates.cnot(), [0, 1])
        expected_state = np.array([[1 / np.sqrt(2)], [0], [0], [1 / np.sqrt(2)]])
        self.assertTrue(np.allclose(circuit.state, expected_state))

    def test_pauli_x_gate(self):
        """
        Test the application of the Pauli-X gate.
        """
        circuit = QuantumCircuit(1)
        circuit.apply_gate(QuantumGates.pauli_x(), [0])
        expected_state = np.array([[0], [1]])
        self.assertTrue(np.allclose(circuit.state, expected_state))

    def test_measurement(self):
        """
        Test the measurement of a quantum state.
        """
        circuit = QuantumCircuit(1)
        circuit.apply_gate(QuantumGates.hadamard(), [0])
        outcome = circuit.measure()
        self.assertIn(outcome, [0, 1])

    def test_measure_qubit(self):
        """
        Test the measurement of a specific qubit.
        """
        circuit = QuantumCircuit(2)
        circuit.apply_gate(QuantumGates.hadamard(), [0])
        outcome = circuit.measure_qubit(0)
        self.assertIn(outcome, [0, 1])

    def test_to_density_matrix(self):
        """
        Test the conversion of a pure state to a density matrix.
        """
        circuit = QuantumCircuit(1)
        circuit.apply_gate(QuantumGates.hadamard(), [0])
        density_matrix = circuit.to_density_matrix()
        expected_density_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertTrue(np.allclose(density_matrix, expected_density_matrix))

if __name__ == "__main__":
    unittest.main()