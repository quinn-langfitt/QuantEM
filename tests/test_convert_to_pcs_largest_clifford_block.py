'''
File for unit tests for the convert_to_PCS_circ_largest_clifford function. 
1. Class is for initial testing with unittest framework

2. Pytest fixtures are used for mock testing the function.
'''

import unittest
import pytest
import numpy as np
from qiskit import QuantumCircuit
from quantem.utils import convert_to_PCS_circ_largest_clifford
from qiskit_addon_utils.slicing import slice_by_depth
from quantem.utils import find_largest_clifford_block

def hydrogen_trial_circuit(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.x(1)

    qc.rx(np.pi / 2, 0)
    qc.h(1)
    qc.h(2)
    qc.h(3)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    qc.rz(1.0, 3)

    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    qc.rx(-np.pi / 2, 0)
    qc.h(1)
    qc.h(2)
    qc.h(3)

    return qc


'''
Initial test cases for convert_to_PCS_circ_largest_clifford function using unittest framework
'''
class InitialTestConvertToPCSLargestClifford(unittest.TestCase):

    def test_convert_to_PCS_largest_clifford_basic(self):
        num_qubits = 4
        num_checks = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.s(2)
        qc.t(0)  # non-Clifford gate
        qc.h(1)
        qc.cx(1, 2)
        qc.s(3)

        sign_list, protected_circ = convert_to_PCS_circ_largest_clifford(qc, num_qubits, num_checks)

        self.assertIsInstance(sign_list, list)
        self.assertIsInstance(protected_circ, QuantumCircuit)
        self.assertGreaterEqual(protected_circ.num_qubits, qc.num_qubits)

        slices = slice_by_depth(qc, 1)
        start, end, clifford_slices = find_largest_clifford_block(slices)
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)

        self.assertGreaterEqual(len(protected_circ.data), len(qc.data))
        self.assertLessEqual(len(sign_list), num_checks)

    def test_no_clifford_block_returns_original(self):
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.t(1)

        sign_list, protected_circ = convert_to_PCS_circ_largest_clifford(qc, 2, 1)

        self.assertIsNone(sign_list)
        self.assertEqual(protected_circ, qc)

    def test_all_clifford_block(self):
        num_qubits = 3
        num_checks = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.s(2)
        qc.cx(1, 2)
        qc.h(2)

        sign_list, protected_circ = convert_to_PCS_circ_largest_clifford(qc, num_qubits, num_checks)

        self.assertIsInstance(sign_list, list)
        self.assertIsInstance(protected_circ, QuantumCircuit)
        self.assertLessEqual(len(sign_list), num_checks)
        self.assertGreater(len(sign_list), 0)
        self.assertGreaterEqual(protected_circ.num_qubits, qc.num_qubits)
        self.assertGreaterEqual(len(protected_circ.data), len(qc.data))

    def test_hydrogen_trial_circuit(self):
        num_qubits = 4
        num_checks = 2
        qc = hydrogen_trial_circuit(num_qubits)

        sign_list, protected_circ = convert_to_PCS_circ_largest_clifford(qc, num_qubits, num_checks)

        self.assertIsInstance(sign_list, list)
        self.assertIsInstance(protected_circ, QuantumCircuit)
        self.assertGreaterEqual(protected_circ.num_qubits, qc.num_qubits)

        slices = slice_by_depth(qc, 1)
        start, end, clifford_slices = find_largest_clifford_block(slices)
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)

        self.assertGreaterEqual(len(protected_circ.data), len(qc.data))
        self.assertLessEqual(len(sign_list), num_checks)


'''
Mock test cases for convert_to_PCS_circ_largest_clifford function using pytest framework
'''

@pytest.fixture
def test_circuit():
    return hydrogen_trial_circuit(4)

@pytest.fixture(params=[1, 2, 3])
def num_checks(request):
    return request.param

def test_clifford_block_found(mocker, test_circuit, num_checks):
    q_num = test_circuit.num_qubits
    mocker.patch("quantem.utils.slice_by_depth", return_value=[QuantumCircuit(q_num) for _ in range(12)])
    mocker.patch("quantem.utils.find_largest_clifford_block", return_value=(6, 8, [QuantumCircuit(q_num) for _ in range(3)]))
    mocker.patch("quantem.utils.combine_slices", return_value=QuantumCircuit(q_num))
    mocker.patch("quantem.utils.convert_to_PCS_circ", return_value=(["+1" for _ in range(num_checks)] , QuantumCircuit(q_num + num_checks)))
    mocker.patch("quantem.utils.expand_circuit", return_value=QuantumCircuit(q_num + num_checks))

    sign_list, final_circ = convert_to_PCS_circ_largest_clifford(test_circuit, q_num, num_checks)
    assert sign_list == ["+1" for _ in range(num_checks)]
    assert isinstance(final_circ, QuantumCircuit)
    assert isinstance(sign_list, list)
    assert len(sign_list) == num_checks
    assert final_circ.num_qubits == q_num + num_checks

def test_no_clifford_block_found(mocker):
    circ = QuantumCircuit(2)
    mocker.patch("quantem.utils.slice_by_depth", return_value=[])
    mocker.patch("quantem.utils.find_largest_clifford_block", return_value=(None, None, []))

    sign_list, final_circ = convert_to_PCS_circ_largest_clifford(circ, 2, 2)
    assert sign_list is None
    assert final_circ == circ

if __name__ == "__main__":
    unittest.main()
