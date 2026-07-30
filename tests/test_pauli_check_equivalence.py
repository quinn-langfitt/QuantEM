"""Deterministic operator-equivalence tests for generated Pauli checks."""

from dataclasses import replace
import sys
from unittest.mock import MagicMock

# Allow equivalence tests to run without rebuilding the local Rust extension.
mock_rust = MagicMock()
mock_rust.sabre = MagicMock()
sys.modules["quantem.rust"] = mock_rust
sys.modules["quantem.rust.sabre"] = mock_rust.sabre
sys.modules["qiskit_addon_utils"] = MagicMock()
sys.modules["qiskit_addon_utils.slicing"] = MagicMock()
sys.modules["mapomatic"] = MagicMock()

import pytest
from qiskit import QuantumCircuit

from quantem import PauliCheck, QEDCompiler, QEDStrategy, verify_pauli_check


@pytest.mark.parametrize("gate_name", ["x", "y", "z", "h", "s", "sdg"])
def test_generated_single_qubit_checks_satisfy_exact_identity(gate_name):
    circuit = QuantumCircuit(1)
    getattr(circuit, gate_name)(0)

    result = QEDCompiler().compile(
        circuit,
        strategy=QEDStrategy.PCS,
        num_checks=1,
    )

    check = result.metadata["pauli_checks"][0]
    assert verify_pauli_check(circuit, check)


@pytest.mark.parametrize("gate_name", ["cx", "swap"])
def test_generated_two_qubit_checks_satisfy_exact_identity(gate_name):
    circuit = QuantumCircuit(2)
    getattr(circuit, gate_name)(0, 1)

    result = QEDCompiler().compile(
        circuit,
        strategy=QEDStrategy.PCS,
        num_checks=2,
    )

    assert all(
        verify_pauli_check(circuit, check)
        for check in result.metadata["pauli_checks"]
    )


def test_verifier_checks_recorded_negative_phase():
    circuit = QuantumCircuit(1)
    circuit.x(0)

    result = QEDCompiler().compile(
        circuit,
        strategy=QEDStrategy.PCS,
        num_checks=1,
        only_Z_checks=True,
    )
    check = result.metadata["pauli_checks"][0]

    assert check.phase == -1
    assert check.sign == "-1"
    assert verify_pauli_check(circuit, check)
    assert not verify_pauli_check(circuit, replace(check, phase=1))


def test_verifier_rejects_corrupted_pauli_operator():
    circuit = QuantumCircuit(1)
    circuit.h(0)

    result = QEDCompiler().compile(
        circuit,
        strategy=QEDStrategy.PCS,
        num_checks=1,
    )
    check = result.metadata["pauli_checks"][0]
    corrupted = replace(check, right="Z")

    assert check.right == "X"
    assert verify_pauli_check(circuit, check)
    assert not verify_pauli_check(circuit, corrupted)


@pytest.mark.parametrize(
    ("option_name", "allowed_symbols"),
    [
        ("only_X_checks", set("IX")),
        ("only_Z_checks", set("IZ")),
    ],
)
def test_check_filter_claim_is_auditable(option_name, allowed_symbols):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    result = QEDCompiler().compile(
        circuit,
        strategy=QEDStrategy.PCS,
        num_checks=1,
        **{option_name: True},
    )
    check = result.metadata["pauli_checks"][0]

    assert set(check.right) <= allowed_symbols
    assert verify_pauli_check(circuit, check)


@pytest.mark.parametrize(
    "invalid_check",
    [
        PauliCheck(left="X", right="X", phase=1),
        "not a check",
    ],
)
def test_verifier_validates_check_input(invalid_check):
    circuit = QuantumCircuit(2)

    expected_error = ValueError if isinstance(invalid_check, PauliCheck) else TypeError
    with pytest.raises(expected_error):
        verify_pauli_check(circuit, invalid_check)


@pytest.mark.parametrize("invalid_atol", [-1.0, float("inf"), float("nan")])
def test_verifier_rejects_unsafe_tolerance(invalid_atol):
    circuit = QuantumCircuit(1)
    check = PauliCheck(left="X", right="X", phase=1)

    with pytest.raises(ValueError, match="finite, non-negative"):
        verify_pauli_check(circuit, check, atol=invalid_atol)
