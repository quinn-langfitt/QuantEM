"""Exact operator-equivalence checks for generated Pauli checks."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli

from quantem.pauli_checks import PauliCheck
from quantem.validation.compiler_inputs import validate_quantum_circuit


def verify_pauli_check(
    circuit: QuantumCircuit,
    check: PauliCheck,
    *,
    atol: float = 1e-10,
) -> bool:
    """Verify the exact PCS identity ``R U L = phase * U``.

    This oracle constructs Pauli matrices directly through Qiskit rather than
    using QuantEM's controlled-check circuit builder. It is intended for small
    reference circuits because dense operator size grows exponentially.
    """
    validate_quantum_circuit(circuit)
    if not isinstance(check, PauliCheck):
        raise TypeError("check must be a PauliCheck")
    if len(check.left) != circuit.num_qubits:
        raise ValueError("left Pauli width must match the circuit width")
    if len(check.right) != circuit.num_qubits:
        raise ValueError("right Pauli width must match the circuit width")
    if isinstance(atol, bool) or not isinstance(atol, Real):
        raise TypeError("atol must be a non-negative real number")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite, non-negative real number")

    payload = Operator(circuit).data
    left = Operator(Pauli(check.left)).data
    right = Operator(Pauli(check.right)).data
    propagated = right @ payload @ left
    expected = check.phase * payload
    return bool(np.allclose(propagated, expected, rtol=0.0, atol=float(atol)))
