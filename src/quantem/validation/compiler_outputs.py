"""Consistency validation for QuantEM compiler outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qiskit.circuit import QuantumCircuit

from quantem.pauli_checks import PauliCheck


class InvalidCompilerOutputError(RuntimeError):
    """Raised when a compiler transformation returns an inconsistent result."""


def _violation(strategy_name: str, message: str) -> None:
    raise InvalidCompilerOutputError(
        f"{strategy_name} produced an invalid compiler output: {message}"
    )


def reject_input_mutation(
    strategy_name: str,
    input_before: QuantumCircuit,
    input_after: QuantumCircuit,
) -> None:
    """Assert a compiler transformation did not mutate the caller's circuit.

    Callers invoke this once per public compilation request rather than once
    per emitted output, so a multi-circuit run (e.g. PCE) pays for a single
    circuit comparison instead of one per check count.
    """
    if input_after != input_before:
        _violation(strategy_name, "the input circuit was mutated")


def _validate_common_output(
    *,
    strategy_name: str,
    input_circuit: QuantumCircuit,
    output_circuit: Any,
    metadata: Any,
) -> Mapping[str, Any]:
    if not isinstance(output_circuit, QuantumCircuit):
        _violation(strategy_name, "the output must be a QuantumCircuit")
    if output_circuit is input_circuit:
        _violation(strategy_name, "the output must not alias the input circuit")
    if not isinstance(metadata, Mapping):
        _violation(strategy_name, "metadata must be a mapping")

    expected_total = output_circuit.num_qubits
    if metadata.get("total_qubits") != expected_total:
        _violation(
            strategy_name,
            "metadata total_qubits does not match the output circuit",
        )

    expected_overhead = expected_total - input_circuit.num_qubits
    if metadata.get("qubit_overhead") != expected_overhead:
        _violation(
            strategy_name,
            "metadata qubit_overhead does not match the circuit width change",
        )
    if expected_overhead < 0:
        _violation(strategy_name, "the output has fewer qubits than the input")

    return metadata


def _require_sized_field(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
    expected_length: int,
    strategy_name: str,
) -> Sequence[Any]:
    value = metadata.get(field_name)
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != expected_length
    ):
        _violation(
            strategy_name,
            f"metadata {field_name} must contain exactly "
            f"{expected_length} entries",
        )
    return value


def validate_pcs_output(
    *,
    input_circuit: QuantumCircuit,
    output_circuit: Any,
    metadata: Any,
    num_checks: int,
) -> None:
    """Validate structural guarantees made by PCS compilation."""
    strategy_name = "PCS"
    metadata = _validate_common_output(
        strategy_name=strategy_name,
        input_circuit=input_circuit,
        output_circuit=output_circuit,
        metadata=metadata,
    )
    if metadata.get("num_checks") != num_checks:
        _violation(strategy_name, "metadata num_checks does not match the request")
    sign_list = _require_sized_field(
        metadata,
        field_name="sign_list",
        expected_length=num_checks,
        strategy_name=strategy_name,
    )
    pauli_checks = _require_sized_field(
        metadata,
        field_name="pauli_checks",
        expected_length=num_checks,
        strategy_name=strategy_name,
    )
    for index, check in enumerate(pauli_checks):
        if not isinstance(check, PauliCheck):
            _violation(
                strategy_name,
                f"metadata pauli_checks entry {index} must be a PauliCheck",
            )
        if len(check.left) != input_circuit.num_qubits:
            _violation(
                strategy_name,
                f"left Pauli width for check {index} does not match the payload",
            )
        if len(check.right) != input_circuit.num_qubits:
            _violation(
                strategy_name,
                f"right Pauli width for check {index} does not match the payload",
            )
        if sign_list[index] != check.sign:
            _violation(
                strategy_name,
                f"sign_list entry {index} does not match its PauliCheck phase",
            )
    if metadata.get("ancilla_qubits") != num_checks:
        _violation(strategy_name, "PCS requires one ancilla per check")
    if output_circuit.num_qubits != input_circuit.num_qubits + num_checks:
        _violation(strategy_name, "the output width does not match the check count")


def validate_afpc_output(
    *,
    input_circuit: QuantumCircuit,
    output_circuit: Any,
    metadata: Any,
    num_checks: int,
) -> None:
    """Validate structural guarantees made by AFPC compilation."""
    strategy_name = "AFPC"
    metadata = _validate_common_output(
        strategy_name=strategy_name,
        input_circuit=input_circuit,
        output_circuit=output_circuit,
        metadata=metadata,
    )
    if metadata.get("num_checks") != num_checks:
        _violation(strategy_name, "metadata num_checks does not match the request")
    for field_name in ("sign_list", "left_mappings", "right_mappings"):
        _require_sized_field(
            metadata,
            field_name=field_name,
            expected_length=num_checks,
            strategy_name=strategy_name,
        )
    if metadata.get("ancilla_qubits") != 0:
        _violation(strategy_name, "AFPC must not report ancilla qubits")
    if output_circuit.num_qubits != input_circuit.num_qubits:
        _violation(strategy_name, "AFPC must preserve the input circuit width")


def validate_iceberg_output(
    *,
    input_circuit: QuantumCircuit,
    output_circuit: Any,
    metadata: Any,
) -> None:
    """Validate structural guarantees made by Iceberg compilation."""
    strategy_name = "ICEBERG"
    metadata = _validate_common_output(
        strategy_name=strategy_name,
        input_circuit=input_circuit,
        output_circuit=output_circuit,
        metadata=metadata,
    )

    if metadata.get("code_qubit_overhead") != 2:
        _violation(strategy_name, "the code must add exactly two code qubits")
    if metadata.get("distance") != 2:
        _violation(strategy_name, "metadata must report code distance 2")

    expected_ancillas = metadata["qubit_overhead"] - 2
    if expected_ancillas < 0:
        _violation(strategy_name, "the output is missing required code qubits")
    if metadata.get("ancilla_qubits") != expected_ancillas:
        _violation(
            strategy_name,
            "metadata ancilla_qubits does not match the non-code overhead",
        )

    register_bundle = metadata.get("register_bundle")
    if register_bundle is None:
        _violation(strategy_name, "metadata is missing register_bundle")

    expected_widths = {"t": 1, "p": input_circuit.num_qubits, "b": 1}
    for register_name, expected_width in expected_widths.items():
        register = getattr(register_bundle, register_name, None)
        if register is None or len(register) != expected_width:
            _violation(
                strategy_name,
                f"register_bundle.{register_name} must have width {expected_width}",
            )
        if register not in output_circuit.qregs:
            _violation(
                strategy_name,
                f"register_bundle.{register_name} is absent from the output circuit",
            )


def validate_pce_output(
    *,
    requested_counts: Sequence[int],
    circuits: Mapping[int, QuantumCircuit],
    metadata: Mapping[int, Mapping[str, Any]],
) -> None:
    """Validate the aggregate key consistency of a PCE compilation."""
    strategy_name = "PCE"
    expected_keys = set(requested_counts)
    if set(circuits) != expected_keys:
        _violation(
            strategy_name,
            "compiled circuit keys do not match the requested check counts",
        )
    if set(metadata) != expected_keys:
        _violation(
            strategy_name,
            "metadata keys do not match the requested check counts",
        )
