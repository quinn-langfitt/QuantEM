"""Shared validation helpers for QuantEM's public compilation contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class PauliCheckOptions:
    """Validated options shared by PCS and AFPC compilation."""

    barriers: bool = True
    reverse: bool = False
    only_X_checks: bool = False
    only_Z_checks: bool = False


@dataclass(frozen=True)
class IcebergOptions:
    """Validated options for Iceberg compilation."""

    optimize_level: int = 3
    attach_readout: bool = True
    syndrome_interval: int = 5
    total_syndrome_cycles: int = 0


def validate_quantum_circuit(
    circuit: QuantumCircuit, *, parameter_name: str = "circuit"
) -> QuantumCircuit:
    """Require a Qiskit ``QuantumCircuit`` at a public compiler boundary."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"{parameter_name} must be a QuantumCircuit")
    return circuit


def validate_bool(value: bool, *, parameter_name: str) -> bool:
    """Validate a strict boolean rather than accepting truthy values."""
    if not isinstance(value, bool):
        raise TypeError(f"{parameter_name} must be a bool")
    return value


def validate_num_checks(num_checks: int, *, parameter_name: str = "num_checks") -> int:
    """Validate and normalize a non-negative Pauli-check count."""
    if isinstance(num_checks, bool) or not isinstance(num_checks, Integral):
        raise TypeError(f"{parameter_name} must be a non-negative integer")
    if num_checks < 0:
        raise ValueError(f"{parameter_name} must be a non-negative integer")
    return int(num_checks)


def validate_check_counts(check_counts: Sequence[int]) -> tuple[int, ...]:
    """Validate the check-count axis used to construct a PCE experiment."""
    if isinstance(check_counts, (str, bytes)) or not isinstance(
        check_counts, Sequence
    ):
        raise TypeError("check_counts must be a sequence of non-negative integers")
    if not check_counts:
        raise ValueError("check_counts must contain at least one value")

    normalized_values = []
    for count in check_counts:
        try:
            normalized_values.append(validate_num_checks(count))
        except TypeError as exc:
            raise TypeError(
                "check_counts must contain non-negative integers"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                "check_counts must contain non-negative integers"
            ) from exc

    normalized = tuple(normalized_values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("check_counts must not contain duplicate values")
    return normalized


def validate_clifford_threshold(clifford_threshold: float) -> float:
    """Validate the AUTO-strategy Clifford-block threshold."""
    if isinstance(clifford_threshold, bool) or not isinstance(
        clifford_threshold, Real
    ):
        raise TypeError("clifford_threshold must be a real number between 0 and 1")
    if not 0 <= clifford_threshold <= 1:
        raise ValueError("clifford_threshold must be between 0 and 1")
    return float(clifford_threshold)


def _reject_unknown_options(
    options: Mapping[str, Any], *, allowed: frozenset[str], strategy_name: str
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"unexpected {strategy_name} option(s): {joined}")


def normalize_pauli_check_options(
    options: Mapping[str, Any], *, strategy_name: str
) -> PauliCheckOptions:
    """Validate and normalize PCS or AFPC keyword options."""
    allowed = frozenset(
        {"barriers", "reverse", "only_X_checks", "only_Z_checks"}
    )
    _reject_unknown_options(
        options, allowed=allowed, strategy_name=strategy_name
    )

    normalized = PauliCheckOptions(
        barriers=validate_bool(
            options.get("barriers", True), parameter_name="barriers"
        ),
        reverse=validate_bool(
            options.get("reverse", False), parameter_name="reverse"
        ),
        only_X_checks=validate_bool(
            options.get("only_X_checks", False),
            parameter_name="only_X_checks",
        ),
        only_Z_checks=validate_bool(
            options.get("only_Z_checks", False),
            parameter_name="only_Z_checks",
        ),
    )
    if normalized.only_X_checks and normalized.only_Z_checks:
        raise ValueError("Cannot specify both only_X_checks and only_Z_checks")
    return normalized


def normalize_iceberg_options(options: Mapping[str, Any]) -> IcebergOptions:
    """Validate and normalize Iceberg keyword options.

    Syndrome placement has two exclusive modes: a positive
    ``syndrome_interval`` with zero total cycles, or ``syndrome_interval=0``
    with a positive ``total_syndrome_cycles``. Supplying only a cycle count
    selects the latter mode.
    """
    allowed = frozenset(
        {
            "optimize_level",
            "attach_readout",
            "syndrome_interval",
            "total_syndrome_cycles",
        }
    )
    _reject_unknown_options(options, allowed=allowed, strategy_name="ICEBERG")

    optimize_level = options.get("optimize_level", 3)
    if isinstance(optimize_level, bool) or not isinstance(
        optimize_level, Integral
    ):
        raise TypeError("optimize_level must be an integer between 0 and 3")
    if optimize_level not in range(4):
        raise ValueError("optimize_level must be an integer between 0 and 3")

    attach_readout = validate_bool(
        options.get("attach_readout", True),
        parameter_name="attach_readout",
    )

    interval_was_supplied = "syndrome_interval" in options
    cycles_were_supplied = "total_syndrome_cycles" in options
    syndrome_interval = options.get("syndrome_interval", 5)
    total_syndrome_cycles = options.get("total_syndrome_cycles", 0)

    if cycles_were_supplied and not interval_was_supplied:
        syndrome_interval = 0

    syndrome_interval = validate_num_checks(
        syndrome_interval, parameter_name="syndrome_interval"
    )
    total_syndrome_cycles = validate_num_checks(
        total_syndrome_cycles, parameter_name="total_syndrome_cycles"
    )

    if syndrome_interval == 0 and total_syndrome_cycles == 0:
        raise ValueError(
            "total_syndrome_cycles must be positive when "
            "syndrome_interval is zero"
        )
    if syndrome_interval > 0 and total_syndrome_cycles > 0:
        raise ValueError(
            "specify either syndrome_interval or total_syndrome_cycles, not both"
        )

    return IcebergOptions(
        optimize_level=int(optimize_level),
        attach_readout=attach_readout,
        syndrome_interval=syndrome_interval,
        total_syndrome_cycles=total_syndrome_cycles,
    )
