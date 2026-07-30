"""Shared validation helpers for QuantEM's public compilation contracts."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real


def validate_num_checks(num_checks: int, *, parameter_name: str = "num_checks") -> int:
    """Validate and normalize a non-negative Pauli-check count."""
    if isinstance(num_checks, bool) or not isinstance(num_checks, Integral):
        raise TypeError(f"{parameter_name} must be a non-negative integer")
    if num_checks < 0:
        raise ValueError(f"{parameter_name} must be a non-negative integer")
    return int(num_checks)


def validate_check_counts(check_counts: Sequence[int]) -> tuple[int, ...]:
    """Validate the check-count axis used to construct a PCE experiment."""
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
