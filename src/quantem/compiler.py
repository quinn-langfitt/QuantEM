from __future__ import annotations

from typing import TYPE_CHECKING

from quantem.iceberg_code import build_iceberg_circuit
from quantem.utils import convert_to_PCS_circ, convert_to_ancilla_free_PCS_circ, find_largest_clifford_block

from qiskit_addon_utils.slicing import slice_by_depth

if TYPE_CHECKING:
    from typing import Mapping, Sequence
    from qiskit import QuantumCircuit

"""
Currently a skeleton for the QED compiler.
"""

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #

CLIFFORD_NAMES = {
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
    "swap",
}

DEFAULT_CLIFFORD_THRESHOLD = 0.4


# ---------------------------------------------------------------------------- #
# Analysis Functions
# ---------------------------------------------------------------------------- #


# def percent_clifford(circ: QuantumCircuit) -> float:
#     """
#     Calculate the fraction of instructions in `circ` whose names are in CLIFFORD_NAMES.
#     Returns a float between 0 and 1.
#     """
#     instructions = circ.data
#     if not instructions:
#         return 0.0
#     count_clifford = sum(
#         1 for instr, *_ in instructions if instr.name in CLIFFORD_NAMES
#     )
#     return count_clifford / len(instructions)

def largest_clifford_block_fraction(circ: QuantumCircuit) -> float:
    """
    Compute the fraction of circuit depth occupied by the largest contiguous Clifford-only block.
    """
    slices = slice_by_depth(circ, 1)
    total_depth = len(slices)

    if total_depth == 0:
        return 0.0

    start, end, block = find_largest_clifford_block(slices)
    block_depth = len(block)
    return block_depth / total_depth


def det_QED_strategy(
    circ: QuantumCircuit, thres: float = DEFAULT_CLIFFORD_THRESHOLD
) -> str:
    """
    Decide which QED strategy to use.

    Current logic:
      - If largest_clifford_block_fraction(circ) >= thres: use PCS
      - Otherwise: use ICEBERG

    This function is intentionally modular: swap out or extend
    the decision logic here without touching placement routines.
    """
    p_cliff = largest_clifford_block_fraction(circ)
    print(f"[det_QED_strategy] largest_clifford_block_fraction = {p_cliff:.2%}")
    return "PCS" if p_cliff >= thres else "ICEBERG"


# ---------------------------------------------------------------------------- #
# Functions for placement
# ---------------------------------------------------------------------------- #


def place_pcs(
    circ: QuantumCircuit, layout: Mapping, gateset: Sequence, n_checks: int = None
) -> QuantumCircuit:
    """
    Insert Pauli Check Sandwiching (PCS) into the given circuit.
    """
    if n_checks is None:
        raise ValueError("place_pcs requires n_checks to be set")
    sign_list, qc = convert_to_PCS_circ(
        circ, circ.num_qubits, n_checks, barriers=True, reverse=False
    )
    return qc, sign_list


def place_afpc(
    circ: QuantumCircuit, layout: Mapping, gateset: Sequence, n_checks: int = None
) -> QuantumCircuit:
    """
    Insert Ancilla-Free Pauli Checks (AFPC) into the circuit.
    Similar structure to place_pcs.
    """
    if n_checks is None:
        raise ValueError("place_afpc requires n_checks to be set")
    sign_list, qc, left_mappings_list, right_mappings_list = (
        convert_to_ancilla_free_PCS_circ(
            circ, circ.num_qubits, n_checks, barriers=True, reverse=False
        )
    )
    return qc, sign_list, left_mappings_list, right_mappings_list


def place_iceberg(
    circ: QuantumCircuit, layout: Mapping, gateset: Sequence, n_checks: int = None
) -> QuantumCircuit:
    """
    Wrap the logical circuit with Iceberg QED code:
      - Transpile, initialize, map logical->physical, syndrome, readout
    """
    # build_iceberg_circuit handles transpile + placement + measurements
    qed_qc, reg_bundle = build_iceberg_circuit(
    circ,
    optimize_level=3,
    attach_readout=True
)
    return qed_qc, reg_bundle


# ---------------------------------------------------------------------------- #
# Overhead Analysis Fucntions
# ---------------------------------------------------------------------------- #


def pcs_analysis(qed_circ: QuantumCircuit):
    raise NotImplementedError("pcs_analysis not yet implemented")


def iceberg_analysis(qed_circ: QuantumCircuit):
    raise NotImplementedError("iceberg_analysis not yet implemented")


# ---------------------------------------------------------------------------- #
# Top-Level Compiler Function
# ---------------------------------------------------------------------------- #


def det_QED(
    circ: QuantumCircuit,
    layout: Mapping,
    gateset: Sequence,
    thres: float = DEFAULT_CLIFFORD_THRESHOLD,
    n_checks: int = None,
    strategy: str | None = None,
) -> QuantumCircuit:
    """
    Top-level QED compiler.

    Args:
        circ: input QuantumCircuit
        layout: qubit layout mapping
        gateset: list of supported gate names/types
        thres: clifford threshold for PCS (only used if strategy is None)
        n_checks: number of checks for PCS (required)
        strategy: explicitly set to "PCS" or "ICEBERG"; if None, strategy is chosen automatically

    Returns:
        QuantumCircuit instrumented with error-detection code
    """
    # 1. Choose strategy
    if strategy is None:
        strategy = det_QED_strategy(circ, thres)
    else:
        strategy = strategy.upper()
        if strategy not in {"PCS", "ICEBERG"}:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be either 'PCS' or 'ICEBERG'.")

    # 2. Place error-detection subcircuits
    if strategy == "PCS":
        qed_circ = place_pcs(circ, layout, gateset, n_checks)
    else:  # ICEBERG
        qed_circ = place_iceberg(circ, layout, gateset, n_checks)

    return qed_circ
