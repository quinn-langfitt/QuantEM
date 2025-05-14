from qiskit import QuantumCircuit
from iceberg_codegen import build_iceberg_circuit
from utils import convert_to_PCS_circ

"""
Currently a skeleton for the QED compiler.
"""

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #

CLIFFORD_NAMES = {
    "x", "y", "z", "h",
    "s", "sdg",
    "cx", "cz",
    "swap",
}

DEFAULT_CLIFFORD_THRESHOLD = 0.9


# ---------------------------------------------------------------------------- #
# Analysis Functions
# ---------------------------------------------------------------------------- #

def percent_clifford(circ: QuantumCircuit) -> float:
    """
    Calculate the fraction of instructions in `circ` whose names are in CLIFFORD_NAMES.
    Returns a float between 0 and 1.
    """
    instructions = circ.data
    if not instructions:
        return 0.0
    count_clifford = sum(1 for instr, *_ in instructions if instr.name in CLIFFORD_NAMES)
    return count_clifford / len(instructions)


def det_QED_strategy(circ: QuantumCircuit, tau: float = DEFAULT_CLIFFORD_THRESHOLD) -> str:
    """
    Decide which QED strategy to use.

    Current logic:
      - If percent_clifford(circ) >= tau: use PCS
      - Otherwise: use ICEBERG

    This function is intentionally modular: swap out or extend
    the decision logic here without touching placement routines.

    Args:
        circ: QuantumCircuit under test
        tau: threshold for percent-Clifford

    Returns:
        "PCS" or "ICEBERG"
    """
    p_cliff = percent_clifford(circ)
    print(f"[det_QED_strategy] percent_clifford = {p_cliff:.2%}")
    return "PCS" if p_cliff >= tau else "ICEBERG"


# ---------------------------------------------------------------------------- #
# Placement Stubs
# ---------------------------------------------------------------------------- #

def place_pcs(circ: QuantumCircuit, layout: dict, gateset: list, n_checks: int = None) -> QuantumCircuit:
    """
    Insert Pauli Check Sandwiching (PCS) into the given circuit.
    """
    if n_checks is None:
        raise ValueError("place_pcs requires n_checks to be set")
    sign_list, qc = convert_to_PCS_circ(
        circ, circ.num_qubits, n_checks,
        barriers=False, reverse=False
    )
    return qc, sign_list


def place_afpc(circ: QuantumCircuit, layout: dict, gateset: list, n_checks: int = None) -> QuantumCircuit:
    """
    (Optional) Insert Ancilla-Free Pauli Checks (AFPC) into the circuit.
    Similar structure to place_pcs.
    """
    raise NotImplementedError


def place_iceberg(circ: QuantumCircuit, layout: dict, gateset: list, n_checks: int = None) -> QuantumCircuit:
    """
    Wrap the logical circuit with Iceberg QED code:
      - Transpile, initialize, map logical->physical, syndrome, readout
    """
    # build_iceberg_circuit handles transpile + placement + measurements
    phys_qc, regs = build_iceberg_circuit(
        circ,
        optimize_level=3,
        attach_syndrome=True,
        attach_readout=True,
    )
    return phys_qc, regs


# ---------------------------------------------------------------------------- #
# Top-Level Compiler Function
# ---------------------------------------------------------------------------- #

def det_QED(circ: QuantumCircuit,
            layout: dict,
            gateset: list,
            tau: float = DEFAULT_CLIFFORD_THRESHOLD,
            n_checks: int = None) -> QuantumCircuit:
    """
    Top-level QED compiler.

    Args:
        circ: input QuantumCircuit
        layout: qubit layout mapping
        gateset: list of supported gate names/types
        tau: clifford threshold for PCS
        n_checks: number of checks for PCS or ICEBERG (algorithm-specific)

    Returns:
        QuantumCircuit instrumented with error-detection code
    """
    # 1. Choose strategy
    strategy = det_QED_strategy(circ, tau)

    # 2. Place error-detection subcircuits
    if strategy == "PCS":
        qed_circ = place_pcs(circ, layout, gateset, n_checks)
    else:
        qed_circ = place_iceberg(circ, layout, gateset, n_checks)

    # 3. (Optional) Perform overhead analysis here
    #    e.g., count ancilla qubits, measure added gates, etc.

    return qed_circ
