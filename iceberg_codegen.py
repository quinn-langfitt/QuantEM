"""
Iceberg quantum‑error‑detection code utilities
Author: ji.liu@anl.gov
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag

# --------------------------------------------------------------------
# 1.   Transpile to the restricted logical basis
# --------------------------------------------------------------------
def basis_transpile(logical_qc: QuantumCircuit) -> QuantumCircuit:
    """Optimize and decompose into {rx, rz, rxx, rzz} only."""
    return transpile(
        logical_qc,
        basis_gates=["rx", "rz", "rxx", "rzz"],
        optimization_level=3,
    )


# --------------------------------------------------------------------
# 2.   Prepare the encoded |0>_L state and first X‑parity check
# --------------------------------------------------------------------
def initial_state_prep(num_logical: int) -> Tuple[QuantumCircuit, SimpleNamespace]:
    """
    Build initial entangled state for Iceberg QED and return circuit + registers.
    """
    p = QuantumRegister(num_logical, "p")      # payload chain
    t = QuantumRegister(1, "t")                # trigger qubit
    b = QuantumRegister(1, "b")                # boundary qubit
    anc = QuantumRegister(1, "anc_init")
    anc_c = ClassicalRegister(1, "anc_init_c")

    qc = QuantumCircuit(t, p, b, anc, anc_c)

    # GHZ‑like chain entanglement
    qc.h(t[0])
    qc.cx(t[0], p[0])
    for i in range(num_logical - 1):
        qc.cx(p[i], p[i + 1])
    qc.cx(p[-1], b[0])

    # Initial X‑stabilizer
    qc.cx(b[0], anc[0])
    qc.cx(t[0], anc[0])
    qc.measure(anc, anc_c)

    regs = SimpleNamespace(t=t, p=p, b=b)  # returned instead of dict
    return qc, regs


# --------------------------------------------------------------------
# 3.   Logical → physical translation
# --------------------------------------------------------------------
def logical_to_physical(
    logical_qc: QuantumCircuit,
    phys_qc: QuantumCircuit,
    regs: SimpleNamespace,
) -> None:
    """
    Map logical rx/rz/rxx/rzz gates to Iceberg physical operations in‑place.
    """
    dag = circuit_to_dag(logical_qc)

    for node in dag.topological_op_nodes():
        name = node.op.name
        theta = node.op.params[0]

        if name == "rx":
            i = logical_qc.find_bit(node.qargs[0]).index
            phys_qc.rxx(theta, regs.t[0], regs.p[i])

        elif name == "rz":
            i = logical_qc.find_bit(node.qargs[0]).index
            phys_qc.rzz(theta, regs.b[0], regs.p[i])

        elif name == "rxx":
            i1 = logical_qc.find_bit(node.qargs[0]).index
            i2 = logical_qc.find_bit(node.qargs[1]).index
            phys_qc.rxx(theta, regs.p[i1], regs.p[i2])

        elif name == "rzz":
            i1 = logical_qc.find_bit(node.qargs[0]).index
            i2 = logical_qc.find_bit(node.qargs[1]).index
            phys_qc.rzz(theta, regs.p[i1], regs.p[i2])

        else:
            raise ValueError(f"Unsupported logical gate: {name}")


# --------------------------------------------------------------------
# 4.   Syndrome‑extract measurement
# --------------------------------------------------------------------
def add_syndrome_measurement(
    phys_qc: QuantumCircuit, regs: SimpleNamespace
) -> None:
    """Append two‑qubit ancilla circuit that extracts Iceberg stabilizers."""
    n = len(regs.p)
    syn = QuantumRegister(2, "syn")
    syn_c = ClassicalRegister(2, "syn_c")
    phys_qc.add_register(syn, syn_c)

    # A‑type stabilizer arm
    phys_qc.h(syn[1])
    phys_qc.cx(syn[1], regs.t[0])
    phys_qc.cx(regs.t[0], syn[0])
    phys_qc.cx(regs.p[0], syn[0])
    phys_qc.cx(syn[1], regs.p[0])

    # B‑type chain
    for i in range(1, n - 1):
        phys_qc.cx(syn[1], regs.p[i])
        phys_qc.cx(regs.p[i], syn[0])

    # Finishing A‑arm
    phys_qc.cx(syn[1], regs.p[-1])
    phys_qc.cx(regs.p[-1], syn[0])
    phys_qc.cx(regs.b[0], syn[0])
    phys_qc.cx(syn[1], regs.b[0])
    phys_qc.h(syn[1])

    phys_qc.measure(syn, syn_c)


# --------------------------------------------------------------------
# 5.   Logical‑state read‑out
# --------------------------------------------------------------------
def add_logical_measurement(
    phys_qc: QuantumCircuit, regs: SimpleNamespace
) -> None:
    """Add logical‑X/Z measurement using two ancillas."""
    n = len(regs.p)
    meas = QuantumRegister(2, "meas")
    meas_c = ClassicalRegister(2, "meas_c")
    phys_qc.add_register(meas, meas_c)

    phys_qc.h(meas[0])
    phys_qc.cx(meas[0], regs.t[0])
    phys_qc.cx(meas[0], meas[1])

    for q in regs.p:
        phys_qc.cx(meas[0], q)

    phys_qc.cx(meas[0], meas[1])
    phys_qc.cx(meas[0], regs.b[0])
    phys_qc.h(meas[0])
    phys_qc.measure(meas, meas_c)

# --------------------------------------------------------------------
# 6.   One‑call convenience wrapper
# --------------------------------------------------------------------
def build_iceberg_circuit(
    logical_qc: QuantumCircuit,
    *,
    optimize_level: int = 3,
    attach_syndrome: bool = True,
    attach_readout: bool = True,
) -> Tuple[QuantumCircuit, SimpleNamespace]:
    """
    End‑to‑end wrapper:
    1.  Transpile `logical_qc` to the {rx, rz, rxx, rzz} basis.
    2.  Prepare the Iceberg |0>_L state.
    3.  Map logical gates onto physical qubits.
    4.  Optionally append syndrome extraction and final read‑out.

    Parameters
    ----------
    logical_qc
        Original high‑level circuit (any basis).
    optimize_level
        qiskit transpile optimization level (0‑3).
    attach_syndrome
        If True, add stabilizer‑measurement block.
    attach_readout
        If True, add logical‑state measurement block.

    Returns
    -------
    phys_qc
        Complete physical‑level QuantumCircuit.
    regs
        SimpleNamespace with named QuantumRegisters.
    """
    # 1 — transpile
    decomposed = transpile(
        logical_qc,
        basis_gates=["rx", "rz", "rxx", "rzz"],
        optimization_level=optimize_level,
    )

    # 2 — initial state
    phys_qc, regs = initial_state_prep(num_logical=decomposed.num_qubits)

    # 3 — logical → physical mapping
    logical_to_physical(decomposed, phys_qc, regs)

    # 4 — optional extras
    if attach_syndrome:
        add_syndrome_measurement(phys_qc, regs)
    if attach_readout:
        add_logical_measurement(phys_qc, regs)

    return phys_qc, regs

