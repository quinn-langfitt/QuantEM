from __future__ import annotations
'''Copyright © 2025 UChicago Argonne, LLC and Northwestern University All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/quinn-langfitt/QuantEM/blob/main/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

"""
Iceberg quantum‑error‑detection code utilities
"""

from types import SimpleNamespace
from typing import Dict, Tuple, List

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit


class Syndrome_gate(Instruction):
    """Annotated Syndrome_measurement gate"""
    def __init__(self, logical_qubits: int):
        """Apply syndrome measurement for logical qubits
        """
        super().__init__(
            "Syndrome_gate",
            num_qubits=logical_qubits,
            num_clbits=0,
            params=[logical_qubits],
        )

    def _define(self):
        raise NotImplementedError("The gate is not synthesized")




# --------------------------------------------------------------------
# 1.   Prepare the encoded |0>_L state and first X‑parity check
# --------------------------------------------------------------------
def initial_state_prep(num_logical: int) -> Tuple[QuantumCircuit, SimpleNamespace]:
    """
    Build initial entangled state for Iceberg QED and return circuit + registers.
    """
    p = QuantumRegister(num_logical, "p")      # physical qubit
    t = QuantumRegister(1, "t")                # top qubit
    b = QuantumRegister(1, "b")                # bottom qubit
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
# 2.   Logical → physical translation
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
        
        elif node.name == "Syndrome_gate":
            phys_qc.append(Syndrome_gate(len(regs.p) + 2), [regs.t[0]] + regs.p[0:] + [regs.b[0]], None)

        else:
            raise ValueError(f"Unsupported logical gate: {name}")


# --------------------------------------------------------------------
# 3.   Syndrome measurement
# --------------------------------------------------------------------
def add_logical_syndrome_gates(
    log_qc: QuantumCircuit, syndrome_interval: int
) -> None:
    """Append syndrome measurement gates at the logical level that extract Iceberg stabilizers."""
    dag = circuit_to_dag(log_qc)
    log_qubits = dag.qubits
    new_dag = dag.copy_empty_like()

    for i, layer in enumerate(dag.layers()):
        # Insert syndrome gate every `syndrome_interval` layers
        if syndrome_interval > 0 and i % syndrome_interval == 0 and i > 0:
            return_val = new_dag.apply_operation_back(Syndrome_gate(len(log_qubits)), log_qubits)
        subdag = layer["graph"]
        for node in subdag.op_nodes():
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

    return dag_to_circuit(new_dag)


def insert_syndrome_measurement(
    phys_qc: QuantumCircuit, regs: SimpleNamespace
) -> None:
    dag = circuit_to_dag(phys_qc)
    new_dag = dag.copy_empty_like()
    syndrome_meas_count = 0
    for node in dag.op_nodes():
        if node.name == "Syndrome_gate":
            #add the syndrome measurement:

            # Allocate 2 ancilla qubits and 2 classical bits
            syn_reg = QuantumRegister(2, f"syn_{syndrome_meas_count}")
            syn_c_reg = ClassicalRegister(2, f"syn_c_{syndrome_meas_count}")
            new_dag.add_qreg(syn_reg)
            new_dag.add_creg(syn_c_reg)

            # Insert the actual syndrome operation
            # Replace with your own implementation here:
            syndrome_subcircuit = QuantumCircuit(regs.t, regs.p, regs.b, syn_reg, syn_c_reg)
            insert_syndrome_op(syndrome_subcircuit, regs, syn_reg, syn_c_reg)

            # Add the new subcircuit’s operations to the DAG
            for inst, qargs, cargs in syndrome_subcircuit.data:
                new_dag.apply_operation_back(inst, qargs=qargs, cargs=cargs)

            syndrome_meas_count += 1

        else:
            # Copy original instruction
            new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)


    return dag_to_circuit(new_dag)

def insert_syndrome_op(phys_qc, regs, ancillas, cbits):

    
    """Append two‑qubit ancilla circuit that extracts Iceberg stabilizers."""
    n = len(regs.p)
    syn = ancillas
    syn_c = cbits
    qubits = [regs.t[0], *regs.p, regs.b[0]]
    phys_qc.h(syn[1])
    for idx in range(0, n + 2, 2):
        if idx == 0 or idx == n:
             # A‑type stabilizer
            phys_qc.cx(syn[1], qubits[idx])
            phys_qc.cx(qubits[idx], syn[0])
            phys_qc.cx(qubits[idx + 1], syn[0])
            phys_qc.cx(syn[1], qubits[idx + 1])
        else:
            # B‑type chain
            phys_qc.cx(syn[1], qubits[idx])
            phys_qc.cx(qubits[idx], syn[0])
            phys_qc.cx(syn[1], qubits[idx + 1])
            phys_qc.cx(qubits[idx + 1], syn[0])

    phys_qc.h(syn[1])

    phys_qc.measure(syn, syn_c)


# --------------------------------------------------------------------
# 4.   Logical‑state read‑out
# --------------------------------------------------------------------
def add_logical_measurement(
    phys_qc: QuantumCircuit, regs: SimpleNamespace
) -> None:
    """Add logical‑X/Z measurement using two ancillas."""
    n = len(regs.p)
    meas = QuantumRegister(2, "meas")
    meas_c = ClassicalRegister(2, "meas_c")
    meas_val = ClassicalRegister(len(regs.p), "meas_result")
    phys_qc.add_register(meas, meas_c, meas_val)

    phys_qc.h(meas[0])
    phys_qc.cx(meas[0], regs.t[0])
    phys_qc.cx(meas[0], meas[1])

    for q in regs.p:
        phys_qc.cx(meas[0], q)

    phys_qc.cx(meas[0], meas[1])
    phys_qc.cx(meas[0], regs.b[0])
    phys_qc.h(meas[0])
    phys_qc.measure(meas, meas_c)
    phys_qc.measure(regs.p, meas_val)

# --------------------------------------------------------------------
# 5.   One‑call convenience wrapper
# --------------------------------------------------------------------
def build_iceberg_circuit(
    logical_qc: QuantumCircuit,
    optimize_level: int = 3,
    attach_readout: bool = True,
    syndrome_interval: int = 5,
    total_syndrome_cycles: int = 5,
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
    attach_readout
        If True, add logical‑state measurement block.
    syndrome_interval
        The circuit depth between syndrome measurement
    total_syndrome_cycles
        The total number of syndromes measurements which evenly divide the circuit depth

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

    # Validate or derive syndrome interval
    if syndrome_interval == 0:
        if total_syndrome_cycles == 0:
            raise ValueError("Either `syndrome_interval` or `total_syndrome_cycles` must be specified.")
        else:
            syndrome_interval = decomposed.depth() // total_syndrome_cycles
    # add logical syndrome gates as place holder
    log_qc = add_logical_syndrome_gates(decomposed, syndrome_interval) 

    # 2 — initial state
    phys_qc, regs = initial_state_prep(num_logical=decomposed.num_qubits)

    # 3 — logical → physical mapping
    logical_to_physical(log_qc, phys_qc, regs)

    # 4 - insert syndrome measurements:
    new_phys_qc = insert_syndrome_measurement(phys_qc, regs)

    # 4 — optional extras
    if attach_readout:
        add_logical_measurement(new_phys_qc, regs)

    return new_phys_qc, regs

