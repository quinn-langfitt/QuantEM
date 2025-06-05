from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit.library import HGate, CXGate, SGate

if TYPE_CHECKING:
    from typing import Iterable
    from qiskit import QuantumCircuit


def cnot_phase(xc, xt, zc, zt):
    """Implement phase for the CNOT gate"""
    return 1 - 2 * (xc * xt * zc * zt + xc * (1 - xt) * (1 - zc) * zt)


def local_phase(x, z):
    """Implement phase for the local gates"""
    return 1 - 2 * x * z


def conjugate_pauli_through_gates(circuit: QuantumCircuit, input_phase, pauli_str):
    """Transform a Pauli operator through a quantum circuit with exact phase tracking"""

    num_qubits = circuit.num_qubits
    x, z = pauli_string_to_components(pauli_str, num_qubits)
    phase = input_phase  # Now tracking as real phase factor

    for instruction in circuit.data:
        gate = instruction.operation
        qubits = [q.index for q in instruction.qubits]

        match gate:
            case HGate():
                q = qubits[0]
                # Calculate phase contribution before updating components
                phase *= local_phase(x[q], z[q])
                # Swap X and Z components
                x[q], z[q] = z[q], x[q]

            case SGate():
                q = qubits[0]
                # Calculate phase contribution before updating components
                phase *= local_phase(x[q], z[q])
                # Update Z component
                z[q] = (x[q] + z[q]) % 2

            case CXGate():
                c, t = qubits
                # Calculate CNOT phase contribution before updating components
                phase *= cnot_phase(x[c], x[t], z[c], z[t])
                # Update X and Z components
                x[t] = (x[t] + x[c]) % 2
                z[c] = (z[c] + z[t]) % 2

            case _:
                raise ValueError(f"Unsupported gate: {gate.name}")

    pauli_string = components_to_pauli_string(x, z)
    return phase, pauli_string


def compute_right_pauli_checks(circuit, left_pauli_checks):
    """Transform input Pauli checks through a quantum circuit with proper phase"""
    results = []
    for phase, pauli_str in left_pauli_checks:
        final_phase, final_pauli = conjugate_pauli_through_gates(
            circuit, phase, pauli_str
        )
        results.append((final_phase, final_pauli))
    return results


# Helper functions
def pauli_string_to_components(pauli_str, num_qubits):
    x = [0] * num_qubits
    z = [0] * num_qubits
    for i, c in enumerate(pauli_str):
        if i >= num_qubits:
            break
        if c.upper() == "X":
            x[i] = 1
        elif c.upper() == "Z":
            z[i] = 1
        elif c.upper() == "Y":
            x[i], z[i] = 1, 1
    return x, z


def components_to_pauli_string(x: Iterable[int], z: Iterable[int]):
    pauli_string = (
        (
            "X"
            if (x == 1 and z == 0)
            else "Z" if (x == 0 and z == 1) else "Y" if (x == 1 and z == 1) else "I"
        )
        for x, z in zip(x, z)
    )
    return "".join(pauli_string)


if __name__ == "__main__":
    from qiskit import QuantumCircuit

    # Example verification
    qc = QuantumCircuit(6)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.cx(0, 5)

    left_checks = [(1, "XYZZYX")]
    right_checks = compute_right_pauli_checks(qc, left_checks)
    print("Input checks:", left_checks)
    print("Output checks:", right_checks)
