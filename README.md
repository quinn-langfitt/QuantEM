# QuantEM: The Quantum Error Management Compiler

[![Tests](https://github.com/QuantA-group/QuantEM/actions/workflows/test.yml/badge.svg)](https://github.com/QuantA-group/QuantEM/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%3C2.0-6133BD?logo=qiskit&logoColor=white)](https://github.com/Qiskit/qiskit)

Error detection will be an integral part of quantum processing, especially for near-term demonstrations of quantum advantage. Unfortunately, adding quantum error detection (QED) to arbitrary quantum circuits is a daunting task as deep knowledge of detecting codes and required circuit modifications is required.

This project contains the first Python-based compiler that automatically integrates QED into quantum circuits, simplifying the inclusion of QED into quantum error management workflows. High-level quantum programs are translated into error-detectable low-level circuits that are quantum machine compatable. You can read more about the project [here](https://arxiv.org/abs/2509.15505).

## Getting Started

**Prerequisites:** Python 3.9+ and a Rust toolchain (needed to build the compiled extension).

To install the `quantem` package from source:

```sh
pip install .
```

Please visit the `example_notebooks` directory for `quantem` tutorials. The additional dependencies needed to run the notebooks can be installed using:

```sh
pip install .[notebook]
```

## Quickstart

To protect a payload circuit with Pauli Check Sandwiching:

```python
from qiskit import QuantumCircuit
from quantem import QEDCompiler, QEDStrategy

# Build the payload circuit you want to protect
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Automatically integrate quantum error detection
compiler = QEDCompiler()
result = compiler.compile(circuit, strategy=QEDStrategy.PCS, num_checks=2)

print(result.strategy_used)   # QEDStrategy.PCS
print(result.metadata)        # checks, ancilla qubits, and other details
print(result.circuit)         # the error-detectable circuit
```

Pass `strategy=QEDStrategy.AUTO` (the default) to let `QuantEM` choose a strategy based on the circuit, or use `compiler.compile_pce(circuit, check_counts=[1, 2, 3])` to generate a family of circuits for Pauli Check Extrapolation.

## Supported QED Protocols

+ **Pauli Check Sandwiching, PCS**: PCS is technique used to detect and mitigate errors. PCS surrounds a payload circuit, $𝑈$ , with controlled Pauli operator checks that are selected such that $RUL = 𝑈$ . Errors on $𝑈$ can be detected
on an ancilla through phase kickback. The errors that are detected in $U$ anticommute with the Pauli operators in the selected checks.
+ **Ancilla-free Pauli Checks, AFPC**: AFPC does not include an ancilla that is measured for syndrome information. Instead, syndrome information is read out directly from the qubit targeted by Pauli checks. AFPC is effective for application-specific error characterization.
+ **Pauli Check Extrapolation, PCE**: PCE builds on Pauli check techniques to mitigate errors by running a payload circuit with varying numbers of Pauli checks and extrapolating the resulting expectation values to the maximum check limit. Fitting a model across the check counts (e.g. linear or exponential) yields an error-mitigated estimate of the ideal expectation value. See the [PCE demo notebook](example_notebooks/pce_demo.ipynb) for a walkthrough.
+ **Iceberg Code**: The Iceberg code is a distance 2 code, [[k+2, k, 2]] for even k, that scales efficiently with the number of 
logical qubits. The Iceberg code requires only two additional qubits for the encoded state. The code implements fault-tolerant initial state preparation and syndrome measurement circuits capable of detecting any single-qubit error.

## Citation

If `QuantEM` is useful in your work, we'd be grateful if you cited our [paper](https://arxiv.org/abs/2509.15505). A BibTeX entry is also available in [`citation.bib`](citation.bib):

```bibtex
@misc{liu2025quantemquantumerrormanagement,
      title={QuantEM: The quantum error management compiler},
      author={Ji Liu and Quinn Langfitt and Mingyoung Jessica Jeng and Alvin Gonzales and Noble Agyeman-Bobie and Kaiya Jones and Siddharth Vijaymurugan and Daniel Dilley and Zain H. Saleem and Nikos Hardavellas and Kaitlin N. Smith},
      year={2025},
      eprint={2509.15505},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2509.15505},
}
```
