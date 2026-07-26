#### `QuantEM`: the quantum error management compiler

---
Error detection will be an integral part of quantum processing, especially for near-term demonstrations of quantum advantage. Unfortunately, adding quantum error detection (QED) to arbitrary quantum circuits is a daunting task as deep knowledge of detecting codes and required circuit modifications is required.

This project contains the first Python-based compiler that automatically integrates QED into quantum circuits, simplifying the inclusion of QED into quantum error management workflows. High-level quantum programs are translated into error-detectable low-level circuits that are quantum machine compatable. You can read more about the project [here](https://arxiv.org/abs/2509.15505).

## Getting Started

To install the `quantem` package from source:

```sh
pip install .
```

<br>
Please visit the `example_notebooks` directory for `qed_compiler` tutorials. The additional dependencies needed to run the notebooks can be installed using:

```sh
pip install .[notebook]
```

## Supported QED Protocols

+ **Pauli Check Sandwiching, PCS**: PCS is technique used to detect and mitigate errors. PCS surrounds a payload circuit, $𝑈$ , with controlled Pauli operator checks that are selected such that $RUL = 𝑈$ . Errors on $𝑈$ can be detected
on an ancilla through phase kickback. The errors that are detected in $U$ anticommute with the Pauli operators in the selected checks.
+ **Ancilla-free Pauli Checks, AFPC**: AFPC does not include an ancilla that is measured for syndrome information. Instead, syndrome information is read out directly from the qubit targeted by Pauli checks. AFPC is effective for application-specific error characterization.
+ **Pauli Check Extrapolation, PCE**: PCE builds on Pauli check techniques to mitigate errors by running a payload circuit with varying numbers of Pauli checks and extrapolating the resulting expectation values to the maximum check limit. Fitting a model across the check counts (e.g. linear or exponential) yields an error-mitigated estimate of the ideal expectation value. See the [PCE demo notebook](example_notebooks/pce_demo.ipynb) for a walkthrough.
+ **Iceberg Code**: The Iceberg code is a distance 2 code, [[k+2, k, 2]] for even k, that scales efficiently with the number of 
logical qubits. The Iceberg code requires only two additional qubits for the encoded state. The code implements fault-tolerant initial state preparation and syndrome measurement circuits capable of detecting any single-qubit error.
