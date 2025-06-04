#### `qed_compiler`

---
Error detection will be an integral part of quantum processing, especially for near-term demonstrations of quantum advantage. Unfortunately, adding quantum error detection (QED) to arbitrary quantum circuits is a daunting task as deep knowledge of detecting codes and required circuit modifications is required.

This project contains the first Python-based compiler that automatically integrates QED into quantum circuits. High-level quantum programs are translated into error-detectable low-level circuits that are quantum machine compatable.

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

## Supported QEC Protocols

+ **Pauli Check Sandwiching, PCS**: PCS is technique used to detect and mitigate errors. PCS surrounds a payload circuit, $𝑈$ , with controlled Pauli operator checks that are selected such that $RUL = 𝑈$ . Errors on $𝑈$ can be detected
on an ancilla through phase kickback. The errors that are detected in $U$ anticommute with the Pauli operators in the selected checks.
+ **Pauli Check Extrapolation, PCE**:
+ **Ancilla-free Pauli Checks, AFPC**:
+ **Iceberg Code** (*Coming Soon*):
