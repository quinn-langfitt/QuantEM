import sys
from unittest.mock import MagicMock

# Mock the Rust extension module to run tests without building binaries
mock_rust = MagicMock()
mock_rust.sabre = MagicMock()
sys.modules["quantem.rust"] = mock_rust
sys.modules["quantem.rust.sabre"] = mock_rust.sabre

# Mock qiskit_addon_utils
mock_qau = MagicMock()
sys.modules["qiskit_addon_utils"] = mock_qau
sys.modules["qiskit_addon_utils.slicing"] = mock_qau.slicing

# Mock mapomatic
sys.modules["mapomatic"] = MagicMock()

import pytest
from qiskit import QuantumCircuit
from quantem.compiler import QEDCompiler, PCECompilationResult, QEDStrategy
from quantem.pauli_check_extrapolation import analyze_pce_results

class MockCompiler(QEDCompiler):
    """Mock compiler to avoid full PCS logic overhead in unit tests."""
    def _compile_pcs(self, circuit, num_checks, **kwargs):
        # Return a dummy circuit with metadata
        qc = circuit.copy()
        qc.metadata = {"num_checks": num_checks}
        return qc, {"checks_added": num_checks}

def test_compile_pce_structure():
    """Test that compile_pce returns the correct structure."""
    compiler = MockCompiler()
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    check_counts = [1, 2, 4]
    result = compiler.compile_pce(circuit, check_counts=check_counts)

    assert isinstance(result, PCECompilationResult)
    assert len(result.circuits) == 3
    assert result.strategy_used == QEDStrategy.PCS
    assert 1 in result.circuits
    assert 2 in result.circuits
    assert 4 in result.circuits
    assert result.circuits[2].metadata["num_checks"] == 2

def test_analyze_pce_results_exponential():
    """Test extrapolation logic with synthetic exponential data."""
    # Model: y = A + B * C^x
    # Let y -> 1.0 as x -> inf
    # Try y = 1.0 - 0.5 * (0.5)^x
    # x=1: 0.75
    # x=2: 0.875
    # x=3: 0.9375
    # x=4: 0.96875

    data = {
        1: 0.75,
        2: 0.875,
        3: 0.9375,
        4: 0.96875
    }

    # We extrapolate to a large n to see if it approaches 1.0
    result = analyze_pce_results(data, n_max=100, model="exponential")

    # Check if we are close to 1.0 (allow slightly larger tolerance due to fitting)
    assert result["extrapolated_value"] == pytest.approx(1.0, abs=0.05)
    assert result["model"] == "exponential"

def test_analyze_pce_results_linear():
    """Test extrapolation logic with synthetic linear data."""
    # y = 0.6 + 0.1 * x
    data = {
        1: 0.7,
        2: 0.8,
        3: 0.9
    }

    # Extrapolate to x=4 -> 1.0
    result = analyze_pce_results(data, n_max=4, model="linear")

    assert result["extrapolated_value"] == pytest.approx(1.0, abs=0.01)

def test_analyze_pce_results_not_enough_data():
    """Test graceful handling of insufficient data."""
    data = {1: 0.75}
    result = analyze_pce_results(data, n_max=10)

    assert result["model"] == "none (too few points)"
    assert result["extrapolated_value"] == 0.75
