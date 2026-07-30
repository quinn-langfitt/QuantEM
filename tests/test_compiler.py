"""Tests for the QEDCompiler class."""

import pytest
import numpy as np
from dataclasses import asdict
from typing import List
from qiskit import QuantumCircuit

from quantem import QEDCompiler, QEDStrategy, CompilationResult
from quantem.validation import IcebergOptions


class RecordingIcebergCompiler(QEDCompiler):
    """Avoid circuit synthesis when testing request normalization."""

    def _compile_iceberg(self, circuit, options: IcebergOptions):
        return circuit.copy(), asdict(options)


class TestQEDCompiler:
    """Test cases for QEDCompiler."""
    
    @pytest.fixture
    def simple_circuit(self):
        """Create a simple test circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.5, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        return qc
    
    @pytest.fixture
    def clifford_circuit(self):
        """Create a Clifford-heavy circuit."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        qc.cx(2, 3)
        qc.s(3)
        return qc
    
    @pytest.fixture
    def qaoa_circuit(self):
        """Create a 6-qubit QAOA circuit matching the Iceberg notebook example."""
        def generate_all_zz_terms(num_qubits: int) -> List[str]:
            zz_terms = []
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    pauli = ['I'] * num_qubits
                    pauli[i] = 'Z'
                    pauli[j] = 'Z'
                    zz_terms.append(''.join(pauli))
            return zz_terms

        def apply_pauli_rotation(qc: QuantumCircuit, pauli: str, angle: float):
            """Applies e^{-i angle * Pauli/2} for specific structured Paulis."""
            if pauli.count('Z') == 2 and all(p in 'IZ' for p in pauli):
                # Two-qubit ZZ term → RZZ
                idx = [i for i, p in enumerate(pauli) if p == 'Z']
                qc.rzz(2 * angle, idx[0], idx[1])
            elif pauli.count('X') == 1 and all(p in 'IX' for p in pauli):
                # Single-qubit X term → RX
                idx = pauli.index('X')
                qc.rx(2 * angle, idx)

        def build_qaoa_from_paulis(paulis: List[str], params: List[float]) -> QuantumCircuit:
            n_qubits = len(paulis[0])
            qc = QuantumCircuit(n_qubits)
            for pauli, angle in zip(paulis, params):
                apply_pauli_rotation(qc, pauli, angle)
            return qc

        # QAOA for MAXCUT with 6 qubits
        num_qubits = 6
        cost_hamiltonian = generate_all_zz_terms(num_qubits)
        mixer_hamiltonian = ['XIIIII', 'IXIIII', 'IIXIII', 'IIIXII', 'IIIIXI', 'IIIIIX']
        test_paulis = (cost_hamiltonian + mixer_hamiltonian) * 2
        test_params = [3.271] * len(cost_hamiltonian) + [2.874] * len(mixer_hamiltonian) + [3.271] * len(cost_hamiltonian) + [2.874] * len(mixer_hamiltonian)
        
        return build_qaoa_from_paulis(test_paulis, test_params)
    
    def test_compiler_initialization(self):
        """Test basic compiler initialization."""
        compiler = QEDCompiler()
        assert compiler.clifford_threshold == 0.4
        assert compiler.default_num_checks == 2
        
        # Test with custom parameters
        compiler_custom = QEDCompiler(
            clifford_threshold=0.5, 
            default_num_checks=3, 
            verbose=True
        )
        assert compiler_custom.clifford_threshold == 0.5
        assert compiler_custom.default_num_checks == 3

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_invalid_clifford_threshold(self, threshold):
        with pytest.raises(ValueError, match="between 0 and 1"):
            QEDCompiler(clifford_threshold=threshold)

    @pytest.mark.parametrize("num_checks", [-1, 1.5, True])
    def test_invalid_default_num_checks(self, num_checks):
        expected = TypeError if isinstance(num_checks, (float, bool)) else ValueError
        with pytest.raises(expected, match="non-negative integer"):
            QEDCompiler(default_num_checks=num_checks)

    @pytest.mark.parametrize("num_checks", [-1, 1.5, True])
    def test_invalid_compile_num_checks(self, simple_circuit, num_checks):
        expected = TypeError if isinstance(num_checks, (float, bool)) else ValueError
        compiler = QEDCompiler()
        with pytest.raises(expected, match="non-negative integer"):
            compiler.compile(
                simple_circuit,
                strategy=QEDStrategy.PCS,
                num_checks=num_checks,
            )

    @pytest.mark.parametrize("entry_point", ["compile", "compile_pce", "analyze"])
    def test_public_entry_points_require_quantum_circuit(self, entry_point):
        compiler = QEDCompiler()

        with pytest.raises(TypeError, match="must be a QuantumCircuit"):
            if entry_point == "compile":
                compiler.compile("not a circuit", strategy=QEDStrategy.PCS)
            elif entry_point == "compile_pce":
                compiler.compile_pce("not a circuit", check_counts=[0])
            else:
                compiler.analyze_circuit("not a circuit")

    @pytest.mark.parametrize(
        ("strategy", "option"),
        [
            (QEDStrategy.PCS, {"barier": True}),
            (QEDStrategy.AFPC, {"optimize_level": 1}),
            (QEDStrategy.ICEBERG, {"only_X_checks": True}),
        ],
    )
    def test_compile_rejects_unknown_or_misapplied_options(
        self, simple_circuit, strategy, option
    ):
        compiler = QEDCompiler()

        with pytest.raises(TypeError, match="unexpected .* option"):
            compiler.compile(simple_circuit, strategy=strategy, **option)

    @pytest.mark.parametrize(
        ("option_name", "option_value"),
        [
            ("barriers", 1),
            ("reverse", "false"),
            ("only_X_checks", None),
            ("only_Z_checks", 0),
        ],
    )
    def test_pcs_options_require_strict_booleans(
        self, simple_circuit, option_name, option_value
    ):
        compiler = QEDCompiler()

        with pytest.raises(TypeError, match=f"{option_name} must be a bool"):
            compiler.compile(
                simple_circuit,
                strategy=QEDStrategy.PCS,
                **{option_name: option_value},
            )

    def test_iceberg_rejects_num_checks(self):
        compiler = QEDCompiler()

        with pytest.raises(TypeError, match="not supported by ICEBERG"):
            compiler.compile(
                QuantumCircuit(2),
                strategy=QEDStrategy.ICEBERG,
                num_checks=1,
            )

    @pytest.mark.parametrize("parameter_name", ["layout", "gateset"])
    def test_future_parameters_fail_instead_of_being_ignored(
        self, simple_circuit, parameter_name
    ):
        compiler = QEDCompiler()

        with pytest.raises(NotImplementedError, match="not implemented"):
            compiler.compile(
                simple_circuit,
                strategy=QEDStrategy.PCS,
                **{parameter_name: {}},
            )
    
    def test_circuit_analysis(self, simple_circuit):
        """Test circuit analysis functionality."""
        compiler = QEDCompiler()
        analysis = compiler.analyze_circuit(simple_circuit)
        
        assert isinstance(analysis, dict)
        assert "total_depth" in analysis
        assert "num_qubits" in analysis
        assert "num_gates" in analysis
        assert "clifford_block_fraction" in analysis
        assert "recommended_strategy" in analysis
        
        assert analysis["num_qubits"] == 3
        assert analysis["num_gates"] == 7
        assert isinstance(analysis["clifford_block_fraction"], float)
        assert analysis["recommended_strategy"] in [QEDStrategy.PCS, QEDStrategy.ICEBERG]
    
    def test_auto_strategy_selection(self, simple_circuit, clifford_circuit):
        """Test automatic strategy selection."""
        compiler = QEDCompiler()
        
        # Test with simple circuit
        result = compiler.compile(simple_circuit, strategy=QEDStrategy.AUTO)
        assert isinstance(result, CompilationResult)
        assert result.strategy_used in [QEDStrategy.PCS, QEDStrategy.ICEBERG]
        
        # Test with Clifford-heavy circuit
        print(clifford_circuit)
        result_clifford = compiler.compile(clifford_circuit, strategy=QEDStrategy.AUTO)
        print(result_clifford)
        assert isinstance(result_clifford, CompilationResult)
        assert result_clifford.strategy_used in [QEDStrategy.PCS, QEDStrategy.ICEBERG]

    def test_auto_does_not_select_infeasible_iceberg_width(self):
        compiler = QEDCompiler()
        result = compiler.compile(
            QuantumCircuit(1),
            strategy=QEDStrategy.AUTO,
            num_checks=0,
        )

        assert result.strategy_used == QEDStrategy.PCS
    
    def test_pcs_compilation(self, simple_circuit):
        """Test PCS strategy compilation."""
        compiler = QEDCompiler()
        result = compiler.compile(
            simple_circuit, 
            strategy=QEDStrategy.PCS, 
            num_checks=2
        )
        
        assert isinstance(result, CompilationResult)
        assert result.strategy_used == QEDStrategy.PCS
        assert result.circuit.num_qubits >= simple_circuit.num_qubits
        
        # Check metadata
        assert "sign_list" in result.metadata
        assert "num_checks" in result.metadata
        assert "ancilla_qubits" in result.metadata
        assert result.metadata["num_checks"] == 2
    
    def test_afpc_compilation(self, simple_circuit):
        """Test AFPC strategy compilation."""
        compiler = QEDCompiler()
        result = compiler.compile(
            simple_circuit, 
            strategy=QEDStrategy.AFPC, 
            num_checks=2
        )
        
        assert isinstance(result, CompilationResult)
        assert result.strategy_used == QEDStrategy.AFPC
        
        # AFPC should not add ancilla qubits
        assert result.metadata["ancilla_qubits"] == 0
        assert result.metadata["qubit_overhead"] == 0
    
    def test_iceberg_compilation(self, qaoa_circuit):
        """Test Iceberg strategy compilation."""
        compiler = QEDCompiler()
        result = compiler.compile(
            qaoa_circuit, 
            strategy=QEDStrategy.ICEBERG
        )
        
        assert isinstance(result, CompilationResult)
        assert result.strategy_used == QEDStrategy.ICEBERG
        
        # Iceberg has two code qubits plus preparation/syndrome/readout ancillas.
        assert result.metadata["code_qubit_overhead"] == 2
        assert result.metadata["qubit_overhead"] == (
            result.circuit.num_qubits - qaoa_circuit.num_qubits
        )
        assert result.metadata["ancilla_qubits"] == (
            result.metadata["qubit_overhead"] - 2
        )
        assert result.metadata["distance"] == 2

    def test_iceberg_rejects_odd_logical_width(self):
        compiler = QEDCompiler()
        with pytest.raises(ValueError, match="positive, even"):
            compiler.compile(
                QuantumCircuit(3),
                strategy=QEDStrategy.ICEBERG,
            )

    @pytest.mark.parametrize(
        ("optimize_level", "expected_error"),
        [(-1, ValueError), (4, ValueError), (True, TypeError), (1.5, TypeError)],
    )
    def test_iceberg_rejects_invalid_optimization_level(
        self, optimize_level, expected_error
    ):
        compiler = QEDCompiler()
        with pytest.raises(expected_error, match="between 0 and 3"):
            compiler.compile(
                QuantumCircuit(2),
                strategy=QEDStrategy.ICEBERG,
                optimize_level=optimize_level,
            )

    def test_iceberg_options_require_strict_boolean(self):
        compiler = QEDCompiler()

        with pytest.raises(TypeError, match="attach_readout must be a bool"):
            compiler.compile(
                QuantumCircuit(2),
                strategy=QEDStrategy.ICEBERG,
                attach_readout=1,
            )

    def test_iceberg_schedule_modes_are_exclusive(self):
        compiler = QEDCompiler()

        with pytest.raises(ValueError, match="either syndrome_interval"):
            compiler.compile(
                QuantumCircuit(2),
                strategy=QEDStrategy.ICEBERG,
                syndrome_interval=2,
                total_syndrome_cycles=2,
            )

    def test_iceberg_total_cycles_selects_cycle_count_mode(self):
        compiler = RecordingIcebergCompiler()

        result = compiler.compile(
            QuantumCircuit(2),
            strategy=QEDStrategy.ICEBERG,
            total_syndrome_cycles=3,
        )

        assert result.metadata["syndrome_interval"] == 0
        assert result.metadata["total_syndrome_cycles"] == 3

    def test_iceberg_rejects_impossible_syndrome_schedule(self):
        compiler = QEDCompiler()
        circuit = QuantumCircuit(2)
        circuit.rxx(0.5, 0, 1)

        with pytest.raises(ValueError, match="cannot exceed"):
            compiler.compile(
                circuit,
                strategy=QEDStrategy.ICEBERG,
                syndrome_interval=0,
                total_syndrome_cycles=2,
            )
    
    def test_default_num_checks(self, simple_circuit):
        """Test that default number of checks is used when not specified."""
        compiler = QEDCompiler(default_num_checks=3)
        result = compiler.compile(simple_circuit, strategy=QEDStrategy.PCS)
        
        assert result.metadata["num_checks"] == 3
    
    def test_invalid_strategy(self, simple_circuit):
        """Test handling of invalid strategy."""
        compiler = QEDCompiler()
        
        # This should work since QEDStrategy is an enum
        with pytest.raises(TypeError):
            compiler.compile(simple_circuit, strategy="INVALID")
    
    def test_compilation_result_structure(self, simple_circuit):
        """Test that CompilationResult has the expected structure."""
        compiler = QEDCompiler()
        result = compiler.compile(simple_circuit, strategy=QEDStrategy.PCS, num_checks=1)
        
        assert hasattr(result, 'circuit')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'metadata')
        
        assert isinstance(result.circuit, QuantumCircuit)
        assert isinstance(result.strategy_used, QEDStrategy)
        assert isinstance(result.metadata, dict)


class TestQEDStrategies:
    """Test the QEDStrategy enum."""
    
    def test_strategy_enum_values(self):
        """Test that strategy enum has expected values."""
        assert QEDStrategy.PCS.value == "pcs"
        assert QEDStrategy.ICEBERG.value == "iceberg"
        assert QEDStrategy.AFPC.value == "afpc"
        assert QEDStrategy.AUTO.value == "auto"
    
    def test_strategy_enum_comparison(self):
        """Test strategy enum comparisons."""
        assert QEDStrategy.PCS == QEDStrategy.PCS
        assert QEDStrategy.PCS != QEDStrategy.ICEBERG


class TestCompilationResult:
    """Test the CompilationResult dataclass."""
    
    def test_compilation_result_creation(self):
        """Test creating a CompilationResult."""
        qc = QuantumCircuit(2)
        result = CompilationResult(
            circuit=qc,
            strategy_used=QEDStrategy.PCS,
            metadata={"test": "value"}
        )
        
        assert result.circuit == qc
        assert result.strategy_used == QEDStrategy.PCS
        assert result.metadata["test"] == "value"
