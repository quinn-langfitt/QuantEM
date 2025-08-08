"""Tests for the QEDCompiler class."""

import pytest
import numpy as np
from typing import List
from qiskit import QuantumCircuit

from quantem import QEDCompiler, QEDStrategy, CompilationResult


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
        
        # Iceberg should add exactly 2 ancilla qubits
        assert result.metadata["ancilla_qubits"] == 2
        assert result.metadata["qubit_overhead"] == 2
        assert result.metadata["distance"] == 2
    
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