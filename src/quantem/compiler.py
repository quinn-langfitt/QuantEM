"""QuantEM Quantum Error Detection Compiler.

This module provides the main QEDCompiler class for automatically integrating
quantum error detection into quantum circuits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from quantem.iceberg_code import build_iceberg_circuit
from quantem.utils import (
    convert_to_PCS_circ,
    convert_to_ancilla_free_PCS_circ,
    find_largest_clifford_block,
)
from qiskit_addon_utils.slicing import slice_by_depth

if TYPE_CHECKING:
    from typing import Mapping, Sequence
    from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------- #
# Configuration and Data Classes
# ---------------------------------------------------------------------------- #

CLIFFORD_NAMES = {
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
    "swap",
}

DEFAULT_CLIFFORD_THRESHOLD = 0.4
DEFAULT_NUM_CHECKS = 2


class QEDStrategy(Enum):
    """Available quantum error detection strategies."""
    PCS = "pcs"
    ICEBERG = "iceberg"
    AFPC = "afpc"
    AUTO = "auto"


@dataclass
class CompilationResult:
    """Result of QED compilation.
    
    Attributes:
        circuit: The compiled quantum circuit with error detection
        strategy_used: The QED strategy that was applied
        metadata: Additional information about the compilation
    """
    circuit: QuantumCircuit
    strategy_used: QEDStrategy
    metadata: Dict[str, Any]

# ---------------------------------------------------------------------------- #
# QED Compiler Class
# ---------------------------------------------------------------------------- #


class QEDCompiler:
    """Quantum Error Detection Compiler.
    
    Automatically integrates quantum error detection into quantum circuits
    by analyzing circuit structure and applying appropriate QED strategies.
    
    Attributes:
        clifford_threshold: Threshold for Clifford gate fraction to use PCS
        default_num_checks: Default number of Pauli checks to insert
        logger: Logger instance for compilation information
    """
    
    def __init__(
        self,
        clifford_threshold: float = DEFAULT_CLIFFORD_THRESHOLD,
        default_num_checks: int = DEFAULT_NUM_CHECKS,
        verbose: bool = False,
    ):
        """Initialize QED compiler.
        
        Args:
            clifford_threshold: Fraction of Clifford gates to trigger PCS strategy
            default_num_checks: Default number of checks for PCS/AFPC
            verbose: Enable verbose logging output
        """
        self.clifford_threshold = clifford_threshold
        self.default_num_checks = default_num_checks
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(name)s] %(levelname)s: %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
    
    def compile(
        self,
        circuit: QuantumCircuit,
        strategy: QEDStrategy = QEDStrategy.AUTO,
        layout: Optional[Mapping] = None,
        gateset: Optional[Sequence] = None,
        num_checks: Optional[int] = None,
        **kwargs,
    ) -> CompilationResult:
        """Compile quantum circuit with error detection.
        
        Args:
            circuit: Input quantum circuit to protect
            strategy: QED strategy to use (AUTO selects automatically)
            layout: Qubit layout mapping (for future use)
            gateset: Supported gate set (for future use)
            num_checks: Number of Pauli checks (uses default if None)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            CompilationResult with protected circuit and metadata
            
        Raises:
            ValueError: If invalid strategy or parameters provided
        """
        if num_checks is None:
            num_checks = self.default_num_checks
        
        # Validate strategy parameter
        if not isinstance(strategy, QEDStrategy):
            raise TypeError(f"strategy must be a QEDStrategy enum, got {type(strategy).__name__}")
            
        # Strategy selection
        if strategy == QEDStrategy.AUTO:
            strategy = self._select_strategy(circuit)
            
        self.logger.info(f"Compiling circuit with {strategy.value.upper()} strategy")
        
        # Apply chosen strategy
        if strategy == QEDStrategy.PCS:
            result_circuit, metadata = self._compile_pcs(
                circuit, num_checks, **kwargs
            )
        elif strategy == QEDStrategy.AFPC:
            result_circuit, metadata = self._compile_afpc(
                circuit, num_checks, **kwargs
            )
        elif strategy == QEDStrategy.ICEBERG:
            result_circuit, metadata = self._compile_iceberg(circuit, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return CompilationResult(
            circuit=result_circuit,
            strategy_used=strategy,
            metadata=metadata,
        )
    
    def analyze_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze circuit properties relevant to QED strategy selection.
        
        Args:
            circuit: Circuit to analyze
            
        Returns:
            Dictionary with analysis results
        """
        slices = slice_by_depth(circuit, 1)
        total_depth = len(slices)
        
        if total_depth == 0:
            clifford_fraction = 0.0
        else:
            start, end, block = find_largest_clifford_block(slices)
            block_depth = len(block) if start is not None else 0
            clifford_fraction = block_depth / total_depth
            
        return {
            "total_depth": total_depth,
            "num_qubits": circuit.num_qubits,
            "num_gates": len(circuit.data),
            "clifford_block_fraction": clifford_fraction,
            "recommended_strategy": (
                QEDStrategy.PCS if clifford_fraction >= self.clifford_threshold 
                else QEDStrategy.ICEBERG
            ),
        }
    
    def _select_strategy(self, circuit: QuantumCircuit) -> QEDStrategy:
        """Automatically select QED strategy based on circuit analysis."""
        analysis = self.analyze_circuit(circuit)
        clifford_fraction = analysis["clifford_block_fraction"]
        
        self.logger.info(
            f"Circuit analysis: {clifford_fraction:.1%} Clifford block fraction"
        )
        
        strategy = analysis["recommended_strategy"]
        self.logger.info(f"Automatically selected {strategy.value.upper()} strategy")
        
        return strategy
        
    def _compile_pcs(
        self, circuit: QuantumCircuit, num_checks: int, **kwargs
    ) -> tuple[QuantumCircuit, Dict[str, Any]]:
        """Apply Pauli Check Sandwiching strategy."""
        try:
            sign_list, qed_circuit = convert_to_PCS_circ(
                circuit, circuit.num_qubits, num_checks, 
                barriers=kwargs.get('barriers', True),
                reverse=kwargs.get('reverse', False)
            )
            
            metadata = {
                "sign_list": sign_list,
                "num_checks": num_checks,
                "ancilla_qubits": num_checks,
                "total_qubits": qed_circuit.num_qubits,
                "qubit_overhead": qed_circuit.num_qubits - circuit.num_qubits,
            }
            
            return qed_circuit, metadata
            
        except Exception as e:
            raise RuntimeError(f"PCS compilation failed: {e}") from e
    
    def _compile_afpc(
        self, circuit: QuantumCircuit, num_checks: int, **kwargs
    ) -> tuple[QuantumCircuit, Dict[str, Any]]:
        """Apply Ancilla-Free Pauli Checks strategy."""
        try:
            sign_list, qed_circuit, left_mappings, right_mappings = (
                convert_to_ancilla_free_PCS_circ(
                    circuit, circuit.num_qubits, num_checks,
                    barriers=kwargs.get('barriers', True),
                    reverse=kwargs.get('reverse', False)
                )
            )
            
            metadata = {
                "sign_list": sign_list,
                "num_checks": num_checks,
                "left_mappings": left_mappings,
                "right_mappings": right_mappings,
                "ancilla_qubits": 0,  # AFPC uses no ancillas
                "total_qubits": qed_circuit.num_qubits,
                "qubit_overhead": 0,
            }
            
            return qed_circuit, metadata
            
        except Exception as e:
            raise RuntimeError(f"AFPC compilation failed: {e}") from e
    
    def _compile_iceberg(
        self, circuit: QuantumCircuit, **kwargs
    ) -> tuple[QuantumCircuit, Dict[str, Any]]:
        """Apply Iceberg code strategy."""
        try:
            qed_circuit, reg_bundle = build_iceberg_circuit(
                circuit,
                optimize_level=kwargs.get('optimize_level', 3),
                attach_readout=kwargs.get('attach_readout', True)
            )
            
            metadata = {
                "register_bundle": reg_bundle,
                "ancilla_qubits": 2,  # Iceberg uses 2 ancillas
                "total_qubits": qed_circuit.num_qubits,
                "qubit_overhead": 2,
                "distance": 2,  # Iceberg is distance-2 code
            }
            
            return qed_circuit, metadata
            
        except Exception as e:
            raise RuntimeError(f"Iceberg compilation failed: {e}") from e


