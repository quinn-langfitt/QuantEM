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

import itertools
import logging
import pickle
import math
from qiskit.transpiler.basepasses import (
    TransformationPass, 
    AnalysisPass
)
from typing import Any
from typing import Callable
from collections import defaultdict
from qiskit.dagcircuit import (
    DAGOutNode, 
    DAGOpNode
)

from qiskit import (
    transpile, 
    QuantumCircuit
)
from typing import List
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager

from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from qiskit.transpiler.passes import RemoveBarriers

import mapomatic as mm
from networkx import Graph
from networkx.algorithms.approximation.clique import maximum_independent_set
from itertools import combinations

from collections import deque

from quantem.pauli_checks import (
    ChecksFinder,
    add_pauli_checks
)

logger = logging.getLogger(__name__)


def check_to_ancilla_free_circ(check_str: str, num_qubits: int) -> tuple[QuantumCircuit, dict]:
    """Convert a Pauli check string into an ancilla-free measurement circuit.
    
    This function generates a circuit that rotates the basis of target qubits to 
    measure the specified Pauli operator (X or Z) in the standard Z-basis.

    Args:
        check_str: String representation of the check (e.g., "+1IXZ").
        num_qubits: Total number of qubits in the circuit.

    Returns:
        tuple: (circuit, mapping)
            - circuit: QuantumCircuit implementing basis rotations.
            - mapping: Dictionary mapping qubit indices to their Pauli basis ('X' or 'Z').
    """
    op_str = check_str[2:]
    qc = QuantumCircuit(num_qubits)
    mapping = {}
    for i, op in enumerate(op_str):
        target = num_qubits - 1 - i
        if op.upper() == "X":
            if mapping.get(target) != "X":
                qc.h(target)
            mapping[target] = "X"
        elif op.upper() == "Z":
            if target not in mapping:
                mapping[target] = "Z"
    return qc, mapping


def convert_to_ancilla_free_PCS_circ(
    circ: QuantumCircuit, 
    num_qubits: int, 
    num_checks: int, 
    barriers: bool = False, 
    reverse: bool = False, 
    only_X_checks: bool = False, 
    only_Z_checks: bool = False
) -> tuple[List[str], QuantumCircuit, List[dict], List[dict]]:
    """Implement Ancilla-Free Pauli Check Sandwiching (AFPC).
    
    Finds and inserts Pauli checks that can be measured directly on data qubits
    without using extra ancilla qubits.

    Args:
        circ: Input quantum circuit.
        num_qubits: Number of qubits in the circuit.
        num_checks: Number of checks to generate.
        barriers: Whether to add barriers around the original circuit.
        reverse: If True, prioritizes higher-weight checks.
        only_X_checks: Restrict to X-type checks only.
        only_Z_checks: Restrict to Z-type checks only.

    Returns:
        tuple: (sign_list, final_circ, left_mappings, right_mappings)
            - sign_list: List of check signs ("+1" or "-1").
            - final_circ: Protected circuit with checks.
            - left_mappings: List of qubit mappings for left checks.
            - right_mappings: List of qubit mappings for right checks.
            
    Raises:
        ValueError: If both only_X_checks and only_Z_checks are True.
    """

    if only_X_checks and only_Z_checks:
        raise ValueError("Cannot specify both only_X_checks and only_Z_checks")

    if only_X_checks:
        characters = ["I", "X"]
    elif only_Z_checks:
        characters = ["I", "Z"]
    else:
        characters = ["I", "X", "Z"]

    candidate_strings = [
        "".join(p)
        for p in itertools.product(characters, repeat=num_qubits)
        if not all(c == "I" for c in p)
    ]

    def weight(s):
        return sum(1 for c in s if c != "I")

    sorted_strings = sorted(candidate_strings, key=weight)
    if reverse:
        sorted_strings.reverse()

    test_finder = ChecksFinder(num_qubits, circ)
    selected_pairs = []
    left_mappings_list = []
    right_mappings_list = []
    sign_list = []

    for s in sorted_strings:
        try:
            result = test_finder.find_checks_sym(pauli_group_elem=list(s))
            left_str, right_str = result.p1_str, result.p2_str
            _, lm = check_to_ancilla_free_circ(left_str, num_qubits)
            _, rm = check_to_ancilla_free_circ(right_str, num_qubits)
            conflict = False
            # Rule 1: For left check, if any accepted candidate already has a mapping for a qubit,
            # the new candidate must have the same mapping on that qubit.
            for accepted in left_mappings_list:
                for q in lm:
                    if q in accepted and accepted[q] != lm[q]:
                        conflict = True
                        break
                if conflict:
                    break
            # Rule 2: For right check, each qubit can appear only once across accepted candidates.
            for accepted in right_mappings_list:
                for q in rm:
                    if q in accepted:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            left_mappings_list.append(lm)
            right_mappings_list.append(rm)
            selected_pairs.append([left_str, right_str])
            sign_list.append(left_str[:2])
            if len(selected_pairs) >= num_checks:
                break
        except Exception as e:
            continue
    if len(selected_pairs) < num_checks:
        logger.warning("Less checks found than required: %d < %d", len(selected_pairs), num_checks)

    final_left_check_circ = QuantumCircuit(num_qubits)
    for lm in left_mappings_list:
        for q in range(num_qubits):
            if lm.get(q, "Z") == "X":
                final_left_check_circ.h(q)
    right_check_circ = QuantumCircuit(num_qubits)
    for pair in selected_pairs:
        rc, _ = check_to_ancilla_free_circ(pair[1], num_qubits)
        right_check_circ = right_check_circ.compose(rc)

    final_circ = final_left_check_circ.copy()
    if barriers:
        final_circ.barrier()
    final_circ = final_circ.compose(circ)
    if barriers:
        final_circ.barrier()
    final_circ = final_circ.compose(right_check_circ)

    return sign_list, final_circ, left_mappings_list, right_mappings_list


def convert_to_PCS_circ(
    circ: QuantumCircuit, 
    num_qubits: int, 
    num_checks: int, 
    barriers: bool = False, 
    reverse: bool = False, 
    only_X_checks: bool = False, 
    only_Z_checks: bool = False
) -> tuple[List[str], QuantumCircuit]:
    """Implement Pauli Check Sandwiching (PCS) with ancilla qubits.
    
    Wraps the input circuit with Pauli checks that propagate through the circuit
    to detect errors. Uses additional ancilla qubits for syndrome measurement.

    Args:
        circ: The input quantum circuit to protect.
        num_qubits: Number of compute qubits.
        num_checks: Number of check layers to insert.
        barriers: Whether to add barriers around the component blocks.
        reverse: If True, searches for checks in reverse order.
        only_X_checks: Restrict to X-type right checks.
        only_Z_checks: Restrict to Z-type right checks.

    Returns:
        tuple: (sign_list, final_circ)
            - sign_list: List of check signs ("+1", "-1").
            - final_circ: The protected circuit including check ancillas.
            
    Raises:
        ValueError: If both only_X_checks and only_Z_checks are True.
    """
    total_qubits = num_qubits + num_checks

    if only_X_checks and only_Z_checks:
        raise ValueError("Cannot specify both only_X_checks and only_Z_checks")

    if only_X_checks:
        characters = ["I", "X"]
    elif only_Z_checks:
        characters = ["I", "Z"]
    else:
        characters = ["I", "X", "Z"]

    strings = [
        "".join(p)
        for p in itertools.product(characters, repeat=num_qubits)
        if not all(c == "I" for c in p)
    ]

    def weight(pauli_string):
        return sum(1 for char in pauli_string if char != "I")

    sorted_strings = sorted(strings, key=weight)
    if reverse:
        sorted_strings.reverse()

    test_finder = ChecksFinder(num_qubits, circ)
    p1_list = []
    found_checks = 0  # Counter for successful checks found

    for string in sorted_strings:
        string_list = list(string)
        try:
            result = test_finder.find_checks_sym(pauli_group_elem=string_list)
            p1_list.append([result.p1_str, result.p2_str])
            found_checks += 1
            logger.debug("Found check %d: %s, %s", found_checks, result.p1_str, result.p2_str)
            if found_checks >= num_checks:
                logger.debug("Required number of checks found")
                logger.debug("p1_list = %s", p1_list)
                break  # Stop the loop if we have found enough checks
        except Exception as e:
            continue  # Skip to the next iteration if an error occurs

    if found_checks < num_checks:
        logger.warning("Less checks found than required: %d < %d", found_checks, num_checks)

    initial_layout = {}
    for i in range(0, num_qubits):
        initial_layout[i] = [i]

    final_layout = {}
    for i in range(0, num_qubits):
        final_layout[i] = [i]

    # add pauli check on two sides:
    # specify the left and right pauli strings
    pcs_qc_list = []
    sign_list = []
    pl_list = []
    pr_list = []

    for i in range(0, num_checks):
        pl = p1_list[i][0][2:]
        pr = p1_list[i][1][2:]
        if i == 0:
            temp_qc = add_pauli_checks(
                circ,
                pl,
                pr,
                initial_layout,
                final_layout,
                False,
                False,
                False,
                False,
                barriers,
            )
            save_qc = add_pauli_checks(
                circ,
                pl,
                pr,
                initial_layout,
                final_layout,
                False,
                False,
                False,
                False,
                barriers,
            )
            prev_qc = temp_qc
        else:
            temp_qc = add_pauli_checks(
                prev_qc,
                pl,
                pr,
                initial_layout,
                final_layout,
                False,
                False,
                False,
                False,
                barriers,
            )
            save_qc = add_pauli_checks(
                prev_qc,
                pl,
                pr,
                initial_layout,
                final_layout,
                False,
                False,
                False,
                False,
                barriers,
            )
            prev_qc = temp_qc
        pl_list.append(pl)
        pr_list.append(pr)
        # sign_list.append(pauli_list[i][0][:2])
        sign_list.append(p1_list[i][0][:2])
        pcs_qc_list.append(save_qc)

    qc = pcs_qc_list[-1]  # return circuit with 'num_checks' implemented

    return sign_list, qc


def find_largest_clifford_block(slices):
    """
    Given a list of circuit slices (each slice is a Qiskit circuit or CircuitInstruction),
    returns a tuple: (start_index, end_index, block_slices) corresponding to the largest contiguous block
    of slices that contain only Clifford gates.

    If no slice is entirely Clifford, returns (None, None, []).

    In case two contiguous blocks have the same depth (i.e. same number of slices),
    the block with the larger total number of gates is chosen.
    """
    # Create a list indicating for each slice whether it is entirely Clifford,
    # and a parallel list with the gate count for that slice (only if it's Clifford).
    slice_is_clifford = []
    gate_counts = []  # number of gates in each slice (if Clifford, else 0)

    for slice_item in slices:
        # Try to get the underlying instruction data.
        if hasattr(slice_item, "data"):
            slice_data = slice_item.data
        elif hasattr(slice_item, "definition"):
            slice_data = slice_item.definition.data
        else:
            slice_is_clifford.append(False)
            gate_counts.append(0)
            continue

        is_all_clifford = True
        count = 0
        for instr in slice_data:
            gate_name = instr.operation.name
            count += 1
            if not is_clifford_gate(gate_name):
                is_all_clifford = False
                break
        slice_is_clifford.append(is_all_clifford)
        # Only count the gates if the slice is entirely Clifford.
        gate_counts.append(count if is_all_clifford else 0)

    max_len = 0  # maximum number of contiguous slices (depth)
    max_gate_count = 0  # total number of gates in that block
    max_start = None  # starting index of the best block

    current_len = 0  # current contiguous slice count
    current_gate_count = 0  # current total gate count in the block
    current_start = 0

    for i, flag in enumerate(slice_is_clifford):
        if flag:
            if current_len == 0:
                current_start = i
            current_len += 1
            current_gate_count += gate_counts[i]
        else:
            # Compare current block to the best so far.
            if (current_len > max_len) or (
                current_len == max_len and current_gate_count > max_gate_count
            ):
                max_len = current_len
                max_gate_count = current_gate_count
                max_start = current_start
            # Reset current block counters.
            current_len = 0
            current_gate_count = 0
    # Final check in case the longest block is at the end.
    if (current_len > max_len) or (
        current_len == max_len and current_gate_count > max_gate_count
    ):
        max_len = current_len
        max_gate_count = current_gate_count
        max_start = current_start

    if max_start is None:
        # No contiguous Clifford block found.
        return None, None, []

    max_end = max_start + max_len - 1
    block_slices = slices[max_start : max_end + 1]
    return max_start, max_end, block_slices


# Helper function for determining if a gate is Clifford.
def is_clifford_gate(gate_name):
    """
    Returns True if gate_name (string) is considered a Clifford gate.
    Adjust the set as needed.
    """
    clifford_set = {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "id", "barrier"}
    return gate_name.lower() in clifford_set


def expand_circuit(circ, target_num_qubits):
    """
    Expands the given circuit 'circ' so that it has 'target_num_qubits' total qubits.
    The original circuit's operations are mapped onto the first circ.num_qubits qubits,
    and extra qubits (ancillas) are added without any operations.
    """
    current_num = circ.num_qubits
    if current_num >= target_num_qubits:
        return circ
    new_circ = QuantumCircuit(target_num_qubits, circ.num_clbits)
    new_circ.compose(circ, qubits=list(range(current_num)), inplace=True)
    return new_circ


def convert_to_PCS_circ_largest_clifford(circ, num_qubits, num_checks):
    """
    Given a circuit, finds the largest contiguous block (slice) where every gate is Clifford.
    Then, it applies the PCS check-insertion (via convert_to_PCS_circ) only on that block,
    and reassembles the final circuit by replacing the original block with the protected block.

    Parameters:
      circ: The original QuantumCircuit.
      num_qubits: Number of compute qubits.
      num_checks: Number of check layers to insert.

    Returns:
      (sign_list, final_circ) where final_circ is the circuit with checks inserted
      only in the largest Clifford block.
    """
    slices = slice_by_depth(circ, 1)  # slice into depth 1 circuits

    start_idx, end_idx, clifford_block_slices = find_largest_clifford_block(slices)
    if start_idx is None:
        logger.info("No contiguous Clifford block found. Returning original circuit")
        return None, circ
    logger.debug("Largest Clifford block is from slice %d to %d", start_idx, end_idx)

    protected_block = combine_slices(clifford_block_slices, include_barriers=False)

    sign_list, protected_block_with_checks = convert_to_PCS_circ(
        protected_block, num_qubits, num_checks
    )

    pre_slices = slices[:start_idx]  # slices before the Clifford block.
    post_slices = slices[end_idx + 1 :]  # slices after the Clifford block.

    if pre_slices:
        pre_circ = combine_slices(pre_slices, include_barriers=False)
    else:
        pre_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits)

    if post_slices:
        post_circ = combine_slices(post_slices, include_barriers=False)
    else:
        post_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits)

    # The protected block with checks may include extra ancilla qubits.
    target_qubits = protected_block_with_checks.num_qubits

    # Expand pre_circ and post_circ if necessary so that all parts have target_qubits.
    if pre_circ.num_qubits < target_qubits:
        pre_circ = expand_circuit(pre_circ, target_qubits)
    if post_circ.num_qubits < target_qubits:
        post_circ = expand_circuit(post_circ, target_qubits)

    # Compose the final circuit.
    pre_circ.barrier()
    final_circ = pre_circ.compose(protected_block_with_checks)
    final_circ.barrier()
    final_circ = final_circ.compose(post_circ)

    return sign_list, final_circ


#################################################################################################
# mapping utils that implement VF2 from mapomatic: https://github.com/qiskit-community/mapomatic
#################################################################################################


def find_nonoverlapping_layouts(layouts):
    used = set()
    non_overlapping = []
    for layout in layouts:
        layout_set = set(layout)
        if layout_set.isdisjoint(used):
            non_overlapping.append(layout)
            used.update(layout_set)
    return non_overlapping


def get_VF2_layouts(qc, backend):
    trans_qc = transpile(qc, backend, optimization_level=3)
    small_qc = mm.deflate_circuit(trans_qc)
    layouts = mm.matching_layouts(small_qc, backend)  # Runs VF2

    mapping_ranges = find_nonoverlapping_layouts(layouts)
    return mapping_ranges, small_qc


##################################################################################################################
# mapping utils that simply partition the chip into connected regions; doesn't take any input from circuit
##################################################################################################################


def build_graph(coupling_map):
    graph = {}
    for q1, q2 in coupling_map:
        graph.setdefault(q1, []).append(q2)
        graph.setdefault(q2, []).append(q1)
    return graph


def generate_mapping_ranges_dfs(num_qubits_circuit, num_qubits_backend, coupling_map):
    graph = build_graph(coupling_map)
    ranges = []
    used = set()

    for start in range(num_qubits_backend):
        if start in used:
            continue
        region = []
        stack = [start]
        while stack and len(region) < num_qubits_circuit:
            node = stack.pop()
            if node not in used:
                used.add(node)
                region.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in used:
                        stack.append(neighbor)
        if len(region) == num_qubits_circuit:
            ranges.append(region)
        if len(used) >= num_qubits_backend:
            break
    return ranges


def generate_mapping_ranges_bfs(num_qubits_circuit, num_qubits_backend, coupling_map):
    graph = build_graph(coupling_map)
    ranges = []
    used = set()

    for start in range(num_qubits_backend):
        if start in used:
            continue
        region = []
        queue = deque([start])
        while queue and len(region) < num_qubits_circuit:
            node = queue.popleft()
            if node not in used:
                used.add(node)
                region.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in used:
                        queue.append(neighbor)
        if len(region) == num_qubits_circuit:
            ranges.append(region)
        if len(used) >= num_qubits_backend:
            break
    return ranges


