import qiskit
import itertools
import pickle
import math
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from typing import Any
from typing import Callable
from collections import defaultdict
from qiskit.dagcircuit import DAGOutNode, DAGOpNode

from qiskit import *
from typing import List
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager

#######################################################################################
# Taken from original Pauli Check Sandwhiching code: https://arxiv.org/abs/2206.00215
#######################################################################################

# class PushOperator:
#     '''Class for finding checks and pushing operations through in symbolic form.'''
#     @staticmethod
#     def x(op2):
#         '''Pushes x through op2.'''
#         ops = {
#             "X": [1, "X"],
#             "Y": [-1, "X"],
#             "Z": [-1, "X"],
#             "H": [1, "Z"],
#             "S": [1, "Y"],
#             "SDG": [-1, "Y"]
#         }
#         return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
#     @staticmethod
#     def y(op2):
#         '''Pushes y through op2.'''
#         ops = {
#             "X": [-1, "Y"],
#             "Y": [1, "Y"],
#             "Z": [-1, "Y"],
#             "H": [-1, "Y"],
#             "S": [-1, "X"],
#             "SDG": [1, "X"]
#         }
#         return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
#     @staticmethod        
#     def z(op2):
#         '''Pushes z through op2.'''
#         ops = {
#             "X": [-1, "Z"],
#             "Y": [-1, "Z"],
#             "Z": [1, "Z"],
#             "H": [1, "X"],
#             "S": [1, "Z"],
#             "SDG": [1, "Z"],
#             "RZ": [1, "Z"]
#         }
#         return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")

#     @staticmethod
#     def cx(op1):
#         '''Pushes op1 through cx.'''
#         ops = {
#             ("I", "I"): [1, "I", "I"],
#             ("I", "X"): [1, "I", "X"],
#             ("I", "Y"): [1, "Z", "Y"],
#             ("I", "Z"): [1, "Z", "Z"],
#             ("X", "I"): [1, "X", "X"],
#             ("X", "X"): [1, "X", "I"],
#             ("X", "Y"): [1, "Y", "Z"],
#             ("X", "Z"): [-1, "Y", "Y"],
#             ("Y", "I"): [1, "Y", "X"],
#             ("Y", "X"): [1, "Y", "I"],
#             ("Y", "Y"): [-1, "X", "Z"],
#             ("Y", "Z"): [1, "X", "Y"],
#             ("Z", "I"): [1, "Z", "I"],
#             ("Z", "X"): [1, "Z", "X"],
#             ("Z", "Y"): [1, "I", "Y"],
#             ("Z", "Z"): [1, "I", "Z"]
#         }
#         return ops.get(tuple(op1), None) or Exception(f"{op1[0]} , {op1[1]} wasn't a Pauli element.")

#     @staticmethod
#     def swap(op1):
#         '''Passes op1 through swap.'''
#         return [1] + list(reversed(op1))

    
# def get_weight(pauli_string):
#     '''Gets the weight of a Pauli string. Returns: int'''
#     count = 0
#     for character in pauli_string:
#         if character != "I":
#             count += 1
#     return count

# class ChecksResult:
#     def __init__(self, p2_weight, p1_str, p2_str):
#         self.p2_weight = p2_weight
#         self.p1_str = p1_str
#         self.p2_str = p2_str
        
# class CheckOperator:
#     '''Stores the check operation along with the phase. operations is a list of strings.'''

#     def __init__(self, phase: int, operations: List[str]):
#         self.phase = phase
#         self.operations = operations

# class TempCheckOperator(CheckOperator):
#     '''A temporary class for storing the check operation along with the phase and other variables.'''

#     def __init__(self, phase: int, operations: List[str]):
#         super().__init__(phase, operations)
#         self.layer_idx = 1

# class ChecksFinder:
#     '''Finds checks symbolically.'''

#     def __init__(self, number_of_qubits: int, circ):
#         self.circ_reversed = circ.inverse()
#         self.number_of_qubits = number_of_qubits

#     def find_checks_sym(self, pauli_group_elem: List[str]) -> ChecksResult:
#         '''Finds p1 and p2 elements symbolically.'''
#         circ_reversed = self.circ_reversed
#         pauli_group_elem_ops = list(pauli_group_elem)
#         p2 = CheckOperator(1, pauli_group_elem_ops)
#         p1 = CheckOperator(1, ["I" for _ in range(len(pauli_group_elem))])
#         temp_check_reversed = TempCheckOperator(1, list(reversed(pauli_group_elem_ops)))

#         circ_dag = circuit_to_dag(circ_reversed)
#         layers = list(circ_dag.multigraph_layers())
#         num_layers = len(layers)

#         while True:
#             layer = layers[temp_check_reversed.layer_idx]
#             for node in layer:
#                 if isinstance(node, DAGOpNode):
#                     self.handle_operator_node(node, temp_check_reversed)
#             if self.should_return_result(temp_check_reversed, num_layers):
#                 p1.phase = temp_check_reversed.phase
#                 p1.operations = list(reversed(temp_check_reversed.operations))
#                 return self.get_check_strs(p1, p2)
#             temp_check_reversed.layer_idx += 1

#     def handle_operator_node(self, node, temp_check_reversed: TempCheckOperator):
#         '''Handles operations for nodes of type "op".'''
#         current_qubits = self.get_current_qubits(self, node)
#         current_ops = [temp_check_reversed.operations[qubit] for qubit in current_qubits]
#         node_op = node.name.upper()
#         self.update_current_ops(current_ops, node_op, temp_check_reversed, current_qubits)

#     def should_return_result(self, temp_check_reversed: TempCheckOperator, num_layers: int) -> bool:
#         '''Checks if we have reached the last layer.'''
#         return temp_check_reversed.layer_idx == num_layers - 1

#     @staticmethod
#     def update_current_ops(op1: List[str], op2: str, temp_check_reversed: TempCheckOperator, current_qubits: List[int]):
#         '''Finds the intermediate check. Always push op1 through op2. '''
#         result = ChecksFinder.get_result(op1, op2)
#         temp_check_reversed.phase *= result[0]
#         for idx, op in enumerate(result[1:]):
#             temp_check_reversed.operations[current_qubits[idx]] = op

#     @staticmethod
#     def get_result(op1: List[str], op2: str) -> List[str]:
#         '''Obtain the result based on the values of op1 and op2.'''
#         if len(op1) == 1:
#             return ChecksFinder.single_qubit_operation(op1[0], op2)
#         else:
#             return ChecksFinder.double_qubit_operation(op1, op2)

#     @staticmethod
#     def single_qubit_operation(op1: str, op2: str) -> List[str]:
#         '''Process the single qubit operations.'''
#         if op1 == "X":
#             return PushOperator.x(op2)
#         elif op1 == "Y":
#             return PushOperator.y(op2)
#         elif op1 == "Z":
#             return PushOperator.z(op2)
#         elif op1 == "I":
#             return [1, "I"]
#         else:
#             raise ValueError(f"{op1} is not I, X, Y, or Z.")

#     @staticmethod
#     def double_qubit_operation(op1: List[str], op2: str) -> List[str]:
#         '''Process the double qubit operations.'''
#         if op2 == "CX":
#             return PushOperator.cx(op1)
#         elif op2 == "SWAP":
#             return PushOperator.swap(op1)
#         else:
#             raise ValueError(f"{op2} is not cx or swap.")

#     @staticmethod
#     def get_check_strs(p1: CheckOperator, p2: CheckOperator) -> ChecksResult:
#         '''Turns p1 and p2 to strings results.'''
#         p1_str = ChecksFinder.get_formatted_str(p1)
#         p2_str = ChecksFinder.get_formatted_str(p2)
#         check_result = ChecksResult(get_weight(p2.operations), p1_str, p2_str)
#         return check_result

#     @staticmethod
#     def get_formatted_str(check_operator: CheckOperator) -> str:
#         '''Format the phase and operations into a string.'''
#         operations = check_operator.operations
#         phase = check_operator.phase
#         phase_str = f"+{phase}" if len(str(phase)) == 1 else str(phase)
#         operations.insert(0, phase_str)
#         return "".join(operations)
    
# #     @staticmethod
# #     def get_current_qubits(node):
# #         '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
# #         # We have to check for single or two qubit gates.
# #         if node.name in ["x", "y", "z", "h", "s", "sdg", "rz"]:
# #             return [node.qargs[0].index]
# #         elif node.name in ["cx", "swap"]:
# #             return [node.qargs[0].index, node.qargs[1].index]
# #         else:
# #             assert False, "Overlooked a node operation."
            
#     # Use for new qiskit version (Qiskit verions >= 1.0)
#     @staticmethod
#     def get_current_qubits(self, node):
#         '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
#         circ_dag = circuit_to_dag(self.circ_reversed)
#         dag_qubit_map = {bit: index for index, bit in enumerate(circ_dag.qubits)}
#         # We have to check for single or two qubit gates.
#         # print("node name = ", node.name)
#         if node.name in ["x", "y", "z", "h", "s", "sdg", "rz", "rx"]:
#             return [dag_qubit_map[node.qargs[0]]]
#         elif node.name in ["cx", "swap"]:
#             return [dag_qubit_map[node.qargs[0]], dag_qubit_map[node.qargs[1]]]
#         else:
#             assert False, "Overlooked a node operation."
            

###################################
# Automatic check injection utils.
###################################

from pauli_checks import *

# def check_to_ancilla_free_circ(check_str, num_qubits):
#     """
#     Given a check string (e.g. "+1XIZX" where the first two characters are the phase),
#     returns a QuantumCircuit on num_qubits that applies a single-qubit gate on each qubit as follows:
#       - If the corresponding character is "X", apply an H gate.
#       - If the character is "Z" or "I", do nothing.
    
#     The check string is assumed to be ordered so that the first non-phase character 
#     corresponds to the last qubit (num_qubits-1), the second to the next-to-last, and so on.
#     """
#     # Remove the phase portion (assumed to be the first two characters).
#     op_str = check_str[2:]
#     qc = QuantumCircuit(num_qubits)
#     for i, op in enumerate(op_str):
#         target_qubit = num_qubits - 1 - i
#         if op.upper() == "X":
#             qc.h(target_qubit)
#         # For 'Z' or 'I', no operation is added.
#     return qc

# def convert_to_ancilla_free_PCS_circ(circ, num_qubits, num_checks, barriers=False):
#     """
#     Converts the given circuit to an ancilla-free PCS circuit.
    
#     The process is:
#       1. Find the check pairs (p1 and p2) using the existing ChecksFinder logic.
#       2. For each check pair, convert the left-check string and right-check string into 
#          ancilla-free layers as follows:
#            - "X" maps to an H gate.
#            - "Z" (and "I") map to no operation.
#       3. Prepend the left-check layers to the circuit and append the right-check layers to the circuit.
#          If barriers=True, a single barrier is inserted between the left-check block and payload,
#          and a single barrier between the payload and the right-check block.
#          The final circuit uses the same number of qubits as the original (no ancillas added).
    
#     Returns a tuple (sign_list, final_circ).
#     """

#     # Find check pairs using the same logic as convert_to_PCS_circ.
#     characters = ['I', 'X', 'Z']
#     candidate_strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits)
#                          if not all(c == 'I' for c in p)]
    
#     def weight(pauli_string):
#         return sum(1 for char in pauli_string if char != 'I')
    
#     sorted_strings = sorted(candidate_strings, key=weight)
    
#     test_finder = ChecksFinder(num_qubits, circ)
#     p1_list = []   # will store [p1_str, p2_str] pairs (each string includes a phase in the first two characters)
#     found_checks = 0
    
#     for string in sorted_strings:
#         string_list = list(string)
#         try:
#             result = test_finder.find_checks_sym(pauli_group_elem=string_list)
#             p1_list.append([result.p1_str, result.p2_str])
#             found_checks += 1
#             print(f"Found check {found_checks}: {result.p1_str}, {result.p2_str}")
#             if found_checks >= num_checks:
#                 print("Required number of checks found.")
#                 break
#         except Exception as e:
#             print(f"Failed to find checks for {string_list}: {e}")
#             continue

#     if found_checks < num_checks:
#         print("Warning: Less checks found than required.")
    
#     # Build left-check and right-check circuits for each layer.
#     left_checks = []
#     right_checks = []
#     sign_list = []
    
#     for i in range(num_checks):
#         left_str = p1_list[i][0]
#         right_str = p1_list[i][1]
#         sign_list.append(left_str[:2])
#         left_checks.append(check_to_ancilla_free_circ(left_str, num_qubits))
#         right_checks.append(check_to_ancilla_free_circ(right_str, num_qubits))
    
#     # Combine the left-check circuits sequentially (without internal barriers).
#     left_check_circ = QuantumCircuit(num_qubits)
#     for lc in left_checks:
#         left_check_circ = left_check_circ.compose(lc)
    
#     # Similarly, combine the right-check circuits.
#     right_check_circ = QuantumCircuit(num_qubits)
#     for rc in right_checks:
#         right_check_circ = right_check_circ.compose(rc)
    
#     # Reassemble the final circuit.
#     final_circ = left_check_circ.copy()
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(circ)
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(right_check_circ)
    
#     return sign_list, final_circ

# def check_to_ancilla_free_circ(check_str, num_qubits):
#     op_str = check_str[2:]
#     qc = QuantumCircuit(num_qubits)
#     mapping = {}
#     for i, op in enumerate(op_str):
#         target = num_qubits - 1 - i
#         if op.upper() == "X":
#             qc.h(target)
#             mapping[target] = "X"
#         elif op.upper() == "Z":
#             mapping[target] = "Z"
#     return qc, mapping

def check_to_ancilla_free_circ(check_str, num_qubits):
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
        # print("mapping =", mapping)
    return qc, mapping

def convert_to_ancilla_free_PCS_circ(circ, num_qubits, num_checks, barriers=False, reverse=False):
    import itertools
    from qiskit import QuantumCircuit

    characters = ['I', 'X', 'Z']
    candidate_strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits)
                         if not all(c == 'I' for c in p)]
    def weight(s): 
        return sum(1 for c in s if c != 'I')
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
        print("Warning: Less checks found than required.")
    
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


# Most recent
# def convert_to_ancilla_free_PCS_circ(circ, num_qubits, num_checks, barriers=False, reverse=False):
#     import itertools
#     from qiskit import QuantumCircuit

#     characters = ['I', 'X', 'Z']
#     candidate_strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits)
#                          if not all(c == 'I' for c in p)]
#     def weight(s): 
#         return sum(1 for c in s if c != 'I')
#     sorted_strings = sorted(candidate_strings, key=weight)
#     if reverse:
#         sorted_strings.reverse()
    
#     test_finder = ChecksFinder(num_qubits, circ)
#     selected_pairs = []
#     agg_left = {}
#     agg_right = {}
#     sign_list = []
    
#     for s in sorted_strings:
#         try:
#             result = test_finder.find_checks_sym(pauli_group_elem=list(s))
#             left_str, right_str = result.p1_str, result.p2_str
#             # Convert candidate check strings to ancilla-free circuits and get mappings.
#             _, lm = check_to_ancilla_free_circ(left_str, num_qubits)
#             _, rm = check_to_ancilla_free_circ(right_str, num_qubits)
#             conflict = False
#             # Rule 1: For left check, no qubit should get both X and Z.
#             for q, check in lm.items():
#                 if q in agg_left and agg_left[q] != check:
#                     conflict = True
#                     break
#             # Rule 2: For right check, each qubit can only appear once.
#             for q in rm.keys():
#                 if q in agg_right:
#                     conflict = True
#                     break
#             if conflict:
#                 continue
#             # Accept candidate: update aggregate mappings.
#             for q, check in lm.items():
#                 agg_left[q] = check
#             for q, check in rm.items():
#                 agg_right[q] = check
#             selected_pairs.append([left_str, right_str])
#             sign_list.append(left_str[:2])
#             if len(selected_pairs) >= num_checks:
#                 break
#         except Exception as e:
#             continue
#     if len(selected_pairs) < num_checks:
#         print("Warning: Less checks found than required.")
    
#     # Build the left-check circuit from the aggregated left mapping.
#     final_left_check_circ = QuantumCircuit(num_qubits)
#     for q in range(num_qubits):
#         if agg_left.get(q, "Z") == "X": # If any candidate requested an X on qubit q, final check is X (apply H), else Z (do nothing)
#             final_left_check_circ.h(q)
#     # For right checks, we simply combine the candidate right-check circuits from selected_pairs.
#     right_check_circ = QuantumCircuit(num_qubits)
#     for pair in selected_pairs:
#         rc, _ = check_to_ancilla_free_circ(pair[1], num_qubits)
#         right_check_circ = right_check_circ.compose(rc)
    
#     final_circ = final_left_check_circ.copy()
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(circ)
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(right_check_circ)

#     return sign_list, final_circ, agg_left, agg_right


# def convert_to_ancilla_free_PCS_circ(circ, num_qubits, num_checks, barriers=False):
#     import itertools
#     from qiskit import QuantumCircuit

#     characters = ['I', 'X', 'Z']
#     candidate_strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits)
#                          if not all(c == 'I' for c in p)]
#     def weight(s): return sum(1 for c in s if c != 'I')
#     sorted_strings = sorted(candidate_strings, key=weight)

#     test_finder = ChecksFinder(num_qubits, circ)
#     p1_list = []
#     found_checks = 0
#     for s in sorted_strings:
#         try:
#             result = test_finder.find_checks_sym(pauli_group_elem=list(s))
#             p1_list.append([result.p1_str, result.p2_str])
#             found_checks += 1
#             print(f"Found check {found_checks}: {result.p1_str}, {result.p2_str}")
#             if found_checks >= num_checks:
#                 break
#         except Exception as e:
#             print(f"Failed for {list(s)}: {e}")
#             continue
#     if found_checks < num_checks:
#         print("Warning: Less checks found than required.")

#     left_mappings = []
#     right_mappings = []
#     sign_list = []
#     for i in range(num_checks):
#         left_str = p1_list[i][0]
#         right_str = p1_list[i][1]
#         sign_list.append(left_str[:2])
#         _, lm = check_to_ancilla_free_circ(left_str, num_qubits)
#         _, rm = check_to_ancilla_free_circ(right_str, num_qubits)
#         left_mappings.append(lm)
#         right_mappings.append(rm)

#     # Aggregate left mappings: if any layer applies an X on a qubit, that qubit gets an H.
#     final_left_mapping = {}
#     for q in range(num_qubits):
#         final_left_mapping[q] = "Z"
#         for mapping in left_mappings:
#             if mapping.get(q) == "X":
#                 final_left_mapping[q] = "X"
#                 break
#     final_left_check_circ = QuantumCircuit(num_qubits)
#     for q in range(num_qubits):
#         if final_left_mapping[q] == "X":
#             final_left_check_circ.h(q)

#     # For right checks, compose them sequentially.
#     right_check_circ = QuantumCircuit(num_qubits)
#     for rm in right_mappings:
#         idx = right_mappings.index(rm)
#         rc, _ = check_to_ancilla_free_circ(p1_list[idx][1], num_qubits)
#         right_check_circ = right_check_circ.compose(rc)

#     final_circ = final_left_check_circ.copy()
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(circ)
#     if barriers:
#         final_circ.barrier()
#     final_circ = final_circ.compose(right_check_circ)

#     return sign_list, final_circ, left_mappings, right_mappings

        

def convert_to_PCS_circ(circ, num_qubits, num_checks, barriers=False, reverse=False):
    total_qubits = num_qubits + num_checks
    
    characters = ['I', 'X', 'Z']
    strings = [''.join(p) for p in itertools.product(characters, repeat=num_qubits) if not all(c == 'I' for c in p)]
    # print("strings to try ", strings)
    # print()

    def weight(pauli_string):
        return sum(1 for char in pauli_string if char != 'I')
    
    sorted_strings = sorted(strings, key=weight)
    if reverse:
        sorted_strings.reverse()
    # print("Sorted by weight:", sorted_strings)
    
    test_finder = ChecksFinder(num_qubits, circ)
    p1_list = []
    found_checks = 0  # Counter for successful checks found

    for string in sorted_strings: 
        # print("attempting ", string)
        string_list = list(string)
        try:
            result = test_finder.find_checks_sym(pauli_group_elem=string_list)
            p1_list.append([result.p1_str, result.p2_str])
            found_checks += 1
            print(f"Found check {found_checks}: {result.p1_str}, {result.p2_str}")
            if found_checks >= num_checks:
                print("Required number of checks found.")
                print("p1_list = ", p1_list)
                break  # Stop the loop if we have found enough checks
        except Exception as e:
            # print(f"Failed to find checks for {string_list}: {e}")
            continue  # Skip to the next iteration if an error occurs

    if found_checks < num_checks:
        print("Warning: Less checks found than required.")

    
    # print("p1_list = ", p1_list)
    # sorted_list = sorted(p1_list, key=lambda s: s[1].count('I'))
    # # pauli_list = sorted_list[-num_qubits -1:-1]
    # pauli_list = sorted_list[-num_checks -1:-1]

    # # print("sorted list: ", sorted_list)
    # print("pauli list: ", pauli_list)
    
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
        # pl = pauli_list[i][0][2:]
        pl = p1_list[i][0][2:]
        # pr = pauli_list[i][1][2:]
        pr = p1_list[i][1][2:]
        if i == 0:
            temp_qc = add_pauli_checks(circ, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            save_qc = add_pauli_checks(circ, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            prev_qc = temp_qc
        else:
            temp_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            save_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, barriers) 
            prev_qc = temp_qc
        pl_list.append(pl)
        pr_list.append(pr)
        # sign_list.append(pauli_list[i][0][:2])
        sign_list.append(p1_list[i][0][:2])
        pcs_qc_list.append(save_qc)

    qc = pcs_qc_list[-1] # return circuit with 'num_checks' implemented

    return sign_list, qc

def checks_on_checks_circ(circ, num_qubits, num_checks, barriers=False):
    total_qubits = num_qubits + num_checks
        
    # add pauli check on two sides:
    # specify the left and right pauli strings
    pcs_qc_list = []
    sign_list = []
    pl_list = []
    pr_list = []

    for i in range(0, num_checks):
        initial_layout = {}
        for j in range(0, num_qubits+i):
            initial_layout[j] = [j]
    
        final_layout = {}
        for j in range(0, num_qubits+i):
            final_layout[j] = [j]
        
        # pl = pauli_list[i][0][2:]
        # pl = p1_list[i][0][2:]
        pl = i*"I" + "X" + "I" * (num_qubits-1)
        print(len(pl))
        pr = i*"I" + "X" + "I" * (num_qubits-1)
        # pr = pauli_list[i][1][2:]
        # pr = p1_list[i][1][2:]
        if i == 0:
            temp_qc = add_pauli_checks(circ, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            save_qc = add_pauli_checks(circ, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            prev_qc = temp_qc
        else:
            temp_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, barriers)
            save_qc = add_pauli_checks(prev_qc, pl, pr, initial_layout, final_layout, False, False, False, False, barriers) 
            prev_qc = temp_qc
        pl_list.append(pl)
        pr_list.append(pr)
        # sign_list.append(pauli_list[i][0][:2])
        # sign_list.append(p1_list[i][0][:2])
        pcs_qc_list.append(save_qc)

    qc = pcs_qc_list[-1] # return circuit with 'num_checks' implemented

    return sign_list, qc

# def find_largest_clifford_block(slices):
#     """
#     Given a list of circuit slices (each slice is a Qiskit circuit or CircuitInstruction),
#     returns a tuple: (start_index, end_index, block_slices) corresponding to the largest contiguous block 
#     of slices that contain only Clifford gates.
    
#     If no slice is entirely Clifford, returns (None, None, []).
#     """
#     # Create a list to mark for each slice whether it is entirely Clifford.
#     slice_is_clifford = []
#     for slice_item in slices:
#         # Try to get the underlying instruction data.
#         if hasattr(slice_item, "data"):
#             slice_data = slice_item.data
#         elif hasattr(slice_item, "definition"):
#             slice_data = slice_item.definition.data
#         else:
#             # If neither attribute is available, skip this slice.
#             slice_is_clifford.append(False)
#             continue

#         # Assume the slice is Clifford until we find a non-Clifford gate.
#         is_all_clifford = True
#         for instr in slice_data:
#             gate_name = instr.operation.name
#             if not is_clifford_gate(gate_name):
#                 is_all_clifford = False
#                 break
#         slice_is_clifford.append(is_all_clifford)
    
#     # Now, find the longest contiguous block of True values in slice_is_clifford.
#     max_len = 0
#     max_start = None
#     current_len = 0
#     current_start = 0
    
#     for i, flag in enumerate(slice_is_clifford):
#         if flag:
#             if current_len == 0:
#                 current_start = i
#             current_len += 1
#         else:
#             if current_len > max_len:
#                 max_len = current_len
#                 max_start = current_start
#             current_len = 0
#     # Final check in case the longest block is at the end.
#     if current_len > max_len:
#         max_len = current_len
#         max_start = current_start
    
#     if max_start is None:
#         # No contiguous Clifford block found.
#         return None, None, []
    
#     max_end = max_start + max_len - 1
#     block_slices = slices[max_start:max_end+1]
#     return max_start, max_end, block_slices

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
    
    max_len = 0           # maximum number of contiguous slices (depth)
    max_gate_count = 0    # total number of gates in that block
    max_start = None      # starting index of the best block
    
    current_len = 0       # current contiguous slice count
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
            if (current_len > max_len) or (current_len == max_len and current_gate_count > max_gate_count):
                max_len = current_len
                max_gate_count = current_gate_count
                max_start = current_start
            # Reset current block counters.
            current_len = 0
            current_gate_count = 0
    # Final check in case the longest block is at the end.
    if (current_len > max_len) or (current_len == max_len and current_gate_count > max_gate_count):
        max_len = current_len
        max_gate_count = current_gate_count
        max_start = current_start

    if max_start is None:
        # No contiguous Clifford block found.
        return None, None, []
    
    max_end = max_start + max_len - 1
    block_slices = slices[max_start:max_end+1]
    return max_start, max_end, block_slices

# Helper function for determining if a gate is Clifford.
def is_clifford_gate(gate_name):
    """
    Returns True if gate_name (string) is considered a Clifford gate.
    Adjust the set as needed.
    """
    clifford_set = {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "id", "barrier"}
    return gate_name.lower() in clifford_set


from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from qiskit.transpiler.passes import RemoveBarriers

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
    slices = slice_by_depth(circ, 1) # slice into depth 1 circuits
    
    start_idx, end_idx, clifford_block_slices = find_largest_clifford_block(slices)
    if start_idx is None:
        print("No contiguous Clifford block found. Returning original circuit.")
        return None, circ
    print(f"Largest Clifford block is from slice {start_idx} to {end_idx}.")
    
    protected_block = combine_slices(clifford_block_slices, include_barriers=False)

    sign_list, protected_block_with_checks = convert_to_PCS_circ(protected_block, num_qubits, num_checks)
    # print(protected_block_with_checks.draw())
    
    pre_slices = slices[:start_idx] # slices before the Clifford block.
    post_slices = slices[end_idx+1:] # slices after the Clifford block.
    
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


# def append_control_paulis_to_circuit(circuit, pauli_string, ancilla_index):
#     """
#     Appends controlled Pauli operations to the quantum circuit based on the pauli_string input.
#     """
#     for index, char in enumerate(reversed(pauli_string)):
#         if ancilla_index == 0:
#             index += 1 # index needs to increase to adjust for shifted qubits
#         if char == 'X':
#             circuit.cx(ancilla_index, index)
#         elif char == 'Y':
#             circuit.cy(ancilla_index, index)
#         elif char == 'Z':
#             circuit.cz(ancilla_index, index)

# def add_pauli_checks(circuit, left_check, right_check, ancilla_position='bottom'):
#     num_ancillas = 1
#     original_size = circuit.num_qubits
#     pcs_circuit = qiskit.QuantumCircuit(original_size + num_ancillas)
    
#     # Determine ancilla index based on the position
#     if ancilla_position == 'top':
#         ancilla_index = 0
#         payload_qubits = [i + 1 for i in range(original_size)] # Move all operations from the input circuit down one qubit
#     elif ancilla_position == 'bottom':
#         ancilla_index = original_size
#         payload_qubits = range(original_size) # Use the original qubit indices as no shift is needed
#     else:
#         raise ValueError("Invalid ancilla position. Choose 'top' or 'bottom'.")

#     # Apply left check
#     pcs_circuit.h(ancilla_index)
#     append_control_paulis_to_circuit(pcs_circuit, left_check, ancilla_index)
#     pcs_circuit.barrier()

#     pcs_circuit.compose(circuit, qubits=payload_qubits, inplace=True)

#     # Apply right check
#     pcs_circuit.barrier()
#     append_control_paulis_to_circuit(pcs_circuit, right_check, ancilla_index)
#     pcs_circuit.h(ancilla_index)

#     return pcs_circuit

# def convert_to_pcs_circ(circ, top_qubit_check, bottom_qubit_check):
#     # print("top qubit check ", top_qubit_check)
#     # print("bottom qubit check ", bottom_qubit_check)

#     # print("payload circ ")
#     # print(circ)

#     # Find left and right checks
#     check_finder = ChecksFinder(circ.num_qubits, circ)
    
#     top_result = check_finder.find_checks_sym(pauli_group_elem = list(top_qubit_check))
#     pl_top, pr_top = top_result.p1_str, top_result.p2_str
#     print(pl_top, pr_top)

#     bottom_result = check_finder.find_checks_sym(pauli_group_elem = list(bottom_qubit_check))
#     pl_bottom, pr_bottom = bottom_result.p1_str, bottom_result.p2_str
#     print(pl_bottom, pr_bottom)

#     left_check = pl_bottom[2:]
#     right_check = pr_bottom[2:]
#     temp_pcs_circ = add_pauli_checks(circ, left_check, right_check, ancilla_position='bottom')
    
#     # print("circuit after bottom check")
#     # print(temp_pcs_circ)

#     left_check = pl_top[2:]
#     right_check = pr_top[2:]
#     pcs_circ = add_pauli_checks(temp_pcs_circ, left_check, right_check, ancilla_position='top')
#     # print("circuit after top check")
#     # print(pcs_circ)

#     return pcs_circ

#################################################################################################
# mapping utils that implement VF2 from mapomatic: https://github.com/qiskit-community/mapomatic
#################################################################################################

import mapomatic as mm
from networkx import Graph
from networkx.algorithms.approximation.clique import maximum_independent_set
from itertools import combinations

# Transpiles qc multiple times to find qc with least number of two-qubit gates
def find_best_trans_qc(qc_list, num_trials):
    pass

# def find_nonoverlapping_layouts(layouts):
#     V = [frozenset(s) for s in layouts]
#     E = [(x,y) for x,y in combinations(V, 2) if x&y] # "if x&y" means "if intersection of sets x and y is non-empty"
#     G = Graph()
#     G.add_nodes_from(V)
#     G.add_edges_from(E)
    
#     max_indep_set = maximum_independent_set(G)
#     # print( max_indep_set)
    
#     # To maintain original order and convert frozensets back to lists
#     ordered_max_indep_set = [list(layout) for layout in layouts if frozenset(layout) in max_indep_set]
    
#     # print(ordered_max_indep_set)
#     mapping_ranges = ordered_max_indep_set
#     return mapping_ranges

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
    layouts = mm.matching_layouts(small_qc, backend) # Runs VF2
    # layouts = mm.matching_layouts(qc, backend)
    # print("layouts:")
    # print(layouts)

    # mapping_ranges = find_nonoverlapping_layouts(layouts) # Runs maximum independent set to approximate maximum number of non-overlapping regions
    mapping_ranges = find_nonoverlapping_layouts(layouts) 
    # mapping_ranges = layouts
    return mapping_ranges, small_qc


##################################################################################################################
# mapping utils that simply partition the chip into connected regions; doesn't take any input from circuit
##################################################################################################################


from collections import deque

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
        # print("Starting DFS from node", start)
        while stack and len(region) < num_qubits_circuit:
            node = stack.pop()
            if node not in used:
                used.add(node)
                region.append(node)
                # print("Visited node", node, "-> region:", region)
                for neighbor in graph.get(node, []):
                    if neighbor not in used:
                        # print("  Adding neighbor", neighbor, "to stack")
                        stack.append(neighbor)
        if len(region) == num_qubits_circuit:
            ranges.append(region)
            # print("Complete region found:", region)
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
        # print("Starting BFS from node", start)
        while queue and len(region) < num_qubits_circuit:
            node = queue.popleft()
            if node not in used:
                used.add(node)
                region.append(node)
                # print("Visited node", node, "-> region:", region)
                for neighbor in graph.get(node, []):
                    if neighbor not in used:
                        # print("  Adding neighbor", neighbor, "to queue")
                        queue.append(neighbor)
        if len(region) == num_qubits_circuit:
            ranges.append(region)
            # print("Complete region found:", region)
        if len(used) >= num_qubits_backend:
            break
    return ranges


# def build_graph(coupling_map):
#     """Build an undirected graph from the coupling map."""
#     graph = {}
#     for q1, q2 in coupling_map:
#         if q1 not in graph:
#             graph[q1] = []
#         if q2 not in graph:
#             graph[q2] = []
#         graph[q1].append(q2)
#         graph[q2].append(q1)
#     return graph

# def is_connected(subset, graph):
#     """Check if all qubits in subset are connected using BFS."""
#     if not subset:
#         return False
#     visited = set()
#     queue = deque([next(iter(subset))])  # Start BFS from any element in the subset
    
#     while queue:
#         node = queue.popleft()
#         if node in visited:
#             continue
#         visited.add(node)
#         # Enqueue all unvisited neighbors that are also in the subset
#         queue.extend([neighbor for neighbor in graph[node] if neighbor in subset and neighbor not in visited])
    
#     return visited == set(subset)  # Check if we've visited all nodes in the subset

# def generate_mapping_ranges(num_qubits_circuit, num_qubits_backend, coupling_map):
#     graph = build_graph(coupling_map)
#     ranges = []
#     used_qubits = set()  # To track which qubits have already been used

#     # Iterate over all qubits in the backend
#     for start in range(num_qubits_backend):
#         if start in used_qubits:
#             continue  # Skip if this qubit is already used in a previous range

#         # Attempt to create a range starting from this qubit
#         current_range = []
#         queue = deque([start])
#         while queue and len(current_range) < num_qubits_circuit:
#             qubit = queue.popleft()
#             if qubit not in used_qubits:
#                 used_qubits.add(qubit)
#                 current_range.append(qubit)
#                 # Enqueue all connected, unused qubits
#                 queue.extend([neighbor for neighbor in graph[qubit] if neighbor not in used_qubits])

#         # Only consider this range if it has the required number of qubits
#         if len(current_range) == num_qubits_circuit:
#             ranges.append(current_range)
#             if len(used_qubits) >= num_qubits_backend:
#                 break  # Stop if we've used up all available qubits

#     return ranges



# def generate_mapping_ranges(num_qubits_circuit, num_qubits_backend, coupling_map):
#     graph = build_graph(coupling_map)
#     ranges = []
#     used_qubits = set()  # To track which qubits have already been used

#     # Iterate over all qubits in the backend
#     for start in range(num_qubits_backend):
#         if start in used_qubits:
#             continue  # Skip if this qubit is already used in a previous range

#         # Attempt to create a range starting from this qubit
#         current_range = []
#         queue = deque([start])
#         while queue and len(current_range) < num_qubits_circuit:
#             qubit = queue.popleft()
#             if qubit not in used_qubits:
#                 used_qubits.add(qubit)
#                 current_range.append(qubit)
#                 # Enqueue all connected, unused qubits
#                 queue.extend([neighbor for neighbor in graph[qubit] if neighbor not in used_qubits])

#         # Only consider this range if it has the required number of qubits
#         if len(current_range) == num_qubits_circuit:
#             # Reorder current_range so that consecutive qubits are connected
#             ordered_range = reorder_connected(current_range, graph)
#             # ordered_range = current_range
#             ranges.append(ordered_range)
#             if len(used_qubits) >= num_qubits_backend:
#                 break  # Stop if we've used up all available qubits

#     return ranges

# def reorder_connected(current_range, graph):
#     """
#     Reorders the qubits in current_range so that consecutive qubits are connected.
#     """
#     # Build the subgraph induced by current_range
#     subgraph = {node: [neighbor for neighbor in graph[node] if neighbor in current_range] for node in current_range}

#     # Attempt to find a Hamiltonian path in the subgraph
#     def backtrack(path, visited):
#         if len(path) == len(current_range):
#             return path  # Found a path covering all nodes
#         current_node = path[-1]
#         for neighbor in subgraph[current_node]:
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 path.append(neighbor)
#                 result = backtrack(path, visited)
#                 if result is not None:
#                     return result
#                 path.pop()
#                 visited.remove(neighbor)
#         return None

#     # Try starting from each node in current_range
#     for start_node in current_range:
#         path = [start_node]
#         visited = set([start_node])
#         result = backtrack(path, visited)
#         if result is not None:
#             return result  # Return the first successful path

#     # If no Hamiltonian path is found, return the original ordering
#     return current_range



##############
# VQE utils
##############

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
import numpy as np

def construct_qcc_circuit(entanglers: list):
    '''This function defines the QCC ansatz circuit for VQE. Here we construct exponential blocks using
    entanglers from QMF state as a proof of principle demonstration.
    
    Args:
        entanglers: list storing Pauli words for construction of qcc_circuit.
        backend: statevector, qasm simulator or a real backend.
        truncation: a threshold number to truncate the blocks. Default: None.
    Returns:
        qcc_circuit
    '''
    num_blocks = len(entanglers)
    num_blocks = 2
    # print("num blocks = ", num_blocks)
    # p = ParameterVector('p', num_blocks)
    p = [1.16692654, 0.27223177, -0.93402707, -0.92067998, 0.06852241, -0.42444632, -0.41270851, -0.01068001]
    
    num_qubits = len(entanglers[0])
    qcc_circuit = QuantumCircuit(num_qubits)
    for i in range(num_blocks):
        circuit = QuantumCircuit(num_qubits)
        key = entanglers[i]
        coupler_map = []
        
        # We first construct coupler_map according to the key.
        for j in range(num_qubits):
            if key[num_qubits-1-j] != 'I':
                coupler_map.append(j)
                
        # Then we construct the circuit.
        if len(coupler_map) == 1:
            # there is no CNOT gate.
            c = coupler_map[0]
            if key[num_qubits-1-c] == 'X':
                circuit.h(c)
                circuit.rz(p[i], c)
                circuit.h(c)
            elif key[num_qubits-1-c] == 'Y':
                circuit.rx(-np.pi/2, c)
                circuit.rz(p[i], c)
                circuit.rx(np.pi/2, c)
                
            qcc_circuit += circuit
        else:
            # Here we would need CNOT gate.
            for j in coupler_map:
                if key[num_qubits-1-j] == 'X':
                    circuit.h(j)
                elif key[num_qubits-1-j] == 'Y':
                    circuit.rx(-np.pi/2, j)
                    
            for j in range(len(coupler_map) - 1):
                circuit.cx(coupler_map[j], coupler_map[j+1])
                
            param_gate = QuantumCircuit(num_qubits)
            param_gate.rz(p[i], coupler_map[-1])
            
            #qcc_circuit += circuit + param_gate + circuit.inverse()
            qcc_circuit.compose(circuit, inplace=True)
            qcc_circuit.compose(param_gate, inplace=True)
            qcc_circuit.compose(circuit.inverse(), inplace=True)
    
    return qcc_circuit


def hf_circ(num_qubits, num_checks):
    total_qubits = num_qubits + num_checks
    hf_circuit = QuantumCircuit(total_qubits)

    hf_circuit.x(0)
    hf_circuit.x(1)
    hf_circuit.x(2)
    hf_circuit.x(3)
    
    entanglers = ['XXIIIIXY', 'IIXXXYII', 'IXXIXIIY', 'XIIXIXYI',
                  'XXIIXYII', 'IXIXIXIY', 'XIXIXIYI', 'IIXXIIXY']

    parameterized_circuit = hf_circuit.compose(construct_qcc_circuit(entanglers))
    
    return parameterized_circuit 

        










