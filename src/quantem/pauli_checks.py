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

import pickle
from qiskit.transpiler.basepasses import AnalysisPass
from collections import defaultdict
from qiskit.dagcircuit import DAGOpNode
from qiskit import (
    QuantumCircuit,
    ClassicalRegister
)
from typing import List
from copy import deepcopy
from qiskit.converters import circuit_to_dag

from qiskit.quantum_info import Operator

class PushOperator:
    '''Class for finding checks and pushing operations through in symbolic form.'''
    @staticmethod
    def x(op2):
        '''Pushes x through op2.'''
        ops = {
            "X": [1, "X"],
            "Y": [-1, "X"],
            "Z": [-1, "X"],
            "H": [1, "Z"],
            "S": [1, "Y"],
            "SDG": [-1, "Y"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
    @staticmethod
    def y(op2):
        '''Pushes y through op2.'''
        ops = {
            "X": [-1, "Y"],
            "Y": [1, "Y"],
            "Z": [-1, "Y"],
            "H": [-1, "Y"],
            "S": [-1, "X"],
            "SDG": [1, "X"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")
    
    @staticmethod        
    def z(op2):
        '''Pushes z through op2.'''
        ops = {
            "X": [-1, "Z"],
            "Y": [-1, "Z"],
            "Z": [1, "Z"],
            "H": [1, "X"],
            "S": [1, "Z"],
            "SDG": [1, "Z"],
            "RZ": [1, "Z"]
        }
        return ops.get(op2, None) or Exception(f"{op2} gate wasn't matched in the DAG.")

    @staticmethod
    def cx(op1):
        '''Pushes op1 through cx.'''
        ops = {
            ("I", "I"): [1, "I", "I"],
            ("I", "X"): [1, "I", "X"],
            ("I", "Y"): [1, "Z", "Y"],
            ("I", "Z"): [1, "Z", "Z"],
            ("X", "I"): [1, "X", "X"],
            ("X", "X"): [1, "X", "I"],
            ("X", "Y"): [1, "Y", "Z"],
            ("X", "Z"): [-1, "Y", "Y"],
            ("Y", "I"): [1, "Y", "X"],
            ("Y", "X"): [1, "Y", "I"],
            ("Y", "Y"): [-1, "X", "Z"],
            ("Y", "Z"): [1, "X", "Y"],
            ("Z", "I"): [1, "Z", "I"],
            ("Z", "X"): [1, "Z", "X"],
            ("Z", "Y"): [1, "I", "Y"],
            ("Z", "Z"): [1, "I", "Z"]
        }
        return ops.get(tuple(op1), None) or Exception(f"{op1[0]} , {op1[1]} wasn't a Pauli element.")

    @staticmethod
    def swap(op1):
        '''Passes op1 through swap.'''
        return [1] + list(reversed(op1))

    
def get_weight(pauli_string):
    '''Gets the weight of a Pauli string. Returns: int'''
    count = 0
    for character in pauli_string:
        if character != "I":
            count += 1
    return count

class ChecksResult:
    def __init__(self, p2_weight, p1_str, p2_str):
        self.p2_weight = p2_weight
        self.p1_str = p1_str
        self.p2_str = p2_str
        
class CheckOperator:
    '''Stores the check operation along with the phase. operations is a list of strings.'''

    def __init__(self, phase: int, operations: List[str]):
        self.phase = phase
        self.operations = operations

class TempCheckOperator(CheckOperator):
    '''A temporary class for storing the check operation along with the phase and other variables.'''

    def __init__(self, phase: int, operations: List[str]):
        super().__init__(phase, operations)
        self.layer_idx = 1

class ChecksFinder:
    '''Finds checks symbolically.'''

    def __init__(self, number_of_qubits: int, circ):
        self.circ_reversed = circ.inverse()
        self.number_of_qubits = number_of_qubits

    def find_checks_sym(self, pauli_group_elem: List[str]) -> ChecksResult:
        '''Finds p1 and p2 elements symbolically.'''
        circ_reversed = self.circ_reversed
        pauli_group_elem_ops = list(pauli_group_elem)
        p2 = CheckOperator(1, pauli_group_elem_ops)
        p1 = CheckOperator(1, ["I" for _ in range(len(pauli_group_elem))])
        temp_check_reversed = TempCheckOperator(1, list(reversed(pauli_group_elem_ops)))

        circ_dag = circuit_to_dag(circ_reversed)
        layers = list(circ_dag.multigraph_layers())
        num_layers = len(layers)

        while True:
            layer = layers[temp_check_reversed.layer_idx]
            for node in layer:
                if isinstance(node, DAGOpNode): 
                    self.handle_operator_node(node, temp_check_reversed)
            if self.should_return_result(temp_check_reversed, num_layers):
                p1.phase = temp_check_reversed.phase
                p1.operations = list(reversed(temp_check_reversed.operations))
                return self.get_check_strs(p1, p2)
            temp_check_reversed.layer_idx += 1

    def handle_operator_node(self, node, temp_check_reversed: TempCheckOperator):
        '''Handles operations for nodes of type "op".'''
        current_qubits = self.get_current_qubits(node)
        current_ops = [temp_check_reversed.operations[qubit] for qubit in current_qubits]
        node_op = node.name.upper()
        self.update_current_ops(current_ops, node_op, temp_check_reversed, current_qubits)

    def should_return_result(self, temp_check_reversed: TempCheckOperator, num_layers: int) -> bool:
        '''Checks if we have reached the last layer.'''
        return temp_check_reversed.layer_idx == num_layers - 1

    @staticmethod
    def update_current_ops(op1: List[str], op2: str, temp_check_reversed: TempCheckOperator, current_qubits: List[int]):
        '''Finds the intermediate check. Always push op1 through op2. '''
        result = ChecksFinder.get_result(op1, op2)
        temp_check_reversed.phase *= result[0]
        for idx, op in enumerate(result[1:]):
            temp_check_reversed.operations[current_qubits[idx]] = op

    @staticmethod
    def get_result(op1: List[str], op2: str) -> List[str]:
        '''Obtain the result based on the values of op1 and op2.'''
        if len(op1) == 1:
            return ChecksFinder.single_qubit_operation(op1[0], op2)
        else:
            return ChecksFinder.double_qubit_operation(op1, op2)

    @staticmethod
    def single_qubit_operation(op1: str, op2: str) -> List[str]:
        '''Process the single qubit operations.'''
        if op1 == "X":
            return PushOperator.x(op2)
        elif op1 == "Y":
            return PushOperator.y(op2)
        elif op1 == "Z":
            return PushOperator.z(op2)
        elif op1 == "I":
            return [1, "I"]
        else:
            raise ValueError(f"{op1} is not I, X, Y, or Z.")

    @staticmethod
    def double_qubit_operation(op1: List[str], op2: str) -> List[str]:
        '''Process the double qubit operations.'''
        if op2 == "CX":
            return PushOperator.cx(op1)
        elif op2 == "SWAP":
            return PushOperator.swap(op1)
        else:
            raise ValueError(f"{op2} is not cx or swap.")

    @staticmethod
    def get_check_strs(p1: CheckOperator, p2: CheckOperator) -> ChecksResult:
        '''Turns p1 and p2 to strings results.'''
        p1_str = ChecksFinder.get_formatted_str(p1)
        p2_str = ChecksFinder.get_formatted_str(p2)
        check_result = ChecksResult(get_weight(p2.operations), p1_str, p2_str)
        return check_result

    @staticmethod
    def get_formatted_str(check_operator: CheckOperator) -> str:
        '''Format the phase and operations into a string.'''
        operations = check_operator.operations
        phase = check_operator.phase
        phase_str = f"+{phase}" if len(str(phase)) == 1 else str(phase)
        operations.insert(0, phase_str)
        return "".join(operations)
      
    # Use for new qiskit version (Qiskit verions >= 1.0)
    def get_current_qubits(self, node):
        '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
        circ_dag = circuit_to_dag(self.circ_reversed)
        dag_qubit_map = {bit: index for index, bit in enumerate(circ_dag.qubits)}

        # We have to check for single or two qubit gates.
        if node.name in ["x", "y", "z", "h", "s", "sdg", "rz", "rx"]:
            return [dag_qubit_map[node.qargs[0]]]
        elif node.name in ["cx", "swap"]:
            return [dag_qubit_map[node.qargs[0]], dag_qubit_map[node.qargs[1]]]
        else:
            assert False, "Overlooked a node operation."            
            
def append_paulis_to_circuit(circuit, pauli_string):
    """
    Appends Pauli operations to the quantum circuit based on the pauli_string input.
    """
    for index, char in enumerate(reversed(pauli_string)):
        if char == 'I':
            circuit.id(index)
        elif char == 'X':
            circuit.x(index)
        elif char == 'Y':
            circuit.y(index)
        elif char == 'Z':
            circuit.z(index)
            
def append_control_paulis_to_circuit(circuit, pauli_string, ancilla_index, mapping):
    """
    Appends controlled Pauli operations to the quantum circuit based on the pauli_string input.
    """
    for orign_index, char in enumerate(reversed(pauli_string)):
        index = mapping[orign_index]
        if char == 'X':
            circuit.cx(ancilla_index, index)
        elif char == 'Y':
            circuit.cy(ancilla_index, index)
        elif char == 'Z':
            circuit.cz(ancilla_index, index)

def verify_circuit_with_pauli_checks(circuit, left_check, right_check):
    """
    Verifies that the original circuit is equivalent to a new circuit that includes left and right Pauli checks.
    The equivalence is verified by comparing the unitary matrix representations of both circuits.
    """
    assert len(circuit.qubits) == len(left_check) == len(right_check), "Number of qubits in circuit and checks must be equal."

    verification_circuit = QuantumCircuit(len(circuit.qubits))
    
    append_paulis_to_circuit(verification_circuit, left_check)
    verification_circuit.compose(circuit, inplace=True)
    append_paulis_to_circuit(verification_circuit, right_check)

    original_operator = Operator(circuit)
    verification_operator = Operator(verification_circuit)

    return verification_circuit, original_operator.equiv(verification_operator)


def add_pauli_checks(circuit, left_check, right_check, initial_layout, final_layout, pauli_meas = False, single_side = False, qubit_measure = False, ancilla_measure = False, barriers = False, increase_size = 0):
    # initial_layout: mapping from original circuit index to the physical qubit index
    # final_layout: mapping from original circuit index to the final physical qubit index
    if initial_layout is None:
        # Number of qubits in circuit and checks must be equal.
        assert len(circuit.qubits) == len(left_check) == len(right_check)
        # First verify the paulis are correct:
        _, equal = verify_circuit_with_pauli_checks(circuit, left_check, right_check)
        assert(equal)
    ancilla_index = len(circuit.qubits)
    if increase_size > 0:
        ancilla_index = len(circuit.qubits) - increase_size
    if pauli_meas is False:
        if increase_size > 0:
            check_circuit = QuantumCircuit(len(circuit.qubits))
        else:
            check_circuit = QuantumCircuit(ancilla_index + 1)
        check_circuit.h(ancilla_index)
        append_control_paulis_to_circuit(check_circuit, left_check, ancilla_index, initial_layout)
    if barriers is True:
        check_circuit.barrier()
    check_circuit.compose(circuit, inplace=True)
    if barriers is True:
        check_circuit.barrier()
    if single_side is False:
        append_control_paulis_to_circuit(check_circuit, right_check, ancilla_index, final_layout)
    if pauli_meas is False:
        check_circuit.h(ancilla_index)
    
    if ancilla_measure is True:
        # add one measurement for the ancilla measurement
        ancilla_cr = ClassicalRegister(1, str(right_check))
        check_circuit.add_register(ancilla_cr)
        check_circuit.measure(ancilla_index, ancilla_cr[0])
        
    if qubit_measure is True:
        meas_cr = ClassicalRegister(len(left_check), "meas")
        check_circuit.add_register(meas_cr)
        for i in range(0, len(left_check)):
            check_circuit.measure(final_layout[i], meas_cr[i])

    return check_circuit
    

class IdentifyOutputMapping(AnalysisPass):
    """identify the output mapping """

    def __init__(self):
        """
        """
        super().__init__()
    def run(self, dag):
        """

        """
        self.property_set["output_mapping"] = defaultdict()
        for node in dag.topological_nodes():
            if isinstance(node, DAGOpNode) and node.name == 'measure':
                self.property_set["output_mapping"][node.cargs[0].index] = node.qargs[0]

class SavePropertySet(AnalysisPass):
    """Printing the propertyset."""

    def __init__(self, file_name = "property_set"):
        super().__init__()
        self.file_name = file_name
    def run(self, dag):
        """Run the PrintPropertySet pass on `dag`.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            write the list of the remote gates in the property_set.
        """
        # Initiate the commutation set
        f = open(self.file_name + '.pkl', 'wb')
        pickle.dump(self.property_set, f)
        f.close()
    
def update_cnot_dist(input_dict, ctrl_index, targ_index):
    output_dict = {}
    # The postprocessing process is equivalent to applying a cnot to qubit index i, and j.
    for key in input_dict.keys():
        new_key = list(key)
        if key[ctrl_index] == '1':
            if key[targ_index] == '0':
                new_key[targ_index] = '1'
            elif key[targ_index] == '1':
                new_key[targ_index] = '0'
            else:
                assert(0)
        output_dict[''.join(new_key)] = input_dict[key]
    return output_dict

def single_side_postprocess(input_dict, right_checks, qubits, layer_index):
    output_dict = input_dict.copy()
    for index in range(0, len(right_checks)):
        check = right_checks[index]
        if check == 'Z':
            ctrl_index = index # qubits - index - 1 
            targ_index = - 2 * layer_index - 1
            print(ctrl_index, targ_index)
            output_dict = update_cnot_dist(output_dict, ctrl_index, targ_index)
            # change the corresponding qubit in the distribution
    return output_dict

def calc_common_index(string_1, string_2):
    common_indexes = []
    for i in range(0, len(string_1)):
        if string_1[i] == string_2[i] and string_1[i] != 'I':
            common_indexes.append(i)
    return common_indexes


def append_meas_paulis_strings_to_circuit(circuit, pauli_strings, mapping, common_pauli_idxs, barrier):
    """
    Appends Pauli measurements to the quantum circuit based on the pauli_string input.
    """
    for pauli_index in range(0, len(pauli_strings)):
        pauli = pauli_strings[pauli_index]
        # add classical registers
        ancilla_cr = ClassicalRegister(1, str(pauli))
        circuit.add_register(ancilla_cr)
        
        meas_indexes = []
        # add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
                meas_indexes.append(index)
            elif char == 'Y':
                circuit.sdg(index)
                circuit.h(index)
                meas_indexes.append(index)
            elif char == 'Z':
                meas_indexes.append(index)
        for idx in range(0, len(meas_indexes)):
            if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
                circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])        

        # One measurement
        circuit.measure(mapping[common_pauli_idxs[pauli_index]], ancilla_cr[0])
    
        for idx in range(len(meas_indexes) - 1, -1,  -1):
            if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
                circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])  

        # add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
            elif char == 'Y':
                circuit.h(index)
                circuit.s(index)  
        if barrier is True:
            circuit.barrier()
                
                
def calc_common(string_1, string_2):
    count = 0
    for i in range(0, len(string_1)):
        if string_1[i] == string_2[i] and string_1[i] != 'I':
            count += 1
    return count

def find_largest_common(curr_pauli, indexed_list):
    max_val = 0
    best_id = 0
    for pauli_idx in range(0, len(indexed_list)):
        pauli_test = indexed_list[pauli_idx][0][2:]
        if curr_pauli != pauli_test:
            val = calc_common(curr_pauli, pauli_test)
        else:
            val = 0
        if val > max_val:
            best_id = pauli_idx
            max_val = val
    print(max_val, best_id)
    return max_val, best_id

def remove_pauli(pauli, sorted_list):
    output_list = []
    for i in sorted_list:
        if i[0][2:] != pauli:
            output_list.append(i)
    return output_list

def append_linear_meas_paulis_strings_to_circuit(circuit, pauli_strings, mapping, common_pauli_idxs, barrier):
    """
    Appends Pauli measurements to the quantum circuit based on the pauli_string input.
    """
    for pauli_index in range(0, len(pauli_strings)):
        pauli = pauli_strings[pauli_index]
        #add classical registers
        ancilla_cr = ClassicalRegister(1, str(pauli))
        circuit.add_register(ancilla_cr)
        
        meas_indexes = []
        #add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
                meas_indexes.append(index)
            elif char == 'Y':
                circuit.sdg(index)
                circuit.h(index)
                meas_indexes.append(index)
            elif char == 'Z':
                meas_indexes.append(index)
        for idx in range(0, len(meas_indexes) - 1):
            circuit.cx(meas_indexes[idx], meas_indexes[idx + 1])        

        #One measurement
        circuit.measure(meas_indexes[-1], ancilla_cr[0])

        for idx in range(len(meas_indexes) - 2, -1,  -1):
            circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
        
        # add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
            elif char == 'Y':
                circuit.h(index)
                circuit.s(index)  
        if barrier is True:
            circuit.barrier()

def postselect_counts(counts: dict, sign_list:list, num_ancillas: int )->dict:
    '''
    Assumes that ancillas are on the left.

    Args:
        num_ancillas: number of ancilla qubits, i.e., number of checks.
        counts: dict of distribution or counts.
    '''
    no_checks = num_ancillas
    sign_list_local = deepcopy(sign_list)
    sign_list_local.reverse()
    err_free_checks = ""

    for i in sign_list_local:
        if i == "+1":
            err_free_checks += "0"
        else:
            err_free_checks += "1"
    final_counts = {}
    
    for key in counts.keys():
        if err_free_checks == key[:no_checks]:
            new_key = key[no_checks:]
            final_counts[new_key] = counts[key]
    return final_counts
    
