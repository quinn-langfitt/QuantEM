import qiskit
import itertools
import pickle
import math
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from typing import Any
from typing import Callable
from collections import defaultdict
from qiskit.dagcircuit import DAGOutNode, DAGOpNode

from qiskit import (
    QuantumCircuit,
    ClassicalRegister
)
from typing import List
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager

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

        # Place counter somewhere here?

        while True:
            # print("layers = " , layers)
            # print()
            layer = layers[temp_check_reversed.layer_idx]
            # print("single layer = ", layer)
            # print()

            for node in layer:
                # print("node = ", node.name)
                # print()
                if isinstance(node, DAGOpNode): 
                    self.handle_operator_node(node, temp_check_reversed)
            if self.should_return_result(temp_check_reversed, num_layers):
                p1.phase = temp_check_reversed.phase
                p1.operations = list(reversed(temp_check_reversed.operations))
                return self.get_check_strs(p1, p2)
            temp_check_reversed.layer_idx += 1
            # print("layer index for temp check = ", temp_check_reversed.layer_idx)

    def handle_operator_node(self, node, temp_check_reversed: TempCheckOperator):
        '''Handles operations for nodes of type "op".'''
        current_qubits = self.get_current_qubits(self, node)
        current_ops = [temp_check_reversed.operations[qubit] for qubit in current_qubits]
        node_op = node.name.upper()
        self.update_current_ops(current_ops, node_op, temp_check_reversed, current_qubits)

    def should_return_result(self, temp_check_reversed: TempCheckOperator, num_layers: int) -> bool:
        '''Checks if we have reached the last layer.'''
        return temp_check_reversed.layer_idx == num_layers - 1

    @staticmethod
    def update_current_ops(op1: List[str], op2: str, temp_check_reversed: TempCheckOperator, current_qubits: List[int]):
        '''Finds the intermediate check. Always push op1 through op2. '''
        # print("pushing op1: ", op1, " through op2: ", op2)
        result = ChecksFinder.get_result(op1, op2)
        # print("result: ", result)
        # print()
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
    
#     @staticmethod
#     def get_current_qubits(node):
#         '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
#         # We have to check for single or two qubit gates.
#         if node.name in ["x", "y", "z", "h", "s", "sdg", "rz"]:
#             return [node.qargs[0].index]
#         elif node.name in ["cx", "swap"]:
#             return [node.qargs[0].index, node.qargs[1].index]
#         else:
#             assert False, "Overlooked a node operation."
            
    # Use for new qiskit version (Qiskit verions >= 1.0)
    @staticmethod
    def get_current_qubits(self, node):
        '''Finding checks: Symbolic: get the current qubits whose operations that will be passed through.'''
        circ_dag = circuit_to_dag(self.circ_reversed)
        dag_qubit_map = {bit: index for index, bit in enumerate(circ_dag.qubits)}
        # We have to check for single or two qubit gates.
        # print("node.name = ", node.name)
        if node.name in ["x", "y", "z", "h", "s", "sdg", "rz", "rx"]:
            return [dag_qubit_map[node.qargs[0]]]
        elif node.name in ["cx", "swap"]:
            return [dag_qubit_map[node.qargs[0]], dag_qubit_map[node.qargs[1]]]
        else:
            assert False, "Overlooked a node operation."

    def get_max_checks_linear(self, pauli_labels, max_layers):
        print('searching...')
        count=0
        p2_weights=[]
        pauli_str_p1s=[]
        pauli_str_p2s=[]
        paulis_by_weight=sorted(pauli_labels, key=get_weight, reverse=True)
        # print(paulis_by_weight)
        for elem in paulis_by_weight:
            # temp_p2_circ=find_p2s(elem, circ)
            if get_weight(elem)>0:
                checks_result=self.find_checks_sym(elem)
                if checks_result:
                    count+=1
                    self._append_result(checks_result, p2_weights, pauli_str_p1s, pauli_str_p2s)
                    # Only get this number of layers. This limit (which works for weight one checks)
                    # comes from the fact that we only need an X and Y C_2 for each compute qubit.
                    # All the Pauli terms anticommute with X or Y. However, the search may
                    # pickup an X, Y, and Z for one qubit so we need more.
                    if count==max_layers:
                        break
        return count, p2_weights, pauli_str_p1s, pauli_str_p2s

    def get_min_checks_linear(self, pauli_labels, max_layers):
        print('searching...')
        count=0
        p2_weights=[]
        pauli_str_p1s=[]
        pauli_str_p2s=[]
        paulis_by_weight=sorted(pauli_labels, key=get_weight, reverse=False)
        # print(paulis_by_weight)
        for elem in paulis_by_weight:
            # temp_p2_circ=find_p2s(elem, circ)
            if get_weight(elem)>0:
                checks_result=self.find_checks_sym(elem)
                if checks_result:
                    count+=1
                    self._append_result(checks_result, p2_weights, pauli_str_p1s, pauli_str_p2s)
                    # Only get this number of layers. This limit (which works for weight one checks)
                    # comes from the fact that we only need an X and Y C_2 for each compute qubit.
                    # All the Pauli terms anticommute with X or Y. However, the search may
                    # pickup an X, Y, and Z for one qubit so we need more.
                    if count==max_layers:
                        break
        return count, p2_weights, pauli_str_p1s, pauli_str_p2s

            
            
def append_paulis_to_circuit(circuit, pauli_string):
    """
    Appends Pauli operations to the quantum circuit based on the pauli_string input.
    """
    for index, char in enumerate(reversed(pauli_string)):
        if char == 'I':
            # circuit.i(index)
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

# def add_pauli_checks(circuit, left_check, right_check, initial_layout, final_layout, single_side = False, qubit_measure = False, ancilla_measure = False, barriers = False):
#     #initial_layout: mapping from original circuit index to the physical qubit index
#     #final_layout: mapping from original circuit index to the final physical qubit index
#     if initial_layout is None:
#         #Number of qubits in circuit and checks must be equal.
#         assert len(circuit.qubits) == len(left_check) == len(right_check)
#         #First verify the paulis are correct:
#         verification_circuit, equal = verify_circuit_with_pauli_checks(circuit, left_check, right_check)
#         assert(equal)
#     ancilla_index = len(circuit.qubits)
#     check_circuit = QuantumCircuit(ancilla_index + 1)
#     check_circuit.h(ancilla_index)
#     append_control_paulis_to_circuit(check_circuit, left_check, ancilla_index, initial_layout)
#     if barriers is True:
#         check_circuit.barrier()
#     check_circuit.compose(circuit, inplace=True)
#     if barriers is True:
#         check_circuit.barrier()
#     if single_side is False:
#         append_control_paulis_to_circuit(check_circuit, right_check, ancilla_index, final_layout)
#     check_circuit.h(ancilla_index)
    
#     if ancilla_measure is True:
#         #add one measurement for the ancilla measurement
#         ancilla_cr = ClassicalRegister(1, str(right_check))
#         check_circuit.add_register(ancilla_cr)
#         check_circuit.measure(ancilla_index, ancilla_cr[0])
        
#     if qubit_measure is True:
#         meas_cr = ClassicalRegister(len(left_check), "meas")
#         check_circuit.add_register(meas_cr)
#         for i in range(0, len(left_check)):
#             check_circuit.measure(final_layout[i], meas_cr[i])
#     return check_circuit


def add_pauli_checks(circuit, left_check, right_check, initial_layout, final_layout, pauli_meas = False, single_side = False, qubit_measure = False, ancilla_measure = False, barriers = False, increase_size = 0):
    #initial_layout: mapping from original circuit index to the physical qubit index
    #final_layout: mapping from original circuit index to the final physical qubit index
    if initial_layout is None:
        #Number of qubits in circuit and checks must be equal.
        assert len(circuit.qubits) == len(left_check) == len(right_check)
        #First verify the paulis are correct:
        verification_circuit, equal = verify_circuit_with_pauli_checks(circuit, left_check, right_check)
        assert(equal)
    ancilla_index = len(circuit.qubits)
    if increase_size > 0:
        ancilla_index = len(circuit.qubits) - increase_size
    if pauli_meas is False:
#         check_circuit = QuantumCircuit(ancilla_index)
#         append_meas_paulis_to_circuit(check_circuit, left_check, initial_layout)
#     else:
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
        #add one measurement for the ancilla measurement
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
            # if isinstance(node, DAGOpNode) and node.name is 'measure':
            if isinstance(node, DAGOpNode) and node.name == 'measure':
                #print(node.qargs, node.cargs)
                self.property_set["output_mapping"][node.cargs[0].index] = node.qargs[0]
#         return self.property_set["remote_gates"]

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
        
def gen_initial_layout(orign_qc, mapped_qc):
    initial_layout = {}
    for i in range(0, len(orign_qc.qubits)):
        initial_layout[i] = mapped_qc.layout.initial_layout.get_virtual_bits()[orign_qc.qubits[i]]
    return initial_layout

def gen_final_layout(orign_qc, mapped_qc):
    final_layout = {}
    pm = PassManager(IdentifyOutputMapping())
    pm.append(SavePropertySet("property_set"))
    new_circ = pm.run(mapped_qc)
    with open('property_set.pkl', 'rb') as f:
        p_set = pickle.load(f)
    print(p_set['output_mapping'])
    for i in range(0, len(orign_qc.qubits)):
        qubit = orign_qc.qubits[i]
        final_layout[i] = p_set['output_mapping'][i].index
    return final_layout

# def filter_results(dictionary, qubits, indexes, sign_list):
#     new_dict = {}
#     for key in dictionary.keys():
#         new_key = ''
#         for i in range(len(key)):
#             meas_index = math.floor((i - qubits)/2)
# #             print(i, meas_index)
# #             print(meas_index in indexes)
#             if meas_index in indexes and key[i] == sign_list[meas_index]:
#                 #the key equals the sign
# #                 print("not found")
#                 new_key = ''
#                 break
#             if meas_index not in indexes:
#                 new_key += key[i]
#         if new_key != '':
#             new_dict[new_key] = dictionary[key]
#     return new_dict
    
def update_cnot_dist(input_dict, ctrl_index, targ_index):
    output_dict = {}
    #the postprocessing process is equivalent to applying a cnot to qubit index i, and j.
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
            ctrl_index = index#qubits - index - 1 
            targ_index = - 2 * layer_index - 1
            print(ctrl_index, targ_index)
            output_dict = update_cnot_dist(output_dict, ctrl_index, targ_index)
            #change the corresponding qubit in the distribution
    return output_dict

def complete_postprocess(input_dist, qubits, check_count, pr_list):
    output_dict = input_dist
    for i in range(0, check_count):
        output_dict = single_side_postprocess(output_dict, right_checks = pr_list[i], qubits = qubits, layer_index = i)
    return output_dict   

def rightchecks_postprocess(input_dist, qubits, check_count, pr_list):
    output_dict = input_dist
    for i in range(0, check_count):
        output_dict = single_side_postprocess(output_dict, right_checks = pr_list[i], qubits = qubits, layer_index = i)
    return output_dict   



#compose have problem for inplace measurements, because one qubit can't be measured by two classical bits
def add_meas_pauli_checks(circuit, left_checks, initial_layout, final_layout, barriers, common_pauli_idxs, barrier = False):

    check_circuit = QuantumCircuit(len(circuit.qubits))
    append_meas_paulis_strings_to_circuit(check_circuit, left_checks, initial_layout, common_pauli_idxs, barrier)
    check_circuit.compose(circuit, inplace=True)
    meas_cr = ClassicalRegister(len(left_checks[0]), "meas")
    check_circuit.add_register(meas_cr)
    for i in range(0, len(left_checks[0])):
        check_circuit.measure(final_layout[i], meas_cr[i])
    return check_circuit

def calc_common_index(string_1, string_2):
    common_indexes = []
    new_string = []
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
        for idx in range(0, len(meas_indexes)):
            if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
                circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])        
#         if pauli_index == 0:
#             #apply a chain of CNOTs
#             for idx in range(0, len(meas_indexes) - 1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the previous pauli:
#             prev_pauli = pauli_strings[pauli_index - 1]
#             common_indexes = calc_common_index(prev_pauli[::-1], pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("prev common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
# #             print(new_meas_indexes)
#             #apply a chain of CNOTs
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])

        #One measurement
#         circuit.measure(meas_indexes[-1], ancilla_cr[0])
        circuit.measure(mapping[common_pauli_idxs[pauli_index]], ancilla_cr[0])
    
        for idx in range(len(meas_indexes) - 1, -1,  -1):
            if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
                circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])  
#         if pauli_index == len(pauli_strings) - 1:
#             #apply a chain of CNOTs
#             for idx in range(len(meas_indexes) - 2, -1,  -1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the next pauli:
#             next_pauli = pauli_strings[pauli_index + 1]
#             common_indexes = calc_common_index(pauli[::-1], next_pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("next common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
#             #apply a chain of CNOTs
#             for idx in range(len(new_meas_indexes) - 2, -1,  -1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
                
#         #Then apply a chain of CNOTs?
#         for idx in range(len(meas_indexes) - 2, -1,  -1):
#             print(meas_indexes, idx)
#             circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
        #add Rotations
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
    new_string = []
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
    
def search_for_pauli_list(init_pauli, sorted_list):
    output_paulis = []
    output_paulis_sign = []
    indexed_list = []
    for index in range(0, len(sorted_list)):
        temp_list = sorted_list[index].copy()
        temp_list.append(index)
        if temp_list[0][2:] != init_pauli:
            indexed_list.append(temp_list)
        else:
            output_paulis.append([temp_list[0][2:], temp_list[1][2:]])
            output_paulis_sign.append([temp_list[0], temp_list[1]])
#     print(indexed_list)
    curr_pauli = init_pauli
    while len(indexed_list) > 0:
        print(indexed_list)
        max_val, best_id = find_largest_common(curr_pauli, indexed_list)
        del_pauli = indexed_list[best_id][0][2:]
        output_paulis.append([del_pauli, indexed_list[best_id][1][2:]])
        output_paulis_sign.append([indexed_list[best_id][0], indexed_list[best_id][1]])
        indexed_list = remove_pauli(del_pauli, indexed_list)
        curr_pauli = del_pauli
    return output_paulis, output_paulis_sign


#compose have problem for inplace measurements, because one qubit can't be measured by two classical bits
def add_linear_meas_pauli_checks(circuit, left_checks, initial_layout, final_layout, barriers, common_pauli_idxs, barrier = False):

    check_circuit = QuantumCircuit(len(circuit.qubits))
    append_linear_meas_paulis_strings_to_circuit(check_circuit, left_checks, initial_layout, common_pauli_idxs, barrier)
    check_circuit.compose(circuit, inplace=True)
    meas_cr = ClassicalRegister(len(left_checks[0]), "meas")
    check_circuit.add_register(meas_cr)
    for i in range(0, len(left_checks[0])):
        check_circuit.measure(final_layout[i], meas_cr[i])
    return check_circuit

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
            # if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
                circuit.cx(meas_indexes[idx], meas_indexes[idx + 1])        
#         if pauli_index == 0:
#             #apply a chain of CNOTs
#             for idx in range(0, len(meas_indexes) - 1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the previous pauli:
#             prev_pauli = pauli_strings[pauli_index - 1]
#             common_indexes = calc_common_index(prev_pauli[::-1], pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("prev common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
# #             print(new_meas_indexes)
#             #apply a chain of CNOTs
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])


    
        # prev_pauli = None
        # common_prev_indexes = []
        # post_pauli = None
        # common_post_indexes = []
        # if pauli_index != 0:
        #     prev_pauli = pauli_strings[pauli_index - 1]
        #     common_prev_indexes = calc_common_index(prev_pauli[::-1], pauli[::-1])
        #     common_prev_meas_indexes = [mapping[i] for i in common_prev_indexes]
        #     print("prev common idx", common_prev_indexes, common_prev_meas_indexes)
        # if pauli_index != len(pauli_strings):
        #     post_pauli = pauli_strings[pauli_index - 1]
        #     common_post_indexes = calc_common_index(post_pauli[::-1], pauli[::-1])
        #     common_post_meas_indexes = [mapping[i] for i in common_post_indexes]
        #     print("post common idx", common_prev_indexes, common_post_meas_indexes)
            
            
            

        #One measurement
#         circuit.measure(meas_indexes[-1], ancilla_cr[0])
        circuit.measure(meas_indexes[-1], ancilla_cr[0])

        for idx in range(len(meas_indexes) - 2, -1,  -1):
            circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
        # for idx in range(len(meas_indexes) - 1, -1,  -1):
        #     if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
        #         circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])  
        
#         if pauli_index == len(pauli_strings) - 1:
#             #apply a chain of CNOTs
#             for idx in range(len(meas_indexes) - 2, -1,  -1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the next pauli:
#             next_pauli = pauli_strings[pauli_index + 1]
#             common_indexes = calc_common_index(pauli[::-1], next_pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("next common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
#             #apply a chain of CNOTs
#             for idx in range(len(new_meas_indexes) - 2, -1,  -1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
                
#         #Then apply a chain of CNOTs?
#         for idx in range(len(meas_indexes) - 2, -1,  -1):
#             print(meas_indexes, idx)
#             circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
        #add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
            elif char == 'Y':
                circuit.h(index)
                circuit.s(index)  
        if barrier is True:
            circuit.barrier()


#compose have problem for inplace measurements, because one qubit can't be measured by two classical bits
def add_linear_opt_meas_pauli_checks(circuit, left_checks, initial_layout, final_layout, barriers, common_pauli_idxs, barrier = False):

    check_circuit = QuantumCircuit(len(circuit.qubits))
    append_linear_opt_meas_paulis_strings_to_circuit(check_circuit, left_checks, initial_layout, common_pauli_idxs, barrier)
    check_circuit.compose(circuit, inplace=True)
    meas_cr = ClassicalRegister(len(left_checks[0]), "meas")
    check_circuit.add_register(meas_cr)
    for i in range(0, len(left_checks[0])):
        check_circuit.measure(final_layout[i], meas_cr[i])
    return check_circuit

def append_linear_opt_meas_paulis_strings_to_circuit(circuit, pauli_strings, mapping, common_pauli_idxs, barrier):
    """
    Appends Pauli measurements to the quantum circuit based on the pauli_string input.
    """
    shared_pauli_idxs = []
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
        # for idx in range(0, len(meas_indexes) - 1):
        #     # if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
        #         circuit.cx(meas_indexes[idx], meas_indexes[idx + 1])        
#         if pauli_index == 0:
#             #apply a chain of CNOTs
#             for idx in range(0, len(meas_indexes) - 1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the previous pauli:
#             prev_pauli = pauli_strings[pauli_index - 1]
#             common_indexes = calc_common_index(prev_pauli[::-1], pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("prev common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
# #             print(new_meas_indexes)
#             #apply a chain of CNOTs
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])


    
        prev_pauli = None
        common_prev_indexes = []
        post_pauli = None
        common_post_indexes = []
        if pauli_index != 0:
            prev_pauli = pauli_strings[pauli_index - 1]
            common_prev_indexes = calc_common_index(prev_pauli[::-1], pauli[::-1])
        if pauli_index != len(pauli_strings):
            post_pauli = pauli_strings[pauli_index - 1]
            common_post_indexes = calc_common_index(post_pauli[::-1], pauli[::-1])
        common_prev_meas_indexes = [mapping[i] for i in common_prev_indexes]
        print("prev common idx", common_prev_indexes, common_prev_meas_indexes)  
        common_post_meas_indexes = [mapping[i] for i in common_post_indexes]
        print("post common idx", common_post_indexes, common_post_meas_indexes)
        merged_meas_indexes = common_prev_meas_indexes.copy()
        
        for item in common_post_meas_indexes:
            if item not in merged_meas_indexes:
                merged_meas_indexes.append(item)

        common_meas_indexes = []

        if pauli_index > 0:
            prev_shared_index = shared_pauli_idxs[pauli_index - 1]
        else:
            prev_shared_index = -1

        for item in common_post_meas_indexes:
            if item in common_prev_meas_indexes:
                common_meas_indexes.append(item)
            
        if len(common_meas_indexes) == 0:
            common_meas_indexes = [merged_meas_indexes[-1]]

        #if the previously shared index is in the current common meas_indexes
        if prev_shared_index in common_meas_indexes:
            shared_index = prev_shared_index
        else:
            shared_index = common_meas_indexes[-1]
        shared_pauli_idxs.append(shared_index)
                
        ladder_meas_indexes = [item for item in meas_indexes if item not in merged_meas_indexes] + [shared_index]
        ladder_meas_indexes.sort()

        #first apply the star shape circuit for cancellation
        for idx in range(0, len(merged_meas_indexes) - 1):
            if merged_meas_indexes[idx] != shared_index:
                circuit.cx(merged_meas_indexes[idx], shared_index)    

        #then apply the ladder shape circuit:
        for idx in range(0, len(ladder_meas_indexes) - 1):
            circuit.cx(ladder_meas_indexes[idx], ladder_meas_indexes[idx + 1])
        
        #One measurement
#         circuit.measure(meas_indexes[-1], ancilla_cr[0])
        circuit.measure(ladder_meas_indexes[-1], ancilla_cr[0])

        #apply the inversed ladder shape circuti
        for idx in range(len(ladder_meas_indexes) - 2, -1,  -1):
            circuit.cx(ladder_meas_indexes[idx], ladder_meas_indexes[idx + 1])

        #apply the inversed star shape circuit for cancellation
        for idx in range(len(merged_meas_indexes) - 1, -1,  -1):
            if merged_meas_indexes[idx] != shared_index:
                circuit.cx(merged_meas_indexes[idx], shared_index)  
        
        # for idx in range(len(meas_indexes) - 1, -1,  -1):
        #     if meas_indexes[idx] != mapping[common_pauli_idxs[pauli_index]]:
        #         circuit.cx(meas_indexes[idx], mapping[common_pauli_idxs[pauli_index]])  
        
#         if pauli_index == len(pauli_strings) - 1:
#             #apply a chain of CNOTs
#             for idx in range(len(meas_indexes) - 2, -1,  -1):
#                 circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
#         else:
#             #compare with the next pauli:
#             next_pauli = pauli_strings[pauli_index + 1]
#             common_indexes = calc_common_index(pauli[::-1], next_pauli[::-1])
#             common_meas_indexes = [mapping[i] for i in common_indexes]
#             print("next common idx", common_indexes, meas_indexes, common_meas_indexes, meas_indexes)
#             new_meas_indexes =  common_meas_indexes + [i for i in meas_indexes if i not in common_meas_indexes]
#             #apply a chain of CNOTs
#             for idx in range(len(new_meas_indexes) - 2, -1,  -1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
#             for idx in range(0, len(new_meas_indexes) - 1):
#                 circuit.cx(new_meas_indexes[idx],new_meas_indexes[idx + 1])
                
#         #Then apply a chain of CNOTs?
#         for idx in range(len(meas_indexes) - 2, -1,  -1):
#             print(meas_indexes, idx)
#             circuit.cx(meas_indexes[idx],meas_indexes[idx + 1])
        #add Rotations
        for orign_index, char in enumerate(reversed(pauli)):
            index = mapping[orign_index]
            if char == 'X':
                circuit.h(index)
            elif char == 'Y':
                circuit.h(index)
                circuit.s(index)  
        if barrier is True:
            circuit.barrier()
            

def filter_results(dictionary, qubits, indexes, sign_list):
    new_dict = {}
    for key in dictionary.keys():
        new_key = ''
        for i in range(len(key)):
            meas_index = math.floor((i - qubits)/2)
#             print(i, meas_index)
#             print(meas_index in indexes)
            if meas_index in indexes and key[i] == sign_list[meas_index]:
                #the key equals the sign
#                 print("not found")
                new_key = ''
                break
            if meas_index not in indexes:
                new_key += key[i]
        if new_key != '':
            new_dict[new_key] = dictionary[key]
    return new_dict

def pauli_strings_commute(pauli_str1, pauli_str2):
    """
    Determine if two Pauli strings commute.
    
    :param pauli_str1: A string representing the first Pauli operator.
    :param pauli_str2: A string representing the second Pauli operator.
    :return: True if the Pauli strings commute, False otherwise.
    """
    if len(pauli_str1) != len(pauli_str2):
        raise ValueError("Pauli strings must be of the same length.")
    
    commute = True  # Assume they commute until proven otherwise
    
    anticommute_count = 0
    
    for i in range(len(pauli_str1)):
        if pauli_str1[i] != pauli_str2[i] and pauli_str1[i] != 'I' and pauli_str2[i] != 'I':
            # Found anti-commuting Pauli matrices
            commute = False
            anticommute_count += 1

    if anticommute_count % 2 == 0:
        commute = True
    
    return commute

def postselect_counts(counts: dict, num_ancillas: int)->dict:
    '''Assumes that ancillas are on the left.
    Args:
        num_ancillas: number of ancilla qubits, i.e., number of checks.
        counts: dict of distribution or counts.'''
    final_counts = {}
    for bitstring, count in counts.items():
        data_str=bitstring[num_ancillas::]
        ancilla_str=bitstring[:num_ancillas:]
        if  ancilla_str=="0"*num_ancillas: 
            final_counts[data_str] = final_counts.get(data_str, 0) + count
    return final_counts