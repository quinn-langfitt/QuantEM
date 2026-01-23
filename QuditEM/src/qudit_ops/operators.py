import cirq
import numpy as np
import itertools
from typing import Sequence, Tuple

#TODO: Generalize to qudits of any dimension: Done
#TODO: Add controlled gates: Done
#TODO: Create generalized chrestenson gate: Done
#TODO: Y gate, Rx,y,z gates for qudits
#TODO: Perhaps create a python file for noise mapping and simulation in qudit space

# Core root of unity
omega = np.exp(2j* np.pi)  # = i

class Chrestenson(cirq.Gate):
    def __init__(self, dim: int = 2):
        self.d = dim

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        h = np.zeros((self.d, self.d), dtype=complex)
        for i in range(self.d):
            for j in range(self.d):
                h[i, j] =  np.exp(2j*np.pi*(1/self.d)*(i * j))
        h /= np.sqrt(self.d) 
        return h

    def _circuit_diagram_info_(self, args):
        return f"Cr"

class Chrestenson_dg(cirq.Gate):
    def __init__(self,dim: int = 2):
        self.d = dim
    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        # Return the conjugate transpose (adjoint) of the Chrestenson gate unitary.
        c = Chrestenson(dim=self.d)
        u = c._unitary_()
        return np.conjugate(u).T

    def _circuit_diagram_info_(self, args):
        return f"Cr†"
    


'''
Generalized gates for qudits of dimension = d
'''

class ZGate(cirq.Gate):
    '''
    Generalized Z gate for qudits of dimension d.
    Applies a Z gate to the target qudit
    '''
    #Applies a generalized Z gate to a qudit.
    def __init__(self, power: int, dim: int = 2):
        self.p = power
        self.d = dim

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        unitary = np.eye(self.d, dtype=complex)
        for i in range(self.d):
                unitary[i,i] = np.exp(2j*np.pi*self.p*(i%self.d)/self.d)
        return unitary

    def _circuit_diagram_info_(self, args):
        return 'Z'+str(self.p) if self.p != 1 else 'Z'

class ModAdd(cirq.Gate):
    '''
    Generalized modulo add gate for qudits of dimension d.
    Maps basis state |i⟩ to |(i + a) mod d >
    '''
    #Applies modulo add to a qudit of dimension d.
    def __init__(self, add: int, dim: int = 3):
        self.d = dim
        self.a = add

    def _qid_shape_(self):
        return (self.d,)
    
    def _unitary_(self):
        # Create a unitary matrix that adds `self.a` modulo `self.d`
        unitary = np.zeros((self.d, self.d), dtype=complex)
        for i in range(self.d):
            j = (i + self.a) % self.d
            unitary[j, i] = 1
        return unitary
        # return np.array([[0, 0, 0, 1],
        #                  [1, 0, 0, 0],
        #                  [0, 1, 0, 0],
        #                  [0, 0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return f"[{'+'*self.a}]"
    
class FlipGate(cirq.Gate):
    """
    Generalized flip gate that maps basis state |i⟩ to |j⟩ in a qudit of dimension d.
    
    """
    def __init__(self, i: int, j: int, dim: int = 2):
        # try:
        #     0 <= i < d and 0 <= j < d
        # except AssertionError:
        #     raise ValueError("i and j must be valid basis states for given dimension d")
        self.d = dim
        self.i = i
        self.j = j

    def _qid_shape_(self):
        return (self.d,)

    def _unitary_(self):
        X_flip = np.eye(self.d, dtype=complex)
        X_flip[[self.i, self.j]] = X_flip[[self.j, self.i]]
        return X_flip

    def _circuit_diagram_info_(self, args):
        return f"X({self.i},{self.j})"

class CFlipGate(cirq.Gate):
    """
    Generalized controlled flip gate that maps basis state |i⟩ to |j⟩ in a qudit of dimension d,
    conditioned on a control qudit being in state |c⟩.
    """
    def __init__(self, c: int, i: int, j: int, dim: int = 2):
        self.d = dim
        # try:
        #     0 <= c < dim and 0 <= i < dim and 0 <= j < dim
        # except AssertionError:
        #     raise ValueError("c, i, and j must be valid basis states for given dimension d")
        self.c = c
        self.i = i
        self.j = j

    def _qid_shape_(self):
        return (self.d,self.d)

    def _unitary_(self):
        a = (self.d*self.c)+self.i
        b = (self.d*self.c)+self.j
        X_flip = np.eye(self.d**2, dtype=complex)
        X_flip[[a, b]] = X_flip[[b, a]]
        return X_flip

    def _circuit_diagram_info_(self, args):
        return f"({self.c})",f"X({self.i},{self.j})"

class CModPlusGate(cirq.Gate):
    '''
    Generalized controlled modulo plus gate for qudits of dimension d.
    '''
    def __init__(self,c: int, dim: int = 2):
        self.d = dim
        self.c = c

    def _qid_shape_(self):
        return (self.d, self.d)
    
    def _unitary_(self):
        # Create a unitary matrix that adds 1 to the target qudit if the control qudit is in state c
        # X_mod = np.zeros((self.d**2, self.d**2), dtype=complex)
        # X_block = np.roll(np.eye(self.d, dtype=complex), -1, axis=1)  # d x d shift operator

        # for block in range(self.d):
        #     for row in range(self.d):
        #         for col in range(self.d):
        #             global_row = block * self.d + row
        #             global_col = block * self.d + col
        #             if block == self.c:
        #                 X_mod[global_row, global_col] = X_block[row, col]
        #             elif row == col:
        #                 X_mod[global_row, global_col] = 1  # Identity on other blocks

        X_mod = np.zeros((self.d**2, self.d**2), dtype=complex)
        for i in range(self.d**2):
            if (self.c*self.d) <= i < (self.c*self.d + self.d):
                j = (i + 1)% self.d + self.c*self.d
                X_mod[j,i] = 1
            else:
                X_mod[i, i] = 1


        # X_mod = np.eye(self.d**2, dtype=complex)
        # for i in range(self.d):
        #     for j in range(self.d):
        #         if (i + self.c) % self.d == j:
        #             X_mod[(self.d * i) + j, (self.d * i) + ((j + 1) % self.d)] = 1
        return X_mod

    def _circuit_diagram_info_(self, args):
        return f"({self.c})", f"[+]"


# print(cirq.unitary(CQuquartPlusGate(1, dim=4)))

class CModPlusSquaredGate(cirq.Gate):
    '''
    Generalized controlled modulo plus squared gate for qudits of dimension d.
    Maps basis state |i⟩ to |(i + 2) mod d
    '''
    def __init__(self,c: int, dim: int  = 2):
        self.d = dim
        self.c = c
        
    def _qid_shape_(self):
        return (self.d, self.d)
    def _unitary_(self):
        X_mod = np.zeros((self.d**2, self.d**2), dtype=complex)
        for i in range(self.d**2):
            if (self.c*self.d) <= i < (self.c*self.d + self.d):
                j = (i + 2)% self.d + self.c*self.d
                X_mod[j,i] = 1
            else:
                X_mod[i, i] = 1
        return X_mod

    def _circuit_diagram_info_(self, args):
        return f"({self.c})", f"[++]"
    
class CModPlusCubedGate(cirq.Gate):
    '''
    Generalized controlled modulo plus cubed gate for qudits of dimension d.
    Maps basis state |i⟩ to |(i + 3) mod d
    '''
    def __init__(self,c: int, dim: int  = 2):
        self.d = dim
        self.c = c
        if c == 0:
            raise ValueError("Invalid index for control qudit.")
    def _qid_shape_(self):
        return (self.d, self.d)
    def _unitary_(self):
        X_mod = np.zeros((self.d**2, self.d**2), dtype=complex)
        for i in range(self.d**2):
            if (self.c*self.d) <= i < (self.c*self.d + self.d):
                j = (i + 3)% self.d + self.c*self.d
                X_mod[j,i] = 1
            else:
                X_mod[i, i] = 1
        return X_mod

    def _circuit_diagram_info_(self, args):
        return f"({self.c})", f"[+++]"
    
class CZGate(cirq.Gate):
    """
    Generalized controlled Z gate that applies a Z gate to the target qudit
    conditioned on the control qudit being in state |c⟩.
    """
    def __init__(self, c: int, power: int=1, dim: int = 2):
        self.d = dim
        self.c = c
        self.p = power

    def _qid_shape_(self):
        return (self.d, self.d)

    def _unitary_(self):
        unitary = np.eye(self.d**2, dtype=complex)
        for i in range(self.c*self.d, self.c*self.d + self.d):
                unitary[i,i] = np.exp(2j*np.pi*self.p*(i%self.d)/self.d)
        return unitary

    def _circuit_diagram_info_(self, args):
        if self.p == 1:
            return f"({self.c})", "Z"
        else:
            return f"({self.c})", f"Z{self.p}"
# print(cirq.unitary(CZGate(4, 1)))

class CModAdd(cirq.Gate):
    '''
    Generalized controlled modulo add gate for qudits of dimension d.
    Maps basis state |i⟩ to |(i + a) mod d
    conditioned on the control qudit being in state |c⟩.
    '''
    def __init__(self, c: int, add: int, dim: int = 3):
        self.d = dim
        self.a = add
        self.c = c

    def _qid_shape_(self):
        return (self.d, self.d)
    
    def _unitary_(self):
        # Create a unitary matrix that adds `self.a` modulo `self.d`
        unitary = np.zeros((self.d**2,self.d**2), dtype=complex)
        for i in range(self.d**2):
            if (self.c*self.d) <= i < (self.c*self.d + self.d):
                j = (i + self.a) % self.d + self.c*self.d
            else:
                j = i
            unitary[j, i] = 1 
        return unitary

    def _circuit_diagram_info_(self, args):
        return f"({self.c})", f"[{'+'*self.a}]"
    
class ZXcomm_rel():
    def __init__(self, z: int, x: int, dim: int = 2):
        self.z = z
        self.x = x
        self.d = dim
        if self.z>= self.d or self.x >= self.d:
            raise ValueError("Invalid indices for commutation relation.")
    def __str__(self):
        """
        Prints out the commutation relation between Z gate and Modulo Add gate.
        """
        Z = cirq.unitary(ZGate(self.z,dim=self.d))
        X = cirq.unitary(ModAdd(self.x,dim=self.d))
        ZX = Z @ X
        XZ = X @ Z

        nz_ZX = ZX[ZX != 0]
        nz_XZ = XZ[XZ != 0]
        np.set_printoptions(precision=5,suppress=True)
        factor = np.around(nz_ZX[-1]/nz_XZ[-1], decimals=3)
        
        return (
            f"Z{self.z}_X{self.x}\n"
            f"{ZX}\n"
            f"X{self.x}_Z{self.z}\n"
            f"{XZ}\n\n"
            f" Z{self.z}_X{self.x} = {factor}*X{self.x}_Z{self.z}"
            )

SHIFT = ModAdd(1, dim=3)._unitary_()
# omega = np.exp(1j * 2 / 3 * np.pi)
CLOCK = ZGate(1, dim=3)._unitary_()



class QutritDepolarizingChannel(cirq.Gate):
    r"""A channel that depolarizes one qutrit.
    """

    def __init__(self, p: float) -> None:
        """Constructs a depolarization channel on a qutrit.

        Args:
            p: The probability that one of the shift/clock matrices is applied. Each of
                the 8 shift/clock gates is applied independently with probability
                $p / 8$.
            n_qubits: the number of qubits.

        Raises:
            ValueError: if p is not a valid probability.
        """

        error_probabilities = {}

        p_depol = p / 8
        p_identity = 1.0 - p
        for gate_pows in itertools.product(range(3), range(3)):
            if gate_pows == (0, 0):
                error_probabilities[gate_pows] = p_identity
            else:
                error_probabilities[gate_pows] = p_depol
        self.error_probabilities = error_probabilities
        self._p = p

    def _qid_shape_(self):
        return (3,)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        op = lambda shift_pow, clock_pow: np.linalg.matrix_power(SHIFT, shift_pow) @ np.linalg.matrix_power(CLOCK,
                                                                                                            clock_pow)
        return [(self.error_probabilities[(shift_pow, clock_pow)], op(shift_pow, clock_pow))
                for (shift_pow, clock_pow) in self.error_probabilities.keys()]

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def _circuit_diagram_info_(self, args):
        if args.precision is not None:
            return (f"D3({self._p:.{args.precision}g})",)
        else:
            return (f"D3({self._p})",)
        return result

    @property
    def p(self) -> float:
        """The probability that one of the qutrit gates is applied.

        Each of the 8 Pauli gates is applied independently with probability
        $p / 8$.
        """
        return self._p

    @property
    def n_qubits(self) -> int:
        """The number of qubits"""
        return 1

class QuditDepolarizingChannel(cirq.Gate):
    r"""A channel that depolarizes one qudit.
    """

    def __init__(self, p: float,dim: int) -> None:
        """Constructs a depolarization channel on a qutrit.

        Args:
            p: The probability that one of the shift/clock matrices is applied. Each of
                the 8 shift/clock gates is applied independently with probability
                $p / 8$.
            n_qubits: the number of qubits.

        Raises:
            ValueError: if p is not a valid probability.
        """
        error_probabilities = {}
        f_dim = float(dim)
        # p = ((f_dim**2 - 1)/f_dim**2)*p

        p_depol = p / (f_dim**2 - 1)
        p_identity = 1.0 - p
        for gate_pows in itertools.product(range(dim), range(dim)): 
            if gate_pows == (0, 0):
                error_probabilities[gate_pows] = p_identity
            else:
                error_probabilities[gate_pows] = p_depol
        self.error_probabilities = error_probabilities
        self._p = p
        self._d = dim
    def _qid_shape_(self):
        return (self._d,)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        DSHIFT = ModAdd(1, dim=self._d)._unitary_()
        DCLOCK = ZGate(1, dim=self._d)._unitary_()
        op = lambda shift_pow, clock_pow: np.linalg.matrix_power(DSHIFT, shift_pow) @ np.linalg.matrix_power(DCLOCK, clock_pow)

        return [(self.error_probabilities[(shift_pow, clock_pow)], op(shift_pow, clock_pow))
                for (shift_pow, clock_pow) in self.error_probabilities.keys()]

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def _circuit_diagram_info_(self, args):
        if args.precision is not None:
            return (f"D{self._d}({self._p:.{args.precision}g})",)
        else:
            return (f"D{self._d}({self._p})",)

    @property
    def p(self) -> float:
        """The probability that one of the qutrit gates is applied.

        Each of the d**2 - 1 Pauli gates is applied independently with probability
        $p / d**2 - 1$.
        """
        return self._p
    
    @property
    def dim(self) -> int:
        """The dimension of the qudit"""
        return self._d

    @property
    def n_qubits(self) -> int:
        """The number of qubits"""
        return 1