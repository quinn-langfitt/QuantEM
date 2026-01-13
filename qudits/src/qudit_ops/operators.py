import cirq
import numpy as np


#TODO: Generalize to qudits of any dimension: Done
#TODO: Add controlled gates: Done
#TODO: Create generalized christensen gate: Done
#TODO: Y gate, Rx,y,z gates for qudits
#TODO: Perhaps create a python file for noise mapping and simulation in qudit space

# Core root of unity
omega = np.exp(2j* np.pi)  # = i

class Christensen(cirq.Gate):
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
        return f"C{self.d}"


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
        return f"[{"+"*self.a}]"
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
        return f"({self.c})", f"[{"+"*self.a}]"
    


# H = cirq.unitary(Christensen(dim=4))
# result = np.allclose(
# cirq.unitary(Christensen(dim=4)) @ cirq.unitary(ModAdd(1, dim=4)) @ H.conj().T,
# cirq.unitary(ZGate(1, dim=4))
# )
# print(result)

    