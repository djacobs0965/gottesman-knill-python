import numpy as np
import galois
import random

GF = galois.GF(2)

# helper array for PauliString.to_string function
a_to_str = ["", "i", "-", "-i"]


# helper function to implement multiplication over the finite field of n-bit strings mod 2.
def mult(n, m):
    return np.sum(np.logical_and(n, m))


####################################
# a Pauli string is represented in form i^(a)X^(b)Z^(c), where a is 0-3, b and c are in {0,1}^n
# NOTE: we use int for dtype for b and c for compatibility with the galois library.
# TODO: use galois.GF for this class instead of modulo operator
####################################
class PauliString:
    def __init__(self, n, z_qubit, a=0):
        self.n = n
        self.a = a
        self.b = np.zeros(n, dtype="int")
        self.c = np.zeros(n, dtype="int")
        self.c[z_qubit] = 1

    # applies S gate to register r
    def apply_s(self, r):
        if self.b[r]:
            self.a = (self.a + 1) % 4
            self.c[r] = (self.c[r] + 1) % 2

    # applies H gate to register r
    def apply_h(self, r):
        tmp = self.b[r]
        self.b[r] = self.c[r]
        self.c[r] = tmp

    # applies CNOT gate to register r
    def apply_cnot(self, r1, r2):
        self.b[r2] = (self.b[r1] + self.b[r2]) % 2
        self.c[r1] = (self.c[r1] + self.c[r2]) % 2

    # applies Z gate to reigster r
    def apply_z(self, r):
        self.apply_s(r)
        self.apply_s(r)

    # applies X gate to register r
    def apply_x(self, r):
        self.apply_h(r)
        self.apply_z(r)
        self.apply_h(r)

    # updates this Pauli string by left multiplying by another Pauli string
    def left_multiply(self, other):
        self.a = self.a + other.a + (2 * mult(self.c, other.b))
        self.b = mult(self.b, other.b)
        self.c = mult(self.c, other.c)

    # returns a string representation of the Pauli string
    def to_string(self):
        a = self.a
        s = []
        for j in range(self.n):
            if self.b[j] and self.c[j]:
                s.append("Y")
                a = (a - 1) % 4
            elif self.b[j]:
                s.append("X")
            elif self.c[j]:
                s.append("Z")
            else:
                s.append("I")
        s.insert(0, a_to_str[a])
        return "".join(s)


#####################################
# A CliffordState is a collection of PauliString objects which store a set of generators for the stabilizer subgroup corresponding to a quantum state.
#####################################
class CliffordState:
    def __init__(self, n):
        self.n = n
        self.generators = []
        for j in range(n):
            self.generators.append(PauliString(n, j))

    # Apply S gate to register r
    def apply_s(self, r):
        for g in self.generators:
            g.apply_s(r)

    # Apply H gate to register r
    def apply_h(self, r):
        for g in self.generators:
            g.apply_h(r)

    # Apply CNOT gate to register r
    def apply_cnot(self, r1, r2):
        for g in self.generators:
            g.apply_cnot(r1, r2)

    # Apply Z gate to register r
    def apply_z(self, r):
        for g in self.generators:
            g.apply_z(r)

    # Apply X gate to register r
    def apply_x(self, r):
        for g in self.generators:
            g.apply_x(r)

    # Update the generators to ensure that there is at most 1 anticommuting Pauli string
    def normalize(self, r):
        anticommuter = -1
        for j in range(self.n):
            if self.generators[j].b[r]:
                if anticommuter == -1:
                    anticommuter = j
                else:
                    self.generators[j].multiply(self.generators[anticommuter])
        if anticommuter != -1:
            self.generators[0], self.generators[anticommuter] = self.generators[anticommuter], self.generators[0]


    # Perform a measurement on register r
    def measure_register(self, r):
        self.normalize(r)
        # anticommuting case
        if self.generators[0].b[r]:
            val = random.randint(0, 1)
            if val == 0:
                self.generators[0] = PauliString(self.n, r, 0)
            else:
                self.generators[0] = PauliString(self.n, r, 2)
            return val
        # all commuting case
        # We create a matrix representing our state (ignoring phase) and solve for a set of generators that multiply to Z_r.
        # We use the the Normal equation to solve the overdetermined system
        else: 
            np_mat = np.zeros((self.n, 2 * self.n), dtype="int")
            mat = GF(np_mat)
            for j, g in enumerate(self.generators):
                arr = np.concatenate((g.b, g.c))
                mat[j] = GF(arr)
        np_e = np.zeros(2 * self.n, dtype="int")
        e = GF(np_e)
        e[self.n + r] = 1
        mat_t = np.transpose(mat)
        mat_normal = np.matmul(mat, mat_t)
        mat_normal_inv = np.linalg.inv(mat_normal)
        vals = np.matmul(mat, e)
        sol = np.matmul(mat_normal_inv, vals)
        # If we find a set of the generators that multiply to Z_r,
        # then we compute the phase of their product to determine the measurement outcome
        if np.array_equal(np.matmul(mat_t, sol), e):
            a = 0
            for j, x in enumerate(sol):
                if x:
                    a += self.generators[j].a
            sign = a % 4
            self.generators[r] = PauliString(self.n, r, a)
            return sign // 2
        # If our least squares did not return an exact solution, something went wrong, since we have $n$ generators, which should
        # be enough to genearte the entire Clifford subspace
        # TODO: verify this claim
        else:
            print("error")

    # perform measurement on the entire n-qubit state
    def measure(self):
        state = []
        for j in range(self.n):
            state.append(self.measure_register(j))
        return state

    # return a string representation of the set of generators
    def to_string(self):
        s = []
        for g in self.generators:
            s.append(g.to_string())
        return str(s)
    


### Some functions to demonstrate the algorithm:

# circuit takes an array of strings describing gates and returns the output of applying the gates then measuring
def circuit(n, gates):
  state = CliffordState(n)
  for gate in gates:
    r1 = gate[1]
    g = gate[0]
    if g == 's':
      state.apply_s(r1)
    elif g == 'h':
      state.apply_h(r1)
    elif g == 'c':
      state.apply_cnot(r1, gate[2])
    elif g == 'x':
      state.apply_x(r1)
    elif g == 'z':
      state.apply_z(r1)
  return state

# generates then performs measurement on the n-qubit entangled state |0...0> + |1...1>
def generate_entangled(n):
  state = CliffordState(n)
  state.apply_h(0)
  for j in range(0, n-1):
    state.apply_cnot(j, j+1)
  return state.measure()

# generates then performs measurement on the n-qubit maximally mixed state
def generate_mixed(n):
  state = CliffordState(n)
  for j in range(0, n):
    state.apply_h(j)
  return state.measure()
