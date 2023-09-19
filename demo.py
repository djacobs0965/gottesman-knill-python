from gottesman_knill import *
import sys

# run this demo using:
# python demo.py ["mixed"|"entangled"] num_qubits num_trials

# check that an option is provided
if len(sys.argv) != 4:
    sys.exit(1)

# Set the maximum allowed number of qubits
max_n = 4
max_trials = 10000

f_str = sys.argv[1]
n_str = sys.argv[2]
trials_str = sys.argv[3]
f = None
n = None
if f_str == 'mixed' and n_str.isdigit() and int(n_str) <= max_n and trials_str.isdigit() and int(trials_str) <= max_trials:
    f = generate_mixed
elif f_str == 'entangled' and n_str.isdigit() and int(n_str) <= max_n and trials_str.isdigit() and int(trials_str) <= max_trials:
    f = generate_entangled
else:
    sys.exit(2)

n = int(n_str)
trials = int(trials_str)
shape = [2] * n
output = np.zeros(shape, dtype="int")
for j in range(trials):
    state = f(n)
    output[tuple(state)] += 1
print(output)