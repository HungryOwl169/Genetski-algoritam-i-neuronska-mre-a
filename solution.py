import sys
import numpy as np
import random

data = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1)
inputs = data[:, :-1]
inputs = inputs.T
expected_outputs = data[:, -1]

test_data = np.genfromtxt(sys.argv[4], delimiter=',', skip_header=1)
test_inputs = test_data[:, :-1]
test_inputs = test_inputs.T
test_expected_outputs = test_data[:, -1]

#np.random.seed(42)

class NeuralNetwork:
    def __init__(self, layer_sizes, weights=None, biases=None):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.fitness = None
        self.error = None
        if (weights == None):
            self.weights = [0.01 * np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        else: 
            self.weights = weights
        if (biases == None):
            self.biases = [0.01 * np.random.randn(y, 1) for y in layer_sizes[1:]]
        else:
            self.biases = biases
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, expected_outputs):
        a = x 
        i = 0
        for b, w in zip(self.biases, self.weights):
            net = np.dot(w, a) + b

            if i == len(self.weights) - 1:
                a = net
            else:
                a = self.sigmoid(net)

            i += 1
        err = np.mean((a-expected_outputs)**2)
        self.error = err
        self.fitness = 1/err
        return a
        
#nn = NeuralNetwork((len(inputs), 5, 1))
#print(nn.forward(inputs))
#print(nn.fitness)

def evaluate(P):
    for nn in P:
        nn.forward(inputs, expected_outputs)
    P_evaluated = sorted(P, key=lambda nn: nn.fitness, reverse=True)
    return P_evaluated

def select_parents(P):
    total_fitness = sum(network.fitness for network in P)
    parents = []

    for _ in range(2):
        pick = random.uniform(0, total_fitness)
        current = 0
        for network in P:
            current += network.fitness
            if current > pick:
                parents.append(network)
                break
    
    return parents

def cross(R1, R2):
    R3 = NeuralNetwork(layer_sizes)
    R3.weights = [(w1 + w2) / 2 for w1, w2 in zip(R1.weights, R2.weights)]
    R3.biases = [(b1 + b2) / 2 for b1, b2 in zip(R1.biases, R2.biases)]
    return R3

def mutate(network, K, p):
    mutated_weights = network.weights.copy()
    mutated_biases = network.biases.copy()

    for i in range(len(mutated_weights)):
        for j in range(len(mutated_weights[i])):
            if random.random() < p:
                gaussian_noise = np.random.normal(0, K)
                mutated_weights[i][j] += gaussian_noise

    for i in range(len(mutated_biases)):
        for j in range(len(mutated_biases[i])):
            if random.random() < p:
                gaussian_noise = np.random.normal(0, K)
                mutated_biases[i][j] += gaussian_noise

    D = NeuralNetwork(layer_sizes, mutated_weights, mutated_biases)
    return D

vel_pop = int(sys.argv[8])
input_dimension = len(inputs)
#layer_sizes = (input_dimension, 5, 1)
if (sys.argv[6] == '5s'):
    layer_sizes = (input_dimension, 5, 1)
elif (sys.argv[6] == '20s'):
    layer_sizes = (input_dimension, 20, 1)
elif (sys.argv[6] == '5s5s'):
    layer_sizes = (input_dimension, 5, 5, 1)
P = []
elitism = int(sys.argv[10])
p = float(sys.argv[12])
K = float(sys.argv[14])
iter = int(sys.argv[16])

for _ in range(vel_pop):
    nn = NeuralNetwork(layer_sizes)
    P.append(nn)

P = evaluate(P)

curr_iteration = 1
for _ in range(iter):
    P1 = P[:elitism]
    while(len(P1) < vel_pop):
        [R1, R2] = select_parents(P)
        D1 = cross(R1, R2)
        D2 = cross(R1, R2)
        D1 = mutate(D1, K, p)
        D2 = mutate(D2, K, p)
        P1.append(D1)
        P1.append(D2)
    P = P1.copy()
    P = evaluate(P)
    if (curr_iteration % 2000 == 0):
        print(f'[Train error @{curr_iteration}]: {round(P[0].error, 6)}')
    curr_iteration += 1
    
P[0].forward(test_inputs, test_expected_outputs)
print(f'[Test error]: {round(P[0].error, 6)}')
