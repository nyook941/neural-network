from Neural_Network import NeuralNetwork
from Train_Neural_Network import TrainNeuralNetwork
from Neuron import Neuron

nn = NeuralNetwork([1], [1], 1)
print(nn)
tnn = TrainNeuralNetwork(nn, [([1], 1)])
# tnn.backpropagate(-1, 1)

# Initialize parameters
input = 1.0
w1 = 0.3  # initial weight
w2 = 0.3
b1 = 0.0    # initial bias
b2 = 0.0
learning_rate = 0.1  # learning rate
target = 1  # desired output when input is 1