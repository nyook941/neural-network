from Neural_Network import NeuralNetwork
from Train_Neural_Network import TrainNeuralNetwork

nn = NeuralNetwork([1], [1], 1)
print(nn)
tnn = TrainNeuralNetwork(nn, [([1], 1)])
tnn.backpropagate(-1)