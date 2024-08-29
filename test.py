from Neural_Network import NeuralNetwork
from Train_Neural_Network import NeuralNetworkTrainer

nn = NeuralNetwork([1], [20], 3)
tnn = NeuralNetworkTrainer(nn, [([1], [1, 1, 0]), ([0], [0, 0, 1])])
tnn.trainSet(1000, 100)
print(nn.forwardPass([1]))
print(nn.forwardPass([0]))