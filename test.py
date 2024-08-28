from Neural_Network import NeuralNetwork
from Train_Neural_Network import NeuralNetworkTrainer

nn = NeuralNetwork([1], [1], 2)
tnn = NeuralNetworkTrainer(nn, [([0], [0, 0]), ([1], [1, 1])])
tnn.trainSet()