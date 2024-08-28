from Neural_Network import NeuralNetwork
from Train_Neural_Network import NeuralNetworkTrainer

nn = NeuralNetwork([1], [2], 2)
tnn = NeuralNetworkTrainer(nn, [([1], [1, 1])])
tnn.trainSet(1)