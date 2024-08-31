from Neural_Network import NeuralNetwork
from Train_Neural_Network import NeuralNetworkTrainer

nn = NeuralNetwork([0, 0, 0], [20], 1)
tnn = NeuralNetworkTrainer(nn, [([0, 0, 0], [0]), ([0, 0, 1], [0]), ([1, 0, 0], [0]), ([1, 0, 1], [1]), ([0, 0.5, 0], [0]), ([0, 0.5, 1], [1]), ([1, 0.5, 0], [1]), ([1, 0.5, 1], [1]), ([0, 1, 0], [0]), ([0, 1, 0], [1]), ([0, 1, 0], [1]), ([0, 1, 0], [0])]
)
tnn.trainSet(10000, 100)
print(nn.forwardPass([0, 0, 0]))
print(nn.forwardPass([1, 1, 1]))