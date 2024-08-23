import math

class Neuron:
    bias = 0
    activation = 0
    def __init__(self, activation) -> None:
        self.bias = 0
        self.activation = activation

    @staticmethod
    def calculateActivation(previousActivations, weights, bias) -> float:
        z = 0
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                z += weights[i][j] * previousActivations[j]
        z += bias
        return Neuron.sigmoid(z)
    
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 +  math.exp(-x))
    
    @staticmethod
    def dSigmoid(x: float) -> float:
        sigmoid = Neuron.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    def __str__(self) -> str:
        return f'{{ activation: {self.activation}, bias: {self.bias} }}'
