import math
from typing import List

class Neuron:
    bias = 0
    activation = 0
    def __init__(self, activation) -> None:
        self.bias = 0
        self.activation = activation

    @staticmethod
    def calculateActivation(previousActivations: List[float], neuronIndex: int, weights: List[List[float]], bias: float) -> float:
        z = 0
        for prevNeuronIndex in range(len(previousActivations)):
            z += weights[neuronIndex][prevNeuronIndex] * previousActivations[prevNeuronIndex]
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
