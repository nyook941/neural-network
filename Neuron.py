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
        return max(0, z)

    def __str__(self) -> str:
        return f'{{ activation: {self.activation}, bias: {self.bias} }}'
