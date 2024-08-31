import numpy as np

class Layer:
    def __init__(self, neuronAmount: int, previousActivations=None) -> None:
        self.neuronAmount = neuronAmount
        if previousActivations is not None:
            self.weights = np.random.uniform(0, 0.3, size=(neuronAmount, previousActivations.size))
            self.biases = np.zeros((neuronAmount, 1))
            self.activations = self.calculateActivations(previousActivations)
        else:
            self.weights = None
            self.biases = None
            self.activations = np.ones((neuronAmount, 1))

    def calculateActivations(self, previousActivations):
        z = self.weights @ previousActivations + self.biases
        return self.sigmoid(z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __str__(self):
        green = '\033[92m'
        reset = '\033[0m'
        return (f"{green}neuronAmount={reset}{self.neuronAmount},\n"
                f"{green}weights={reset}\n{self.weights if self.weights is not None else 'None'},\n"
                f"{green}biases={reset}\n{self.biases if self.biases is not None else 'None'},\n"
                f"{green}activations={reset}\n{self.activations if self.activations is not None else 'None'})")