import numpy as np

class Layer:
    def __init__(self, neuronAmount: int, previousActivations = None) -> None:
        self.weights = np.random.uniform(0, 0.3, size=(self.neuronAmount, previousActivations.size)) if previousActivations else None
        self.biases = np.zeros(neuronAmount, 1) if previousActivations else None
        self.activations = self.calculateActivations() if previousActivations else np.ones(neuronAmount, 1)
        self.neuronAmount = neuronAmount

    def calculateActivations(self, previousActivations):
        z = self.weights @ previousActivations + self.biases
        return self.sigmoid(z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))