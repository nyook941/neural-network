from typing import List
from Neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, neuronAmount, previousActivations, inputLayer=False) -> None:
        self.neurons = []
        self.weights = (
            None
            if inputLayer
            else [
                [np.random.randn() for _ in range(len(previousActivations))]
                for _ in range(neuronAmount)
            ]
        )
        for i in range(neuronAmount):
            self.neurons.append(
                Neuron(
                    previousActivations[i] 
                    if inputLayer 
                    else Neuron.calculateActivation(previousActivations, self.weights, 0)
                )
            )

    def previousActivations(self) -> List[float]:
        previousActivations = []
        for neuron in self.neurons:
            previousActivations.append(neuron.activation)
        return previousActivations

    def __str__(self) -> str:
        return '{\n Neurons: [\n  ' + '\n  '.join(str(neuron) for neuron in self.neurons) +'\n ]' + f'\n Weights:\n  {str(self.weights)}\n}}'