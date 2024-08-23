from typing import List
from Neuron import Neuron
import random

class Layer:
    def __init__(self, neuronAmount, previousActivations, inputLayer=False) -> None:
        self.neurons: Neuron = []
        self.weights = (
            None
            if inputLayer
            else [
                [random.uniform(0.0, 1.0) for _ in range(len(previousActivations))]
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

    def getActivationList(self) -> List[float]:
        activations = []
        for neuron in self.neurons:
            activations.append(neuron.activation)
        return activations
    
    def __str__(self) -> str:
        s = "\033[32mNeurons:\033[0m\n\t"
        for neuron in self.neurons:
            s += str(neuron) + ",\n\t"
        return s + f"\033[32mWeights:\033[0m\n\t{self.weights}\n"