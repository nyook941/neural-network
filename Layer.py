from typing import List
from Neuron import Neuron
import random

class Layer:
    def __init__(self, neuronAmount: int, previousActivations: List[float], inputLayer=False) -> None:
        self.neurons: Neuron = []
        self.weights = (
            None
            if inputLayer
            else [
                [random.uniform(0.0, 1.0) for _ in range(len(previousActivations))]
                for _ in range(neuronAmount)
            ]
        )
        for neuronIndex in range(neuronAmount):
            self.neurons.append(
                Neuron(
                    previousActivations[neuronIndex]
                    if inputLayer
                    else Neuron.calculateActivation(previousActivations, neuronIndex, self.weights, 0)
                )
            )

    def getActivationList(self) -> List[float]:
        activations = []
        for neuron in self.neurons:
            activations.append(neuron.activation)
        return activations
    
    def getBiasList(self) -> List[float]:
        biases = []
        for neuron in self.neurons:
            biases.append(neuron.bias)
        return biases
    
    def __str__(self) -> str:
        s = "\033[32mNeurons:\033[0m\n\t"
        for neuron in self.neurons:
            s += str(neuron) + ",\n\t"
        if self.weights:
            s += f"\033[32mWeights:\033[0m\n\t"
            for neuronWeights in self.weights:
                s += str(neuronWeights) + "\n\t"
        return s + "\n"