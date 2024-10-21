import numpy as np
from Layer import Layer
from typing import List

class Network:
    def __init__(self, inputLayerNeurons, hiddenLayers: List[int], outputLayerNeurons) -> None:
        self.input = Layer(inputLayerNeurons)
        hidden: List[Layer] = []
        for i in range(len(hiddenLayers)):
            if i == 0:
                hidden.append(Layer(hiddenLayers[i], self.input.activations))
            hidden.append(Layer(hiddenLayers[i], hidden[i-1].activations))
        self.output = Layer(outputLayerNeurons, hidden[-1].activations)
        self.layers: List[Layer] = np.array([self.input] + hidden + [self.output], dtype=object)

    def __str__(self):
        blue = '\033[94m'
        reset = '\033[0m'
        layer_strings = []

        # Adding the input layer
        layer_strings.append(f"{blue}Input layer:\n{reset}{self.input}")

        # Adding hidden layers
        for i, layer in enumerate(self.layers[1:-1], start=1):
            layer_strings.append(f"{blue}Hidden layer {i}:\n{reset}{layer}")

        # Adding the output layer
        layer_strings.append(f"{blue}Output layer:\n{reset}{self.output}")

        return "\n".join(layer_strings)
    
    def feedForward(self, inputMatrix):
        if inputMatrix.shape != self.input.activations.shape:
            raise Exception("Provided input shape does not match input shape of nn")
        self.input.activations = inputMatrix
        for i in range(1, len(self.layers)):
            previousActivations = self.layers[i-1].activations
            self.layers[i].activations = self.layers[i].calculateActivations(previousActivations)
        return self.output.activations
