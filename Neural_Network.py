from Layer import Layer

class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayers=[16, 16], outputLayerNeurons=10) -> None:
        # Add input layer
        self.layers = [Layer(len(inputLayer), inputLayer, True)]

        # Add hidden layers
        for neuronAmount in hiddenLayers:
            self.layers.append(Layer(neuronAmount, self.layers[-1].previousActivations()))

        # Add output layers
        self.layers.append(Layer(outputLayerNeurons, self.layers[-1].previousActivations()))

    def __str__(self) -> str:
        return '\n '.join(str(layer) for layer in self.layers)
    
nn = NeuralNetwork([1, 1, 1], [1], 1)
print(nn)