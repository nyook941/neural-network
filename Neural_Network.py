from typing import List
from Layer import Layer

class NeuralNetwork:
    def __init__(self, inputLayer: List[float], hiddenLayers: List[int], outputLayerNeurons: int) -> None:
        # Add input layer
        self.layers = [Layer(len(inputLayer), inputLayer, True)]

        # Add hidden layers
        for neuronAmount in hiddenLayers:
            self.layers.append(Layer(neuronAmount, self.layers[-1].getActivationList()))

        # Add output layers
        self.layers.append(Layer(outputLayerNeurons, self.layers[-1].getActivationList()))

    def calculateCost(self, expectedOutputs: List[float]):
        outputLayer = self.layers[-1].getActivationList()
        if len(expectedOutputs) != len(outputLayer):
            raise ValueError(f"Expected array of length {len(outputLayer)} for parameter 'expectedOutputs' but got length {len(expectedOutputs)} instead")
        
        sum = 0
        for i in range(len(expectedOutputs)):
            sum += (outputLayer[i] - expectedOutputs[i]) ** 2

        return sum

        

    def __str__(self) -> str:
        s = f"\033[34mInput Layer:\033[0m\n\t{self.layers[0]}"
        for i in range(1, len(self.layers)-1):
            s += f"\033[34mHidden Layer {i}:\033[0m\n\t{self.layers[i]}"
        return s + f"\033[34mOutput Layer:\033[0m\n\t{self.layers[-1]}"
    
nn = NeuralNetwork([1, 1, 1], [1], 1)
print(nn)
print(nn.calculateCost([1]))