from typing import List
from Layer import Layer
from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, inputLayer: List[float], hiddenLayers: List[int], outputLayerNeurons: int) -> None:
        # Add input layer
        self.layers = [Layer(len(inputLayer), inputLayer, True)]

        # Add hidden layers
        for neuronAmount in hiddenLayers:
            self.layers.append(Layer(neuronAmount, self.layers[-1].getActivationList()))

        # Add output layers
        self.layers.append(Layer(outputLayerNeurons, self.layers[-1].getActivationList()))

    def forwardPass(self, inputs: List[float], layerIndex: int = 1) -> List[float]:
        self.layers[layerIndex-1].setActivations(inputs)
        if layerIndex == len(self.layers):
            return inputs
        layer = self.layers[layerIndex]
        activationList = []
        for currentLayerIndex in range(len(layer.neurons)):
            sum = 0
            for previousLayerIndex in range(len(inputs)):
                sum += layer.weights[currentLayerIndex][previousLayerIndex] * inputs[previousLayerIndex]
            activationList.append(Neuron.sigmoid(sum))
        return self.forwardPass(activationList, layerIndex+1)
    
    def setWeights(self, newWeights, learningRate=0.1):
        for layerIndex in range(len(newWeights)):
            layerWeights = self.layers[layerIndex+1].weights
            for currentLayerIndex in range(len(layerWeights)):
                for prevLayerIndex in range(len(layerWeights[currentLayerIndex])):
                    weightGradient = newWeights[layerIndex][currentLayerIndex][prevLayerIndex]
                    layerWeights[currentLayerIndex][prevLayerIndex] -= learningRate * weightGradient

    def setBiases(self, newBiases, learningRate=0.1):
        for layerIndex in range(len(newBiases)):
            layerBiases = self.layers[layerIndex+1].getBiasList()
            for currentLayerIndex in range(len(layerBiases)):
                biasGradient = newBiases[layerIndex][currentLayerIndex]
                layerBiases[currentLayerIndex] -= learningRate * biasGradient

    def __str__(self) -> str:
        s = f"\033[34mInput Layer:\033[0m\n\t{self.layers[0]}"
        for i in range(1, len(self.layers)-1):
            s += f"\033[34mHidden Layer {i}:\033[0m\n\t{self.layers[i]}"
        return s + f"\033[34mOutput Layer:\033[0m\n\t{self.layers[-1]}"