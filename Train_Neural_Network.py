from Neural_Network import NeuralNetwork
from Neuron import Neuron
from typing import List

class TrainNeuralNetwork:
    def __init__(self, nn: NeuralNetwork, trainingSet) -> None:
        self.nn = nn
        self.trainingSet = trainingSet
        self.sampleIndex = 0

    def train(self):
        pass
        #  TODO

    def backpropagate(self, layerIndex: int):
        layer = self.nn.layers[layerIndex]
        previousActivationList = self.nn.layers[layerIndex-1].getActivationList()
        weightNudges = []
        biasNudges = []
        previousActivationNudges = []
        for previousActivation in previousActivationList:
            weightSum = 0
            biasSum = 0
            previousActivationSum = 0
            for i in range(len(layer.weights[0])):
                for j in range(len(layer.weights)):
                    bias = layer.neurons[j].bias
                    weight = layer.weights[i][j]
                    z = weight * previousActivation + bias
                    dActivation = Neuron.dSigmoid(z)
                    dCost = 2 * (Neuron.sigmoid(z) - self.trainingSet[self.sampleIndex][1])
                    weightSum += dActivation * dCost * previousActivation
                    biasSum += dCost * dActivation
                    previousActivationSum += weight * dCost * dActivation
            weightNudges.append(weightSum)
            biasNudges.append(biasSum)
            previousActivationNudges.append(previousActivationSum)
        print(weightNudges)
        print(biasNudges)
        print(previousActivationNudges)