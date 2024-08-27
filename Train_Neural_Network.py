from Neural_Network import NeuralNetwork
from Neuron import Neuron
from typing import List

class TrainNeuralNetwork:
    def __init__(self, nn: NeuralNetwork, trainingSet) -> None:
        self.nn = nn
        self.trainingSet = trainingSet
        self.sampleIndex = 0
        self.activationGradients = []
        self.biasGradients = []
        self.weightGradients = []
        self.cost = self.calculateCost()

    def trainSet(self, set):
        pass
        #  TODO

    def backpropagate(self, layerIndex: int, activationGradients: List[float]):
        biases = self.nn.layers[layerIndex].getBiases()
        weights = self.nn.layers[layerIndex].weights
        layerWeightGradient = weights.copy()
        layerBiasGraidient = biases.copy()
        layerActivationGradients = activationGradients.copy()
        prevLayerActivations = self.nn.layers[layerIndex-1].getActivationList()

        for prevLayerIndex in range(len(prevLayerActivations)):
            previousActivation = prevLayerActivations[prevLayerIndex]
            for currentLayerIndex in range(len(activationGradients)):
                # get values
                bias = biases[currentLayerIndex]
                weight = weights[currentLayerIndex][prevLayerIndex]
                z = previousActivation * weight + bias
                dz_dw = previousActivation
                da_dz = Neuron.dSigmoid(z)
                errorTerm = activationGradients[currentLayerIndex]

                # calculate bias gradients
                biasGradient = da_dz * errorTerm
                layerBiasGraidient[currentLayerIndex] = biasGradient

                # calculate weight gradients
                weightGradient = dz_dw * biasGradient
                layerWeightGradient[currentLayerIndex][prevLayerIndex] = weightGradient

                # calculate activation gradients
                activationGradient = weight * biasGradient
                layerActivationGradients[currentLayerIndex] = activationGradient

    def getCostMatrix(self):
        output = self.nn.layers[-1].getActivationList()
        costMatrix = []
        for i in range(len(output)):
            expectedValue = self.trainingSet[i][1]
            costMatrix.append((output[i] - expectedValue) ** 2)
        return costMatrix


    def getMSE():
        pass
