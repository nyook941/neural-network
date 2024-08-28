from Neural_Network import NeuralNetwork
from Neuron import Neuron
from typing import List

class NeuralNetworkTrainer:
    def __init__(self, nn: NeuralNetwork, trainingSet: List[tuple]) -> None:
        self.nn = nn
        self.trainingSet = trainingSet
        self.biasGradients = []
        self.weightGradients = []

    def trainSet(self):
        for sampleIndex in range(len(self.trainingSet)):
            sample = self.trainingSet[sampleIndex]
            self.nn.forwardPass(sample[0])
            print(self.nn)
            costMatrix = self.getCostMatrixDerivative(sampleIndex)
            self.backpropagate(len(self.nn.layers)-1, costMatrix)
            self.nn.setWeights(self.weightGradients)
        # print(self.weightGradients)
        # print(self.biasGradients)

    def backpropagate(self, layerIndex: int, activationGradients: List[float]):
        if layerIndex == 0:
            return
        
        biases = self.nn.layers[layerIndex].getBiasList()
        weights = self.nn.layers[layerIndex].weights
        layerWeightGradients = [[0 for i in range(len(weights[0]))] for _ in range(len(weights))]
        layerBiasGradients = [0 for _ in range(len(biases))]
        layerActivationGradients = []
        prevLayerActivations = self.nn.layers[layerIndex-1].getActivationList()

        for prevLayerIndex in range(len(prevLayerActivations)):
            previousActivation = prevLayerActivations[prevLayerIndex]
            activationGradient = 0
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
                layerBiasGradients[currentLayerIndex] = biasGradient

                # calculate weight gradients
                weightGradient = dz_dw * biasGradient
                layerWeightGradients[currentLayerIndex][prevLayerIndex] = weightGradient

                # calculate activation gradients
                activationGradient += weight * biasGradient
            layerActivationGradients.append(activationGradient)
            self.weightGradients = layerWeightGradients + self.weightGradients
            self.biasGradients = layerBiasGradients + self.biasGradients
        
        self.backpropagate(layerIndex-1, layerActivationGradients)

    def getCostMatrix(self, sampleIndex):
        output = self.nn.layers[-1].getActivationList()
        costMatrix = []
        for i in range(len(output)):
            expectedValue = self.trainingSet[sampleIndex][1][i]
            costMatrix.append((output[i] - expectedValue) ** 2)
        return costMatrix
    
    def getCostMatrixDerivative(self, sampleIndex):
        output = self.nn.layers[-1].getActivationList()
        costMatrix = []
        for i in range(len(output)):
            expectedValue = self.trainingSet[sampleIndex][1][i]
            costMatrix.append((output[i] - expectedValue) * 2)
        return costMatrix

    def getMSE():
        pass
