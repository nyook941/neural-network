from Neural_Network import NeuralNetwork
from Neuron import Neuron
from typing import List

class NeuralNetworkTrainer:
    def __init__(self, nn: NeuralNetwork, trainingSet: List[tuple], learningRate=0.1) -> None:
        self.nn = nn
        self.trainingSet = trainingSet
        self.learningRate = learningRate
        self.biasGradients = []
        self.weightGradients = []
        self.initializeWeightGradientSize()
        self.initializeBiasGadientSize()

    def initializeWeightGradientSize(self):
        self.weightGradients = []
        for layerIndex in range(1, len(self.nn.layers)):
            self.weightGradients.append([])
            layer = self.nn.layers[layerIndex]
            for currentNeuronIndex in range(len(layer.weights)):
                self.weightGradients[layerIndex-1].append([])
                for prevNeruonIndex in range(len(layer.weights[0])):
                    self.weightGradients[layerIndex-1][currentNeuronIndex].append(0)

    def addWeightGradients(self, layerIndex, layerWeightGradients):
        layer = self.weightGradients[layerIndex-1]
        for currentNeuronIndex in range(len(layerWeightGradients)):
            for previousNeuronIndex in range(len(layerWeightGradients[currentNeuronIndex])):
                layer[currentNeuronIndex][previousNeuronIndex] += layerWeightGradients[currentNeuronIndex][previousNeuronIndex]

    def averageWeightGradients(self):
        for layerIndex in range(len(self.weightGradients)):
            for currentNeuronIndex in range(len(self.weightGradients[layerIndex])):
                for prevNeuronIndex in range(len(self.weightGradients[layerIndex][currentNeuronIndex])):
                    self.weightGradients[layerIndex][currentNeuronIndex][prevNeuronIndex] /= len(self.trainingSet)

    def initializeBiasGadientSize(self):
        self.biasGradients = []
        for currentLayer in range(1, len(self.nn.layers)):
            self.biasGradients.append([])
            for neuronIndex in range(len(self.nn.layers[currentLayer].neurons)):
                self.biasGradients[currentLayer-1].append(0)

    def addBiasLayerGradients(self, layerIndex, layerBiasGradients):
        layer = self.biasGradients[layerIndex-1]
        for currentNeuronIndex in range(len(layerBiasGradients)):
            layer[currentNeuronIndex] += layerBiasGradients[currentNeuronIndex]

    def averageBiasGradients(self):
        for layerIndex in range(len(self.biasGradients)):
            for curretnNeuronIndex in range(len(self.biasGradients[layerIndex])):
                self.biasGradients[layerIndex][curretnNeuronIndex] /= len(self.trainingSet)

    def trainSet(self, epochs, epochPrintInterval=10):
        for epoch in range(epochs):
            for sampleIndex in range(len(self.trainingSet)):
                sample = self.trainingSet[sampleIndex]
                self.nn.forwardPass(sample[0])
                costMatrix = self.getCostMatrixDerivative(sampleIndex)
                self.backpropagate(len(self.nn.layers)-1, costMatrix)
            # Average and set the weight gradients
            self.averageWeightGradients()
            self.nn.setWeights(self.weightGradients, self.learningRate)

            # Average and set the biases
            self.averageBiasGradients()
            self.nn.setBiases(self.biasGradients, self.learningRate)

            if epoch % epochPrintInterval == 0:
                print(f"\033[35mEpoch {epoch}\33[0m")
                print(f"MSE: {self.getMSE()}\n")

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
                layerWeightGradients[currentLayerIndex][prevLayerIndex] += weightGradient

                # calculate activation gradients
                activationGradient += weight * biasGradient
            layerActivationGradients.append(activationGradient)
        self.addWeightGradients(layerIndex, layerWeightGradients)
        self.addBiasLayerGradients(layerIndex, layerBiasGradients)
        
        self.backpropagate(layerIndex-1, layerActivationGradients)

    def getCostMatrix(self, sampleIndex) -> List[float]:
        self.nn.forwardPass(self.trainingSet[sampleIndex][0])
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
    
    def getMSE(self):
        costs = []
        for sampleIndex in range(len(self.trainingSet)):
            costMatrix = self.getCostMatrix(sampleIndex)
            costs.append(sum(costMatrix))
        return sum(costs) / len(self.trainingSet)