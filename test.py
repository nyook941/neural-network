from Network import Network
import numpy as np

nn = Network(
    inputLayerNeurons=1,
    hiddenLayers=[10, 10],
    outputLayerNeurons=1
    )
print(nn)
print()
print(nn.feedForward(np.array([[0]])))