import numpy as np
from Layer import Layer
from typing import List

class Network:
    def __init__(self, inputLayerNeurons, hiddenLayers: List[int], outputLayerNeurons) -> None:
        self.input = Layer(inputLayerNeurons)
        hidden = [Layer(_) for _ in hiddenLayers]
        self.output = Layer(outputLayerNeurons)
        self.layers = np.array([self.input] + hidden + [self.output], dType=object)