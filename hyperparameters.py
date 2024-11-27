import torch.nn as nn

#! NOT USING THIS FOR NOW


class CNNHyperparameters:
    def __init__(self):
        self.learning_rate = 0.001
        self.conv_layers = [16, 32, 64]
        self.fc_layers = [1024, 400]
        self.activation = nn.ReLU()


class ResNetHyperparameters:
    def __init__(self):
        pass


class KNNHyperparameters:
    def __init__(self):
        pass


class SVMHyperparameters:
    def __init__(self):
        pass


class ViTHyperparameters:
    def __init__(self):
        pass
