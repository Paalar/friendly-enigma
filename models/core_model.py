import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from config import config


class Net(nn.Module):
    def __init__(self, input_length, output_length, tune_config):
        super(Net, self).__init__()
        self.tune_config = tune_config
        config_layers = self.get_config("hidden_layers")
        config_layers.append(output_length)
        activations = [get_activation(name) for name in self.get_config("activations")]

        first_layer = nn.Linear(input_length, config_layers[0])
        linear_layers = [
            nn.Linear(config_layers[index], config_layers[index + 1])
            for index in range(len(config_layers) - 1)
        ]
        linear_layers.insert(0, first_layer)
        self.layers = [
            layer for pair in zip(linear_layers, activations) for layer in pair
        ]

    def forward(self, data_input):
        output = data_input
        for layer in self.layers:
            output = layer(output)
        return output


    def get_config(self, name):
        return (
            self.tune_config[name].sample()
            if not self.tune_config == None
            else config[name]
        )

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        return nn.ReLU()
