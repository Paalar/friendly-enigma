from torch import nn
import torch
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, input_length, output_length):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_length, 64)
        self.hidden_1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.2)
        self.hidden_2 = nn.Linear(128, output_length)

    def forward(self, data_input):
        input_layer = self.input_layer(data_input)
        activation_input = F.relu(input_layer)
        normalized_input = activation_input

        hidden_1 = self.hidden_1(normalized_input)
        activation_1 = F.relu(hidden_1)
        normalized_1 = activation_1
        dropped = self.dropout(normalized_1)

        last_hidden = self.hidden_2(dropped)
        activation_last = F.relu(last_hidden)
        normalized_last = activation_last
        return normalized_last
