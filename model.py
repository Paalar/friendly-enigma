from torch import nn
import torch
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, input_length, output_length):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_length, 120)
        self.hidden_1 = nn.Linear(120, 84)
        self.hidden_2 = nn.Linear(84, output_length)

    def forward(self, data_input):
        input_layer = F.relu(self.input_layer(data_input))
        hidden_1 = F.relu(self.hidden_1(input_layer))
        last_hidden = F.relu(self.hidden_2(hidden_1))
        return last_hidden
