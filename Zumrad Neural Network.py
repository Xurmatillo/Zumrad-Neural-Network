import torch
import torch.nn as nn
import torch.nn.functional as F

class Zumrad(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Zumrad, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
