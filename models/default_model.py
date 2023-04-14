import torch.nn as nn
import torch.nn.functional as F


class DEFAULT_MODEL(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.ff1 = nn.Linear(3, 120)
        self.ff2 = nn.Linear(120, 200)
        self.ff3 = nn.Linear(200, 7)

    def forward(self, x):
        output = F.relu(self.ff1(x))
        output = F.relu(self.ff2(output))
        return self.ff3(output)
