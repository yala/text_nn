import torch
import torch.nn as nn
import pdb

class Empty(torch.nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

    def forward(self, x):
        return x
