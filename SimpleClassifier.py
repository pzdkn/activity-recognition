# Define Architecture
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(70, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = x.view(-1, 70)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model has %d trainable parameters" % total_params)

if __name__ == '__main__':
    snet = SimpleClassifier(10)
    snet.double()
    snet.print_num_params()
    x = np.random.random((1, 7, 10))
    x = torch.from_numpy(x)
    y = snet(x)
    print(y.shape)
    print(y)