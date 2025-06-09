import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def get_flat_weights(self) -> list:
        return torch.cat([param.data.view(-1).cpu() for param in self.parameters()]).tolist()

    def set_flat_weights(self, flat_weights: list):
        idx = 0
        with torch.no_grad():
            for param in self.parameters():
                num_el = param.numel()
                param.copy_(Tensor(flat_weights[idx: idx + num_el]).view_as(param))
                idx += num_el


class MnistNet(Net):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class KWSNet(Net):
    def __init__(self):
        super(KWSNet, self).__init__()
        # self.fc1 = nn.Linear(1053, 25)
        self.fc1 = nn.Linear(1053, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# class KWSNet(Net):
#     def __init__(self):
#         super(KWSNet, self).__init__()
#         self.c1 = nn.Conv2d(1, 32, 3)
#         self.c2 = nn.Conv2d(32, 64, 3)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(15169, 4)
#
#
#     def forward(self, x):
#         x = self.c1(x)
#         x = F.relu(x)
#         x = self.c2(x)
#         x = F.relu(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         output = F.log_softmax(x, dim=1)
#         return output
