import torch
import torch.nn as nn
import torch.nn.functional as F

#https://pytorch.org/functorch/stable/notebooks/ensembling.html
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(800,512)
        self.fc2 = nn.Linear(512, 400)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

#https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
class all_neumf(nn.Module):
    def __init__(self):
        super(all_neumf, self).__init__()
        self.mlp = MLP()
        self.neumf = nn.Linear(800, 1)
    
    def forward(self, x, y):
        x = self.mlp(x)
        x = torch.cat((y, x), 1)
        x = self.neumf(x)
        x = F.relu(x)
        return x        
    