import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_input(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(800,1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 400)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
    
class all_neumf_input(nn.Module):
#https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    def __init__(self):
        super(all_neumf_input, self).__init__()
        self.mlp = MLP_input()
        self.neumf = nn.Linear(800, 1)
    
    def forward(self, x, y):
        x = self.mlp(x)
        x = torch.cat((y, x), 1)
        x = self.neumf(x)
        x = F.relu(x)
        return x        
    