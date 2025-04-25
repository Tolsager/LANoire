import torch
from torch import nn

class BimodalGMU(torch.nn.Module):
    def __init__(self, fl_m1: int, fl_m2: int, hidden_size: int = 200):
        super().__init__()
        self.fc_z = nn.Linear(fl_m1+fl_m2, hidden_size)
        self.fc_h1 = nn.Linear(fl_m1, hidden_size)
        self.fc_h2 = nn.Linear(fl_m2, hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, m1, m2):
        z = self.sigmoid(self.fc_z(torch.cat([m1, m2], dim=1)))
        h1 = self.tanh(self.fc_h1(m1))
        h2 = self.tanh(self.fc_h2(m2))
        h = z * h1 + (1 - z) * h2
        return h


class TrimodalGMU(torch.nn.Module):
    def __init__(self, fl_m1: int, fl_m2: int, fl_m3, hidden_size: int = 200):
        super().__init__()
        self.fc_z1 = nn.Linear(fl_m1+fl_m2+fl_m3, hidden_size)
        self.fc_z2 = nn.Linear(fl_m1+fl_m2+fl_m3, hidden_size)
        self.fc_z3 = nn.Linear(fl_m1+fl_m2+fl_m3, hidden_size)
        self.fc_h1 = nn.Linear(fl_m1, hidden_size)
        self.fc_h2 = nn.Linear(fl_m2, hidden_size)
        self.fc_h3 = nn.Linear(fl_m3, hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, m1, m2, m3):
        cat = torch.cat([m1, m2, m3], dim=1)
        z1 = self.sigmoid(self.fc_z1(cat))
        z2 = self.sigmoid(self.fc_z2(cat))
        z3 = self.sigmoid(self.fc_z3(cat))
        h1 = self.tanh(self.fc_h1(m1))
        h2 = self.tanh(self.fc_h2(m2))
        h3 = self.tanh(self.fc_h3(m3))
        h = z1 * h1 + z2 * h2 + z3 * h3
        return h

        
