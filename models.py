import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out, dropout):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(H2, D_out),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class LinearModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out)
        )

    def forward(self, x):
        pred = self.layers(x)
        return pred