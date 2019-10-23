import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution
from IPython.core.debugger import set_trace
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nnodes, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 1)
        #self.linear = nn.Linear(nnodes, 1)
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(nnodes, int(nnodes/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(nnodes/2), 1),
        )
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = x.squeeze(2)
        x = self.linear(x)
        return x

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nnodes, dropout):
#         super(GCN, self).__init__()
#         self.dropout = dropout
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, 1)
#         self.linear = nn.Sequential(
#             nn.Dropout(self.dropout),
#             nn.Linear(nnodes, int(nnodes/2)),
#             nn.ReLU(),
#             nn.Dropout(self.dropout),
#             nn.Linear(int(nnodes/2), 1),
#         )
        

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         x = x.squeeze(2)
#         x = self.linear(x)
#         return x
    
class MLP(nn.Module):
    def __init__(self, nnodes, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear = nn.Sequential(
            nn.Linear(nnodes, int(nnodes/2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(nnodes/2), int(nnodes/4)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(nnodes/4), 1)
        )
        
    def forward(self, x):
        return self.linear(x)
    
class MultiDrugMLP(nn.Module):
    def __init__(self, ngenes, ndrugs, dropout):
        super(MultiDrugMLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ngenes, int(ngenes/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(ngenes/2), ndrugs),
        )
        
    def forward(self, x):
        return self.linear(x)
    
