import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from IPython.core.debugger import set_trace


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        
        # This is a hacky implementation of batch sparse matrix multiplication
        # This is slow, and it's on FB's list to implement as a (faster) native operation
        # See https://github.com/pytorch/pytorch/issues/10043
        support = support.unbind()
        output = torch.stack([torch.spmm(adj, support_part) for support_part in support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def batch_mm(matrix, vector_batch):
    batch_size = vector_batch.shape[0]
    # Stack the vector batch into columns. (b, n, 1) -> (n, b)
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, b) -> (b, m, 1)
    return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1, 1)
