import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module, BatchNorm1d

import numpy


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        self.stdv = numpy.sqrt(1-beta**2) / math.sqrt(in_features*1.0)
        self.beta = beta

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0.0, 1.0)  # .uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.normal_(0.0, 1.0)  # .uniform_(-alpha, alpha)

    def forward(self, input):
        if self.bias is not None:
            return F.linear(input, self.weight) * self.stdv + self.bias * self.beta
        else:
            return F.linear(input, self.weight) * self.stdv


class LinearNet(Module):
    def __init__(self, widths, non_lin=torch.relu, bias=True, beta=0.1, batch_norm=False):
        super(LinearNet, self).__init__()
        self.widths = widths
        self.depth = len(self.widths)-1
        self.non_lin = non_lin
        self.beta = beta
        self.batch_norm = batch_norm

        self.pre_alpha = [None for i in range(self.depth)]
        self.alpha = [None for i in range(self.depth)]

        self.linears = []
        for i in range(self.depth):
            lin = Linear(widths[i], widths[i+1], bias, beta)
            self.add_module('lin'+str(i).zfill(2), lin)
            self.linears += [lin]

        if self.batch_norm:
            self.bns = []
            for i in range(1, self.depth):
                # , track_running_stats=False)
                bn = BatchNorm1d(widths[i], affine=False, eps=0.1)
                self.add_module('bn'+str(i).zfill(2), bn)
                self.bns += [bn]

    def reset_parameters(self):
        for l in self.linears:
            l.reset_parameters()

    def forward(self, x):
        self.alpha[0] = x
        for i in range(self.depth-1):
            self.pre_alpha[i+1] = self.linears[i](self.alpha[i])

            self.alpha[i+1] = self.non_lin(self.pre_alpha[i+1])
            if self.batch_norm:
                self.alpha[i+1] = self.bns[i](self.alpha[i+1])

        return self.linears[self.depth-1](self.alpha[self.depth-1])
