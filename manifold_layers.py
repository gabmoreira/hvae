import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from poincareball import *
from geoopt import ManifoldParameter


class GeodesicLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(GeodesicLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, x):
        x = x.unsqueeze(-2).expand(*x.shape[:-(len(x.shape) - 2)], self.out_features, self.in_features)
        res = self.manifold.normdist2plane(x, self.bias, self.weight, signed=True, norm=self.weight_norm)
        return res


class ExpZero(nn.Module):
    def __init__(self, manifold):
        super(ExpZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.expmap0(input)


class LogZero(nn.Module):
    def __init__(self, manifold):
        super(LogZero, self).__init__()
        self.manifold = manifold

    def forward(self, input):
        return self.manifold.logmap0(input)

