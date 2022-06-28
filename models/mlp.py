import torch.nn as nn
from cogdl.layers import  MLP as MLPLayer
from cogdl.data import Graph

from cogdl.models import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        dropout=0.0,
        activation="relu",
        norm=None,
        act_first=False,
        bias=True,
    ):
        super(MLP, self).__init__()
        self.nn = MLPLayer(in_feats, out_feats, hidden_size, num_layers, dropout, activation, norm, act_first, bias)

    def forward(self, x):
        if isinstance(x, Graph):
            x = x.x
        return self.nn(x)

    def predict(self, data):
        return self.forward(data.x)
