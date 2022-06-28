from cogdl.layers import SGCLayer

from cogdl.models import BaseModel


class sgc(BaseModel):
    def __init__(self, in_feats, out_feats):
        super(sgc, self).__init__()
        self.nn = SGCLayer(in_feats, out_feats)
        self.cache = dict()

    def forward(self, graph):
        graph.sym_norm()

        x = self.nn(graph, graph.x)
        return x

    def predict(self, data):
        return self.forward(data)

