import torch
from torch.nn import ELU

from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.dense import dense_mincut_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import Sequential

from .base import BaseGraphNeuralNetwork


class MinCutGCN(BaseGraphNeuralNetwork):
    def __init__(self, node_dim, edge_dim, node_hidden_dim, 
                 edge_hidden_dim, output_dim, 
                 num_layers=2, dropout=0.5, pooling=dense_mincut_pool):
        super().__init__(node_dim, edge_dim, node_hidden_dim, edge_hidden_dim, output_dim, num_layers, dropout, pooling)
        
        mp = [
            (GraphConv(node_dim, node_hidden_dim), 'x, edge_index, edge_weight -> x'),
            ELU(inplace=True),
        ]
        
        for _ in range(num_layers - 1):
            mp.append((GraphConv(node_hidden_dim, node_hidden_dim), 'x, edge_index, edge_weight -> x'))
            mp.append(ELU(inplace=True))
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        
        self.mlp = torch.nn.Sequential()
        self.mlp.append(Linear(node_hidden_dim, output_dim))

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        
        adj = to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = self.pooling(x, adj, s)

        out = torch.softmax(s, dim=-1)

        return out, mc_loss, o_loss
