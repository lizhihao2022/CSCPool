from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import ELU
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.dense import dense_mincut_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import Sequential

from .base import BaseGraphNeuralNetwork


def dense_csc_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
    coarsen_node: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    coarsen_node = coarsen_node.unsqueeze(0) if coarsen_node.dim() == 2 else coarsen_node

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    
    # Coarsen node regularization.
    _, coarsen_node_idx = torch.topk(coarsen_node, k=50, dim=1)
    coarsen_node_idx = coarsen_node_idx.flatten()
    p = torch.zeros((x.shape[1], len(coarsen_node_idx)), device=x.device)
    diag = torch.eye(adj.shape[1], device=x.device)
    loop_adj = adj + diag
    loop_adj = F.normalize(loop_adj, dim=1)
    p = loop_adj[:, :, coarsen_node_idx]
    p = F.normalize(p, dim=1)
    
    coarsen_adj = torch.matmul(torch.matmul(p.transpose(1, 2), adj), p)
    coarsen_loss = -(_rank3_trace(coarsen_adj) / _rank3_trace(loop_adj))
    coarsen_loss = torch.mean(coarsen_loss)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss, coarsen_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out


class CSCGCN(BaseGraphNeuralNetwork):
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
        
        out = torch.softmax(s, dim=-1)

        return out

    def contrastive_loss(self, x, s, tau=0.07):
        # Compute cosine similarity
        sim = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
        
        # neg_idx = torch.randint(0, x.shape[0], (x.shape[0], 10))
        # neg_x = x[neg_idx]
        # sim = torch.einsum('ij, kj -> ik', x, neg_x)
            
        # Compute cluster assignments
        s = torch.softmax(s, dim=-1)
        _, cluster_assignments = s.max(dim=-1)

        # Compute mask for positive and negative samples
        pos_mask = cluster_assignments.unsqueeze(1) == cluster_assignments.unsqueeze(0)
        # neg_mask = ~pos_mask

        # Compute contrastive loss
        pos_sim = sim.masked_select(pos_mask)
        # neg_sim = sim.masked_select(neg_mask)
        loss = -torch.log(torch.exp(pos_sim / tau).sum() / torch.exp(sim / tau).sum())
        
        return loss
    
    def loss(self, x, edge_index, edge_weight=None, **kwargs):
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        
        adj = to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = self.pooling(x, adj, s)

        contrastive_loss = self.contrastive_loss(x, s)
        
        loss = mc_loss + o_loss + contrastive_loss

        return loss, mc_loss, o_loss, contrastive_loss
    
    def get_cluster_assignments(self, x, edge_index, edge_weight=None, **kwargs):
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        
        s = torch.softmax(s, dim=-1)
        _, cluster_assignments = s.max(dim=-1)
        
        return cluster_assignments
