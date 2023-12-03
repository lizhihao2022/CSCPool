import torch.nn as nn
import torch.nn.functional as F


class BaseGraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        node_hidden_dim,
        edge_hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.5,
        pooling=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.output_dim = output_dim

        self.node_encoder = nn.Linear(node_dim, node_hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, edge_hidden_dim)
        self.fc_out = nn.Linear(node_hidden_dim, output_dim)

        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = F.relu(self.node_encoder(x))
        edge_weight = F.relu(self.edge_encoder(edge_weight))

        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling is not None:
            x = self.pooling(x, **kwargs)

        out = self.fc_out(x)

        return out
    
    def skip_connection(self, x, edge_index, edge_weight=None, **kwargs):
        x_skip = x  # Save the input features for skip connection
        x = F.relu(self.node_encoder(x))
        edge_weight = F.relu(self.edge_encoder(edge_weight))
        
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.pooling is not None:
            x = self.pooling(x, **kwargs)
        
        # Add skip connection
        x = x + x_skip
        
        out = self.fc_out(x)
        
        return out
