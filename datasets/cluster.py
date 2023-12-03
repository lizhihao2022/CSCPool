from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn2_conv import gcn_norm

AVAILABLE_DATASETS = ['Cora', 'CiteSeer', 'PubMed']


class ClusterDataset:
    def __init__(self, name, data_dir):
        if name not in AVAILABLE_DATASETS:
            raise NotImplementedError("Dataset {} not implemented".format(name))
        
        dataset = self.load_data(name, data_dir)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.data = self.preprocess(dataset[0])
    
    def load_data(self, name, data_dir):
        dataset = Planetoid(
            root=data_dir, 
            name=name, 
            transform=NormalizeFeatures(), 
            pre_transform=None,
        )
        return dataset
        
    def preprocess(self, data):
        self.num_features = self.num_features
        self.num_classes = self.num_classes
        data.edge_index, data.edge_weight = gcn_norm(data.edge_index, edge_weight=data.edge_weight, 
                                                     num_nodes=data.num_nodes, add_self_loops=False, 
                                                     dtype=data.x.dtype)
        
        return data
