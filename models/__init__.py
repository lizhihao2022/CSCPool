from .graph_nerual_networks import MinCutGCN, CSCGCN, DiffGCN

ALL_MODELS = {
    "MinCutGCN": MinCutGCN,
    "AMGGCN": CSCGCN,
    "DiffGCN": DiffGCN,
}
