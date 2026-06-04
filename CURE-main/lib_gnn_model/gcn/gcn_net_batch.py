import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, dim))
        self.convs.append(GCNConv(dim, out_channels))

    def forward(self, x, edge_index):
        # 第一层 GCN + ReLU
        node_embeddings = F.relu(self.convs[0](x, edge_index))

        # 新增：dropout（训练时生效）
        node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)

        # 第二层（输出 logits）
        logits = self.convs[1](node_embeddings, edge_index)

        return logits, node_embeddings