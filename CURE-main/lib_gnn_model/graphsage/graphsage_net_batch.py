import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=64, dropout=0.5):
        super().__init__()

        self.dropout = dropout

        # 第一层：生成 node_embeddings
        self.conv1 = SAGEConv(in_channels, dim)

        # 第二层：生成 logits
        self.conv2 = SAGEConv(dim, out_channels)

    def forward(self, x, edge_index):
        # 第一层
        node_embeddings = self.conv1(x, edge_index)
        node_embeddings = F.relu(node_embeddings)

        # dropout
        node_embeddings = F.dropout(
            node_embeddings, p=self.dropout, training=self.training
        )

        # 第二层
        logits = self.conv2(node_embeddings, edge_index)

        return logits, node_embeddings