import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv


class GINNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.5):
        super(GINNet, self).__init__()

        self.dropout = dropout

        # 第一层：用于生成 node_embeddings
        self.conv1 = GINConv(Linear(input_dim, hidden_dim, bias=False))

        # 第二层：用于生成 logits
        self.conv2 = GINConv(Linear(hidden_dim, out_dim, bias=False))

    def forward(self, x, edge_index):
        # 第一层
        node_embeddings = self.conv1(x, edge_index)
        node_embeddings = F.relu(node_embeddings)

        # dropout（只在训练时生效）
        node_embeddings = F.dropout(
            node_embeddings, p=self.dropout, training=self.training
        )

        # 第二层
        logits = self.conv2(node_embeddings, edge_index)

        return logits, node_embeddings