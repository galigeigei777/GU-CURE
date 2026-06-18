import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv



class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=64, num_layers=2, dropout=0.6, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        self.convs = torch.nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(in_channels, dim, heads=heads, dropout=dropout))

        # 中间层（如有）
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(dim * heads, dim, heads=heads, dropout=dropout))

        # 最后一层
        self.convs.append(GATConv(dim * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, return_all_emb=False):
        """
        如果 return_all_emb=True，则返回 (logits, [每层embedding])
        否则仅返回 logits。
        """
        all_embs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            all_embs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层得到 logits
        logits = self.convs[-1](x, edge_index)
        all_embs.append(logits)

        if return_all_emb:
            return logits, all_embs
        else:
            return logits,all_embs[-1]





# class GATNet(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, dim=256, num_layers=2, dropout=0.6):
#         super().__init__()
#
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GATConv(in_channels, dim))
#         self.convs.append(GATConv(dim, out_channels))
#
#     def forward(self, x, edge_index):
#         # x = F.dropout(x, p=self.dropout, training=self.training)
#         for i in range(self.num_layers - 1):
#             x = F.relu(self.convs[i](x, edge_index))
#             # x = F.dropout(x, p=self.dropout, training=self.training)
#
#         x = self.convs[-1](x, edge_index)
#
#         return x