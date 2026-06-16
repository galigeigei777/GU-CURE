import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP


class APPNPNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 dropout=0.5, K=10, alpha=0.1):
        super(APPNPNet, self).__init__()
        self.dropout = float(dropout)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.prop = APPNP(K=int(K), alpha=float(alpha))
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if hasattr(self.prop, "reset_parameters"):
            self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        feature = self.prop(h, edge_index)
        output = self.lin2(feature)

        return output, feature