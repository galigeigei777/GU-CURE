import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


class SIGNNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 num_layers=2, dropout=0.5):
        super(SIGNNet, self).__init__()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.lins = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels, bias=False)
            for _ in range(self.num_layers + 1)
        ])

        self.lin1 = nn.Linear((self.num_layers + 1) * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    @staticmethod
    def _propagate_once(x, edge_index):
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index[0], edge_index[1]

        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = x.new_zeros(x.size())
        out.index_add_(0, col, x[row] * norm.view(-1, 1))
        return out

    def _build_sign_features(self, x, edge_index):
        xs = [x]
        x_k = x
        for _ in range(self.num_layers):
            x_k = self._propagate_once(x_k, edge_index)
            xs.append(x_k)
        return xs

    def forward(self, x, edge_index):
        xs = self._build_sign_features(x, edge_index)

        outs = []
        for x_i, lin in zip(xs, self.lins):
            out = F.relu(lin(x_i))
            out = F.dropout(out, p=self.dropout, training=self.training)
            outs.append(out)

        h = torch.cat(outs, dim=-1)
        feature = self.lin1(h)
        output = self.lin2(feature)

        return output, feature