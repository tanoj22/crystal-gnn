import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool


class GaussianExpansion(nn.Module):
    """Expand scalar distances into a vector of Gaussian basis functions."""

    def __init__(self, start=0.0, stop=6.0, num_gaussians=40):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CrystalGNN(nn.Module):
    """CGCNN-style model for predicting a scalar property from a crystal graph."""

    def __init__(self, node_dim=64, edge_dim=40, n_conv=3, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(100, node_dim)
        self.gaussian = GaussianExpansion(start=0.0, stop=6.0, num_gaussians=edge_dim)
        self.convs = nn.ModuleList(
            [CGConv(node_dim, dim=edge_dim) for _ in range(n_conv)]
        )
        self.head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
        )

    def forward(self, data):
        # embedding
        x = self.embedding(data.z)

        # gaussian
        edge_attr = self.gaussian(data.edge_attr)

        # message passing
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)

        # pooling
        x = global_mean_pool(x, data.batch)

        # head
        return self.head(x).view(-1)
