import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class BaselineGCN(nn.Module):
    """
    Modèle GCN simple servant de référence.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super().__init__()

        # Encodeur GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Décodeurs
        self.decoder_pop = nn.Linear(out_channels, 1)
        self.decoder_country = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        z = self.conv2(x, edge_index)

        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)

        return pop_pred, country_pred, z


class AnomalyDetectorGCN(nn.Module):
    """
    GCN plus profond avec décodeurs MLP.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super().__init__()

        self.convs = nn.ModuleList([
            GCNConv(
                in_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(num_layers)
        ])

        self.decoder_pop = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

        self.decoder_country = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)

        z = x

        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)

        return pop_pred, country_pred, z


class ImprovedGAT(nn.Module):
    """
    Modèle basé sur des couches GAT avec normalisation et décodeurs MLP.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, num_heads=4):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        self.convs.append(
            GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=1,
                concat=False
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.decoder_pop = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

        self.decoder_country = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = bn(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)

        z = x

        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)

        return pop_pred, country_pred, z
