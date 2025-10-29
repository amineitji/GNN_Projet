import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ImprovedGATWithAnomalyGuidedLearning(nn.Module):
    """
    ⭐ GAT avec apprentissage guidé par détection d'anomalies de liens
    VERSION CORRIGÉE - Fix du problème de dimensions
    """

    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, num_heads=4):
        super().__init__()

        # Couches GNN
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Last layer
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Decoders pour nœuds
        self.decoder_pop = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.BatchNorm1d(hidden_channels), nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
        self.decoder_country = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.BatchNorm1d(hidden_channels), nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )

        # ⭐ CORRIGÉ : Link Anomaly Detector avec le bon nombre de features
        # Features: src_emb + dst_emb + geo_dist + pop_ratio + degree_diff + emb_similarity + direction_flag
        # = hidden_channels * 2 + 5
        self.link_anomaly_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 5, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # Passer par les couches GNN
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = bn(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)

        embeddings = x

        # Prédictions sur les nœuds
        pop_pred = self.decoder_pop(embeddings)
        country_pred = self.decoder_country(embeddings)

        return pop_pred, country_pred, embeddings

    def compute_link_features(self, embeddings, node_features, src_idx, dst_idx):
        """
        ⭐ CORRIGÉ : Version uniforme qui crée 5 features
        """
        src_emb = embeddings[src_idx]
        dst_emb = embeddings[dst_idx]

        src_feat = node_features[src_idx]
        dst_feat = node_features[dst_idx]

        # Distance géographique
        geo_dist = torch.sqrt(
            (src_feat[:, 0] - dst_feat[:, 0]) ** 2 +
            (src_feat[:, 1] - dst_feat[:, 1]) ** 2
        ).view(-1, 1)

        # Population
        pop_src = src_feat[:, 2].view(-1, 1)
        pop_dst = dst_feat[:, 2].view(-1, 1)
        pop_ratio = ((pop_dst + 1) / (pop_src + 1)).view(-1, 1)

        # Degré
        degree_diff = torch.abs(src_feat[:, 3] - dst_feat[:, 3]).view(-1, 1)

        # Similarité embeddings
        emb_similarity = F.cosine_similarity(src_emb, dst_emb, dim=-1).view(-1, 1)

        # Direction petite → grande
        direction_flag = (pop_src < pop_dst).float().view(-1, 1)

        # ⭐ IMPORTANT : 5 features (pas 4 !)
        link_features = torch.cat([
            src_emb, dst_emb,              # 2 * hidden_channels
            geo_dist,                       # 1
            pop_ratio,                      # 1
            degree_diff,                    # 1
            emb_similarity,                 # 1
            direction_flag                  # 1
        ], dim=-1)

        return link_features

    def compute_link_features_vectorized(self, embeddings, node_features, src_idx, dst_idx):
        """
        ⭐ CORRIGÉ : Version vectorisée qui crée AUSSI 5 features (cohérent avec compute_link_features)
        """
        # 1. Embeddings
        src_emb = embeddings[src_idx]
        dst_emb = embeddings[dst_idx]

        # 2. Node features (lon, lat, log_pop, degree)
        src_feat = node_features[src_idx]
        dst_feat = node_features[dst_idx]

        # 3. Calcul vectorisé des features du lien
        geo_dist = torch.sqrt(
            (src_feat[:, 0] - dst_feat[:, 0]) ** 2 +
            (src_feat[:, 1] - dst_feat[:, 1]) ** 2
        ).unsqueeze(1)

        # ⭐ CORRIGÉ : Utiliser pop_ratio au lieu de pop_diff
        pop_src = src_feat[:, 2].unsqueeze(1)
        pop_dst = dst_feat[:, 2].unsqueeze(1)
        pop_ratio = ((pop_dst + 1) / (pop_src + 1))

        degree_diff = torch.abs(src_feat[:, 3] - dst_feat[:, 3]).unsqueeze(1)
        emb_similarity = F.cosine_similarity(src_emb, dst_emb, dim=-1).unsqueeze(1)

        # ⭐ NOUVEAU : Direction flag (petite → grande)
        direction_flag = (pop_src < pop_dst).float()

        # 4. Concaténation finale (5 features + embeddings)
        link_features = torch.cat([
            src_emb, dst_emb,              # 2 * hidden_channels
            geo_dist,                       # 1
            pop_ratio,                      # 1
            degree_diff,                    # 1
            emb_similarity,                 # 1
            direction_flag                  # 1
        ], dim=-1)

        return link_features

    def detect_link_anomalies(self, embeddings, node_features, src_idx, dst_idx):
        """
        Détecte les anomalies de liens
        Score proche de 1 = lien suspect
        Score proche de 0 = lien normal
        """
        link_features = self.compute_link_features(embeddings, node_features, src_idx, dst_idx)
        anomaly_score = self.link_anomaly_detector(link_features)
        return anomaly_score.squeeze()