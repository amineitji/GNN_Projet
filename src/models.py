import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# --- Modèle 1: GCN de base ---
# Un modèle simple pour servir de référence (baseline)
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super().__init__()
        # Encodeur: 2 couches GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Décodeurs: Simples couches linéaires
        # Décodeur pour la population (régression -> 1 valeur)
        self.decoder_pop = nn.Linear(out_channels, 1)
        # Décodeur pour le pays (classification -> N valeurs/classes)
        self.decoder_country = nn.Linear(out_channels, num_classes)
        
    def forward(self, x, edge_index):
        # Propagation à travers l'encodeur
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        z = self.conv2(x, edge_index) # 'z' est l'embedding final du nœud
        
        # Prédiction avec les décodeurs
        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)
        
        # Retourne les prédictions ET l'embedding (pour la visualisation t-SNE)
        return pop_pred, country_pred, z


# --- Modèle 2: GCN plus profond ---
# Une version améliorée du GCN, plus profonde et avec des décodeurs plus complexes
# pour la tâche de détection d'anomalies.
class AnomalyDetectorGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super().__init__()
        # Encodeur: N couches GCN
        self.convs = nn.ModuleList([GCNConv(in_channels if i==0 else hidden_channels, hidden_channels) 
                                    for i in range(num_layers)])
        
        # Décodeurs: Petits réseaux de neurones (MLP) au lieu d'une seule couche
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
        # Propagation à travers l'encodeur
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        # 'x' est maintenant l'embedding final 'z'
        z = x 
        
        # Prédiction avec les décodeurs
        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)
        
        return pop_pred, country_pred, z


# --- Modèle 3: GAT amélioré (Votre proposition finale) ---
# Utilise des couches GAT (Graph Attention Networks) qui peuvent
# apprendre à donner plus ou moins d'importance à certains voisins.
# Ajoute aussi de la BatchNorm pour stabiliser l'entraînement.
class ImprovedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes, num_heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # --- Encodeur GAT ---
        # Première couche: (features -> hidden) avec multi-têtes
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Couches intermédiaires
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Couche finale: (hidden*heads -> hidden) avec une seule tête (concat=False)
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # --- Décodeurs MLP (similaires au modèle 2 mais avec BatchNorm) ---
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
        
    def forward(self, x, edge_index):
        # Propagation à travers l'encodeur GAT
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = bn(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.elu(x) # ELU est souvent utilisé avec GAT
                x = F.dropout(x, p=0.3, training=self.training)
        z = x # 'z' est l'embedding final
        
        # Prédiction avec les décodeurs
        pop_pred = self.decoder_pop(z)
        country_pred = self.decoder_country(z)
        
        return pop_pred, country_pred, z