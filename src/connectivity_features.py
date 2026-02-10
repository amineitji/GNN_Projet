"""
Module de feature engineering pour enrichir les features de connectivité.
Ajoute des descripteurs basés sur la structure du réseau afin d'améliorer
la prédiction de la population.
"""

import torch
import numpy as np


def compute_connectivity_features(data):
    """
    Ajoute des features de connectivité enrichies à data.x.

    Features ajoutées :
    - in_degree : nombre de connexions entrantes
    - out_degree : nombre de connexions sortantes
    - weighted_degree : degré pondéré par la population des voisins
    - international_degree : nombre de connexions vers d'autres pays
    - hub_score : score composite indiquant le rôle de hub
    """

    print("\nCalcul des features de connectivité...")

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. Degré entrant et sortant
    in_degree = torch.zeros(num_nodes, device=data.x.device)
    out_degree = torch.zeros(num_nodes, device=data.x.device)

    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        out_degree[src] += 1
        in_degree[dst] += 1

    print(f"   Degré moyen : {out_degree.mean().item():.2f}")
    print(f"   Degré max   : {out_degree.max().item():.0f}")

    # 2. Degré pondéré par la population des voisins
    weighted_degree = torch.zeros(num_nodes, device=data.x.device)

    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        weighted_degree[src] += data.population[dst]

    weighted_degree = torch.log1p(weighted_degree)
    print(f"   Degré pondéré moyen : {weighted_degree.mean().item():.2f}")

    # 3. Connexions internationales
    international_degree = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        for neighbor in neighbors:
            if data.country_labels[node] != data.country_labels[neighbor.item()]:
                international_degree[node] += 1

    international_ratio = international_degree / (out_degree + 1e-6)
    print(f"   Connexions internationales moyennes : {international_degree.mean().item():.2f}")

    # 4. Score de hub (score composite)
    hub_scores = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) == 0:
            continue

        degree_score = out_degree[node] / out_degree.max()

        neighbor_countries = [data.country_labels[n.item()] for n in neighbors]
        diversity_score = len(set(neighbor_countries)) / max(len(neighbor_countries), 1)

        distances = []
        for neighbor in neighbors:
            dist = torch.sqrt(
                (data.x[node, 0] - data.x[neighbor, 0]) ** 2 +
                (data.x[node, 1] - data.x[neighbor, 1]) ** 2
            )
            distances.append(dist.item())

        avg_distance = sum(distances) / len(distances)
        distance_score = min(avg_distance / 0.5, 1.0)

        hub_scores[node] = (
            0.5 * degree_score +
            0.3 * diversity_score +
            0.2 * distance_score
        )

    print(f"   Score hub moyen : {hub_scores.mean().item():.3f}")

    # 5. Densité locale (coefficient de clustering)
    local_density = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node].tolist()
        if len(neighbors) < 2:
            continue

        connections = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                is_connected = (
                    ((edge_index[0] == n1) & (edge_index[1] == n2)).any() or
                    ((edge_index[0] == n2) & (edge_index[1] == n1)).any()
                )
                if is_connected:
                    connections += 1

        max_possible = len(neighbors) * (len(neighbors) - 1) / 2
        local_density[node] = connections / max_possible if max_possible > 0 else 0

    print(f"   Densité locale moyenne : {local_density.mean().item():.3f}")

    # Assemblage final
    in_degree_norm = torch.log1p(in_degree)
    out_degree_norm = torch.log1p(out_degree)

    new_features = torch.stack([
        in_degree_norm,
        out_degree_norm,
        weighted_degree,
        international_degree,
        international_ratio,
        hub_scores,
        local_density
    ], dim=1)

    original_dim = data.x.size(1)
    data.x = torch.cat([data.x, new_features], dim=1)

    print("Features de connectivité ajoutées")
    print(f"   Dimension originale : {original_dim}")
    print(f"   Dimension finale    : {data.x.size(1)}")

    return data


def print_feature_statistics(data):
    """
    Affiche des statistiques descriptives sur les features.
    """
    print("\nStatistiques des features")
    print("=" * 60)

    feature_names = [
        "Longitude", "Latitude", "Log Population", "Degree (original)",
        "In-Degree (log)", "Out-Degree (log)", "Weighted Degree",
        "International Degree", "International Ratio", "Hub Score", "Local Density"
    ]

    for i, name in enumerate(feature_names[:data.x.size(1)]):
        values = data.x[:, i]
        print(
            f"{name:20} | Mean: {values.mean():.3f} | Std: {values.std():.3f} | "
            f"Min: {values.min():.3f} | Max: {values.max():.3f}"
        )


def find_top_hubs(data, k=10):
    """
    Identifie les principaux hubs selon le hub_score.
    """
    if data.x.size(1) < 10:
        print("Features de connectivité non calculées")
        return

    hub_scores = data.x[:, 9]
    top_k_indices = torch.topk(hub_scores, k).indices

    print(f"\nTop {k} hubs détectés")
    print("=" * 60)

    for rank, idx in enumerate(top_k_indices, 1):
        city = data.city_names[idx.item()]
        country = data.country_names[idx.item()]
        population = data.population[idx.item()].item()
        hub_score = hub_scores[idx].item()
        degree = data.x[idx, 3].item()

        print(
            f"#{rank:2}. {city:20} ({country:15}) | "
            f"Pop: {population:>10,.0f} | Hub Score: {hub_score:.3f} | Degree: {degree:.0f}"
        )
