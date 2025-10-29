"""
⭐ Module de Feature Engineering pour Enrichir les Features de Connectivité
Ajoute des features basées sur la structure du réseau pour mieux prédire la population
"""

import torch
import numpy as np


def compute_connectivity_features(data):
    """
    Ajoute des features de connectivité enrichies au data.x

    Features ajoutées:
    1. in_degree: Nombre de connexions entrantes
    2. out_degree: Nombre de connexions sortantes
    3. weighted_degree: Degré pondéré par la population des voisins
    4. international_degree: Nombre de connexions vers d'autres pays
    5. hub_score: Score composite indiquant si c'est un hub majeur

    Args:
        data: PyG Data object avec edge_index, x, population, country_labels

    Returns:
        data: Data object avec features enrichies (data.x augmenté)
    """

    print("\n🔧 Calcul des features de connectivité enrichies...")

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # ═══════════════════════════════════════════════════════════
    # 1. DEGRÉ ENTRANT ET SORTANT
    # ═══════════════════════════════════════════════════════════

    in_degree = torch.zeros(num_nodes, device=data.x.device)
    out_degree = torch.zeros(num_nodes, device=data.x.device)

    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        out_degree[src] += 1
        in_degree[dst] += 1

    print(f"   ├─ Degré moyen: {out_degree.mean().item():.2f}")
    print(f"   ├─ Max degré: {out_degree.max().item():.0f}")

    # ═══════════════════════════════════════════════════════════
    # 2. DEGRÉ PONDÉRÉ PAR POPULATION DES VOISINS
    # ═══════════════════════════════════════════════════════════
    # Intuition: Si mes voisins sont importants, je suis probablement important

    weighted_degree = torch.zeros(num_nodes, device=data.x.device)

    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        # Ajouter la population du voisin
        weighted_degree[src] += data.population[dst]

    # Normaliser avec log pour éviter les valeurs trop grandes
    weighted_degree = torch.log1p(weighted_degree)

    print(f"   ├─ Degré pondéré moyen: {weighted_degree.mean().item():.2f}")

    # ═══════════════════════════════════════════════════════════
    # 3. CONNEXIONS INTERNATIONALES
    # ═══════════════════════════════════════════════════════════
    # Intuition: Les hubs internationaux sont généralement de grandes villes

    international_degree = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        # Trouver tous les voisins de ce nœud
        neighbors = edge_index[1][edge_index[0] == node]

        for neighbor in neighbors:
            # Si pays différent, compter
            if data.country_labels[node] != data.country_labels[neighbor.item()]:
                international_degree[node] += 1

    # Pourcentage de connexions internationales
    international_ratio = international_degree / (out_degree + 1e-6)

    print(f"   ├─ Connexions internationales moyennes: {international_degree.mean().item():.2f}")

    # ═══════════════════════════════════════════════════════════
    # 4. SCORE DE HUB
    # ═══════════════════════════════════════════════════════════
    # Score composite qui combine plusieurs indicateurs

    hub_scores = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]

        if len(neighbors) == 0:
            continue

        # A. Degré normalisé (0-1)
        degree_score = out_degree[node] / out_degree.max()

        # B. Diversité géographique (pays différents)
        neighbor_countries = [data.country_labels[n.item()] for n in neighbors]
        unique_countries = len(set(neighbor_countries))
        diversity_score = unique_countries / max(len(neighbor_countries), 1)

        # C. Distance moyenne aux voisins (hubs = longue distance)
        distances = []
        for neighbor in neighbors:
            dist = torch.sqrt(
                (data.x[node, 0] - data.x[neighbor, 0]) ** 2 +
                (data.x[node, 1] - data.x[neighbor, 1]) ** 2
            )
            distances.append(dist.item())
        avg_distance = sum(distances) / len(distances) if distances else 0
        distance_score = min(avg_distance / 0.5, 1.0)  # Normaliser sur [0, 1]

        # Score combiné
        hub_scores[node] = (
                0.5 * degree_score +  # Degré est le plus important
                0.3 * diversity_score +  # Diversité internationale
                0.2 * distance_score  # Connexions longue distance
        )

    print(f"   ├─ Score hub moyen: {hub_scores.mean().item():.3f}")
    print(f"   └─ Top 5 hubs scores: {hub_scores.topk(5).values.tolist()}")

    # ═══════════════════════════════════════════════════════════
    # 5. DENSITÉ LOCALE
    # ═══════════════════════════════════════════════════════════
    # Nombre de connexions entre mes voisins (coefficient de clustering)

    local_density = torch.zeros(num_nodes, device=data.x.device)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node].tolist()

        if len(neighbors) < 2:
            continue

        # Compter les connexions entre voisins
        connections_between_neighbors = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                # Vérifier si n1 et n2 sont connectés
                is_connected = ((edge_index[0] == n1) & (edge_index[1] == n2)).any() or \
                               ((edge_index[0] == n2) & (edge_index[1] == n1)).any()
                if is_connected:
                    connections_between_neighbors += 1

        # Coefficient de clustering
        max_possible = len(neighbors) * (len(neighbors) - 1) / 2
        local_density[node] = connections_between_neighbors / max_possible if max_possible > 0 else 0

    print(f"   └─ Densité locale moyenne: {local_density.mean().item():.3f}")

    # ═══════════════════════════════════════════════════════════
    # ASSEMBLAGE FINAL
    # ═══════════════════════════════════════════════════════════

    # Normaliser les features pour éviter des échelles trop différentes
    in_degree_norm = torch.log1p(in_degree)
    out_degree_norm = torch.log1p(out_degree)

    # Stack toutes les nouvelles features
    new_features = torch.stack([
        in_degree_norm,  # Feature 4 (ancien: 0-3 = lon, lat, log_pop, degree)
        out_degree_norm,  # Feature 5
        weighted_degree,  # Feature 6
        international_degree,  # Feature 7
        international_ratio,  # Feature 8
        hub_scores,  # Feature 9
        local_density  # Feature 10
    ], dim=1)

    # Concaténer avec les features existantes
    original_dim = data.x.size(1)
    data.x = torch.cat([data.x, new_features], dim=1)

    print(f"\n✅ Features enrichies ajoutées!")
    print(f"   ├─ Dimension originale: {original_dim}")
    print(f"   ├─ Nouvelles features: 7")
    print(f"   └─ Dimension finale: {data.x.size(1)}")

    return data


def print_feature_statistics(data):
    """
    Affiche des statistiques sur les features pour vérification
    """
    print("\n📊 Statistiques des Features:")
    print("=" * 60)

    feature_names = [
        "Longitude", "Latitude", "Log Population", "Degree (original)",
        "In-Degree (log)", "Out-Degree (log)", "Weighted Degree",
        "International Degree", "International Ratio", "Hub Score", "Local Density"
    ]

    for i, name in enumerate(feature_names[:data.x.size(1)]):
        values = data.x[:, i]
        print(f"{name:20} | Mean: {values.mean():.3f} | Std: {values.std():.3f} | "
              f"Min: {values.min():.3f} | Max: {values.max():.3f}")


def find_top_hubs(data, k=10):
    """
    Identifie les top hubs selon le hub_score
    """
    if data.x.size(1) < 10:  # Si features enrichies pas encore ajoutées
        print("⚠️  Features de connectivité pas encore calculées")
        return

    hub_scores = data.x[:, 9]  # Feature 9 = hub_score
    top_k_indices = torch.topk(hub_scores, k).indices

    print(f"\n🏆 Top {k} Hubs Détectés:")
    print("=" * 60)

    for rank, idx in enumerate(top_k_indices, 1):
        city = data.city_names[idx.item()]
        country = data.country_names[idx.item()]
        population = data.population[idx.item()].item()
        hub_score = hub_scores[idx].item()
        degree = data.x[idx, 3].item()  # Degree original

        print(f"#{rank:2}. {city:20} ({country:15}) | "
              f"Pop: {population:>10,.0f} | Hub Score: {hub_score:.3f} | Degree: {degree:.0f}")


