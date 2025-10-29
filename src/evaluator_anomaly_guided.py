import torch
import torch.nn.functional as F
import numpy as np


class AnomalyEvaluatorWithLinkGuidance:
    """
    ⭐ Evaluator amélioré qui détecte et analyse les anomalies de liens
    Focus sur : petites villes → grandes villes de pays différents
    """

    def __init__(self, model, data, device='cpu', weights=(0.4, 0.3, 0.3)):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.pop_weight, self.country_weight, self.link_weight = weights

    @torch.no_grad()
    def compute_anomaly_scores(self):
        """Calcule tous les scores d'anomalie"""
        self.model.eval()
        pop_pred, country_pred, embeddings = self.model(self.data.x, self.data.edge_index)

        # 1. Anomalies de population
        log_pop_true = torch.log1p(self.data.population)
        log_pop_pred = pop_pred.squeeze()
        pop_error = torch.abs(log_pop_pred - log_pop_true).cpu().numpy()

        # 2. Anomalies de pays
        country_true = torch.LongTensor(self.data.country_labels).to(self.device)
        country_probs = torch.softmax(country_pred, dim=1)
        true_probs = country_probs[range(len(country_true)), country_true]
        country_score = (1 - true_probs).cpu().numpy()

        # 3. ⭐ Anomalies de liens (basées sur les voisins suspects)
        link_scores = self.compute_link_based_node_anomalies(embeddings)

        # 4. Score combiné
        combined = (self.pop_weight * pop_error +
                    self.country_weight * country_score +
                    self.link_weight * link_scores)

        return {
            'population_error': pop_error,
            'country_score': country_score,
            'link_score': link_scores,
            'combined_score': combined,
            'embeddings': embeddings.cpu().numpy()
        }

    def compute_link_based_node_anomalies(self, embeddings):
        """
        ⭐ AMÉLIORÉ : Calcule le score d'anomalie d'un nœud basé sur ses liens suspects

        Un nœud est suspect si :
        1. Il a beaucoup de connexions vers de grandes villes d'autres pays
        2. Il a des connexions à très grande distance
        3. Ses voisins sont très incohérents (populations variées, pays différents)
        """
        node_anomaly_scores = torch.zeros(self.data.num_nodes, device=self.device)

        for node_idx in range(self.data.num_nodes):
            # Trouver les voisins sortants (où node_idx est source)
            neighbors = self.data.edge_index[1][self.data.edge_index[0] == node_idx]

            if len(neighbors) == 0:
                # Nœud isolé => suspect
                node_anomaly_scores[node_idx] = 0.8
                continue

            # Calculer les scores d'anomalie pour tous les liens
            link_anomaly_scores = self.model.detect_link_anomalies(
                embeddings,
                self.data.x,
                torch.full((len(neighbors),), node_idx, device=self.device),
                neighbors
            )

            # ⭐ Métriques avancées
            mean_score = link_anomaly_scores.mean()
            max_score = link_anomaly_scores.max()
            num_suspicious = (link_anomaly_scores > 0.7).sum().float() / len(neighbors)

            # Score final : combinaison pondérée
            node_anomaly_scores[node_idx] = (
                    0.4 * mean_score +  # Score moyen
                    0.3 * max_score +  # Pire cas
                    0.3 * num_suspicious  # Proportion de liens suspects
            )

        return node_anomaly_scores.cpu().numpy()

    def get_top_anomalies(self, scores, k=10):
        """Retourne les k nœuds avec les scores d'anomalie les plus élevés"""
        top_idx = np.argsort(scores)[-k:][::-1]
        return [{
            'index': idx,
            'city': self.data.city_names[idx],
            'country': self.data.country_names[idx],
            'population': self.data.population[idx].item(),
            'score': scores[idx]
        } for idx in top_idx]

    def get_anomalous_links(self, embeddings, threshold=0.7, top_k=50):
        """
        ⭐ AMÉLIORÉ : Retourne les liens les plus suspects SANS DOUBLONS
        Si pas assez de liens > threshold, prend les top meilleurs
        """
        edge_index = self.data.edge_index

        # Calculer le score d'anomalie de tous les liens
        link_anomaly_scores = self.model.detect_link_anomalies(
            embeddings,
            self.data.x,
            edge_index[0],
            edge_index[1]
        )

        # ⭐ NOUVEAU : Si pas assez de liens au-dessus du threshold,
        # prendre directement les top scores
        anomalous_mask = link_anomaly_scores > threshold
        num_above_threshold = anomalous_mask.sum().item()

        if num_above_threshold < top_k * 2:  # On a besoin de 2x car doublons
            # Prendre les top au lieu de filtrer par threshold
            print(f"   ⚠️  Seulement {num_above_threshold} liens > {threshold}")
            print(f"   ➜  On prend les {top_k * 2} meilleurs liens à la place")
            sorted_all = torch.argsort(link_anomaly_scores, descending=True)
            sorted_indices = sorted_all[:top_k * 2].cpu().numpy()
        else:
            anomalous_indices = torch.where(anomalous_mask)[0].cpu().numpy()
            sorted_indices = anomalous_indices[
                np.argsort(-link_anomaly_scores[anomalous_indices].detach().cpu().numpy())
            ]

        # ⭐ Éviter les doublons
        seen_pairs = set()
        results = []

        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()

            # Créer une clé normalisée
            pair_key = tuple(sorted([src, dst]))

            # Si déjà vu, skip
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            # Informations détaillées
            pop_src = self.data.population[src].item()
            pop_dst = self.data.population[dst].item()
            country_src = self.data.country_names[src]
            country_dst = self.data.country_names[dst]

            # Calculs
            pop_ratio = pop_dst / (pop_src + 1)
            geo_dist = torch.sqrt(
                (self.data.x[src, 0] - self.data.x[dst, 0]) ** 2 +
                (self.data.x[src, 1] - self.data.x[dst, 1]) ** 2
            ).item()

            # Raison de la suspicion
            reasons = []
            if pop_ratio > 100:
                reasons.append(f"Ratio pop: {pop_ratio:.0f}x")
            if country_src != country_dst:
                reasons.append("Pays différents")
            if geo_dist > 0.4:
                reasons.append(f"Distance: {geo_dist:.2f}")

            results.append({
                'src_index': src,
                'src_city': self.data.city_names[src],
                'src_country': country_src,
                'src_population': pop_src,
                'dst_index': dst,
                'dst_city': self.data.city_names[dst],
                'dst_country': country_dst,
                'dst_population': pop_dst,
                'anomaly_score': link_anomaly_scores[idx].item(),
                'pop_ratio': pop_ratio,
                'geo_distance': geo_dist,
                'reasons': ', '.join(reasons)
            })

        print(f"   ✅ Trouvé {len(results)} paires uniques\n")
        return results

    def analyze_suspicious_patterns(self, embeddings, threshold=0.7):
        """
        ⭐ NOUVEAU : Analyse les patterns de liens suspects
        """
        anomalous_links = self.get_anomalous_links(embeddings, threshold=threshold, top_k=100)

        if len(anomalous_links) == 0:
            return {
                'total_suspicious': 0,
                'patterns': {}
            }

        # Statistiques
        small_to_large = sum(1 for link in anomalous_links if link['pop_ratio'] > 50)
        cross_country = sum(1 for link in anomalous_links if link['src_country'] != link['dst_country'])
        long_distance = sum(1 for link in anomalous_links if link['geo_distance'] > 0.4)

        # Pays sources les plus suspects
        src_countries = {}
        for link in anomalous_links:
            country = link['src_country']
            src_countries[country] = src_countries.get(country, 0) + 1

        return {
            'total_suspicious': len(anomalous_links),
            'small_to_large_cities': small_to_large,
            'cross_country_links': cross_country,
            'long_distance_links': long_distance,
            'top_source_countries': sorted(src_countries.items(), key=lambda x: x[1], reverse=True)[:5],
            'avg_pop_ratio': np.mean([link['pop_ratio'] for link in anomalous_links]),
            'avg_geo_distance': np.mean([link['geo_distance'] for link in anomalous_links])
        }

    def print_anomalous_links_report(self, embeddings, threshold=0.7, top_k=20):
        """
        ⭐ NOUVEAU : Génère un rapport détaillé des liens suspects
        """
        print("\n" + "=" * 80)
        print("🔍 RAPPORT D'ANALYSE DES LIENS SUSPECTS")
        print("=" * 80)

        # Analyse globale
        patterns = self.analyze_suspicious_patterns(embeddings, threshold=threshold)

        print(f"\n📊 Statistiques globales:")
        print(f"   ├─ Liens suspects détectés: {patterns['total_suspicious']}")
        print(f"   ├─ Petite → Grande ville: {patterns['small_to_large_cities']}")
        print(f"   ├─ Entre pays différents: {patterns['cross_country_links']}")
        print(f"   ├─ Longue distance: {patterns['long_distance_links']}")
        print(f"   ├─ Ratio population moyen: {patterns['avg_pop_ratio']:.1f}x")
        print(f"   └─ Distance géographique moyenne: {patterns['avg_geo_distance']:.3f}")

        print(f"\n🌍 Top 5 pays sources de liens suspects:")
        for country, count in patterns['top_source_countries']:
            print(f"   ├─ {country}: {count} liens")

        # Top liens suspects
        anomalous_links = self.get_anomalous_links(embeddings, threshold=threshold, top_k=top_k)

        print(f"\n🔝 Top {min(top_k, len(anomalous_links))} liens les plus suspects:")
        print("=" * 80)

        for i, link in enumerate(anomalous_links[:top_k], 1):
            print(f"\n#{i}. Score: {link['anomaly_score']:.4f}")
            print(f"   Source: {link['src_city']}, {link['src_country']} ({link['src_population']:,.0f} hab.)")
            print(f"   Destin: {link['dst_city']}, {link['dst_country']} ({link['dst_population']:,.0f} hab.)")
            print(f"   Raisons: {link['reasons']}")

        print("\n" + "=" * 80)