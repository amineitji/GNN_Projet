import torch
import numpy as np


class AnomalyEvaluator:
    def __init__(self, model, data, device='cpu', pop_weight=0.6):
        """
        Initialise l'évaluateur d'anomalies.

        - pop_weight : pondération utilisée pour le calcul du score final
          (cohérente avec l'entraînement du modèle).
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.pop_weight = pop_weight

    @torch.no_grad()
    def compute_anomaly_scores(self):
        """
        Calcule les scores d'anomalie pour l'ensemble des nœuds du graphe.
        """
        self.model.eval()

        # Passage avant pour obtenir les prédictions et les embeddings
        pop_pred, country_pred, embeddings = self.model(
            self.data.x, self.data.edge_index
        )

        # Score d'anomalie basé sur l'erreur de population
        log_pop_true = torch.log1p(self.data.population)
        log_pop_pred = pop_pred.squeeze()
        pop_error = torch.abs(log_pop_pred - log_pop_true).cpu().numpy()

        # Score d'anomalie basé sur l'incertitude de classification du pays
        country_true = torch.LongTensor(self.data.country_labels).to(self.device)
        country_probs = torch.softmax(country_pred, dim=1)
        true_probs = country_probs[
            range(len(country_true)), country_true
        ]
        country_score = (1 - true_probs).cpu().numpy()

        # Combinaison des scores
        combined = (
                self.pop_weight * pop_error +
                (1 - self.pop_weight) * country_score
        )

        return {
            'population_error': pop_error,
            'country_score': country_score,
            'combined_score': combined,
            'embeddings': embeddings.cpu().numpy()
        }

    def get_top_anomalies(self, scores, k=10):
        """
        Trie les nœuds par score d'anomalie et retourne les k plus élevés.
        """
        top_idx = np.argsort(scores)[-k:][::-1]

        return [{
            'index': idx,
            'city': self.data.city_names[idx],
            'country': self.data.country_names[idx],
            'population': self.data.population[idx].item(),
            'score': scores[idx]
        } for idx in top_idx]
