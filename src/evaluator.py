import torch
import numpy as np


class AnomalyEvaluator:
    def __init__(self, model, data, device='cpu', pop_weight=0.6):
        """
        Initialise l'évaluateur.
        - pop_weight: Le 'alpha' utilisé pour calculer le score d'anomalie final.
          (similaire à l'alpha de l'entraînement)
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.pop_weight = pop_weight # Pondération pour le score de population
        
    @torch.no_grad()
    def compute_anomaly_scores(self):
        """
        Calcule les scores d'anomalie pour TOUS les nœuds du graphe.
        """
        self.model.eval()
        # Fait une passe forward pour obtenir les prédictions et les embeddings
        pop_pred, country_pred, embeddings = self.model(self.data.x, self.data.edge_index)
        
        # --- Score d'anomalie 1: Erreur de Population ---
        # Calcule l'erreur absolue entre la prédiction et la vérité
        log_pop_true = torch.log1p(self.data.population)
        log_pop_pred = pop_pred.squeeze()
        # Un nœud avec une grande 'pop_error' est anormal du point de vue de la population
        pop_error = torch.abs(log_pop_pred - log_pop_true).cpu().numpy()
        
        # --- Score d'anomalie 2: Erreur/Incertitude de Pays ---
        country_true = torch.LongTensor(self.data.country_labels).to(self.device)
        # Applique Softmax pour obtenir des probabilités (de 0 à 1)
        country_probs = torch.softmax(country_pred, dim=1)
        # Récupère la probabilité que le modèle a assignée à la VRAIE classe
        true_probs = country_probs[range(len(country_true)), country_true]
        # Le score est 1 - cette probabilité.
        # Si le modèle est très confiant (prob=0.99), le score est bas (0.01).
        # Si le modèle est très incertain (prob=0.1), le score est élevé (0.9).
        country_score = (1 - true_probs).cpu().numpy()
        
        # --- Score Combiné ---
        # Combine les deux scores d'erreur en un score d'anomalie final
        # C'est la métrique principale pour la détection d'anomalies.
        combined = (self.pop_weight * pop_error) + ((1 - self.pop_weight) * country_score)
        
        return {
            'population_error': pop_error,
            'country_score': country_score,
            'combined_score': combined,
            'embeddings': embeddings.cpu().numpy() # Pour la visualisation t-SNE
        }
    
    def get_top_anomalies(self, scores, k=10):
        """
        Tri les nœuds par leur score d'anomalie et retourne les 'k' pires.
        Ce sont les anomalies les plus probables.
        """
        # Récupère les indices des k plus grands scores
        top_idx = np.argsort(scores)[-k:][::-1] 
        
        # Formate les résultats pour qu'ils soient lisibles
        return [{
            'index': idx,
            'city': self.data.city_names[idx],
            'country': self.data.country_names[idx],
            'population': self.data.population[idx].item(),
            'score': scores[idx]
        } for idx in top_idx]