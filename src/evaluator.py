import torch
import numpy as np


class AnomalyEvaluator:
    def __init__(self, model, data, device='cpu', pop_weight=0.6):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.pop_weight = pop_weight # Pondération pour le score de population
        
    @torch.no_grad()
    def compute_anomaly_scores(self):
        self.model.eval()
        pop_pred, country_pred, embeddings = self.model(self.data.x, self.data.edge_index)
        
        # Population error
        log_pop_true = torch.log1p(self.data.population)
        log_pop_pred = pop_pred.squeeze()
        pop_error = torch.abs(log_pop_pred - log_pop_true).cpu().numpy()
        
        # Country error
        country_true = torch.LongTensor(self.data.country_labels).to(self.device)
        country_probs = torch.softmax(country_pred, dim=1)
        true_probs = country_probs[range(len(country_true)), country_true]
        country_score = (1 - true_probs).cpu().numpy()
        
        # Combined
        # <<< MODIFICATION >>>
        # Utilisation des poids définis dans __init__
        combined = (self.pop_weight * pop_error) + ((1 - self.pop_weight) * country_score)
        
        return {
            'population_error': pop_error,
            'country_score': country_score,
            'combined_score': combined,
            'embeddings': embeddings.cpu().numpy()
        }
    
    def get_top_anomalies(self, scores, k=10):
        top_idx = np.argsort(scores)[-k:][::-1]
        return [{
            'index': idx,
            'city': self.data.city_names[idx],
            'country': self.data.country_names[idx],
            'population': self.data.population[idx].item(),
            'score': scores[idx]
        } for idx in top_idx]