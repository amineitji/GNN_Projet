import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, model, data, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        # Historique pour tracer les courbes d'apprentissage
        self.history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_acc': []}
        
    def train_epoch(self, optimizer, alpha=0.6):
        """
        Effectue une seule époque d'entraînement.
        'alpha' est le poids donné à la loss de population.
        '1 - alpha' sera le poids pour la loss de pays.
        """
        self.model.train() # Passe le modèle en mode entraînement (active le dropout)
        optimizer.zero_grad()
        
        # Étape 1: Obtenir les prédictions du modèle
        pop_pred, country_pred, _ = self.model(self.data.x, self.data.edge_index)
        
        # --- Étape 2: Calculer les deux 'Loss' (erreurs) ---
        # Calcule les erreurs UNIQUEMENT sur les nœuds du 'train_mask'
        
        # Loss 1: Erreur sur la POPULATION (Régression)
        # On prédit sur le log de la population pour stabiliser
        log_pop_true = torch.log1p(self.data.population[self.data.train_mask])
        log_pop_pred = pop_pred[self.data.train_mask].squeeze()
        # On utilise la Mean Squared Error (MSE)
        loss_pop = F.mse_loss(log_pop_pred, log_pop_true)
        
        # Loss 2: Erreur sur le PAYS (Classification)
        country_true = torch.LongTensor(self.data.country_labels[self.data.train_mask.cpu().numpy()]).to(self.device)
        # On utilise la Cross Entropy Loss
        loss_country = F.cross_entropy(country_pred[self.data.train_mask], country_true)
        
        # --- Étape 3: Combiner les 'Loss' (Cœur de l'approche) ---
        # C'est la tâche multi-objectifs. Le modèle doit devenir bon
        # aux DEUX tâches en même temps.
        loss = alpha * loss_pop + (1 - alpha) * loss_country
        
        # Étape 4: Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad() # Désactive le calcul de gradient pour l'évaluation
    def evaluate(self, mask):
        """
        Évalue les performances du modèle sur un ensemble de données (validation ou test).
        """
        self.model.eval() # Passe le modèle en mode évaluation (désactive le dropout)
        pop_pred, country_pred, _ = self.model(self.data.x, self.data.edge_index)
        
        # Calcule la MSE sur la population (similaire à train_epoch)
        log_pop_true = torch.log1p(self.data.population[mask])
        log_pop_pred = pop_pred[mask].squeeze()
        mse = F.mse_loss(log_pop_pred, log_pop_true).item()
        
        # Calcule l'Accuracy (précision) sur le pays
        country_true = torch.LongTensor(self.data.country_labels[mask.cpu().numpy()]).to(self.device)
        acc = (country_pred[mask].argmax(1) == country_true).float().mean().item()
        
        return mse, acc
    
    def fit(self, epochs=200, lr=0.01, alpha=0.6, patience=30):
        """
        La boucle d'entraînement principale.
        """
        Path('models').mkdir(exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        # Scheduler pour réduire le learning rate si la loss ne s'améliore pas
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(epochs), desc='Training')
        for epoch in pbar:
            train_loss = self.train_epoch(optimizer, alpha)
            val_mse, val_acc = self.evaluate(self.data.val_mask)
            
            scheduler.step(val_mse) # Le scheduler surveille la loss de validation (MSE)
            
            # Sauvegarde des métriques pour les graphiques
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_mse)
            self.history['val_mse'].append(val_mse)
            self.history['val_acc'].append(val_acc)
            
            pbar.set_postfix({'train': f'{train_loss:.4f}', 'val_mse': f'{val_mse:.4f}', 'val_acc': f'{val_acc:.3f}'})
            
            # --- Early Stopping & Sauvegarde du meilleur modèle ---
            # Si la loss de validation s'améliore, on sauvegarde le modèle
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best.pt')
            else:
                # Sinon, on incrémente le compteur de patience
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break
        
        # Recharge le meilleur modèle sauvegardé avant de terminer
        self.model.load_state_dict(torch.load('models/best.pt'))
        return self.history