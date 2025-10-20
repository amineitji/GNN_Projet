import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class Trainer:
    def __init__(self, model, data, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_acc': []}
        
    def train_epoch(self, optimizer, alpha=0.6):
        self.model.train()
        optimizer.zero_grad()
        
        pop_pred, country_pred, _ = self.model(self.data.x, self.data.edge_index)
        
        # Loss population
        log_pop_true = torch.log1p(self.data.population[self.data.train_mask])
        log_pop_pred = pop_pred[self.data.train_mask].squeeze()
        loss_pop = F.mse_loss(log_pop_pred, log_pop_true)
        
        # Loss country
        country_true = torch.LongTensor(self.data.country_labels[self.data.train_mask.cpu().numpy()]).to(self.device)
        loss_country = F.cross_entropy(country_pred[self.data.train_mask], country_true)
        
        loss = alpha * loss_pop + (1 - alpha) * loss_country
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        pop_pred, country_pred, _ = self.model(self.data.x, self.data.edge_index)
        
        log_pop_true = torch.log1p(self.data.population[mask])
        log_pop_pred = pop_pred[mask].squeeze()
        mse = F.mse_loss(log_pop_pred, log_pop_true).item()
        
        country_true = torch.LongTensor(self.data.country_labels[mask.cpu().numpy()]).to(self.device)
        acc = (country_pred[mask].argmax(1) == country_true).float().mean().item()
        return mse, acc
    
    def fit(self, epochs=200, lr=0.01, alpha=0.6, patience=30):
        Path('models').mkdir(exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(epochs), desc='Training')
        for epoch in pbar:
            train_loss = self.train_epoch(optimizer, alpha)
            val_mse, val_acc = self.evaluate(self.data.val_mask)
            
            scheduler.step(val_mse)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_mse)
            self.history['val_mse'].append(val_mse)
            self.history['val_acc'].append(val_acc)
            
            pbar.set_postfix({'train': f'{train_loss:.4f}', 'val_mse': f'{val_mse:.4f}', 'val_acc': f'{val_acc:.3f}'})
            
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break
        
        self.model.load_state_dict(torch.load('models/best.pt'))
        return self.history