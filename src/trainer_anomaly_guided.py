import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np


class TrainerWithAnomalyGuidedLearning:

    def __init__(self, model, data, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_acc': []
        }

    def identify_suspicious_links(self):
        edge_index = self.data.edge_index
        node_features = self.data.x
        populations = self.data.population
        countries = self.data.country_labels

        suspicious_scores = []

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()

            pop_src = populations[src].item()
            pop_dst = populations[dst].item()
            country_src = countries[src]
            country_dst = countries[dst]

            pop_ratio = (pop_dst + 1) / (pop_src + 1)

            geo_dist = torch.sqrt(
                (node_features[src, 0] - node_features[dst, 0]) ** 2 +
                (node_features[src, 1] - node_features[dst, 1]) ** 2
            ).item()

            lat_diff = abs(node_features[src, 1].item() - node_features[dst, 1].item())
            lon_diff = abs(node_features[src, 0].item() - node_features[dst, 0].item())

            diff_country = 1 if country_src != country_dst else 0
            small_to_large = 1 if pop_src < pop_dst else 0

            suspicion_score = 0.0

            if small_to_large and diff_country:
                ratio_score = min(pop_ratio / 100.0, 1.0)
                suspicion_score += 0.4 * ratio_score

            if geo_dist > 0.1:
                distance_score = min(geo_dist / 0.5, 1.0)
                if diff_country:
                    suspicion_score += 0.35 * distance_score
                else:
                    suspicion_score += 0.20 * distance_score

            if lon_diff > 0.3:
                suspicion_score += 0.15 * min(lon_diff / 0.5, 1.0)

            if pop_ratio > 50:
                suspicion_score += 0.10 * min(pop_ratio / 100.0, 1.0)

            if small_to_large and diff_country and geo_dist > 0.4:
                suspicion_score = min(suspicion_score * 1.3, 1.0)

            suspicious_scores.append(suspicion_score)

        suspicious_scores = torch.tensor(suspicious_scores, device=self.device)

        return suspicious_scores

    def prepare_link_batches(self, num_samples=2000):
        suspicion_scores = self.identify_suspicious_links()

        normal_threshold = torch.quantile(suspicion_scores, 0.4)
        suspect_threshold = torch.quantile(suspicion_scores, 0.85)

        normal_mask = suspicion_scores < normal_threshold
        suspect_mask = suspicion_scores > suspect_threshold

        normal_indices = torch.where(normal_mask)[0]
        suspect_indices = torch.where(suspect_mask)[0]

        num_suspect = min(num_samples // 2, len(suspect_indices))
        num_normal = min(num_samples // 3, len(normal_indices))
        num_neg = num_samples - num_normal - num_suspect

        if num_normal > 0:
            normal_sample = normal_indices[torch.randperm(len(normal_indices))[:num_normal]]
        else:
            normal_sample = torch.tensor([], dtype=torch.long, device=self.device)

        if num_suspect > 0:
            suspect_sample = suspect_indices[torch.randperm(len(suspect_indices))[:num_suspect]]
        else:
            suspect_sample = torch.tensor([], dtype=torch.long, device=self.device)

        neg_edges = self.sample_negative_edges(num_neg)

        self.normal_sample = normal_sample
        self.suspect_sample = suspect_sample
        self.neg_edges = neg_edges
        self.suspicion_scores = suspicion_scores

    def sample_negative_edges(self, num_samples):
        if num_samples <= 0:
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(2, 0)

        neg_edges = []
        edge_set = set(map(tuple, self.data.edge_index.t().cpu().numpy()))

        attempts = 0
        max_attempts = num_samples * 10

        while len(neg_edges) < num_samples and attempts < max_attempts:
            src = torch.randint(0, self.data.num_nodes, (1,)).item()
            dst = torch.randint(0, self.data.num_nodes, (1,)).item()

            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                neg_edges.append([src, dst])

            attempts += 1

        if len(neg_edges) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(2, 0)

        return torch.tensor(neg_edges, device=self.device).t()

    def train_epoch(self, optimizer, alpha=0.4, beta=0.3, gamma=0.3):
        self.model.train()
        optimizer.zero_grad()

        pop_pred, country_pred, embeddings = self.model(self.data.x, self.data.edge_index)

        log_pop_true = torch.log1p(self.data.population[self.data.train_mask])
        log_pop_pred = pop_pred[self.data.train_mask].squeeze()
        loss_pop = F.mse_loss(log_pop_pred, log_pop_true)

        country_true = torch.LongTensor(
            self.data.country_labels[self.data.train_mask.cpu().numpy()]
        ).to(self.device)
        loss_country = F.cross_entropy(country_pred[self.data.train_mask], country_true)

        all_src, all_dst, all_labels = [], [], []

        if len(self.normal_sample) > 0:
            normal_edges = self.data.edge_index[:, self.normal_sample]
            all_src.append(normal_edges[0])
            all_dst.append(normal_edges[1])
            all_labels.append(torch.zeros(len(self.normal_sample), device=self.device))

        if len(self.suspect_sample) > 0:
            suspect_edges = self.data.edge_index[:, self.suspect_sample]
            all_src.append(suspect_edges[0])
            all_dst.append(suspect_edges[1])
            all_labels.append(self.suspicion_scores[self.suspect_sample])

        if self.neg_edges.size(1) > 0:
            all_src.append(self.neg_edges[0])
            all_dst.append(self.neg_edges[1])
            all_labels.append(torch.ones(self.neg_edges.size(1), device=self.device))

        if len(all_src) > 0:
            all_src = torch.cat(all_src)
            all_dst = torch.cat(all_dst)
            all_labels = torch.cat(all_labels)

            link_features = self.model.compute_link_features(
                embeddings, self.data.x, all_src, all_dst
            )
            anomaly_pred = self.model.link_anomaly_detector(link_features).squeeze()
            loss_link = F.mse_loss(anomaly_pred, all_labels)
        else:
            loss_link = torch.tensor(0.0, device=self.device)

        loss = alpha * loss_pop + beta * loss_country + gamma * loss_link
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        pop_pred, country_pred, _ = self.model(self.data.x, self.data.edge_index)

        log_pop_true = torch.log1p(self.data.population[mask])
        log_pop_pred = pop_pred[mask].squeeze()
        mse = F.mse_loss(log_pop_pred, log_pop_true).item()

        country_true = torch.LongTensor(
            self.data.country_labels[mask.cpu().numpy()]
        ).to(self.device)
        acc = (country_pred[mask].argmax(1) == country_true).float().mean().item()

        return mse, acc

    def fit(self, epochs=200, lr=0.01, alpha=0.4, beta=0.3, gamma=0.3, patience=30):
        Path('models').mkdir(exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        pbar = tqdm(range(epochs), desc='Training')
        for epoch in pbar:
            train_loss = self.train_epoch(optimizer, alpha, beta, gamma)
            val_mse, val_acc = self.evaluate(self.data.val_mask)

            scheduler.step(val_mse)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_mse)
            self.history['val_mse'].append(val_mse)
            self.history['val_acc'].append(val_acc)

            pbar.set_postfix({
                'train': f'{train_loss:.4f}',
                'val_mse': f'{val_mse:.4f}',
                'val_acc': f'{val_acc:.3f}'
            })

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_anomaly_guided.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(torch.load('models/best_anomaly_guided.pt'))
        return self.history