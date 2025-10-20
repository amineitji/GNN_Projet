import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from pathlib import Path

sns.set_style("whitegrid")


class Visualizer:
    def __init__(self, dpi=300):
        self.dpi = dpi
        Path('viz').mkdir(exist_ok=True)
        
    def plot_training_curves(self, history, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Curves', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history['val_mse'], label='MSE', linewidth=2, color='orange')
        ax2 = axes[1].twinx()
        ax2.plot(history['val_acc'], label='Acc', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE', color='orange')
        ax2.set_ylabel('Acc', color='green')
        axes[1].set_title('Metrics', fontweight='bold')
        axes[1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_anomaly_distribution(self, scores, threshold=None, save_path='viz/anomaly_dist.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        if threshold:
            axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Q95: {threshold:.3f}')
            axes[0].legend()
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Score Distribution', fontweight='bold')
        
        axes[1].boxplot(scores, vert=True)
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Boxplot', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_tsne(self, embeddings, labels, scores=None, save_path='viz/tsne.png'):
        tsne = TSNE(n_components=2, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)
        axes[0].set_title('t-SNE by Country', fontweight='bold')
        
        if scores is not None:
            sc = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=scores, cmap='YlOrRd', alpha=0.6, s=30)
            axes[1].set_title('t-SNE by Score', fontweight='bold')
            plt.colorbar(sc, ax=axes[1], label='Anomaly Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()