import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.manifold import TSNE
from pathlib import Path

sns.set_theme(style="whitegrid")


class Visualizer:
    def __init__(self, save_dir, dpi=300):
        self.dpi = dpi
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_curves(self, history, save_path_suffix):
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
        save_path = self.save_dir / f'training_{save_path_suffix}.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def plot_anomaly_distribution(self, scores, threshold=None, save_path_suffix='anomaly_dist'):
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
        save_path = self.save_dir / f'{save_path_suffix}.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_tsne(self, embeddings, labels, scores=None, save_path_suffix='tsne'):
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
        save_path = self.save_dir / f'{save_path_suffix}.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    # ==========================================================
    # <<< NOUVELLES FONCTIONS POUR L'ANALYSE DU GRID SEARCH >>>
    # (MAINTENANT CORRIGÉES)
    # ==========================================================

    def _clean_model_names(self, df, best_gat_name):
        """Nettoie les noms pour des graphiques plus lisibles."""
        
        # Noms des modèles d'ablation (pour les deux conventions de nommage)
        pop_only_name_base = f"{best_gat_name} (Pop-Only)"
        country_only_name_base = f"{best_gat_name} (Country-Only)"
        pop_only_name_grid = f"GAT_Best_(Pop-Only)"
        country_only_name_grid = f"GAT_Best_(Country-Only)"

        def clean_name(name):
            if name == best_gat_name:
                # Essayer de parser le nom grid search
                m = re.match(r'GAT_h(\d+)_l(\d+)_head(\d+)_a([\d\.]+)', name)
                if m:
                    return f"BEST GAT (h{m.group(1)}_l{m.group(2)}_hd{m.group(3)}_a{m.group(4)})"
                # Essayer de parser le nom AG grid search
                m_ag = re.match(r'GAT_AG_h(\d+)_l(\d+)_head(\d+)', name)
                if m_ag:
                    return f"BEST GAT-AG (h{m_ag.group(1)}_l{m_ag.group(2)}_hd{m_ag.group(3)})"
                # Sinon, utiliser le nom de base
                return f"BEST GAT ({name})" 

            if name in [pop_only_name_base, pop_only_name_grid]:
                return "Ablation: Pop-Only"
            if name in [country_only_name_base, country_only_name_grid]:
                return "Ablation: Country-Only"
            
            if "Baseline GCN" in name:
                return "Baseline GCN"
            if "AnomalyDetector GCN" in name:
                return "AnomalyDetector GCN"
            if "GAT + Anomaly-Guided" in name:
                return "GAT + Anomaly-Guided (Base)"
            if "GAT_AG" in name: # Attrape les autres runs AG du grid search
                return "GAT-AG (Grid)"
                
            return name # Retourne le nom original s'il n'est pas reconnu
            
        df_copy = df.copy()
        df_copy['clean_name'] = df_copy['model'].apply(clean_name)
        return df_copy

    # --- AJOUT DE LA FONCTION MANQUANTE ---
    def _parse_gat_name(self, model_name):
        """Utilitaire pour extraire les hyperparams du nom du modèle."""
        if not model_name.startswith('GAT_h'):
            return None
        
        match = re.match(r'GAT_h(\d+)_l(\d+)_head(\d+)_a([\d\.]+)', model_name)
        if match:
            return {
                'hidden_channels': int(match.group(1)),
                'num_layers': int(match.group(2)),
                'num_heads': int(match.group(3)),
                'train_alpha': float(match.group(4))
            }
        return None
    # --- FIN DE L'AJOUT ---

    def plot_main_comparison(self, df_full, best_gat_name):
        """
        Graphique 1: Compare les baselines au meilleur GAT trouvé.
        """
        df_combined = df_full[df_full['score_type'] == 'combined_score'].sort_values('q95')
        
        # Inclure le modèle Anomaly-Guided s'il est présent
        model_names = ['Baseline GCN (Ref)', 'AnomalyDetector GCN (Ref)', best_gat_name, 'GAT + Anomaly-Guided']
        
        # Trouver tous les modèles AG du grid search
        ag_models = df_combined[df_combined['model'].str.startswith('GAT_AG')]['model'].unique()
        model_names.extend(ag_models)
        
        df_main = df_combined[df_combined['model'].isin(model_names)]
        
        if df_main.empty:
            print("Warning: Impossible de générer 'plot_main_comparison'. Modèles non trouvés.")
            return

        df_main = self._clean_model_names(df_main, best_gat_name)
        
        # Ne garder que les 5 meilleurs AG et les 2 baselines + meilleur GAT
        df_ag_best = df_main[df_main['clean_name'].str.contains("GAT-AG")].head(5)
        df_others = df_main[~df_main['clean_name'].str.contains("GAT-AG")]
        df_main_plot = pd.concat([df_ag_best, df_others]).sort_values('q95')

        
        plt.figure(figsize=(10, max(4, len(df_main_plot)*1.5) )) # Hauteur dynamique
        g = sns.barplot(
            data=df_main_plot,
            x='q95',
            y='clean_name',
            palette='viridis'
        )
        g.set_title('Comparaison des Modèles (Score Q95 - plus bas = meilleur)', fontweight='bold')
        g.set_xlabel('Q95 du Score d\'Anomalie Combiné')
        g.set_ylabel('Modèle')
        
        save_path = self.save_dir / 'comparison_01_main_models.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ Graphique de comparaison principal sauvegardé sur {save_path.name}")

    def plot_ablation_study(self, df_full, best_gat_name):
        """
        Graphique 2: Compare le meilleur GAT à ses versions d'ablation.
        """
        df_combined = df_full[df_full['score_type'] == 'combined_score']
        
        # Noms des modèles d'ablation
        pop_only_name_base = f"{best_gat_name} (Pop-Only)"
        country_only_name_base = f"{best_gat_name} (Country-Only)"
        pop_only_name_grid = "GAT_Best_(Pop-Only)"
        country_only_name_grid = "GAT_Best_(Country-Only)"
        
        model_names_to_keep = [
            best_gat_name, 
            pop_only_name_base, 
            country_only_name_base,
            pop_only_name_grid,
            country_only_name_grid
        ]
        
        df_ablation = df_combined[df_combined['model'].isin(model_names_to_keep)].copy()

        try:
            best_alpha_series = df_ablation[df_ablation['model'] == best_gat_name]['train_alpha']
            if not best_alpha_series.empty:
                best_alpha = best_alpha_series.iloc[0]
            else:
                best_alpha = -1.0 
        except Exception:
            best_alpha = -1.0

        if best_alpha == 1.0:
            df_ablation = df_ablation[
                ~df_ablation['model'].isin([pop_only_name_base, pop_only_name_grid])
            ]
        if best_alpha == 0.0:
            df_ablation = df_ablation[
                ~df_ablation['model'].isin([country_only_name_base, country_only_name_grid])
            ]
        
        if df_ablation.empty or len(df_ablation) < 2:
            print(f"Warning: Données insuffisantes pour 'plot_ablation_study'. Meilleurs runs: {best_gat_name}")
            return

        df_ablation = self._clean_model_names(df_ablation, best_gat_name)
        
        plt.figure(figsize=(10, max(4, len(df_ablation)*1.5) ))
        g = sns.barplot(
            data=df_ablation.sort_values('q95'),
            x='q95',
            y='clean_name',
            palette='plasma'
        )
        g.set_title('Ablation Study sur la Fonction de Loss (sur le meilleur GAT)', fontweight='bold')
        g.set_xlabel('Q95 du Score d\'Anomalie Combiné (plus bas = meilleur)')
        g.set_ylabel('')
        
        save_path = self.save_dir / 'comparison_02_ablation_study.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ Graphique d'ablation study sauvegardé sur {save_path.name}")

    def plot_grid_search_analysis(self, df_full):
        """
        Graphique 3: Analyse détaillée du Grid Search (Bubble Chart).
        """
        df_gat = df_full[df_full['score_type'] == 'combined_score'].copy()
        
        # Cette ligne va maintenant fonctionner
        parsed_params = df_gat['model'].apply(self._parse_gat_name)
        
        df_gat = df_gat.join(pd.DataFrame(parsed_params.tolist(), index=df_gat.index))
        
        # Garde uniquement les runs valides du grid search GAT
        df_gat = df_gat.dropna(subset=['hidden_channels', 'num_layers', 'num_heads', 'train_alpha'])
        
        if df_gat.empty:
            print("Warning: Impossible de générer 'plot_grid_search_analysis'. Aucun run GAT standard trouvé.")
            return

        g = sns.relplot(
            data=df_gat,
            x='hidden_channels',
            y='num_layers',
            size='num_heads', 
            col='train_alpha', 
            hue='q95', 
            palette='mako_r', 
            kind='scatter',
            sizes=(100, 500),
            alpha=0.8,
            height=5,
            aspect=0.7,
            x_jitter=0.1, 
            y_jitter=0.1
        )
        
        g.fig.suptitle('Analyse du Grid Search (Couleur = Score Q95, plus sombre = meilleur)', y=1.05, fontweight='bold')
        g.set_xlabels('Canaux Cachés (hidden_channels)')
        g.set_ylabels('Nb. Couches (num_layers)')
        g.set_titles(r'Alpha d\'entraînement = {col_name}')
        g._legend.set_title('Score Q95') 
        
        try:
            h, l = g.legend.axes.get_legend_handles_labels()
            size_idx = [i for i, label in enumerate(l) if label == 'num_heads'][0]
            l[size_idx] = 'Nb. Têtes (num_heads)'
            g.legend.set_texts(l)
        except Exception:
             pass 

        save_path = self.save_dir / 'comparison_03_grid_search.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✅ Graphique d'analyse du Grid Search sauvegardé sur {save_path.name}")