#!/usr/bin/env python3
"""
GNN Anomaly Detection - Point d'entrée principal
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data_loader import AirportDataLoader
from src.models import BaselineGCN, AnomalyDetectorGCN, ImprovedGAT
from src.trainer import Trainer
from src.evaluator import AnomalyEvaluator
from src.visualizer import Visualizer


# ==================== CONFIGURATION ====================
DATA_PATH = "data/airportsAndCoordAndPop.graphml.xml"
EPOCHS = 200
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_experiment(model_name, model, data):
    """Lance une expérience complète"""
    print(f"\n{'='*60}")
    print(f"🚀 {model_name}")
    print(f"{'='*60}")
    
    # Entraînement
    trainer = Trainer(model, data, DEVICE)
    history = trainer.fit(epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Évaluation
    evaluator = AnomalyEvaluator(model, data, DEVICE)
    scores = evaluator.compute_anomaly_scores()
    test_scores = scores['combined_score'][data.test_mask.cpu().numpy()]
    
    # Résultats
    results = {
        'model': model_name,
        'mean_score': float(np.mean(test_scores)),
        'std_score': float(np.std(test_scores)),
        'q95': float(np.percentile(test_scores, 95)),
        'q99': float(np.percentile(test_scores, 99))
    }
    
    # Top anomalies
    anomalies = evaluator.get_top_anomalies(scores['combined_score'], k=10)
    
    print(f"\n📊 Top 10 Anomalies:")
    for i, a in enumerate(anomalies, 1):
        print(f"  {i}. {a['city']}, {a['country']} - Score: {a['score']:.3f}")
    
    return results, scores, history


def main():
    print("="*60)
    print("🎯 GNN ANOMALY DETECTION")
    print("="*60)
    print(f"💻 Device: {DEVICE}\n")
    
    # Charger données
    print("📂 Loading data...")
    loader = AirportDataLoader(DATA_PATH)
    data = loader.load_data()
    
    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]
    
    # Définir les modèles à tester
    models = {
        'Baseline GCN': BaselineGCN(in_channels, 32, 64, num_classes),
        'AnomalyDetector GCN': AnomalyDetectorGCN(in_channels, 64, 3, num_classes),
        'Improved GAT': ImprovedGAT(in_channels, 64, 3, num_classes, num_heads=4)
    }
    
    # Lancer expériences
    all_results = []
    all_scores = {}
    all_histories = {}
    
    for name, model in models.items():
        try:
            results, scores, history = run_experiment(name, model, data)
            all_results.append(results)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            continue
    
    # Tableau comparatif
    df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    
    # Sauvegarder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path('results').mkdir(exist_ok=True)
    df.to_csv(f'results/comparison_{timestamp}.csv', index=False)
    print(f"\n💾 Results saved to results/comparison_{timestamp}.csv")
    
    # Visualisations
    print("\n📊 Generating visualizations...")
    viz = Visualizer()
    
    for name, history in all_histories.items():
        viz.plot_training_curves(history, f'viz/training_{name.replace(" ", "_")}.png')
    
    # Meilleur modèle
    best = max(all_results, key=lambda x: x['q95'])
    best_name = best['model']
    best_scores = all_scores[best_name]['combined_score']
    
    viz.plot_anomaly_distribution(best_scores, np.percentile(best_scores, 95))
    viz.plot_tsne(all_scores[best_name]['embeddings'], data.country_labels, best_scores)
    
    print("\n✅ Done! Check 'viz/' folder for visualizations")


if __name__ == "__main__":
    main()