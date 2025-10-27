#!/usr/bin/env python3
"""
GNN Anomaly Detection - Point d'entrée principal
Peut être lancé de deux manières :
  - python main.py          : Lance une nouvelle expérience (configurable)
  - python main.py review   : Ouvre le menu de revue des expériences
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

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

# <<< NOUVEAU >>>
# Paramètres par défaut pour le GAT
DEFAULT_GAT_CONFIG = {
    'hidden_channels': 64,
    'num_layers': 3,
    'num_heads': 4
}


# =====================================================================
# <<< PARTIE 1 : LOGIQUE POUR LANCER UNE NOUVELLE EXPÉRIENCE >>>
# =====================================================================

def run_experiment(model_name, model, data, experiment_alpha=0.6):
    """Lance une expérience complète (une seule)"""
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} (Alpha: {experiment_alpha})")
    print(f"{'='*60}")
    
    trainer = Trainer(model, data, DEVICE)
    history = trainer.fit(epochs=EPOCHS, lr=LEARNING_RATE, alpha=experiment_alpha)
    
    evaluator = AnomalyEvaluator(model, data, DEVICE)
    all_scores = evaluator.compute_anomaly_scores() 
    
    results_list = []
    score_types = ['combined_score', 'population_error', 'country_score']
    
    print("\n📊 Évaluation des scores (sur ensemble test):")
    
    for score_type in score_types:
        test_scores = all_scores[score_type][data.test_mask.cpu().numpy()]
        results = {
            'model': model_name,
            'train_alpha': experiment_alpha,
            'score_type': score_type,
            'mean_score': float(np.mean(test_scores)),
            'std_score': float(np.std(test_scores)),
            'q95': float(np.percentile(test_scores, 95)),
            'q99': float(np.percentile(test_scores, 99))
        }
        results_list.append(results)
        print(f"  - Score: {score_type:<17} | Q95: {results['q95']:.4f} | Q99: {results['q99']:.4f}")

    anomalies = evaluator.get_top_anomalies(all_scores['combined_score'], k=10)
    
    print(f"\n📊 Top 10 Anomalies (basé sur 'combined_score'):")
    for i, a in enumerate(anomalies, 1):
        print(f"  {i}. {a['city']}, {a['country']} - Score: {a['score']:.3f}")
    
    return results_list, all_scores, history


# <<< MODIFIÉ >>>
# Accepte la configuration du GAT en paramètre
def run_full_experiment(gat_config):
    """
    Fonction principale pour lancer une série complète d'expériences.
    """
    print("="*60)
    print("🎯 GNN ANOMALY DETECTION (Mode: Nouvelle Expérience)")
    print("="*60)
    print(f"💻 Device: {DEVICE}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('results') / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Tous les résultats seront sauvegardés dans: {run_dir}")
    
    print("📂 Loading data...")
    loader = AirportDataLoader(DATA_PATH)
    data = loader.load_data()
    
    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]
    
    # <<< MODIFIÉ >>>
    # Les modèles de base sont fixes, le GAT utilise la config
    models_to_run = {
        'Baseline GCN (Ref)': BaselineGCN(in_channels, 32, 64, num_classes),
        'AnomalyDetector GCN (Ref)': AnomalyDetectorGCN(in_channels, 64, 3, num_classes),
        'Improved GAT (Custom)': ImprovedGAT(
            in_channels,
            hidden_channels=gat_config['hidden_channels'],
            num_layers=gat_config['num_layers'],
            num_classes=num_classes,
            num_heads=gat_config['num_heads']
        )
    }
    
    print("\n" + "="*60)
    print(f"🔬 Configuration 'Improved GAT' pour cette exécution:")
    print(f"  - hidden_channels: {gat_config['hidden_channels']}")
    print(f"  - num_layers: {gat_config['num_layers']}")
    print(f"  - num_heads: {gat_config['num_heads']}")
    print("="*60)

    all_results = []
    all_scores = {}
    all_histories = {}
    
    for name, model in models_to_run.items():
        try:
            results_list, scores, history = run_experiment(name, model, data)
            all_results.extend(results_list)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            continue
    
    # ==================== ABLATION STUDY (LOSS) ====================
    print("\n" + "="*60)
    print("🔬 ABLATION STUDY (Loss Function)")
    print(f"Lancement de l'ablation sur la configuration 'Improved GAT (Custom)'")
    print("="*60)
    
    # <<< MODIFIÉ >>>
    # L'ablation utilise la MÊME configuration GAT que celle testée
    ablation_configs = {
        f"Improved GAT (Custom - Pop-Only)": {'alpha': 1.0},
        f"Improved GAT (Custom - Country-Only)": {'alpha': 0.0},
    }
    
    for name, config in ablation_configs.items():
        print(f"\n--- Running Ablation: {name} ---")
        try:
            # Recrée le modèle GAT avec la config de l'utilisateur
            model_abl = ImprovedGAT(
                in_channels,
                hidden_channels=gat_config['hidden_channels'],
                num_layers=gat_config['num_layers'],
                num_classes=num_classes,
                num_heads=gat_config['num_heads']
            )
            results_list, scores, history = run_experiment(
                name, model_abl, data, experiment_alpha=config['alpha']
            )
            all_results.extend(results_list)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            continue

    # Tableau comparatif
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['train_alpha', 'model', 'q95'], ascending=[True, True, False])
    
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON (incl. Ablation Study)")
    print("="*60)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df.to_string(index=False, float_format="%.4f"))
    
    csv_path = run_dir / 'comparison_full_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Tableau de résultats sauvegardé sur {csv_path}")
    
    # Visualisations
    print("\n📊 Generating visualizations...")
    viz = Visualizer(save_dir=run_dir)
    
    for name, history in all_histories.items():
        clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        viz.plot_training_curves(history, f'{clean_name}')
    
    try:
        # Visualise le modèle GAT custom que l'utilisateur vient de configurer
        best_name = 'Improved GAT (Custom)'
        print(f"\n📈 Visualisations (basées sur {best_name}) sauvegardées dans {run_dir}")
        
        best_scores_dict = all_scores[best_name]
        best_combined_scores = best_scores_dict['combined_score']
        
        viz.plot_anomaly_distribution(best_combined_scores, np.percentile(best_combined_scores, 95))
        viz.plot_tsne(best_scores_dict['embeddings'], data.country_labels, best_combined_scores)
        
        print("\n✅ Done! Check 'results/' folder for visualizations")
    except KeyError:
        print(f"\n❌ Erreur : Le modèle '{best_name}' n'a pas pu être évalué. Visualisations ignorées.")
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération des visualisations: {e}")


# =====================================================================
# <<< PARTIE 2 : LOGIQUE POUR REVOIR LES ANCIENNES EXPÉRIENCES >>>
# (Aucun changement ici)
# =====================================================================

def clear_screen():
    """Efface le terminal pour une meilleure lisibilité"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_experiment_files(run_dir):
    """Affiche les fichiers d'une exécution spécifique"""
    clear_screen()
    print("="*70)
    print(f"🔬 Visualisation de l'exécution: {run_dir.name}")
    print("="*70)
    
    files = sorted(list(run_dir.glob('*')))
    csv_files = [f for f in files if f.suffix == '.csv']
    img_files = [f for f in files if f.suffix in ['.png', '.jpg', '.jpeg']]
    
    if csv_files:
        print("\n--- 📊 Tableau de Comparaison ---")
        try:
            df = pd.read_csv(csv_files[0])
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(df.to_string(index=False, float_format="%.4f"))
        except Exception as e:
            print(f"Impossible de lire le CSV: {e}")
    else:
        print("\n--- ❌ Aucun fichier CSV trouvé ---")

    if img_files:
        print("\n" + "--- 🖼️ Visualisations Générées ---")
        for img in img_files:
            print(f"  - {img.name}")
    else:
        print("\n--- ❌ Aucune image trouvée ---")
        
    print("\n" + "="*70)
    print("Appuyez sur 'Entrée' pour revenir au menu principal...")
    input()


def review_experiments():
    """Affiche le menu principal pour sélectionner une exécution"""
    results_dir = Path('results')
    
    while True:
        clear_screen()
        print("="*70)
        print("📋 SÉLECTEUR D'EXPÉRIENCE GNN (Mode: Revue)")
        print("="*70)
        
        if not results_dir.exists():
            print("Le dossier 'results' n'existe pas.")
            print("Veuillez d'abord lancer 'main.py' (sans 'review') pour générer des résultats.")
            return

        runs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)
        
        if not runs:
            print("Aucune exécution trouvée dans le dossier 'results'.")
            print("Veuillez lancer 'main.py' (sans 'review') pour générer des résultats.")
            return
            
        print("Veuillez choisir une exécution à inspecter :")
        for i, run_dir in enumerate(runs):
            print(f"  [{i+1}] {run_dir.name}")
            
        print("\n  [q] Quitter")
        
        choice = input("\nVotre choix : ")
        
        if choice.lower() == 'q':
            break
            
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(runs):
                show_experiment_files(runs[choice_idx])
            else:
                print("Choix invalide.")
                input("Appuyez sur 'Entrée' pour réessayer.")
        except ValueError:
            print("Veuillez entrer un numéro valide.")
            input("Appuyez sur 'Entrée' pour réessayer.")

# =====================================================================
# <<< PARTIE 3 : NOUVELLE LOGIQUE DE PROMPT INTERACTIF >>>
# =====================================================================

def safe_int_input(prompt, default):
    """Demande un entier à l'utilisateur, avec une valeur par défaut."""
    val_str = input(prompt)
    if val_str == "":
        return default
    try:
        return int(val_str)
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return default

def get_interactive_gat_config():
    """
    Demande à l'utilisateur de saisir les hyperparamètres pour le modèle GAT.
    """
    clear_screen()
    print("="*70)
    print("🔧 'IMPROVED GAT' EXPERIMENT CONFIGURATION")
    print("="*70)
    print("Please enter hyperparameters. Leave blank to use default values.")
    
    # <<< MODIFICATION >>> Noms des hyperparamètres en anglais
    hidden = safe_int_input(f"  - hidden_channels (default: {DEFAULT_GAT_CONFIG['hidden_channels']}): ", DEFAULT_GAT_CONFIG['hidden_channels'])
    layers = safe_int_input(f"  - num_layers (default: {DEFAULT_GAT_CONFIG['num_layers']}): ", DEFAULT_GAT_CONFIG['num_layers'])
    heads = safe_int_input(f"  - num_heads (default: {DEFAULT_GAT_CONFIG['num_heads']}): ", DEFAULT_GAT_CONFIG['num_heads'])

    config = {
        'hidden_channels': hidden,
        'num_layers': layers,
        'num_heads': heads
    }
    return config


# =====================================================================
# <<< PARTIE 4 : POINT D'ENTRÉE PRINCIPAL (MODIFIÉ) >>>
# =====================================================================

if __name__ == "__main__":
    # Mode "Revue" : python main.py review
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'review':
        review_experiments()
    else:
        # Mode "Nouvelle Expérience" : python main.py
        # 1. Obtenir la configuration de l'utilisateur
        custom_gat_config = get_interactive_gat_config()
        # 2. Lancer la série complète d'expériences avec cette config
        run_full_experiment(custom_gat_config)


# =====================================================================
# <<< PARTIE 4 : POINT D'ENTRÉE PRINCIPAL (MODIFIÉ) >>>
# =====================================================================

if __name__ == "__main__":
    # Mode "Revue" : python main.py review
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'review':
        review_experiments()
    else:
        # Mode "Nouvelle Expérience" : python main.py
        # 1. Obtenir la configuration de l'utilisateur
        custom_gat_config = get_interactive_gat_config()
        # 2. Lancer la série complète d'expériences avec cette config
        run_full_experiment(custom_gat_config)