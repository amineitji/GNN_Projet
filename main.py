#!/usr/bin/env python3
"""
GNN Anomaly Detection - Point d'entrée principal
Version Interactive :
 - Cas de base
 - Hyperparamètres custom
 - Grid Search
 - Revue des résultats
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import itertools # Pour le Grid Search

from src.data_loader import AirportDataLoader
from src.models import BaselineGCN, AnomalyDetectorGCN, ImprovedGAT
from src.trainer import Trainer
from src.evaluator import AnomalyEvaluator
from src.visualizer import Visualizer # Ce fichier doit contenir les nouvelles fonctions d'analyse


# ==================== CONFIGURATION ====================
DATA_PATH = "data/airportsAndCoordAndPop.graphml.xml"
EPOCHS = 200
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Configuration pour le Cas de Base [1] et le Cas Custom [2] ---
DEFAULT_GAT_CONFIG = {
    'hidden_channels': 64,
    'num_layers': 3,
    'num_heads': 4
}
DEFAULT_ALPHA = 0.6 # Poids de la loss pour ces runs

# --- Configuration pour le Grid Search [3] ---
# Hyperparamètres "très variés" pour le Grid Search
GAT_GRID_SEARCH_CONFIG = {
    'hidden_channels': [16, 32, 64], # Taille de l'embedding
    'num_layers': [2, 3, 4],         # Profondeur du modèle
    'num_heads': [1, 2, 4]           # Nb. de têtes d'attention
}
TRAIN_ALPHAS_GRID = [0.1, 0.3, 0.5, 0.7, 0.9] # Poids de la loss à tester


# =====================================================================
# <<< PARTIE 1 : LOGIQUE D'EXPÉRIENCE DE BASE >>>
# (Cette fonction est utilisée par TOUTES les options d'exécution)
# =====================================================================

def run_experiment(model_name, model, data, experiment_alpha=0.6):
    """Lance une expérience complète (une seule)"""
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} (Alpha: {experiment_alpha})")
    print(f"{'='*60}")
    
    trainer = Trainer(model, data, DEVICE)
    # Note : Le 'patience' est un peu réduit pour des runs plus rapides
    history = trainer.fit(epochs=EPOCHS, lr=LEARNING_RATE, alpha=experiment_alpha, patience=25)
    
    # L'évaluateur utilise un poids fixe (ex: 0.6) pour que les
    # 'combined_score' soient toujours comparables.
    evaluator = AnomalyEvaluator(model, data, DEVICE, pop_weight=0.6)
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


# =====================================================================
# <<< PARTIE 2 : LOGIQUE POUR LES CHOIX [1] et [2] (Base / Custom) >>>
# =====================================================================

def run_full_experiment(gat_config, run_name_suffix="Run"):
    """
    Fonction pour les runs [1] et [2].
    Lance les baselines, UN GAT (défaut ou custom), et l'ablation sur ce GAT.
    """
    print(f"💻 Device: {DEVICE}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('results') / f'run_{timestamp}_{run_name_suffix}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Tous les résultats seront sauvegardés dans: {run_dir}")
    
    print("📂 Loading data...")
    loader = AirportDataLoader(DATA_PATH)
    data = loader.load_data()
    
    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]
    
    # --- 1. Définition des modèles ---
    gat_model_name = f"Improved GAT ({run_name_suffix})"
    models_to_run = {
        'Baseline GCN (Ref)': BaselineGCN(in_channels, 32, 64, num_classes),
        'AnomalyDetector GCN (Ref)': AnomalyDetectorGCN(in_channels, 64, 3, num_classes),
        gat_model_name: ImprovedGAT(
            in_channels,
            hidden_channels=gat_config['hidden_channels'],
            num_layers=gat_config['num_layers'],
            num_classes=num_classes,
            num_heads=gat_config['num_heads']
        )
    }
    
    print("\n" + "="*60)
    print(f"🔬 Configuration '{gat_model_name}' pour cette exécution:")
    print(f"  - hidden_channels: {gat_config['hidden_channels']}")
    print(f"  - num_layers: {gat_config['num_layers']}")
    print(f"  - num_heads: {gat_config['num_heads']}")
    print(f"  - train_alpha: {DEFAULT_ALPHA}")
    print("="*60)

    all_results = []
    all_scores = {}
    all_histories = {}
    
    # --- 2. Lancement des modèles (Comparaison) ---
    for name, model in models_to_run.items():
        try:
            results_list, scores, history = run_experiment(name, model, data, experiment_alpha=DEFAULT_ALPHA)
            all_results.extend(results_list)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            continue
    
    # --- 3. Lancement de l'Ablation Study ---
    print("\n" + "="*60)
    print(f"🔬 ABLATION STUDY (Loss Function) sur '{gat_model_name}'")
    print("="*60)
    
    ablation_configs = {
        f"{gat_model_name} (Pop-Only)": {'alpha': 1.0},
        f"{gat_model_name} (Country-Only)": {'alpha': 0.0},
    }
    
    for name, config in ablation_configs.items():
        if config['alpha'] == DEFAULT_ALPHA: continue # Évite de re-run si alpha est 0 ou 1
        
        print(f"\n--- Running Ablation: {name} ---")
        try:
            # Recrée le modèle GAT
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

    # --- 4. Rapport Final et Visualisations ---
    df = pd.DataFrame(all_results)
    df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)
    
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON (incl. Ablation Study)")
    print("="*60)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_combined.to_string(index=False, float_format="%.4f"))
    
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
        best_name = gat_model_name
        print(f"\n📈 Visualisations (basées sur {best_name}) sauvegardées dans {run_dir}")
        
        best_scores_dict = all_scores[best_name]
        best_combined_scores = best_scores_dict['combined_score']
        
        viz.plot_anomaly_distribution(best_combined_scores, np.percentile(best_combined_scores, 95))
        viz.plot_tsne(best_scores_dict['embeddings'], data.country_labels, best_combined_scores)
        
        # --- Graphiques d'analyse comparative ---
        print("\n" + "-"*60)
        print("📊 Génération des graphiques d'analyse comparative...")
        print("-" * 60)
        viz.plot_main_comparison(df, best_name)
        viz.plot_ablation_study(df, best_name)
        # Note: Le plot_grid_search_analysis ne sera pas très utile ici, mais on le lance
        viz.plot_grid_search_analysis(df)
        print("-" * 60)
        
        print("\n✅ Done! Check 'results/' folder for visualizations")
    except KeyError:
        print(f"\n❌ Erreur : Le modèle '{best_name}' n'a pas pu être évalué. Visualisations ignorées.")
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération des visualisations: {e}")

# =====================================================================
# <<< PARTIE 3 : LOGIQUE POUR LE CHOIX [3] (Grid Search) >>>
# =====================================================================

def run_grid_search_experiment():
    """
    Fonction principale pour lancer la série complète d'expériences 
    (Baselines + Grid Search GAT + Ablation).
    """
    print(f"💻 Device: {DEVICE}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('results') / f'run_{timestamp}_GridSearch'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Tous les résultats seront sauvegardés dans: {run_dir}")
    
    print("📂 Loading data...")
    loader = AirportDataLoader(DATA_PATH)
    data = loader.load_data()
    
    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]
    
    all_results = []
    all_scores = {}
    all_histories = {}

    # --- 1. Expériences de Baseline (Alpha par défaut) ---
    print("\n" + "="*60)
    print("🔬 Phase 1: Entraînement des modèles de référence (Baselines)")
    print("="*60)
    
    baseline_models = {
        'Baseline GCN (Ref)': BaselineGCN(in_channels, 32, 64, num_classes),
        'AnomalyDetector GCN (Ref)': AnomalyDetectorGCN(in_channels, 64, 3, num_classes),
    }
    
    for name, model in baseline_models.items():
        try:
            results_list, scores, history = run_experiment(name, model, data, experiment_alpha=DEFAULT_ALPHA)
            all_results.extend(results_list)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
    
    # --- 2. Grid Search sur ImprovedGAT ---
    print("\n" + "="*60)
    print("🔬 Phase 2: Grid Search sur 'Improved GAT'")
    print("="*60)
    
    # Générer les combinaisons d'hyperparamètres
    h_channels = GAT_GRID_SEARCH_CONFIG['hidden_channels']
    n_layers = GAT_GRID_SEARCH_CONFIG['num_layers']
    n_heads = GAT_GRID_SEARCH_CONFIG['num_heads']
    
    # Crée la grille de toutes les combinaisons possibles (arch + alpha)
    grid = list(itertools.product(h_channels, n_layers, n_heads, TRAIN_ALPHAS_GRID))
    
    total_runs = len(grid)
    print(f"Total 'Improved GAT' experiments to run: {total_runs}")

    best_gat_results = None # Pour stocker la meilleure config GAT
    best_gat_q95 = float('inf')
    
    for i, (hidden, layers, heads, alpha) in enumerate(grid):
        model_name = f"GAT_h{hidden}_l{layers}_head{heads}_a{alpha}"
        print(f"\n--- [Run {i+1}/{total_runs}] ---")
        
        model_gat = ImprovedGAT(
            in_channels,
            hidden_channels=hidden,
            num_layers=layers,
            num_classes=num_classes,
            num_heads=heads
        )
        
        try:
            results_list, scores, history = run_experiment(model_name, model_gat, data, experiment_alpha=alpha)
            all_results.extend(results_list)
            all_scores[model_name] = scores
            all_histories[model_name] = history
            
            current_q95 = [r['q95'] for r in results_list if r['score_type'] == 'combined_score'][0]
            if current_q95 < best_gat_q95:
                best_gat_q95 = current_q95
                best_gat_results = {
                    'name': model_name,
                    'config': {'hidden_channels': hidden, 'num_layers': layers, 'num_heads': heads},
                    'alpha': alpha,
                    'scores': scores,
                    'history': history
                }
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue

    if best_gat_results is None:
        print("❌ Aucun run GAT n'a réussi. Impossible de continuer l'ablation.")
        return

    # --- 3. Annonce du meilleur modèle ---
    best_name = best_gat_results['name']
    best_config = best_gat_results['config']
    print("\n" + "="*60)
    print(f"🏆 Meilleure configuration GAT trouvée:")
    print(f"   - Modèle: {best_name}")
    print(f"   - Q95 (Combined): {best_gat_q95:.4f}")
    print(f"   - Config: {best_config}")
    print(f"   - Alpha: {best_gat_results['alpha']}")
    print("="*60)

    # --- 4. Ablation Study (basée sur le MEILLEUR GAT) ---
    print("\n" + "="*60)
    print("🔬 Phase 4: ABLATION STUDY (Loss Function)")
    print(f"Lancement de l'ablation sur la MEILLEURE configuration 'Improved GAT'")
    print("="*60)
    
    ablation_alphas = {}
    if best_gat_results['alpha'] != 1.0:
         ablation_alphas[f"GAT_Best_h{best_config['hidden_channels']}_(Pop-Only)"] = 1.0
    if best_gat_results['alpha'] != 0.0:
         ablation_alphas[f"GAT_Best_h{best_config['hidden_channels']}_(Country-Only)"] = 0.0

    for name, alpha_abl in ablation_alphas.items():
        print(f"\n--- Running Ablation: {name} ---")
        try:
            model_abl = ImprovedGAT(
                in_channels,
                hidden_channels=best_config['hidden_channels'],
                num_layers=best_config['num_layers'],
                num_classes=num_classes,
                num_heads=best_config['num_heads']
            )
            results_list, scores, history = run_experiment(
                name, model_abl, data, experiment_alpha=alpha_abl
            )
            all_results.extend(results_list)
            all_scores[name] = scores
            all_histories[name] = history
        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    # --- 5. Final Report & Visuals ---
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON (incl. Grid Search & Ablation)")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_combined.to_string(index=False, float_format="%.4f"))

    csv_path = run_dir / 'comparison_full_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Tableau de résultats complet sauvegardé sur {csv_path}")
    
    # --- Visualisations ---
    print("\n📊 Generating visualizations...")
    viz = Visualizer(save_dir=run_dir)
    
    try:
        viz.plot_training_curves(best_gat_results['history'], f'{best_name}_(BEST)')
        viz.plot_training_curves(all_histories['Baseline GCN (Ref)'], 'BaselineGCN_Ref')
        viz.plot_training_curves(all_histories['AnomalyDetector GCN (Ref)'], 'AnomalyDetectorGCN_Ref')
        
        print(f"\n📈 Visualisations (basées sur {best_name}) sauvegardées dans {run_dir}")
        best_scores_dict = best_gat_results['scores']
        best_combined_scores = best_scores_dict['combined_score']
        
        viz.plot_anomaly_distribution(best_combined_scores, np.percentile(best_combined_scores, 95))
        viz.plot_tsne(best_scores_dict['embeddings'], data.country_labels, best_combined_scores)

        # --- Graphiques d'analyse comparative ---
        print("\n" + "-"*60)
        print("📊 Génération des graphiques d'analyse comparative...")
        print("-" * 60)
        viz.plot_main_comparison(df, best_name)
        viz.plot_ablation_study(df, best_name)
        viz.plot_grid_search_analysis(df)
        print("-" * 60)
        
        print("\n✅ Done! Check 'results/' folder for all visualizations")
    
    except KeyError as e:
        print(f"\n❌ Erreur : Le modèle de référence '{e}' n'a pas pu être trouvé. Visualisations ignorées.")
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération des visualisations: {e}")


# =====================================================================
# <<< PARTIE 4 : LOGIQUE POUR LA REVUE [r] >>>
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
        print("\n--- 📊 Tableau de Comparaison (Top 20, 'combined_score' trié par Q95) ---")
        try:
            df = pd.read_csv(csv_files[0])
            df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)
            with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 1000):
                print(df_combined.to_string(index=False, float_format="%.4f"))
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
            
        print("\n  [m] Retour au menu principal")
        
        choice = input("\nVotre choix : ")
        
        if choice.lower() == 'm':
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
# <<< PARTIE 5 : LOGIQUE DE PROMPT INTERACTIF (POUR CHOIX [2]) >>>
# =====================================================================

def safe_int_input(prompt, default):
    """Demande un entier à l'utilisateur, avec une valeur par défaut."""
    val_str = input(prompt)
    if val_str == "":
        return default
    try:
        return int(val_str)
    except ValueError:
        print(f"Entrée invalide. Utilisation de la valeur par défaut: {default}")
        return default

def get_interactive_gat_config():
    """
    Demande à l'utilisateur de saisir les hyperparamètres pour le modèle GAT.
    """
    clear_screen()
    print("="*70)
    print("🔧 CONFIGURATION GAT CUSTOMISÉE")
    print("="*70)
    print("Veuillez entrer vos hyperparamètres. Laissez vide pour les valeurs par défaut.")
    
    hidden = safe_int_input(f"  - Canaux cachés (défaut: {DEFAULT_GAT_CONFIG['hidden_channels']}): ", DEFAULT_GAT_CONFIG['hidden_channels'])
    layers = safe_int_input(f"  - Nombre de couches (défaut: {DEFAULT_GAT_CONFIG['num_layers']}): ", DEFAULT_GAT_CONFIG['num_layers'])
    heads = safe_int_input(f"  - Nombre de têtes (défaut: {DEFAULT_GAT_CONFIG['num_heads']}): ", DEFAULT_GAT_CONFIG['num_heads'])

    config = {
        'hidden_channels': hidden,
        'num_layers': layers,
        'num_heads': heads
    }
    return config


# =====================================================================
# <<< PARTIE 6 : POINT D'ENTRÉE PRINCIPAL (MENU INTERACTIF) >>>
# =====================================================================

def show_main_menu():
    """Affiche le menu principal interactif."""
    while True:
        clear_screen()
        print("="*70)
        print("          PROJET GNN - DÉTECTION D'ANOMALIES")
        print("="*70)
        print("Que souhaitez-vous faire ?\n")
        print("  [1] Lancer l'expérience de base (Baselines + 1 GAT par défaut)")
        print("  [2] Lancer une expérience avec hyperparamètres custom (Interactif)")
        print("  [3] Lancer le Grid Search complet (pour le rapport)")
        print("\n  [r] Revoir les résultats d'une exécution précédente")
        print("  [q] Quitter")
        
        choice = input("\nVotre choix : ")

        if choice == '1':
            clear_screen()
            print("🚀 Lancement de l'expérience de base...")
            run_full_experiment(DEFAULT_GAT_CONFIG, run_name_suffix="BaseRun")
            print("\n✅ Expérience de base terminée.")
            input("Appuyez sur 'Entrée' pour retourner au menu...")

        elif choice == '2':
            custom_config = get_interactive_gat_config()
            clear_screen()
            print("🚀 Lancement de l'expérience customisée...")
            run_full_experiment(custom_config, run_name_suffix="CustomRun")
            print("\n✅ Expérience customisée terminée.")
            input("Appuyez sur 'Entrée' pour retourner au menu...")

        elif choice == '3':
            clear_screen()
            print("🚀 Lancement du Grid Search complet...")
            print(f"Configuration: {GAT_GRID_SEARCH_CONFIG}")
            print(f"Alphas à tester: {TRAIN_ALPHAS_GRID}")
            print("\n" + "="*30 + " ATTENTION " + "="*30)
            print(f"Cela va lancer {len(list(itertools.product(GAT_GRID_SEARCH_CONFIG['hidden_channels'], GAT_GRID_SEARCH_CONFIG['num_layers'], GAT_GRID_SEARCH_CONFIG['num_heads'], TRAIN_ALPHAS_GRID)))} exécutions GAT.")
            confirm = input("Êtes-vous sûr de vouloir continuer ? (o/n): ")
            if confirm.lower() == 'o':
                run_grid_search_experiment()
                print("\n✅ Grid Search terminé.")
            else:
                print("\nOpération annulée.")
            input("Appuyez sur 'Entrée' pour retourner au menu...")

        elif choice.lower() == 'r':
            review_experiments() # Ce module a sa propre boucle

        elif choice.lower() == 'q':
            print("Au revoir !")
            break
            
        else:
            print("Choix invalide.")
            input("Appuyez sur 'Entrée' pour réessayer.")


if __name__ == "__main__":
    # Gère le cas 'python main.py review'
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'review':
        review_experiments()
    else:
        # Lance le menu principal interactif
        show_main_menu()