#!/usr/bin/env python3
"""
GNN Anomaly Detection - Point d'entrée principal
Version Interactive avec Anomaly-Guided Learning
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
import itertools

from src.data_loader import AirportDataLoader
from src.models import BaselineGCN, AnomalyDetectorGCN, ImprovedGAT
from src.trainer import Trainer
from src.evaluator import AnomalyEvaluator
from src.visualizer import Visualizer

from src.connectivity_features import compute_connectivity_features
from src.models_anomaly_guided import ImprovedGATWithAnomalyGuidedLearning
from src.trainer_anomaly_guided import TrainerWithAnomalyGuidedLearning
from src.evaluator_anomaly_guided import AnomalyEvaluatorWithLinkGuidance


class Logger(object):
    """
    Redirige la sortie standard (print) vers un fichier
    tout en la conservant dans le terminal.
    """

    def __init__(self, filepath, original_stdout):
        self.terminal = original_stdout
        self.log_file = open(filepath, "w", encoding='utf-8')
        self.stdout = original_stdout

    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


DATA_PATH = "data/airportsAndCoordAndPop.graphml.xml"
EPOCHS = 200
LEARNING_RATE = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_GAT_CONFIG = {
    'hidden_channels': 64,
    'num_layers': 3,
    'num_heads': 4
}
DEFAULT_ALPHA = 0.6

GAT_GRID_SEARCH_CONFIG = {
    'hidden_channels': [16, 32, 64],
    'num_layers': [2, 3, 4],
    'num_heads': [1, 2, 4]
}
TRAIN_ALPHAS_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def run_experiment(model_name, model, data, experiment_alpha=0.6):
    """Lance une expérience standard (sans Anomaly-Guided)"""
    print(f"\n{'=' * 60}")
    print(f"{model_name} (Alpha: {experiment_alpha})")
    print(f"{'=' * 60}")

    trainer = Trainer(model, data, DEVICE)
    history = trainer.fit(epochs=EPOCHS, lr=LEARNING_RATE, alpha=experiment_alpha, patience=25)

    evaluator = AnomalyEvaluator(model, data, DEVICE, pop_weight=0.6)
    all_scores = evaluator.compute_anomaly_scores()

    results_list = []
    score_types = ['combined_score', 'population_error', 'country_score']

    print("\nÉvaluation des scores (sur ensemble test):")
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

    with torch.no_grad():
        model.eval()
        pop_pred, country_pred, _ = model(data.x, data.edge_index)
        predicted_pops = torch.expm1(pop_pred.squeeze()).cpu().numpy()
        predicted_countries = country_pred.argmax(dim=1).cpu().numpy()

    country_idx_to_name = {}
    for node_idx in range(len(data.country_labels)):
        idx = data.country_labels[node_idx]
        if idx not in country_idx_to_name:
            country_idx_to_name[idx] = data.country_names[node_idx]

    anomalies = evaluator.get_top_anomalies(all_scores['combined_score'], k=10)

    print(f"\nTop 10 Anomalies (basé sur 'combined_score'):")
    print("=" * 80)
    for i, a in enumerate(anomalies, 1):
        idx = a['index']
        observed_pop = a['population']
        predicted_pop = predicted_pops[idx]
        pop_diff = predicted_pop - observed_pop
        pop_diff_pct = (pop_diff / (observed_pop + 1)) * 100

        observed_country = a['country']
        observed_country_idx = data.country_labels[idx]
        predicted_country_idx = predicted_countries[idx]
        predicted_country = country_idx_to_name.get(predicted_country_idx, f"UNKNOWN_{predicted_country_idx}")
        country_match = "[OK]" if observed_country_idx == predicted_country_idx else "[X]"

        pop_err = all_scores['population_error'][idx]
        ctry_scr = all_scores['country_score'][idx]

        print(f"\n  #{i}. {a['city']}, {observed_country}")
        print(f"      Population observée : {observed_pop:>12,.0f} habitants")
        print(f"      Population prédite  : {predicted_pop:>12,.0f} habitants")
        print(f"      Différence          : {pop_diff:>+12,.0f} ({pop_diff_pct:+.1f}%)")
        print(f"      Pays observé        : {observed_country}")
        print(f"      Pays prédit         : {predicted_country} {country_match}")
        print(f"      Score Population    : {pop_err:.4f}")
        print(f"      Score Pays          : {ctry_scr:.4f}")
        print(f"      Score d'anomalie    : {a['score']:.4f}")
    print("\n" + "=" * 80)

    return results_list, all_scores, history


def run_anomaly_guided_experiment(gat_config, data):
    """Lance une expérience avec Anomaly-Guided Learning"""
    print("\n" + "=" * 60)
    print("ANOMALY-GUIDED LEARNING (GAT + Link Detection)")
    print("=" * 60)

    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]

    model = ImprovedGATWithAnomalyGuidedLearning(
        in_channels,
        hidden_channels=gat_config['hidden_channels'],
        num_layers=gat_config['num_layers'],
        num_classes=num_classes,
        num_heads=gat_config['num_heads']
    )

    trainer = TrainerWithAnomalyGuidedLearning(model, data, DEVICE)
    trainer.prepare_link_batches(num_samples=2000)

    history = trainer.fit(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )

    evaluator = AnomalyEvaluatorWithLinkGuidance(
        model, data, DEVICE, weights=(0.5, 0.3, 0.2)
    )
    scores = evaluator.compute_anomaly_scores()

    name = "GAT + Anomaly-Guided"
    results_list = []
    score_types = ['combined_score', 'population_error', 'country_score', 'link_score']

    print("\nÉvaluation des scores:")
    for score_type in score_types:
        test_scores = scores[score_type][data.test_mask.cpu().numpy()]
        results = {
            'model': name,
            'train_alpha': 0.5,
            'score_type': score_type,
            'mean_score': float(np.mean(test_scores)),
            'std_score': float(np.std(test_scores)),
            'q95': float(np.percentile(test_scores, 95)),
            'q99': float(np.percentile(test_scores, 99))
        }
        results_list.append(results)
        print(f"  - Score: {score_type:<17} | Q95: {results['q95']:.4f} | Q99: {results['q99']:.4f}")

    anomalies = evaluator.get_top_anomalies(scores['combined_score'], k=10)

    with torch.no_grad():
        pop_pred, country_pred, embeddings = model(data.x, data.edge_index)
        predicted_pops = torch.expm1(pop_pred.squeeze()).cpu().numpy()
        predicted_countries = country_pred.argmax(dim=1).cpu().numpy()

    country_idx_to_name = {}
    for node_idx in range(len(data.country_labels)):
        idx = data.country_labels[node_idx]
        if idx not in country_idx_to_name:
            country_idx_to_name[idx] = data.country_names[node_idx]

    print(f"\nTop 10 Anomalies de Nœuds (guidé par liens):")
    print("=" * 80)
    for i, a in enumerate(anomalies, 1):
        idx = a['index']
        observed_pop = a['population']
        predicted_pop = predicted_pops[idx]
        pop_diff = predicted_pop - observed_pop
        pop_diff_pct = (pop_diff / (observed_pop + 1)) * 100

        observed_country = a['country']
        observed_country_idx = data.country_labels[idx]
        predicted_country_idx = predicted_countries[idx]
        predicted_country = country_idx_to_name.get(predicted_country_idx, f"UNKNOWN_{predicted_country_idx}")
        country_match = "[OK]" if observed_country_idx == predicted_country_idx else "[X]"

        link_score = scores['link_score'][idx]

        print(f"\n  #{i}. {a['city']}, {observed_country}")
        print(f"      Population observée : {observed_pop:>12,.0f} habitants")
        print(f"      Population prédite  : {predicted_pop:>12,.0f} habitants")
        print(f"      Différence          : {pop_diff:>+12,.0f} ({pop_diff_pct:+.1f}%)")
        print(f"      Pays observé        : {observed_country}")
        print(f"      Pays prédit         : {predicted_country} {country_match}")
        print(f"      Score liens         : {link_score:.4f}")
        print(f"      Score d'anomalie    : {a['score']:.4f}")
    print("\n" + "=" * 80)

    embeddings_tensor = torch.tensor(scores['embeddings'], device=DEVICE)
    evaluator.print_anomalous_links_report(embeddings_tensor, threshold=0.5, top_k=10)

    return results_list, scores, history


def run_full_experiment(gat_config, run_name_suffix="Run", include_anomaly_guided=True):
    """
    Lance les baselines, UN GAT, l'ablation, et optionnellement Anomaly-Guided
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('results') / f'run_{timestamp}_{run_name_suffix}'
    run_dir.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    log_path = run_dir / 'console_log.txt'
    logger = Logger(log_path, original_stdout)
    sys.stdout = logger

    try:
        print(f"Device: {DEVICE}\n")
        print(f"Résultats sauvegardés dans: {run_dir}")
        print(f"Log console sauvegardé dans: {log_path.name}")

        print("Loading data...")
        loader = AirportDataLoader(DATA_PATH)
        data = loader.load_data()

        if include_anomaly_guided:
            data = compute_connectivity_features(data)

        num_classes = len(np.unique(data.country_labels))
        in_channels = data.x.shape[1]

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

        print("\n" + "=" * 60)
        print(f"Configuration '{gat_model_name}':")
        print(f"  - hidden_channels: {gat_config['hidden_channels']}")
        print(f"  - num_layers: {gat_config['num_layers']}")
        print(f"  - num_heads: {gat_config['num_heads']}")
        print(f"  - train_alpha: {DEFAULT_ALPHA}")
        print("=" * 60)

        all_results = []
        all_scores = {}
        all_histories = {}

        for name, model in models_to_run.items():
            try:
                results_list, scores, history = run_experiment(name, model, data, experiment_alpha=DEFAULT_ALPHA)
                all_results.extend(results_list)
                all_scores[name] = scores
                all_histories[name] = history
            except Exception as e:
                print(f"Error with {name}: {e}")
                import traceback
                traceback.print_exc()

        if include_anomaly_guided:
            try:
                results_ag, scores_ag, history_ag = run_anomaly_guided_experiment(gat_config, data)
                all_results.extend(results_ag)
                all_scores["GAT + Anomaly-Guided"] = scores_ag
                all_histories["GAT + Anomaly-Guided"] = history_ag
                print("\nAnomaly-Guided terminé avec succès!")
            except Exception as e:
                print(f"Error with Anomaly-Guided: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print(f"ABLATION STUDY sur '{gat_model_name}'")
        print("=" * 60)

        ablation_configs = {
            f"{gat_model_name} (Pop-Only)": {'alpha': 1.0},
            f"{gat_model_name} (Country-Only)": {'alpha': 0.0},
        }

        for name, config in ablation_configs.items():
            if config['alpha'] == DEFAULT_ALPHA:
                continue

            print(f"\n--- Running Ablation: {name} ---")
            try:
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
                print(f"Error with {name}: {e}")

        df = pd.DataFrame(all_results)
        df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)

        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_combined.to_string(index=False, float_format="%.4f"))

        csv_path = run_dir / 'comparison_full_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nRésultats sauvegardés: {csv_path}")

        print("\nGenerating visualizations...")
        viz = Visualizer(save_dir=run_dir)

        for name, history in all_histories.items():
            clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            viz.plot_training_curves(history, f'{clean_name}')

        try:
            best_scores_dict = all_scores[gat_model_name]
            best_combined_scores = best_scores_dict['combined_score']
            viz.plot_anomaly_distribution(best_combined_scores, np.percentile(best_combined_scores, 95))
            viz.plot_tsne(best_scores_dict['embeddings'], data.country_labels, best_combined_scores)
            viz.plot_main_comparison(df, gat_model_name)
            viz.plot_ablation_study(df, gat_model_name)
            print(f"\nDone! Check '{run_dir}' for results")
        except Exception as e:
            print(f"\nError visualizations: {e}")

    except Exception as e:
        print(f"Une erreur majeure est survenue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        logger.close()


def run_anomaly_guided_grid_search_step(model_name, gat_config, data):
    """
    Une version "légère" de l'expérience Anomaly-Guided pour le Grid Search.
    N'imprime pas les top 10 anomalies pour éviter de polluer le log.
    """
    print(f"\n--- [AG Run] {model_name} ---")

    num_classes = len(np.unique(data.country_labels))
    in_channels = data.x.shape[1]

    model = ImprovedGATWithAnomalyGuidedLearning(
        in_channels,
        hidden_channels=gat_config['hidden_channels'],
        num_layers=gat_config['num_layers'],
        num_classes=num_classes,
        num_heads=gat_config['num_heads']
    )

    trainer = TrainerWithAnomalyGuidedLearning(model, data, DEVICE)
    trainer.prepare_link_batches(num_samples=1000)

    history = trainer.fit(
        epochs=EPOCHS, lr=LEARNING_RATE,
        alpha=0.5, beta=0.3, gamma=0.2,
        patience=25
    )

    evaluator = AnomalyEvaluatorWithLinkGuidance(
        model, data, DEVICE, weights=(0.5, 0.3, 0.2)
    )
    scores = evaluator.compute_anomaly_scores()

    results_list = []
    fixed_alpha_for_ag = 0.5
    score_types = ['combined_score', 'population_error', 'country_score', 'link_score']

    print("  Évaluation (sur ensemble test):")
    for score_type in score_types:
        test_scores = scores[score_type][data.test_mask.cpu().numpy()]
        results = {
            'model': model_name,
            'train_alpha': fixed_alpha_for_ag,
            'score_type': score_type,
            'mean_score': float(np.mean(test_scores)),
            'std_score': float(np.std(test_scores)),
            'q95': float(np.percentile(test_scores, 95)),
            'q99': float(np.percentile(test_scores, 99))
        }
        results_list.append(results)
        print(f"    - {score_type:<17} | Q95: {results['q95']:.4f}")

    return results_list, scores, history


def run_grid_search_experiment():
    """Grid Search complet (Baselines + GAT grid + GAT Anomaly-Guided grid)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path('results') / f'run_{timestamp}_GridSearch_Full'
    run_dir.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    log_path = run_dir / 'console_log.txt'
    logger = Logger(log_path, original_stdout)
    sys.stdout = logger

    try:
        print(f"Device: {DEVICE}\n")
        print(f"Résultats sauvegardés dans: {run_dir}")
        print(f"Log console sauvegardé dans: {log_path.name}")

        print("Loading data...")
        loader = AirportDataLoader(DATA_PATH)
        data = loader.load_data()

        print("\nCalcul des features de connectivité (requis pour Phase 2b)...")
        data = compute_connectivity_features(data)

        num_classes = len(np.unique(data.country_labels))
        in_channels = data.x.shape[1]

        all_results = []
        all_scores = {}
        all_histories = {}

        print("\n" + "=" * 60)
        print("Phase 1: Baselines")
        print("=" * 60)

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
                print(f"Error with {name}: {e}")

        h_channels = GAT_GRID_SEARCH_CONFIG['hidden_channels']
        n_layers = GAT_GRID_SEARCH_CONFIG['num_layers']
        n_heads = GAT_GRID_SEARCH_CONFIG['num_heads']

        best_gat_results = None
        best_gat_q95 = float('inf')

        print("\n" + "=" * 60)
        print("Phase 2a: Grid Search (Standard GAT)")
        print("=" * 60)

        grid_std = list(itertools.product(h_channels, n_layers, n_heads, TRAIN_ALPHAS_GRID))
        total_runs_std = len(grid_std)
        print(f"Total experiments (Standard GAT): {total_runs_std}")

        for i, (hidden, layers, heads, alpha) in enumerate(grid_std):
            model_name = f"GAT_h{hidden}_l{layers}_head{heads}_a{alpha}"
            print(f"\n--- [Run {i + 1}/{total_runs_std}] ---")

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
                print(f"Error with {model_name}: {e}")

        print("\n" + "=" * 60)
        print("Phase 2b: Grid Search (Anomaly-Guided GAT)")
        print("=" * 60)

        grid_ag = list(itertools.product(h_channels, n_layers, n_heads))
        total_runs_ag = len(grid_ag)
        print(f"Total experiments (Anomaly-Guided): {total_runs_ag}")

        for i, (hidden, layers, heads) in enumerate(grid_ag):
            model_name = f"GAT_AG_h{hidden}_l{layers}_head{heads}"
            gat_config = {'hidden_channels': hidden, 'num_layers': layers, 'num_heads': heads}

            try:
                results_list, scores, history = run_anomaly_guided_grid_search_step(
                    model_name, gat_config, data
                )
                all_results.extend(results_list)
                all_scores[model_name] = scores
                all_histories[model_name] = history
            except Exception as e:
                print(f"Error with {model_name}: {e}")

        if best_gat_results is None:
            print("Aucun run GAT standard n'a réussi. Ablation study annulée.")
        else:
            best_name = best_gat_results['name']
            best_config = best_gat_results['config']
            print("\n" + "=" * 60)
            print(f"Meilleure configuration (Standard GAT): {best_name} (Q95: {best_gat_q95:.4f})")
            print("=" * 60)

            print("\n" + "=" * 60)
            print(f"Phase 3: Ablation Study (sur {best_name})")
            print("=" * 60)

            ablation_alphas = {}
            if best_gat_results['alpha'] != 1.0:
                ablation_alphas[f"GAT_Best_(Pop-Only)"] = 1.0
            if best_gat_results['alpha'] != 0.0:
                ablation_alphas[f"GAT_Best_(Country-Only)"] = 0.0

            for name, alpha_abl in ablation_alphas.items():
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
                    print(f"Error with {name}: {e}")

        df = pd.DataFrame(all_results)
        df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)

        print("\n" + "=" * 60)
        print("RESULTS COMPARISON (Grid Search COMPLET)")
        print("=" * 60)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_combined.to_string(index=False, float_format="%.4f"))

        csv_path = run_dir / 'comparison_full_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nRésultats sauvegardés: {csv_path}")

        viz = Visualizer(save_dir=run_dir)
        try:
            best_overall_model_name = df_combined.iloc[0]['model']
            best_overall_history = all_histories[best_overall_model_name]
            best_overall_scores_dict = all_scores[best_overall_model_name]

            print(f"\nMeilleur modèle global (tous types conf.): {best_overall_model_name}")

            viz.plot_training_curves(best_overall_history, f'{best_overall_model_name}_BEST_OVERALL')
            best_combined_scores = best_overall_scores_dict['combined_score']
            viz.plot_anomaly_distribution(best_combined_scores, np.percentile(best_combined_scores, 95))
            viz.plot_tsne(best_overall_scores_dict['embeddings'], data.country_labels, best_combined_scores)

            viz.plot_main_comparison(df, best_overall_model_name)

            if best_gat_results:
                viz.plot_ablation_study(df, best_gat_results['name'])

            viz.plot_grid_search_analysis(df)
            print(f"\nDone! Check '{run_dir}'")
        except Exception as e:
            print(f"\nError visualizations: {e}")

    except Exception as e:
        print(f"Une erreur majeure est survenue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        logger.close()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def show_experiment_files(run_dir):
    clear_screen()
    print("=" * 70)
    print(f"Visualisation: {run_dir.name}")
    print("=" * 70)

    files = sorted(list(run_dir.glob('*')))
    csv_files = [f for f in files if f.suffix == '.csv']
    img_files = [f for f in files if f.suffix in ['.png', '.jpg', '.jpeg']]
    log_files = [f for f in files if f.suffix == '.txt']

    if log_files:
        print("\n--- Log Console ---")
        print(f"  - {log_files[0].name}")

    if csv_files:
        print("\n--- Tableau de Comparaison ---")
        try:
            df = pd.read_csv(csv_files[0])
            df_combined = df[df['score_type'] == 'combined_score'].sort_values(by='q95', ascending=True)
            with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 1000):
                print(df_combined.to_string(index=False, float_format="%.4f"))
        except Exception as e:
            print(f"Impossible de lire le CSV: {e}")

    if img_files:
        print("\n--- Visualisations ---")
        for img in img_files:
            print(f"  - {img.name}")

    print("\n" + "=" * 70)
    input("Appuyez sur 'Entrée' pour revenir...")


def review_experiments():
    results_dir = Path('results')

    while True:
        clear_screen()
        print("=" * 70)
        print("REVUE DES EXPÉRIENCES")
        print("=" * 70)

        if not results_dir.exists():
            print("Le dossier 'results' n'existe pas.")
            return

        runs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)

        if not runs:
            print("Aucune exécution trouvée.")
            return

        print("Choisissez une exécution:")
        for i, run_dir in enumerate(runs):
            print(f"  [{i + 1}] {run_dir.name}")

        print("\n  [m] Retour au menu")

        choice = input("\nVotre choix : ")

        if choice.lower() == 'm':
            break

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(runs):
                show_experiment_files(runs[choice_idx])
            else:
                print("Choix invalide.")
                input("Appuyez sur 'Entrée'...")
        except ValueError:
            print("Veuillez entrer un numéro valide.")
            input("Appuyez sur 'Entrée'...")


def safe_int_input(prompt, default):
    val_str = input(prompt)
    if val_str == "":
        return default
    try:
        return int(val_str)
    except ValueError:
        print(f"Entrée invalide. Défaut: {default}")
        return default


def get_interactive_gat_config():
    clear_screen()
    print("=" * 70)
    print("CONFIGURATION GAT CUSTOMISÉE")
    print("=" * 70)
    print("Entrez vos hyperparamètres (vide = défaut):")

    hidden = safe_int_input(f"  - Canaux cachés (défaut: {DEFAULT_GAT_CONFIG['hidden_channels']}): ",
                            DEFAULT_GAT_CONFIG['hidden_channels'])
    layers = safe_int_input(f"  - Nombre de couches (défaut: {DEFAULT_GAT_CONFIG['num_layers']}): ",
                            DEFAULT_GAT_CONFIG['num_layers'])
    heads = safe_int_input(f"  - Nombre de têtes (défaut: {DEFAULT_GAT_CONFIG['num_heads']}): ",
                           DEFAULT_GAT_CONFIG['num_heads'])

    return {
        'hidden_channels': hidden,
        'num_layers': layers,
        'num_heads': heads
    }


def show_main_menu():
    while True:
        clear_screen()
        print("=" * 70)
        print("    GNN ANOMALY DETECTION + ANOMALY-GUIDED LEARNING")
        print("=" * 70)
        print("\n  [1] Expérience de base (Baselines + GAT + Anomaly-Guided)")
        print("  [2] Expérience custom (Hyperparamètres interactifs)")
        print("  [3] Grid Search complet (MAJ: inclut Anomaly-Guided)")
        print("\n  [r] Revoir les résultats précédents")
        print("  [q] Quitter")

        choice = input("\nVotre choix : ")

        if choice == '1':
            clear_screen()
            print("Lancement expérience de base...")
            run_full_experiment(DEFAULT_GAT_CONFIG, run_name_suffix="BaseRun", include_anomaly_guided=True)
            input("\nTerminé. Appuyez sur 'Entrée'...")

        elif choice == '2':
            custom_config = get_interactive_gat_config()
            clear_screen()
            print("Lancement expérience customisée...")
            run_full_experiment(custom_config, run_name_suffix="CustomRun", include_anomaly_guided=True)
            input("\nTerminé. Appuyez sur 'Entrée'...")

        elif choice == '3':
            clear_screen()
            print("Lancement Grid Search Complet...")

            std_runs = len(list(
                itertools.product(GAT_GRID_SEARCH_CONFIG['hidden_channels'], GAT_GRID_SEARCH_CONFIG['num_layers'],
                                  GAT_GRID_SEARCH_CONFIG['num_heads'], TRAIN_ALPHAS_GRID)))
            ag_runs = len(list(
                itertools.product(GAT_GRID_SEARCH_CONFIG['hidden_channels'], GAT_GRID_SEARCH_CONFIG['num_layers'],
                                  GAT_GRID_SEARCH_CONFIG['num_heads'])))
            total_runs = std_runs + ag_runs

            print(f"\nCela va lancer {std_runs} (Standard GAT) + {ag_runs} (Anomaly-Guided) = {total_runs} exécutions.")
            print("   (Le Grid Search Anomaly-Guided est plus long par exécution)")
            confirm = input("Continuer ? (o/n): ")

            if confirm.lower() == 'o':
                run_grid_search_experiment()
                input("\nTerminé. Appuyez sur 'Entrée'...")
            else:
                print("\nAnnulé.")
                input("Appuyez sur 'Entrée'...")

        elif choice.lower() == 'r':
            review_experiments()

        elif choice.lower() == 'q':
            print("Au revoir !")
            break

        else:
            print("Choix invalide.")
            input("Appuyez sur 'Entrée'...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'review':
        review_experiments()
    else:
        show_main_menu()