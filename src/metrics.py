import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve
)
from typing import Tuple, Dict
import matplotlib.pyplot as plt


def calculate_business_cost(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           fn_cost: float = 1.0,
                           fp_cost: float = 10.0) -> float:
    """
    Calcule le coût métier basé sur les erreurs FN et FP.

    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        fn_cost: Coût d'un faux négatif (défaut: 1)
        fp_cost: Coût d'un faux positif (défaut: 10)

    Returns:
        Coût métier total
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (fn * fn_cost) + (fp * fp_cost)

    return total_cost


def calculate_all_metrics(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Calcule toutes les métriques techniques et métier.

    Args:
        y_true: Vraies étiquettes
        y_pred_proba: Probabilités prédites (entre 0 et 1)
        threshold: Seuil de décision (défaut: 0.5)

    Returns:
        Dictionnaire avec toutes les métriques
    """
    metrics = {}

    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Métriques techniques
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # Coût métier
    metrics['business_cost'] = calculate_business_cost(y_true, y_pred)

    return metrics


def find_optimal_threshold(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           fn_cost: float = 1.0,
                           fp_cost: float = 10.0) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Trouve le seuil optimal qui minimise le coût métier.

    Args:
        y_true: Vraies étiquettes
        y_proba: Probabilités prédites
        fn_cost: Coût d'un faux négatif
        fp_cost: Coût d'un faux positif

    Returns:
        Tuple (seuil optimal, coût minimal, array des seuils, array des coûts)
    """
    # Générer différents seuils
    thresholds = np.linspace(0.0, 1.0, 100)
    costs = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost = calculate_business_cost(y_true, y_pred, fn_cost, fp_cost)
        costs.append(cost)

    costs = np.array(costs)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    minimal_cost = costs[optimal_idx]

    return optimal_threshold, minimal_cost, thresholds, costs


def optimize_threshold(y_true: np.ndarray,
                      y_proba: np.ndarray,
                      metric: str = 'f1',
                      fn_cost: float = 1.0,
                      fp_cost: float = 10.0) -> float:
    """
    Trouve le seuil optimal selon la métrique spécifiée.

    Args:
        y_true: Vraies étiquettes
        y_proba: Probabilités prédites
        metric: Métrique à optimiser ('f1', 'business_cost', 'precision', 'recall')
        fn_cost: Coût d'un faux négatif (pour business_cost)
        fp_cost: Coût d'un faux positif (pour business_cost)

    Returns:
        Seuil optimal
    """
    if metric == 'business_cost':
        optimal_threshold, _, _, _ = find_optimal_threshold(y_true, y_proba, fn_cost, fp_cost)
        return optimal_threshold

    # Pour les autres métriques
    thresholds = np.linspace(0.0, 1.0, 100)
    best_score = -np.inf if metric != 'business_cost' else np.inf
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    return best_threshold


def plot_cost_vs_threshold(thresholds: np.ndarray,
                           costs: np.ndarray,
                           optimal_threshold: float,
                           save_path: str = None):
    """
    Trace la courbe du coût métier en fonction du seuil.

    Args:
        thresholds: Array des seuils testés
        costs: Array des coûts correspondants
        optimal_threshold: Seuil optimal à marquer
        save_path: Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, linewidth=2, label='Coût métier')
    plt.axvline(optimal_threshold, color='red', linestyle='--',
                label=f'Seuil optimal = {optimal_threshold:.3f}')
    plt.xlabel('Seuil de décision')
    plt.ylabel('Coût métier total')
    plt.title('Optimisation du seuil de décision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")

    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str = None):
    """
    Trace la matrice de confusion.

    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        save_path: Chemin pour sauvegarder la figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Bon client (0)', 'Mauvais client (1)'])
    plt.yticks(tick_marks, ['Bon client (0)', 'Mauvais client (1)'])

    # Ajouter les valeurs dans les cellules
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")

    plt.show()


def plot_roc_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   save_path: str = None):
    """
    Trace la courbe ROC.

    Args:
        y_true: Vraies étiquettes
        y_proba: Probabilités prédites
        save_path: Chemin pour sauvegarder la figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée dans {save_path}")

    plt.show()


def print_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray):
    """
    Affiche un rapport de classification détaillé.

    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
    """
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    print(classification_report(y_true, y_pred,
                                target_names=['Bon client', 'Mauvais client']))

    metrics = calculate_all_metrics(y_true, y_pred)

    print("\n" + "="*50)
    print("MÉTRIQUES DÉTAILLÉES")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    print("\n" + "="*50)
    print("MATRICE DE CONFUSION")
    print("="*50)
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']} (coût: {metrics['false_positives'] * 10})")
    print(f"False Negatives: {metrics['false_negatives']} (coût: {metrics['false_negatives'] * 1})")
    print(f"True Positives:  {metrics['true_positives']}")

    print("\n" + "="*50)
    print("COÛT MÉTIER")
    print("="*50)
    print(f"Coût total: {metrics['business_cost']:.2f}")
    print("="*50 + "\n")
