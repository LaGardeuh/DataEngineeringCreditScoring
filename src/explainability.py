import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Any, List
from pathlib import Path


def explain_model_shap(model: Any,
                       X_data: pd.DataFrame,
                       sample_size: int = 100) -> shap.Explainer:
    """
    Crée un explainer SHAP pour le modèle.

    Args:
        model: Modèle entraîné
        X_data: Données pour l'explication
        sample_size: Nombre d'échantillons pour l'explication (pour accélérer)

    Returns:
        Explainer SHAP
    """
    # Échantillonner les données si nécessaire
    if len(X_data) > sample_size:
        X_sample = X_data.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_data

    # Créer l'explainer selon le type de modèle
    model_type = type(model).__name__

    if 'XGB' in model_type or 'LGBM' in model_type:
        explainer = shap.TreeExplainer(model)
    elif 'RandomForest' in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        # Pour les modèles linéaires ou autres
        explainer = shap.Explainer(model.predict, X_sample)

    print(f"Explainer SHAP créé pour {model_type}")

    return explainer


def calculate_shap_values(explainer: shap.Explainer,
                         X_data: pd.DataFrame) -> np.ndarray:
    """
    Calcule les valeurs SHAP pour les données.

    Args:
        explainer: Explainer SHAP
        X_data: Données à expliquer

    Returns:
        Valeurs SHAP
    """
    shap_values = explainer(X_data)

    return shap_values


def plot_shap_summary(shap_values: np.ndarray,
                     X_data: pd.DataFrame,
                     max_display: int = 20,
                     save_path: str = None):
    """
    Affiche le graphique SHAP summary (importance globale).

    Args:
        shap_values: Valeurs SHAP
        X_data: Données
        max_display: Nombre maximum de features à afficher
        save_path: Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, max_display=max_display, show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure SHAP summary sauvegardée dans {save_path}")

    plt.show()


def plot_shap_bar(shap_values: np.ndarray,
                 max_display: int = 20,
                 save_path: str = None):
    """
    Affiche le graphique SHAP bar (importance moyenne absolue).

    Args:
        shap_values: Valeurs SHAP
        max_display: Nombre maximum de features à afficher
        save_path: Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure SHAP bar sauvegardée dans {save_path}")

    plt.show()


def plot_shap_waterfall(shap_values: np.ndarray,
                       index: int = 0,
                       save_path: str = None):
    """
    Affiche le graphique SHAP waterfall pour une prédiction individuelle.

    Args:
        shap_values: Valeurs SHAP
        index: Index de l'observation à expliquer
        save_path: Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[index], show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure SHAP waterfall sauvegardée dans {save_path}")

    plt.show()


def plot_shap_force(shap_values: np.ndarray,
                   index: int = 0,
                   save_path: str = None):
    """
    Affiche le graphique SHAP force plot pour une prédiction individuelle.

    Args:
        shap_values: Valeurs SHAP
        index: Index de l'observation à expliquer
        save_path: Chemin pour sauvegarder la figure
    """
    # Force plot pour une observation
    shap.plots.force(shap_values[index], matplotlib=True, show=False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure SHAP force sauvegardée dans {save_path}")

    plt.show()


def get_top_features_for_client(shap_values: np.ndarray,
                                X_data: pd.DataFrame,
                                client_index: int,
                                top_n: int = 10) -> pd.DataFrame:
    """
    Extrait les top features qui influencent la prédiction d'un client.

    Args:
        shap_values: Valeurs SHAP
        X_data: Données
        client_index: Index du client
        top_n: Nombre de top features à retourner

    Returns:
        DataFrame avec les features et leur impact SHAP
    """
    # Extraire les valeurs SHAP pour ce client
    client_shap = shap_values[client_index].values
    client_data = X_data.iloc[client_index]

    # Créer un DataFrame
    feature_impact = pd.DataFrame({
        'feature': X_data.columns,
        'value': client_data.values,
        'shap_value': client_shap,
        'abs_shap_value': np.abs(client_shap)
    })

    # Trier par impact absolu
    feature_impact = feature_impact.sort_values('abs_shap_value', ascending=False)

    return feature_impact.head(top_n)


def explain_prediction(model: Any,
                      explainer: shap.Explainer,
                      X_data: pd.DataFrame,
                      client_index: int,
                      top_n: int = 10):
    """
    Explique la prédiction pour un client spécifique.

    Args:
        model: Modèle entraîné
        explainer: Explainer SHAP
        X_data: Données
        client_index: Index du client à expliquer
        top_n: Nombre de top features à afficher
    """
    # Prédiction
    prediction = model.predict(X_data.iloc[[client_index]])[0]
    proba = model.predict_proba(X_data.iloc[[client_index]])[0, 1]

    print("="*60)
    print(f"EXPLICATION DE LA PRÉDICTION - Client {client_index}")
    print("="*60)
    print(f"Prédiction: {'Défaut' if prediction == 1 else 'Pas de défaut'}")
    print(f"Probabilité de défaut: {proba:.4f}")
    print("="*60)

    # Calculer les valeurs SHAP
    shap_values = explainer(X_data.iloc[[client_index]])

    # Top features
    top_features = get_top_features_for_client(shap_values, X_data, 0, top_n)

    print(f"\nTop {top_n} features influençant la prédiction:")
    print("-"*60)
    for idx, row in top_features.iterrows():
        direction = "↑ Augmente" if row['shap_value'] > 0 else "↓ Diminue"
        print(f"{row['feature']:30s} = {row['value']:10.2f}")
        print(f"  Impact SHAP: {row['shap_value']:+.4f} {direction} le risque")
        print()

    return shap_values, top_features


def create_explainability_report(model: Any,
                                 X_train: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 sample_indices: List[int] = [0, 1, 2],
                                 output_dir: str = 'reports/figures'):
    """
    Crée un rapport complet d'explicabilité avec graphiques.

    Args:
        model: Modèle entraîné
        X_train: Données d'entraînement
        X_test: Données de test
        sample_indices: Indices des clients à expliquer
        output_dir: Dossier de sortie pour les figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Création du rapport d'explicabilité...")

    # Créer l'explainer
    explainer = explain_model_shap(model, X_train, sample_size=100)

    # Calculer les valeurs SHAP sur le test set
    print("\nCalcul des valeurs SHAP...")
    shap_values = calculate_shap_values(explainer, X_test)

    # 1. Summary plot (global)
    print("\nGénération du summary plot...")
    plot_shap_summary(
        shap_values,
        X_test,
        save_path=output_path / 'shap_global.png'
    )

    # 2. Bar plot (global)
    print("\nGénération du bar plot...")
    plot_shap_bar(
        shap_values,
        save_path=output_path / 'shap_bar.png'
    )

    # 3. Explication locale pour quelques clients
    for idx in sample_indices:
        if idx < len(X_test):
            print(f"\nExplication locale pour le client {idx}...")

            # Waterfall plot
            plot_shap_waterfall(
                shap_values,
                index=idx,
                save_path=output_path / f'shap_local_client_{idx}.png'
            )

            # Explication détaillée
            explain_prediction(model, explainer, X_test, idx)

    print(f"\n✓ Rapport d'explicabilité créé dans {output_dir}")
