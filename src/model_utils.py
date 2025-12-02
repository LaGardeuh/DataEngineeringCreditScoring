import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Any, Dict
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm


def prepare_train_test_split(df: pd.DataFrame,
                             target_col: str = 'TARGET',
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple:
    """
    Sépare les données en ensembles d'entraînement et de test.

    Args:
        df: DataFrame complet
        target_col: Nom de la colonne cible
        test_size: Proportion de l'ensemble de test
        random_state: Graine aléatoire pour la reproductibilité

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    # Séparer features et target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split stratifié pour préserver la distribution des classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Taille train: {X_train.shape}")
    print(f"Taille test: {X_test.shape}")
    print(f"\nDistribution des classes (train):")
    print(y_train.value_counts(normalize=True))
    print(f"\nDistribution des classes (test):")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame = None) -> Tuple:
    """
    Normalise les features avec StandardScaler.

    Args:
        X_train: Features d'entraînement
        X_test: Features de test (optionnel)

    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    return X_train_scaled, X_test_scaled, scaler


def get_model(model_name: str, class_weight: str = 'balanced', **kwargs) -> Any:
    """
    Retourne un modèle initialisé selon le nom.

    Args:
        model_name: Nom du modèle ('logistic', 'random_forest', 'xgboost', 'lightgbm')
        class_weight: Pondération des classes pour gérer le déséquilibre
        **kwargs: Paramètres additionnels pour le modèle

    Returns:
        Modèle initialisé
    """
    if model_name == 'logistic':
        # Paramètres par défaut
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        # Fusionner avec kwargs (kwargs a la priorité)
        params = {**default_params, **kwargs}
        # Ajouter class_weight seulement s'il n'est pas dans kwargs
        if 'class_weight' not in params:
            params['class_weight'] = class_weight
        return LogisticRegression(**params)

    elif model_name == 'random_forest':
        default_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **kwargs}
        if 'class_weight' not in params:
            params['class_weight'] = class_weight
        return RandomForestClassifier(**params)

    elif model_name == 'xgboost':
        # Pour XGBoost, utiliser scale_pos_weight au lieu de class_weight
        default_params = {
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        params = {**default_params, **kwargs}
        return XGBClassifier(**params)

    elif model_name == 'lightgbm':
        default_params = {
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        params = {**default_params, **kwargs}
        if 'class_weight' not in params:
            params['class_weight'] = class_weight
        return LGBMClassifier(**params)

    else:
        raise ValueError(f"Modèle inconnu: {model_name}")


def train_model_with_cv(model: Any,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        cv_folds: int = 5,
                        scoring: str = 'roc_auc') -> Tuple[Any, np.ndarray]:
    """
    Entraîne un modèle avec validation croisée.

    Args:
        model: Modèle à entraîner
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        cv_folds: Nombre de folds pour la validation croisée
        scoring: Métrique de scoring

    Returns:
        Tuple (modèle entraîné, scores de validation croisée)
    """
    # Validation croisée stratifiée
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Calculer les scores de validation croisée
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring)

    print(f"\nScores de validation croisée ({scoring}):")
    print(f"  Moyenne: {cv_scores.mean():.4f}")
    print(f"  Écart-type: {cv_scores.std():.4f}")
    print(f"  Min: {cv_scores.min():.4f}")
    print(f"  Max: {cv_scores.max():.4f}")

    # Entraîner le modèle sur tout l'ensemble d'entraînement
    model.fit(X_train, y_train)

    return model, cv_scores


def train_and_log_model(model_name: str,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       params: Dict = None,
                       experiment_name: str = "credit_scoring") -> Any:
    """
    Entraîne un modèle et log les résultats dans MLflow.

    Args:
        model_name: Nom du modèle
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        params: Paramètres du modèle
        experiment_name: Nom de l'expérience MLflow

    Returns:
        Modèle entraîné
    """
    # Créer ou récupérer l'expérience
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        # Logger les paramètres
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        if params:
            mlflow.log_params(params)

        # Initialiser et entraîner le modèle
        model = get_model(model_name, **(params or {}))
        model, cv_scores = train_model_with_cv(model, X_train, y_train)

        # Logger les scores de validation croisée
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())

        # Prédictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculer et logger les métriques
        from src.metrics import calculate_all_metrics

        train_metrics = calculate_all_metrics(y_train, y_pred_train, y_proba_train)
        test_metrics = calculate_all_metrics(y_test, y_pred_test, y_proba_test)

        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Logger le modèle
        if model_name in ['xgboost']:
            mlflow.xgboost.log_model(model, "model")
        elif model_name in ['lightgbm']:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"\n✓ Modèle {model_name} entraîné et loggé dans MLflow")
        print(f"  AUC (test): {test_metrics['auc_roc']:.4f}")
        print(f"  Coût métier (test): {test_metrics['business_cost']:.2f}")

    return model


def save_model(model: Any, filepath: str):
    """
    Sauvegarde un modèle sur le disque.

    Args:
        model: Modèle à sauvegarder
        filepath: Chemin de sauvegarde
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Modèle sauvegardé dans {filepath}")


def load_model(filepath: str) -> Any:
    """
    Charge un modèle depuis le disque.

    Args:
        filepath: Chemin du modèle

    Returns:
        Modèle chargé
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"Modèle chargé depuis {filepath}")
    return model


def get_feature_importance(model: Any,
                          feature_names: list,
                          top_n: int = 20) -> pd.DataFrame:
    """
    Extrait l'importance des features d'un modèle.

    Args:
        model: Modèle entraîné
        feature_names: Liste des noms de features
        top_n: Nombre de top features à retourner

    Returns:
        DataFrame avec les features et leur importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Ce modèle ne fournit pas d'importance de features")
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_n)
