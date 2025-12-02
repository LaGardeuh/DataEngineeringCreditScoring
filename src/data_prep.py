import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path


def load_data(data_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    Charge tous les fichiers CSV du projet Home Credit Default Risk.

    Args:
        data_dir: Chemin vers le dossier contenant les fichiers CSV

    Returns:
        Dictionnaire contenant tous les DataFrames
    """
    data_path = Path(data_dir)

    dataframes = {}

    # Fichiers principaux
    files = {
        'application_train': 'application_train.csv',
        'application_test': 'application_test.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'credit_card_balance': 'credit_card_balance.csv',
        'installments_payments': 'installments_payments.csv',
        'POS_CASH_balance': 'POS_CASH_balance.csv',
        'previous_application': 'previous_application.csv'
    }

    for name, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            print(f"Chargement de {filename}...")
            dataframes[name] = pd.read_csv(filepath)
            print(f"  Shape: {dataframes[name].shape}")
        else:
            print(f"Attention: {filename} non trouvé dans {data_dir}")

    return dataframes


def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes dans un DataFrame.

    Args:
        df: DataFrame à analyser
        threshold: Seuil de pourcentage de valeurs manquantes pour signaler une colonne

    Returns:
        DataFrame avec les statistiques sur les valeurs manquantes
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })

    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values(
        'missing_percentage', ascending=False
    )

    print(f"\nColonnes avec plus de {threshold*100}% de valeurs manquantes:")
    high_missing = missing_stats[missing_stats['missing_percentage'] > threshold * 100]
    print(high_missing)

    return missing_stats


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon différentes stratégies.

    Args:
        df: DataFrame à traiter
        strategy: Stratégie d'imputation ('median', 'mean', 'mode', 'drop')

    Returns:
        DataFrame avec valeurs manquantes traitées
    """
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    if strategy == 'median':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mean':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'mode':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Pour les variables catégorielles, remplir avec le mode ou une catégorie spéciale
    for col in categorical_cols:
        mode_val = df_clean[col].mode()
        fill_value = mode_val[0] if len(mode_val) > 0 else 'MISSING'
        df_clean[col] = df_clean[col].fillna(fill_value)

    return df_clean


def encode_categorical_variables(df: pd.DataFrame,
                                 encoding_type: str = 'label',
                                 columns: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode les variables catégorielles.

    Args:
        df: DataFrame à encoder
        encoding_type: Type d'encodage ('label', 'onehot')
        columns: Liste des colonnes à encoder (None = toutes les colonnes object)

    Returns:
        Tuple (DataFrame encodé, dictionnaire des encodeurs)
    """
    df_encoded = df.copy()
    encoders = {}

    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    if encoding_type == 'label':
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le

    elif encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)

    return df_encoded, encoders


def create_aggregated_features(main_df: pd.DataFrame,
                               secondary_df: pd.DataFrame,
                               key: str,
                               prefix: str) -> pd.DataFrame:
    """
    Crée des features agrégées à partir d'une table secondaire.

    Args:
        main_df: DataFrame principal
        secondary_df: DataFrame secondaire à agréger
        key: Clé de jointure
        prefix: Préfixe pour les nouvelles colonnes

    Returns:
        DataFrame avec nouvelles features agrégées
    """
    # Colonnes numériques pour l'agrégation
    numeric_cols = secondary_df.select_dtypes(include=[np.number]).columns.tolist()
    if key in numeric_cols:
        numeric_cols.remove(key)

    # Agrégations statistiques
    agg_dict = {col: ['mean', 'max', 'min', 'sum', 'std'] for col in numeric_cols}

    agg_features = secondary_df.groupby(key).agg(agg_dict)
    agg_features.columns = [f'{prefix}_{col[0]}_{col[1]}'.upper() for col in agg_features.columns]
    agg_features = agg_features.reset_index()

    # Ajouter le compte
    count_feature = secondary_df.groupby(key).size().reset_index(name=f'{prefix}_COUNT')
    agg_features = agg_features.merge(count_feature, on=key, how='left')

    # Merger avec le DataFrame principal
    result_df = main_df.merge(agg_features, on=key, how='left')

    return result_df


def merge_all_tables(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Fusionne toutes les tables du dataset Home Credit.

    Args:
        dataframes: Dictionnaire contenant tous les DataFrames

    Returns:
        DataFrame fusionné et enrichi
    """
    # Partir du DataFrame principal
    df_main = dataframes['application_train'].copy()

    print(f"Shape initial: {df_main.shape}")

    # Traiter bureau et bureau_balance
    if 'bureau' in dataframes and 'bureau_balance' in dataframes:
        bureau = dataframes['bureau'].copy()
        bureau_balance = dataframes['bureau_balance'].copy()

        # Agréger bureau_balance
        bureau = create_aggregated_features(bureau, bureau_balance, 'SK_ID_BUREAU', 'BB')

        # Agréger bureau au niveau client
        df_main = create_aggregated_features(df_main, bureau, 'SK_ID_CURR', 'BUREAU')
        print(f"Après bureau: {df_main.shape}")

    # Traiter previous_application
    if 'previous_application' in dataframes:
        prev_app = dataframes['previous_application'].copy()
        df_main = create_aggregated_features(df_main, prev_app, 'SK_ID_CURR', 'PREV')
        print(f"Après previous_application: {df_main.shape}")

    # Traiter installments_payments
    if 'installments_payments' in dataframes:
        installments = dataframes['installments_payments'].copy()
        df_main = create_aggregated_features(df_main, installments, 'SK_ID_CURR', 'INSTAL')
        print(f"Après installments: {df_main.shape}")

    # Traiter credit_card_balance
    if 'credit_card_balance' in dataframes:
        cc_balance = dataframes['credit_card_balance'].copy()
        df_main = create_aggregated_features(df_main, cc_balance, 'SK_ID_CURR', 'CC')
        print(f"Après credit_card: {df_main.shape}")

    # Traiter POS_CASH_balance
    if 'POS_CASH_balance' in dataframes:
        pos_balance = dataframes['POS_CASH_balance'].copy()
        df_main = create_aggregated_features(df_main, pos_balance, 'SK_ID_CURR', 'POS')
        print(f"Après POS_CASH: {df_main.shape}")

    return df_main


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Supprime les doublons du DataFrame.

    Args:
        df: DataFrame à nettoyer
        subset: Liste des colonnes à considérer pour la détection de doublons

    Returns:
        DataFrame sans doublons
    """
    initial_shape = df.shape
    df_clean = df.drop_duplicates(subset=subset)
    removed = initial_shape[0] - df_clean.shape[0]

    if removed > 0:
        print(f"Suppression de {removed} lignes dupliquées")

    return df_clean
