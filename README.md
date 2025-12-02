# Projet Final : Data Engineering & Credit Scoring

## 1. Objectif

Vous êtes Data Scientist au sein d'une société financière spécialisée dans les crédits à la consommation destinés à des clients ayant peu ou pas d'historique de prêt. L'entreprise souhaite mettre en place un outil complet de credit scoring permettant :

- d'évaluer automatiquement la probabilité qu'un client rembourse son crédit
- de classer chaque demande en crédit accordé ou crédit refusé
- de s'appuyer sur des données variées (comportementales, financières externes, etc.)

Les données sont disponibles ici : https://www.kaggle.com/c/home-credit-default-risk/data

### 1.1 Votre mission

- **Construire et optimiser un modèle de scoring** qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- **Analyser les features** qui contribuent le plus au modèle (feature importance globale et locale), afin de permettre à un chargé d'études de mieux comprendre le score attribué.
- **Mettre en œuvre une approche MLOps** de bout en bout, du tracking des expérimentations à la pré-production du modèle.

> **Note** : Vous êtes encouragés à utiliser des kernels Kaggle pour l'analyse exploratoire et le feature engineering, mais vous devez les analyser et les adapter à vos besoins.

### 1.2 Composante MLOps attendue

La mise en œuvre doit inclure au minimum :

- Le tracking des expérimentations avec MLflow dans les notebooks d'entraînement
- L'utilisation de l'interface MLflow UI pour visualiser les runs
- Le stockage centralisé des modèles dans un model registry MLflow
- Le test du serving MLflow

### 1.3 Enjeux métiers à intégrer

Deux éléments majeurs doivent être pris en compte :

1. **Le déséquilibre** entre bons et mauvais clients dans le jeu de données
2. **L'asymétrie du coût métier** : un faux négatif (FN) coûte environ **dix fois plus** qu'un faux positif (FP)

### 1.4 Score métier et optimisation du seuil

- Définir un score métier pour comparer les modèles en minimisant le coût des erreurs FN/FP
- Optimiser le seuil de décision (ne pas utiliser le seuil standard de 0.5)
- Conserver les métriques techniques (AUC, accuracy) à titre de contrôle

### 1.5 Méthodologie de modélisation

- Validation croisée obligatoire
- Optimisation des hyperparamètres (GridSearchCV ou équivalent)
- Vigilance sur l'overfitting (AUC > 0.82 doit alerter)

---

## 2. Étapes du Projet

### 2.1 Étape 1 : Préparer, nettoyer et enrichir les données

**Objectif** : Constituer un dataset propre et enrichi, prêt pour l'entraînement.

**Prérequis** :
- Explorer les données brutes
- Vérifier les formats et valeurs manquantes
- Identifier les colonnes clés pour les jointures
- Prendre en compte le déséquilibre des classes

**Recommandations** :
- Charger chaque fichier séparément et inspecter ses colonnes
- Utiliser pandas pour fusionner les jeux de données
- Visualiser la distribution des classes cibles
- Créer de nouvelles features si nécessaire
- Explorer les possibilités d'imputation avant de supprimer des colonnes

**Points de vigilance** :
- Vérifier les doublons
- Analyser l'importance métier avant de supprimer des colonnes
- Documenter et justifier les imputations
- Gérer les duplications lors des fusions
- Encoder en tenant compte du type de modèle (ordinal vs nominal)

**Outils** : pandas, matplotlib, seaborn, scikit-learn, missingno

---

### 2.2 Étape 2 : Traquer les expérimentations avec MLflow

**Objectif** : Des runs visibles dans l'UI MLflow avec les paramètres testés et les scores obtenus.

**Recommandations** :
- Intégrer `mlflow.start_run()` dans vos notebooks
- Logger les métriques et paramètres principaux
- Utiliser `mlflow.autolog()` si compatible
- Activer l'interface avec `mlflow ui`

**Points de vigilance** :
- Utiliser un environnement virtuel pour éviter les conflits
- Annoter les expériences (tags, noms, commentaires)
- Versionner les modèles enregistrés
- Éviter de sauvegarder des fichiers inutiles

**Outils** : MLflow

---

### 2.3 Étape 3 : Modéliser et expérimenter avec plusieurs algorithmes

**Objectif** : Un ou plusieurs modèles entraînés, avec validation croisée et métriques d'évaluation.

**Recommandations** :
- Commencer par des modèles simples (Logistic Regression, Random Forest)
- Comparer avec des modèles plus puissants (XGBoost, LightGBM, MLP)
- Utiliser `StratifiedKFold` pour conserver la distribution des classes
- Documenter clairement les notebooks
- Stocker les scores et hyperparamètres testés

**Points de vigilance** :
- **Toujours utiliser la validation croisée**
- Utiliser des métriques adaptées : AUC-ROC, Recall, F1-score, Coût métier (FN >> FP)
- **Stratifier** pour éviter le biais vers la classe majoritaire
- Gérer le déséquilibre : `class_weight`, SMOTE, etc.

**Outils** : scikit-learn, XGBoost, LightGBM

---

### 2.4 Étape 4 : Optimiser les hyperparamètres et le seuil métier

**Objectif** : Un modèle avec hyperparamètres optimisés et un seuil métier ajusté.

**Recommandations** :
- Utiliser GridSearchCV ou Optuna
- Définir une fonction de coût pondérant les erreurs FN et FP
- Tester différents seuils (0.1 à 0.9)
- Tracer la courbe coût vs. seuil

**Points de vigilance** :
- Ne pas garder le seuil par défaut (0.5) sans justification
- Tracer le score métier en fonction du seuil
- Ne pas optimiser uniquement sur l'AUC ou l'accuracy
- Tester la robustesse du modèle choisi

**Outils** : scikit-learn (GridSearchCV), Optuna

---

## 3. Rendu Final

### 3.1 Structure du dépôt GitHub

```
credit-scoring/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_explainability.ipynb
│   └── 04_mlflow_serving_test.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── model_utils.py
│   ├── metrics.py
│   └── explainability.py
│
├── model/
│   ├── MLmodel
│   ├── conda.yaml
│   └── model.pkl
│
├── reports/
│   ├── rapport_credit_scoring.pdf
│   └── figures/
│       ├── shap_global.png
│       ├── shap_local.png
│       └── courbe_cout_vs_seuil.png
│
└── mlruns/  # (facultatif)
```

### 3.2 Description des fichiers

| Fichier/Dossier | Description |
|-----------------|-------------|
| `README.md` | Résumé du projet, commandes Docker, commande curl pour l'API, seuil métier choisi, structure du dépôt |
| `requirements.txt` | Dépendances Python |
| `Dockerfile` | Configuration pour servir le modèle (port 1234), ne doit PAS réentraîner |
| `.gitignore` | Exclure : `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `data/`, `*.csv`, `mlruns/` |

### 3.3 Notebooks

| Notebook | Contenu |
|----------|---------|
| `01_data_preparation.ipynb` | Chargement, fusion, nettoyage, encodage, split train/test, analyse du déséquilibre |
| `02_model_training.ipynb` | Modèles (baseline + avancés), gestion déséquilibre, validation croisée, tracking MLflow obligatoire, export vers `model/` |
| `03_explainability.ipynb` | SHAP global et local, export des figures |
| `04_mlflow_serving_test.ipynb` | Test de l'API, vérification des prédictions, calcul de métrique |

### 3.4 Code Python (src/)

| Fichier | Contenu |
|---------|---------|
| `data_prep.py` | Fonctions de chargement, jointure, nettoyage, encodage |
| `model_utils.py` | Fonctions d'entraînement, split, sauvegarde |
| `metrics.py` | AUC, précision, rappel, F1, coût métier |
| `explainability.py` | Calcul SHAP, graphiques globaux et locaux |

### 3.5 Rapport PDF (2-3 pages max)

- Démarche de préparation des données et modélisation
- Résultats principaux (AUC, seuil optimal, coût métier)
- Interprétation des variables importantes
- Capture d'écran MLflow montrant les runs et le modèle choisi

---

## 4. Grille d'évaluation (20 points)

| Catégorie | Critères | Points |
|-----------|----------|--------|
| Structure du dépôt | Arborescence conforme, fichiers obligatoires présents | 2 |
| README.md | Contexte métier, installation, commandes Docker, test API, seuil métier | 3 |
| Notebooks | 4 notebooks complets, documentés, reproductibles | 4 |
| Code Python (src/) | Modularité, fonctions propres et réutilisables | 3 |
| Tracking MLflow | Runs complets, modèle dans le registry, versionnage | 3 |
| Modèle + Docker | MLmodel, conda.yaml, model.pkl, Dockerfile fonctionnel | 3 |
| Rapport PDF | Synthèse claire, résultats, interprétabilité, figures | 2 |

---

## 5. Oral de validation (5 minutes)

**Conditions** :
- Ordinateur personnel prêt
- Conteneur Docker lancé et opérationnel
- Serveur MLflow (tracking + UI) démarré
- Serveur de prédiction actif

**Déroulement** :
- Démonstration du modèle en serving via Docker
- Appel API sur un échantillon du jeu de test
- Évaluation de la maîtrise technique et capacité à expliquer les choix

**Décision** :
- Oral validé : note GitHub confirmée
- Oral non validé : note réduite ou invalidée