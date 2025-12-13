# Credit Scoring - Home Credit Default Risk

Système de scoring crédit basé sur l'apprentissage automatique pour prédire le risque de défaut de paiement.

---

## Contexte

En tant que Data Scientist au sein d'une société financière, ce projet vise à développer un outil de **credit scoring** permettant :

### Enjeu Principal

**Un faux négatif (FN) coûte 10× plus qu'un faux positif (FP)**

- **Faux Négatif (FN)** : Accorder un crédit à un client qui fera défaut → Perte de ~10× le montant du prêt
- **Faux Positif (FP)** : Refuser un crédit à un bon client → Perte de 1× profit potentiel

---

## Modèle Champion : LightGBM

Après comparaison de 4 modèles (Logistic Regression, Random Forest, XGBoost, LightGBM), le modèle **LightGBM** a été sélectionné.

### Performances

| Métrique | Valeur |
|----------|--------|
| **Modèle** | LightGBM |
| **AUC** | 0.7793 |
| **Accuracy** | 0.7246 |
| **Precision** | 0.1826 |
| **Recall** | 0.6935 |
| **F1-Score** | 0.2891 |
| **Coût Métier** | **30,600** |
| **Seuil Optimal** | **0.5152** |

### Règle de Décision

```python
if probabilité_défaut >= 0.5152:
    décision = "REFUSER le crédit"  # Risque élevé
else:
    décision = "ACCEPTER le crédit"  # Risque acceptable
```

### Justification du Seuil

Le seuil de **0.5152** a été optimisé pour minimiser le coût métier total :
```
Coût Total = (Faux Négatifs × 10) + (Faux Positifs × 1)
```

Ce seuil représente le meilleur équilibre entre :
- Minimiser les défauts non détectés (FN) qui coûtent cher
- Accepter un nombre raisonnable de faux positifs (FP)

---

## 📊 Comparaison des Modèles

| Modèle | AUC | Coût Métier | Seuil Optimal |
|--------|-----|-------------|---------------|
| **LightGBM** 🏆 | **0.7793** | **30,600** | 0.5152 |
| XGBoost | 0.7695 | 31,411 | 0.5253 |
| Logistic Regression | 0.7684 | 31,714 | 0.5152 |
| Random Forest | 0.7553 | 32,783 | 0.1616 |

**LightGBM** offre le meilleur compromis avec :
---

## 🛠️ Installation

### Prérequis

- Python 3.11+
- Docker
- Git

### Étape 1 : Cloner le Dépôt

```bash
git clone <url-du-repo>
cd "Projet Final"
```

### Étape 2 : Créer un Environnement Virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate 
```

### Étape 3 : Installer les Dépendances

```bash
pip install -r requirements.txt
```

---

##  Utilisation

### 1. Exploration des Données

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 2. Entraînement des Modèles

```bash
# Démarrer MLflow UI (dans un terminal séparé)
mlflow ui

# Ouvrir le notebook d'entraînement
jupyter notebook notebooks/02_model_training.ipynb
```

Visualiser les expériences : http://localhost:5000

### 3. Analyse d'Explicabilité (SHAP)

```bash
jupyter notebook notebooks/03_explainability.ipynb
```

### 4. Test du Serving MLflow

```bash
jupyter notebook notebooks/04_mlflow_serving_test.ipynb
```

---

## 🐳 Déploiement avec Docker

### Construction de l'Image

```bash
docker build -t credit-scoring:latest .
```

### Lancement du Conteneur

```bash
docker run -p 1234:1234 credit-scoring:latest
```

Le serveur d'inférence sera accessible sur `http://localhost:1234`

### Alternative : Docker Compose

```bash
docker-compose up
```

---

## 🔌 Test de l'API

### Commande curl

```bash
curl -X POST http://localhost:1234/invocations \
  -H 'Content-Type: application/json' \
  -d @sample_request.json
```

### Format de la Requête

```json
{
  "dataframe_split": {
    "columns": ["feature1", "feature2", "..."],
    "data": [[valeur1, valeur2, ...]]
  }
}
```

### Réponse Attendue

```json
[0.3456]  # Probabilité de défaut (entre 0 et 1)
```

**Interprétation** :
- Si probabilité < 0.5152 → **Accepter** le crédit
- Si probabilité ≥ 0.5152 → **Refuser** le crédit
---

## 📁 Structure du Projet

```
Projet Final/
│
├── README.md                      # Ce fichier
├── requirements.txt               # Dépendances Python
├── Dockerfile                     # Configuration Docker
├── docker-compose.yml             # Orchestration Docker
├── .gitignore                     # Fichiers à exclure de Git
│
├── notebooks/                     # Notebooks Jupyter
│   ├── 01_data_preparation.ipynb     # Préparation des données
│   ├── 02_model_training.ipynb       # Entraînement des modèles
│   ├── 03_explainability.ipynb       # Analyse SHAP
│   └── 04_mlflow_serving_test.ipynb  # Test du serving
│
├── src/                           # Code source Python
│   ├── __init__.py
│   ├── data_prep.py                  # Fonctions de préparation
│   ├── model_utils.py                # Utilitaires de modélisation
│   ├── metrics.py                    # Métriques métier
│   └── explainability.py             # Fonctions SHAP
│
├── model/                         # Modèle MLflow (LightGBM)
│   ├── MLmodel                       # Métadonnées MLflow
│   ├── conda.yaml                    # Environnement conda
│   ├── model.pkl                     # Modèle sérialisé (371 KB)
│   ├── python_env.yaml               # Environnement Python
│   └── requirements.txt              # Dépendances du modèle
│
├── models/                        # Modèles entraînés (sauvegarde)
│   ├── lightgbm.pkl
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   └── scaler.pkl
│
├── reports/                       # Rapports et visualisations
│   ├── rapport_credit_scoring.pdf    # Rapport final (2-3 pages)
│   ├── model_comparison.csv          # Comparaison des modèles
│   └── figures/                      # Graphiques
│       ├── shap_global.png              # Importance globale
│       ├── shap_local.png               # Importance locale
│       ├── shap_summary.png
│       └── model_comparison.png
│
├── data/                          # Données (non versionnées)
│   └── application_train_prepared.csv
│
└── mlruns/                        # Tracking MLflow (non versionné)
```

---

## 🔬 MLflow - Suivi des Expérimentations

### Démarrer le Serveur MLflow

```bash
mlflow ui
```

Accéder à l'interface : http://localhost:5000


### Modèles Enregistrés

- **Nom** : `credit_scoring_model`
- **Version active** : LightGBM (la plus récente)
- **Run ID** : Consultable dans MLflow UI

---

## 📈 Métriques et Optimisation

### Fonction de Coût Métier

```python
Coût Total = (FN × 10) + (FP × 1)
```

Où :
- **FN** = Nombre de faux négatifs (défauts non détectés)
- **FP** = Nombre de faux positifs (bons clients refusés)

### Stratégie d'Optimisation

1. **Validation croisée** : StratifiedKFold (5 folds)
2. **Gestion du déséquilibre** : `class_weight='balanced'`
3. **Hyperparamètres** : RandomizedSearchCV
4. **Seuil métier** : Optimisation sur fonction de coût

### Déséquilibre des Classes

- **Bons clients (0)** : 91.9%
- **Mauvais clients (1)** : 8.1%
- **Ratio** : 11.39:1


Crédits: IR4 2027 - Thomas Béchu, Noé Guengant, Malo Kerautret
