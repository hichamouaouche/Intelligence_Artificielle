# Guide de compréhension du projet

## Vue d’ensemble
Ce projet est un notebook de data science orienté intelligence artificielle appliqué à la détection de fraude bancaire. L’objectif principal est de comparer plusieurs approches de classification sur un jeu de données très déséquilibré, puis d’évaluer les modèles avec des métriques adaptées, de calibrer les probabilités et d’expliquer les prédictions avec SHAP.

Le fichier principal est `allworkmodels.ipynb`. Le fichier `allworkmodels.py` reprend la même logique sous forme de script Python exporté.

## Problème traité
Le projet travaille sur le dataset Credit Card Fraud Detection. La tâche consiste à prédire si une transaction est frauduleuse ou non à partir de variables numériques déjà prétraitées. Le défi central n’est pas seulement la classification, mais surtout le déséquilibre extrême entre les classes, avec très peu de fraudes par rapport aux transactions légitimes.

## Structure du pipeline
Le notebook suit quatre grandes étapes:

1. Analyse exploratoire et préparation des données.
2. Entraînement et comparaison de plusieurs modèles.
3. Évaluation avancée et calibration des probabilités.
4. Interprétabilité des prédictions avec SHAP.

## Étape 1: préparation des données
Le projet commence par le chargement des bibliothèques de manipulation et de machine learning. Les données sont lues depuis `data/creditcard.csv`.

Les premiers contrôles portent sur:
- la forme du jeu de données,
- la distribution des classes,
- le pourcentage de fraude,
- la présence éventuelle de valeurs manquantes.

Ensuite, le notebook explore la colinéarité entre variables avec une matrice de corrélation et le VIF. La variable cible est `Class`.

Les étapes de préparation incluent:
- la standardisation de `Time` et `Amount`,
- le split train/test avec stratification,
- un traitement du déséquilibre par SMOTE,
- un traitement du déséquilibre par ADASYN,
- une comparaison entre aucune correction, `class_weight='balanced'`, SMOTE et ADASYN.

Le projet ajoute aussi une visualisation claire des résultats de ces approches avec `seaborn` et `matplotlib` pour comparer les scores des métriques.

## Étape 2: modèles entraînés
Trois familles de modèles sont testées.

### Régression logistique
Une régression logistique avec pénalité Elastic Net sert de baseline. Elle est entraînée avec pondération des classes pour mieux gérer le déséquilibre.

### Random Forest
Un Random Forest est utilisé avec pondération des classes. Le notebook ajoute aussi une analyse de proximité entre observations pour détecter des points atypiques, ce qui donne un score d’outlier basé sur la co-occurrence dans les feuilles terminales.

### XGBoost
XGBoost est le modèle le plus travaillé:
- une optimisation bayésienne avec Optuna cherche de bons hyperparamètres,
- un second entraînement utilise une fonction de perte asymétrique pour pénaliser davantage les faux négatifs,
- le projet compare ces variantes pour voir laquelle détecte le mieux les fraudes.

## Étape 3: évaluation
Le projet n’utilise pas seulement l’accuracy, qui serait trompeuse sur un dataset aussi déséquilibré. Il compare plutôt:
- F1-Macro,
- MCC,
- AUPRC,
- matrices de confusion,
- courbes Precision-Recall.

Une partie importante est aussi la calibration des probabilités. Le notebook compare les prédictions avant et après calibration avec:
- sigmoid / Platt Scaling,
- isotonic regression.

L’idée est de rendre les probabilités plus fiables pour un usage critique.

## Étape 4: interprétabilité
La dernière partie utilise SHAP sur le modèle XGBoost optimisé. Le notebook produit:
- un summary plot global,
- un bar plot d’importance des variables,
- une explication locale d’une transaction frauduleuse,
- des dependence plots pour les variables les plus influentes.

Cette étape sert à comprendre quelles variables poussent une prédiction vers fraude ou non-fraude.

## Dépendances et exécution
Le projet est configuré via `pyproject.toml` et s’exécute avec `uv`.

Commande d’installation:
```bash
uv sync
```

Commande pour ouvrir le notebook:
```bash
uv run jupyter lab allworkmodels.ipynb
```

Le projet utilise notamment:
- pandas,
- matplotlib,
- jupyterlab,
- kagglehub,
- scikit-learn,
- imbalanced-learn,
- xgboost,
- optuna,
- shap,
- statsmodels,
- seaborn,
- numpy.

## Point important
Le jeu de données n’est pas inclus dans le dépôt. Le notebook attend un fichier local `data/creditcard.csv`. Le téléchargement depuis Kaggle est mentionné dans le notebook, mais la cellule de téléchargement est commentée.

## Ce qu’il faut retenir
En résumé, ce projet montre comment construire une chaîne complète de détection de fraude:
- préparation des données,
- gestion du déséquilibre,
- comparaison de modèles,
- visualisation des résultats avec seaborn et matplotlib,
- optimisation des hyperparamètres,
- calibration des probabilités,
- interprétation des résultats.

C’est donc un projet plus proche d’une étude méthodologique que d’une simple démonstration de classification.
