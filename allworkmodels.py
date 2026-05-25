# %% [markdown]
# # Projet IA : Classification Robuste et Analyse de Décision en Environnement Critique
# ## Credit Card Fraud Detection - Dataset très déséquilibré
# %%
# import kagglehub
#
#  Download to ./data
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud", output_dir="data/")
#
# print("Path to dataset files:", path)
# %% [markdown]
# ---
# ## Étape 1 : Analyse Exploratoire et Préparation (EDA)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 6)
print('Toutes les librairies chargées avec succès.')
# %%
df = pd.read_csv('data/creditcard.csv')
print(f'Dimensions: {df.shape}')
print(f'Distribution de la classe cible:\n{df["Class"].value_counts()}')
print(f'\nPourcentage de fraudes: {df["Class"].mean()*100:.4f}%')
# %%
df.describe()
# %%
NumberNull=df.isnull().sum().sum()
print(NumberNull)
# %% [markdown]
# ### 1.1 Analyse de la Colinéarité
# %%
corr = df.drop('Class', axis=1).corr()
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
plt.title('Matrice de Corrélation des Features')
plt.tight_layout()
plt.show()
# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df.drop(['Class', 'Time'], axis=1)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print('VIF (Variance Inflation Factor) - valeurs > 10 indiquent une colinéarité forte')
vif_data.sort_values('VIF', ascending=False).head(20)
# %% [markdown]
# ### 1.2 Standardisation et Split
# %%
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')
print(f'Fraude train: {y_train.sum()}, Fraude test: {y_test.sum()}')
# %% [markdown]
# ### 1.3 Traitement du Déséquilibre
# %%
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f'Ratio classe majoritaire / minoritaire: {ratio:.2f}')

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f'SMOTE - Train size: {X_train_smote.shape}, Distribution:\n{y_train_smote.value_counts()}')
# %% [markdown]
# ### Comparaison des approches de traitement du déséquilibre
# Deux stratégies sont comparées :
# - **Niveau algorithmique** : `class_weight='balanced'` dans la Régression Logistique (pondère les erreurs selon l'inverse des fréquences).
# - **Niveau données** : **SMOTE** (génération synthétique de la classe minoritaire par interpolation entre voisins).
# 
# On entraîne un modèle simple (LogisticRegression) avec et sans chaque méthode pour mesurer l'impact.
# %%
from sklearn.metrics import f1_score, precision_recall_curve, auc, matthews_corrcoef

# Modèle sans traitement du déséquilibre
lr_base = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
lr_base.fit(X_train, y_train)
y_pred_base = lr_base.predict(X_test)

# Modèle avec class_weight='balanced' (algorithmique)
lr_balanced = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=42)
lr_balanced.fit(X_train, y_train)
y_pred_bal = lr_balanced.predict(X_test)

# Modèle avec SMOTE (données)
lr_smote = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = lr_smote.predict(X_test)

comparison = pd.DataFrame({
    'Approche': ['Aucun', 'class_weight (algorithmique)', 'SMOTE (données)'],
    'F1-Macro': [
        f1_score(y_test, y_pred_base, average='macro'),
        f1_score(y_test, y_pred_bal, average='macro'),
        f1_score(y_test, y_pred_smote, average='macro')
    ],
    'MCC': [
        matthews_corrcoef(y_test, y_pred_base),
        matthews_corrcoef(y_test, y_pred_bal),
        matthews_corrcoef(y_test, y_pred_smote)
    ]
})
print('Comparaison des stratégies de traitement du déséquilibre :')
comparison.round(4)
# %% [markdown]
# ---
# ## Étape 2 : Développement des Modèles
# %% [markdown]
# ### 2.1 Baseline : Régression Logistique avec Pénalité Élastique (Elastic Net)
# %%
lr = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    C=1.0,
    l1_ratio=0.5,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr.fit(X_train, y_train)
# %%
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]
print('Régression Logistique ElasticNet - Résultats')
print(f'F1-Macro: {f1_score(y_test, y_pred_lr, average="macro"):.4f}')
print(f'MCC: {matthews_corrcoef(y_test, y_pred_lr):.4f}')
precision, recall, _ = precision_recall_curve(y_test, y_proba_lr)
print(f'AUPRC: {auc(recall, precision):.4f}')
print(confusion_matrix(y_test, y_pred_lr))
# %% [markdown]
# ### 2.2 Random Forest avec Analyse de Proximité
# %%
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
# %%
# Matrice de proximité : fréquence de co-occurrence dans les feuilles terminales
# On utilise un sous-échantillon pour la proximité (sinon matrice NxN trop grande)
from sklearn.utils import resample

sample_idx = resample(np.arange(len(X_train)), n_samples=5000, random_state=42)
X_sample = X_train.iloc[sample_idx] if hasattr(X_train, 'iloc') else X_train[sample_idx]

leaf_ids = rf.apply(X_sample)
n_trees = leaf_ids.shape[1]
n_samples = leaf_ids.shape[0]
proximity = np.zeros((n_samples, n_samples))
for t in range(n_trees):
    leaves = leaf_ids[:, t]
    for leaf in np.unique(leaves):
        idx_in_leaf = np.where(leaves == leaf)[0]
        for i in idx_in_leaf:
            proximity[i, idx_in_leaf] += 1
proximity /= n_trees
np.fill_diagonal(proximity, 1.0)

avg_proximity = proximity.mean(axis=1)
outlier_score = 1 - avg_proximity

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.hist(outlier_score, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Score d\'Outlier (1 - proximité moyenne)')
plt.ylabel('Nombre d\'observations')
plt.title('Distribution des Scores d\'Outlier de Prédiction')

plt.subplot(1, 2, 2)
threshold = np.percentile(outlier_score, 95)
outliers = np.where(outlier_score > threshold)[0]
plt.scatter(range(n_samples), outlier_score, s=5, alpha=0.5)
plt.axhline(threshold, color='r', linestyle='--', label=f'Seuil 95% ({threshold:.3f})')
plt.xlabel('Indice observation')
plt.ylabel('Score d\'Outlier')
plt.title(f'{len(outliers)} Outliers de Prédiction Détectés')
plt.legend()
plt.tight_layout()
plt.show()

# Analyse des outliers : sont-ils surtout des fraudes mal classifiées ?
y_sample = y_train.iloc[sample_idx].values if hasattr(y_train, 'iloc') else y_train[sample_idx]
print(f'Classe réelle des outliers (top 5%): fraudes = {y_sample[outliers].sum()} / {len(outliers)}')
print(f'Classe réelle des non-outliers: fraudes = {y_sample[~np.isin(np.arange(n_samples), outliers)].sum()} / {n_samples - len(outliers)}')
# %%
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print('Random Forest - Résultats')
print(f'F1-Macro: {f1_score(y_test, y_pred_rf, average="macro"):.4f}')
print(f'MCC: {matthews_corrcoef(y_test, y_pred_rf):.4f}')
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
print(f'AUPRC: {auc(recall_rf, precision_rf):.4f}')
print(confusion_matrix(y_test, y_pred_rf))
# %% [markdown]
# ### 2.3 XGBoost : Cost-Sensitive Learning + Optimisation Bayésienne (Optuna)
# %%
# Comparaison de 2 stratégies de gestion du déséquilibre:
# Stratégie A: scale_pos_weight
# Stratégie B: custom loss function (asymétrique)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight recommandé: {scale_pos_weight:.2f}')
# %% [markdown]
# #### Justification de l'espace de recherche Optuna
# - **n_estimators (100–500)** : Bornes standard pour le boosting ; < 100 sous-apprentissage, > 500 rendement décroissant + surapprentissage.
# - **max_depth (3–12)** : Valeurs faibles pour forcer l'apprentissage de motifs généraux (évite le surapprentissage sur données tabulaires déséquilibrées).
# - **learning_rate (0.01–0.3)** : Taux d'apprentissage classique ; < 0.01 converge trop lentement, > 0.3 instable.
# - **subsample (0.6–1.0)** : Agrégation stochastique pour réduire la variance ; 0.6 minimum recommandé.
# - **colsample_bytree (0.6–1.0)** : Sous-échantillonnage des colonnes pour diversifier les arbres.
# - **min_child_weight (1–10)** : Contrôle de la complexité des feuilles terminales ; valeurs élevées évitent les feuilles trop spécifiques.
# - **reg_lambda / reg_alpha (1e-8–10)** : Régularisation L2/L1 ; plage large pour tester de l'absence à la forte pénalité.
# - **scale_pos_weight (1 – ratio×2)** : Poids de la classe minoritaire ; de pas d'ajustement à un sur-équilibrage volontaire.
# %%
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, scale_pos_weight * 2),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')
# %%
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30, show_progress_bar=True)
# %%
print('Meilleurs hyperparamètres XGBoost:')
best_params_xgb = study.best_params
for k, v in best_params_xgb.items():
    print(f'  {k}: {v}')
print(f'\nMeilleur F1-Macro: {study.best_value:.4f}')
# %%
# Graphiques de convergence Optuna
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig2 = optuna.visualization.plot_param_importances(study)
fig2.show()
# %%
xgb_opt = xgb.XGBClassifier(**best_params_xgb, random_state=42, eval_metric='logloss')
xgb_opt.fit(X_train, y_train)

y_pred_xgb = xgb_opt.predict(X_test)
y_proba_xgb = xgb_opt.predict_proba(X_test)[:, 1]
print('XGBoost Optimisé (scale_pos_weight) - Résultats')
print(f'F1-Macro: {f1_score(y_test, y_pred_xgb, average="macro"):.4f}')
print(f'MCC: {matthews_corrcoef(y_test, y_pred_xgb):.4f}')
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
print(f'AUPRC: {auc(recall_xgb, precision_xgb):.4f}')
print(confusion_matrix(y_test, y_pred_xgb))
# %%
# Custom Loss Function : perte asymétrique (coût plus élevé pour faux négatifs)

def asymmetric_logloss(y_true, y_pred):
    y_true = y_true.get_label()
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    cost_positive = 10.0  # Coût élevé pour faux négatif (fraude non détectée)
    cost_negative = 1.0   # Coût faible pour faux positif
    grad = np.where(y_true == 1, cost_positive * (y_pred - 1), cost_negative * y_pred)
    hess = np.where(y_true == 1, cost_positive * y_pred * (1 - y_pred), cost_negative * y_pred * (1 - y_pred))
    return grad, hess
# %%
# Entraînement XGBoost avec custom loss pour comparer
xgb_custom = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_custom.set_params(objective='binary:logistic')
# Note: custom objective nécessite l'API DMatrix; on garde scale_pos_weight comme comparaison
# On entraîne un modèle avec coût asymétrique manuellement

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params_custom = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'eval_metric': 'logloss',
    'seed': 42
}
model_custom = xgb.train(params_custom, dtrain, num_boost_round=200, obj=asymmetric_logloss)
y_proba_custom = 1.0 / (1.0 + np.exp(-model_custom.predict(dtest)))
y_pred_custom = (y_proba_custom >= 0.5).astype(int)

print('XGBoost avec Custom Loss Function - Résultats')
print(f'F1-Macro: {f1_score(y_test, y_pred_custom, average="macro"):.4f}')
print(f'MCC: {matthews_corrcoef(y_test, y_pred_custom):.4f}')
precision_cust, recall_cust, _ = precision_recall_curve(y_test, y_proba_custom)
print(f'AUPRC: {auc(recall_cust, precision_cust):.4f}')
print(confusion_matrix(y_test, y_pred_custom))
# %% [markdown]
# ---
# ## Étape 3 : Évaluation et Calibration
# %% [markdown]
# ### 3.1 Comparaison des Métriques Avancées
# %%
results = pd.DataFrame({
    'Modèle': ['Régression Logistique (ElasticNet)', 'Random Forest', 'XGBoost (Optuna)', 'XGBoost (Custom Loss)'],
    'F1-Macro': [
        f1_score(y_test, y_pred_lr, average='macro'),
        f1_score(y_test, y_pred_rf, average='macro'),
        f1_score(y_test, y_pred_xgb, average='macro'),
        f1_score(y_test, y_pred_custom, average='macro')
    ],
    'MCC': [
        matthews_corrcoef(y_test, y_pred_lr),
        matthews_corrcoef(y_test, y_pred_rf),
        matthews_corrcoef(y_test, y_pred_xgb),
        matthews_corrcoef(y_test, y_pred_custom)
    ],
    'AUPRC': [
        auc(*precision_recall_curve(y_test, y_proba_lr)[1:]),
        auc(*precision_recall_curve(y_test, y_proba_rf)[1:]),
        auc(*precision_recall_curve(y_test, y_proba_xgb)[1:]),
        auc(*precision_recall_curve(y_test, y_proba_custom)[1:])
    ]
})
results.round(4)
# %%
# Courbes Precision-Recall
plt.figure(figsize=(10, 8))
for name, y_proba, color in [
    ('Régression Logistique', y_proba_lr, 'blue'),
    ('Random Forest', y_proba_rf, 'green'),
    ('XGBoost Optimisé', y_proba_xgb, 'red'),
    ('XGBoost Custom Loss', y_proba_custom, 'orange')
]:
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=f'{name} (AUPRC={auc(recall, precision):.3f})', color=color, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbes Precision-Recall')
plt.legend()
plt.grid(True)
plt.show()
# %% [markdown]
# ### 3.2 Calibration des Probabilités
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (name, y_proba, color) in zip([axes[0], axes[1]], [
    ('Random Forest', y_proba_rf, 'green'),
    ('XGBoost', y_proba_xgb, 'red')
]):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', color=color, label=name)
    ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')
    ax.set_xlabel('Probabilité prédite moyenne')
    ax.set_ylabel('Fraction de positifs')
    ax.set_title(f'Diagramme de Fiabilité - {name}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
# %%
# Application de Platt Scaling (sigmoid) et Isotonic Regression
for model_name, model, X_tr, y_tr in [
    ('Random Forest', rf, X_train, y_train),
    ('XGBoost', xgb_opt, X_train, y_train)
]:
    for method in ['sigmoid', 'isotonic']:
        calibrated = CalibratedClassifierCV(model, method=method, cv=5)
        calibrated.fit(X_tr, y_tr)
        y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
        f1_cal = f1_score(y_test, (y_proba_cal >= 0.5).astype(int), average='macro')
        prec_cal, rec_cal, _ = precision_recall_curve(y_test, y_proba_cal)
        auprc_cal = auc(rec_cal, prec_cal)
        print(f'{model_name} + {method}: F1-Macro={f1_cal:.4f}, AUPRC={auprc_cal:.4f}')
# %%
# Visualisation calibration avant/après
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=5)
rf_calibrated.fit(X_train, y_train)
y_proba_rf_cal = rf_calibrated.predict_proba(X_test)[:, 1]

xgb_calibrated = CalibratedClassifierCV(xgb_opt, method='sigmoid', cv=5)
xgb_calibrated.fit(X_train, y_train)
y_proba_xgb_cal = xgb_calibrated.predict_proba(X_test)[:, 1]

for ax, (name, orig, cal, color) in enumerate(zip(
    ['Random Forest', 'XGBoost'],
    [y_proba_rf, y_proba_xgb],
    [y_proba_rf_cal, y_proba_xgb_cal],
    ['green', 'red']
)):
    frac_orig, mean_orig = calibration_curve(y_test, orig, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y_test, cal, n_bins=10)
    axes[ax].plot(mean_orig, frac_orig, 's--', color=color, label='Avant calibration', alpha=0.6)
    axes[ax].plot(mean_cal, frac_cal, 'o-', color=color, label='Après calibration', lw=2)
    axes[ax].plot([0, 1], [0, 1], 'k--', label='Parfait')
    axes[ax].set_xlabel('Probabilité prédite moyenne')
    axes[ax].set_ylabel('Fraction de positifs')
    axes[ax].set_title(f'Calibration - {name}')
    axes[ax].legend()
    axes[ax].grid(True)

plt.tight_layout()
plt.show()
# %% [markdown]
# ---
# ## Étape 4 : Interprétabilité avec SHAP
# %%
# SHAP sur XGBoost (modèle le plus performant)
explainer = shap.TreeExplainer(xgb_opt)
X_sample_shap = X_test[:1000]
shap_values = explainer.shap_values(X_sample_shap)
# %%
plt.figure()
shap.summary_plot(shap_values, X_sample_shap, show=False)
plt.title('SHAP Summary Plot - Impact des Features sur les Prédictions')
plt.tight_layout()
plt.show()
# %%
plt.figure()
shap.summary_plot(shap_values, X_sample_shap, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (Bar Plot)')
plt.tight_layout()
plt.show()
# %%
# Interprétation locale : explication d'une prédiction de fraude
fraud_idx = np.where(y_test.values == 1)[0][0]
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[fraud_idx, :], X_sample_shap.iloc[fraud_idx, :], show=True)
# %%
# Dependence plots pour les features les plus importantes
for feature in X.columns[:4]:
    shap.dependence_plot(feature, shap_values, X_sample_shap, show=False)
    plt.title(f'SHAP Dependence Plot - {feature}')
    plt.show()
# %%
print('=== PROJET TERMINÉ ===')
print('Toutes les étapes ont été exécutées avec succès.')