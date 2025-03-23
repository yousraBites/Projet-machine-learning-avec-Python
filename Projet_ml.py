import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

%matplotlib inline

data = pd.read_csv(r"C:\Users\yousr\Desktop\projet ml\loan_data.csv")
print("Dimension initiale :", data.shape)
nb_age_incorrect = data[data['person_age'] > 100].shape[0]
print("Nombre de lignes avec un âge > 100 :", nb_age_incorrect)
data = data[data['person_age'] <= 100].copy()
print("Dimension après suppression :", data.shape)
np.random.seed(123)
if data.shape[0] > 5000:
    data = data.sample(n=5000, random_state=123)
print("Dimension finale :", data.shape)
bins = [300, 580, 670, 740, 800, 851]
labels = ["faible", "moyen/limite", "bon", "très bon", "excellent"]
data['credit_score'] = pd.cut(data['credit_score'], bins=bins, labels=labels, include_lowest=True, right=False)
print("Répartition de credit_score :\n", data['credit_score'].value_counts())
data['previous_loan_defaults_on_file'] = np.where(data['previous_loan_defaults_on_file'] == "Yes", 1, 0)
data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].astype('category')
categorical_vars = ['person_home_ownership', 'loan_intent', 'credit_score']
for col in categorical_vars:
    data[col] = data[col].astype('category')
quantitative_vars = data[['person_age', 'person_emp_exp', 'loan_percent_income', 'loan_int_rate', 'cb_person_cred_hist_length']]
print("\nStatistiques descriptives (quantitatives) :\n", quantitative_vars.describe())
qualitative_vars = data[['credit_score', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']]
print("\nStatistiques descriptives (qualitatives) :")
for col in qualitative_vars.columns:
    print(f"\n{col} :")
    print(qualitative_vars[col].value_counts())
data['loan_status'] = data['loan_status'].map({0: "rejeté", 1: "approuvé"})
data['loan_status'] = data['loan_status'].astype('category')
print("\nRépartition de la target (loan_status) :")
print(data['loan_status'].value_counts())
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i, col in enumerate(quantitative_vars.columns):
    axs[i].hist(quantitative_vars[col], color='pink', edgecolor='black')
    axs[i].set_title(f"Histogramme de {col}")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i, col in enumerate(quantitative_vars.columns):
    axs[i].boxplot(quantitative_vars[col].dropna(), patch_artist=True)
    axs[i].set_title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()
print("\nNombre de valeurs manquantes par variable :")
print(data.isna().sum())
X = pd.concat([quantitative_vars, pd.get_dummies(qualitative_vars, drop_first=True)], axis=1)
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("\nDimensions du jeu d'entraînement :", X_train.shape)
print("Dimensions du jeu de test :", X_test.shape)
