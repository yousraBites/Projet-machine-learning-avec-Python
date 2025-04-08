import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

df = pd.read_csv("loan_data.csv")
df = df[df["person_age"] <= 100]
df = df.sample(n=5000, random_state=123)
if df["loan_status"].dtype != object:
    df["loan_status"] = df["loan_status"].map({0: "rejeté", 1: "approuvé"})
bins = [300, 580, 670, 740, 800, 851]
labels = ["faible", "moyen", "bon", "très bon", "excellent"]
df["credit_score"] = pd.cut(df["credit_score"], bins=bins, labels=labels, right=False, include_lowest=True)
df["previous_loan_defaults_on_file"] = (df["previous_loan_defaults_on_file"] == "Yes").astype(int).astype("category")
df["person_home_ownership"] = df["person_home_ownership"].astype("category")
df["loan_intent"] = df["loan_intent"].astype("category")
df["credit_score"] = df["credit_score"].astype("category")

quant_vars = ["person_age", "person_emp_exp", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
qual_vars = ["credit_score", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
quantitative_vars = df[quant_vars]
qualitative_vars = df[qual_vars]
target = df["loan_status"].astype("category")

qdummies = pd.get_dummies(qualitative_vars, drop_first=True)
predictors = pd.concat([quantitative_vars, qdummies], axis=1)

X_tv, X_test, y_tv, y_test = train_test_split(predictors, target, test_size=0.2, random_state=123, stratify=target)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1111, random_state=123, stratify=y_tv)
scaler = StandardScaler()
X_train[quant_vars] = scaler.fit_transform(X_train[quant_vars])
X_val[quant_vars] = scaler.transform(X_val[quant_vars])
X_test[quant_vars] = scaler.transform(X_test[quant_vars])
res = []
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    rec = recall_score(y_val, pred, pos_label="rejeté")
    prec = precision_score(y_val, pred, pos_label="rejeté")
    fsc = f1_score(y_val, pred, pos_label="rejeté")
    res.append((k, acc, rec, prec, fsc))
res = pd.DataFrame(res, columns=["k", "accuracy", "recall", "precision", "fscore"])
best_k = int(res.loc[res["accuracy"].idxmax(), "k"])
X_final = pd.concat([X_train, X_val])
y_final = pd.concat([y_train, y_val])
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_final, y_final)
pred_test = knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_test)
cm_knn = confusion_matrix(y_test, pred_test, labels=["rejeté", "approuvé"])
probs_knn = knn.predict_proba(X_test)
auc_knn = roc_auc_score(y_test.map({"rejeté": 0, "approuvé": 1}), probs_knn[:, list(knn.classes_).index("approuvé")])
fpr_knn, tpr_knn, _ = roc_curve(y_test.map({"rejeté": 0, "approuvé": 1}), probs_knn[:, list(knn.classes_).index("approuvé")])

pdict = pd.concat([quantitative_vars, pd.get_dummies(qualitative_vars, drop_first=True)], axis=1)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(pdict, target, test_size=0.2, random_state=123, stratify=target)
nb = GaussianNB()
nb.fit(X_train_nb, y_train_nb)
pred_nb = nb.predict(X_test_nb)
acc_nb = accuracy_score(y_test_nb, pred_nb)
rec_nb = recall_score(y_test_nb, pred_nb, pos_label="approuvé")
prec_nb = precision_score(y_test_nb, pred_nb, pos_label="approuvé")
f1_nb = f1_score(y_test_nb, pred_nb, pos_label="approuvé")
cm_nb = confusion_matrix(y_test_nb, pred_nb, labels=["rejeté", "approuvé"])
probs_nb = nb.predict_proba(X_test_nb)
auc_nb = roc_auc_score(y_test_nb.map({"rejeté": 0, "approuvé": 1}), probs_nb[:, list(nb.classes_).index("approuvé")])
fpr_nb, tpr_nb, _ = roc_curve(y_test_nb.map({"rejeté": 0, "approuvé": 1}), probs_nb[:, list(nb.classes_).index("approuvé")])

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(pdict, target, test_size=0.2, random_state=123, stratify=target)
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_train_dt, y_train_dt)
pred_dt = dt.predict(X_test_dt)
acc_dt = accuracy_score(y_test_dt, pred_dt)
rec_dt = recall_score(y_test_dt, pred_dt, pos_label="approuvé")
prec_dt = precision_score(y_test_dt, pred_dt, pos_label="approuvé")
f1_dt = f1_score(y_test_dt, pred_dt, pos_label="approuvé")
cm_dt = confusion_matrix(y_test_dt, pred_dt, labels=["rejeté", "approuvé"])
probs_dt = dt.predict_proba(X_test_dt)
auc_dt = roc_auc_score(y_test_dt.map({"rejeté": 0, "approuvé": 1}), probs_dt[:, list(dt.classes_).index("approuvé")])
fpr_dt, tpr_dt, _ = roc_curve(y_test_dt.map({"rejeté": 0, "approuvé": 1}), probs_dt[:, list(dt.classes_).index("approuvé")])
path = dt.cost_complexity_pruning_path(X_train_dt, y_train_dt)
ccp_alphas = path.ccp_alphas
dt_models = []
for ccp in ccp_alphas:
    m = DecisionTreeClassifier(random_state=123, ccp_alpha=ccp)
    m.fit(X_train_dt, y_train_dt)
    dt_models.append(m)
pruned_dt = min(dt_models, key=lambda m: 1 - accuracy_score(y_train_dt, m.predict(X_train_dt)))
pred_pruned = pruned_dt.predict(X_test_dt)
acc_pruned = accuracy_score(y_test_dt, pred_pruned)
rec_pruned = recall_score(y_test_dt, pred_pruned, pos_label="approuvé")
prec_pruned = precision_score(y_test_dt, pred_pruned, pos_label="approuvé")
f1_pruned = f1_score(y_test_dt, pred_pruned, pos_label="approuvé")
cm_pruned = confusion_matrix(y_test_dt, pred_pruned, labels=["rejeté", "approuvé"])
probs_pruned = pruned_dt.predict_proba(X_test_dt)
auc_pruned = roc_auc_score(y_test_dt.map({"rejeté": 0, "approuvé": 1}), probs_pruned[:, list(pruned_dt.classes_).index("approuvé")])
fpr_pruned, tpr_pruned, _ = roc_curve(y_test_dt.map({"rejeté": 0, "approuvé": 1}), probs_pruned[:, list(pruned_dt.classes_).index("approuvé")])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
accs, recs, precs, f1s = [], [], [], []
for tr_idx, te_idx in skf.split(pdict, target):
    X_tr, X_te = pdict.iloc[tr_idx], pdict.iloc[te_idx]
    y_tr, y_te = target.iloc[tr_idx], target.iloc[te_idx]
    clf = DecisionTreeClassifier(random_state=123)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    accs.append(accuracy_score(y_te, y_pred))
    recs.append(recall_score(y_te, y_pred, pos_label="approuvé"))
    precs.append(precision_score(y_te, y_pred, pos_label="approuvé"))
    f1s.append(f1_score(y_te, y_pred, pos_label="approuvé"))
acc_cv_dt = np.mean(accs)
rec_cv_dt = np.mean(recs)
prec_cv_dt = np.mean(precs)
f1_cv_dt = np.mean(f1s)
