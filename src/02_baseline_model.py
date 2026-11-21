"""Step 2: baseline tissue-of-origin classifier (multinomial logistic regression).

Gene selection and scaling are fitted on the training set only, inside a pipeline,
so nothing about the test set leaks into the model.
Outputs: results/baseline_metrics.txt, results/baseline_confusion_matrix.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

PROC = Path("data/processed")
RES = Path("results")
RES.mkdir(exist_ok=True)
SEED = 42
N_GENES = 2000   # keep the 2,000 most variable genes (chosen on training data only)


class TopVarianceGenes(BaseEstimator, TransformerMixin):
    """Keep the n most variable genes, learned from whatever data .fit() sees."""
    def __init__(self, n=N_GENES):
        self.n = n
    def fit(self, X, y=None):
        self.idx_ = np.argsort(np.var(X, axis=0))[::-1][: self.n]
        return self
    def transform(self, X):
        return X[:, self.idx_]


expr = pd.read_pickle(PROC / "tumour_expression.pkl")
labels = pd.read_csv(PROC / "labels.csv").set_index("sample").loc[expr.index]
train = labels["split"] == "train"
X_train, X_test = expr[train].values, expr[~train].values
y_train, y_test = labels.loc[train, "cancer_type"].values, labels.loc[~train, "cancer_type"].values

model = make_pipeline(TopVarianceGenes(), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=2000))

# 5-fold cross-validation on the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
print(f"5-fold CV balanced accuracy (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit on all training data, evaluate once on the held-out test set
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
print(f"Test accuracy: {acc:.3f}   Test macro-F1: {macro_f1:.3f}")

report = classification_report(y_test, pred, digits=3)
with open(RES / "baseline_metrics.txt", "w") as f:
    f.write(f"5-fold CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
    f.write(f"Test accuracy: {acc:.3f}\nTest macro-F1: {macro_f1:.3f}\n\n{report}")
print(report)

# Confusion matrix, row-normalised (each row = true cancer type, sums to 1)
classes = sorted(np.unique(y_train))
cm = confusion_matrix(y_test, pred, labels=classes, normalize="true")
plt.figure(figsize=(13, 11))
sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", vmin=0, vmax=1,
            cbar_kws={"label": "Fraction of true class"})
plt.xlabel("Predicted cancer type")
plt.ylabel("True cancer type")
plt.title(f"Baseline logistic regression, {N_GENES} genes (test accuracy {acc:.2f})")
plt.tight_layout()
plt.savefig(RES / "baseline_confusion_matrix.png", dpi=150)
print("Saved results/baseline_confusion_matrix.png")
