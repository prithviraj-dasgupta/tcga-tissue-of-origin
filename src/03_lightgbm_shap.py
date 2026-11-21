"""Step 3: gradient-boosted tree model (LightGBM) + SHAP explanations.

Which genes does the model use to recognise each cancer type, and do they make biological sense?
Same 2,000 training-set genes and the same train/test split as the baseline.
Outputs: results/lgbm_metrics.txt, results/shap_top_genes_per_type.csv,
         results/shap_global_top20.png, results/shap_top_genes_heatmap.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROC = Path("data/processed")
RES = Path("results")
RES.mkdir(exist_ok=True)
SEED = 42
N_GENES = 2000
N_SHAP = 600     # explain a random subset of test tumours to keep SHAP fast
TOP_K = 5        # top genes reported per cancer type

expr = pd.read_pickle(PROC / "tumour_expression.pkl")
labels = pd.read_csv(PROC / "labels.csv").set_index("sample").loc[expr.index]
train = (labels["split"] == "train").values

# Gene selection on the training set only (same rule as the baseline)
genes = expr[train].var().sort_values(ascending=False).index[:N_GENES]
X_train, X_test = expr.loc[train, genes], expr.loc[~train, genes]
y_train, y_test = labels.loc[train, "cancer_type"].values, labels.loc[~train, "cancer_type"].values

print("Training LightGBM (a few minutes)...")
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                       colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                       class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc, macro_f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
print(f"LightGBM test accuracy: {acc:.3f}   macro-F1: {macro_f1:.3f}")
with open(RES / "lgbm_metrics.txt", "w") as f:
    f.write(f"Test accuracy: {acc:.3f}\nTest macro-F1: {macro_f1:.3f}\n\n")
    f.write(classification_report(y_test, pred, digits=3))

# SHAP: how much each gene pushes a tumour towards each cancer type
print(f"Computing SHAP values for {N_SHAP} test tumours...")
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_test), size=min(N_SHAP, len(X_test)), replace=False)
X_shap, y_shap = X_test.iloc[idx], y_test[idx]
sv = shap.TreeExplainer(model).shap_values(X_shap)
if isinstance(sv, list):                      # older SHAP: list of (samples x genes), one per class
    sv = np.stack(sv, axis=-1)                # -> samples x genes x classes
classes = list(model.classes_)

# For each cancer type: among tumours of that type, which genes push hardest towards that type?
rows = []
for k, ct in enumerate(classes):
    mask = y_shap == ct
    if mask.sum() == 0:
        continue
    contrib = pd.Series(sv[mask, :, k].mean(axis=0), index=genes)   # signed: + means "evidence for ct"
    for rank, (gene, val) in enumerate(contrib.sort_values(ascending=False).head(TOP_K).items(), 1):
        rows.append({"cancer_type": ct, "rank": rank, "gene": gene, "mean_shap": round(val, 4),
                     "n_tumours_explained": int(mask.sum())})
top = pd.DataFrame(rows)
top.to_csv(RES / "shap_top_genes_per_type.csv", index=False)

print("\nTop 3 genes per cancer type (evidence FOR that type):")
print(top[top["rank"] <= 3].groupby("cancer_type")["gene"].apply(", ".join).to_string())

# Plot 1: genes that matter most overall (mean |SHAP| across tumours and classes)
global_imp = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=genes).sort_values(ascending=False)
global_imp.head(20)[::-1].plot.barh(figsize=(7, 7), color="#3b6ea8")
plt.xlabel("Mean |SHAP value| (all tumours, all cancer types)")
plt.title("Top 20 genes driving tissue-of-origin predictions")
plt.tight_layout()
plt.savefig(RES / "shap_global_top20.png", dpi=150)
plt.close()

# Plot 2: heatmap - top 2 genes per cancer type, and how much each pushes towards every type
heat_genes = list(dict.fromkeys(top[top["rank"] <= 2]["gene"]))
heat = pd.DataFrame({ct: [sv[y_shap == ct, genes.get_loc(g), k].mean() if (y_shap == ct).any() else np.nan
                          for g in heat_genes] for k, ct in enumerate(classes)}, index=heat_genes)
plt.figure(figsize=(14, max(8, len(heat_genes) * 0.28)))
lim = np.nanpercentile(np.abs(heat.values), 98)
sns.heatmap(heat, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim,
            cbar_kws={"label": "Mean SHAP towards that cancer type"})
plt.xlabel("Cancer type")
plt.ylabel("Gene")
plt.title("Each cancer type's top SHAP genes")
plt.tight_layout()
plt.savefig(RES / "shap_top_genes_heatmap.png", dpi=150)
print("\nSaved results/shap_global_top20.png, results/shap_top_genes_heatmap.png, "
      "results/shap_top_genes_per_type.csv")
