"""Step 4: (a) remove sex-linked genes, a shortcut SHAP revealed, and re-check accuracy;
          (b) how small can the gene panel get before accuracy drops?

Sex-linked genes are found from the data: genes whose expression separates male from female
patients almost perfectly (AUC > 0.9 or < 0.1) in training tumours of cancer types that occur in both sexes.
Panel genes are ranked with the training set only; the test set is used only to score each panel.
Outputs: results/sex_linked_genes_removed.txt, results/panel_size_vs_accuracy.csv,
         results/panel_size_vs_accuracy.png, results/minimal_panel_genes.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RAW, PROC, RES = Path("data/raw"), Path("data/processed"), Path("results")
N_GENES = 2000
SEX_SPECIFIC = {"BRCA", "CESC", "OV", "UCEC", "UCS", "PRAD", "TGCT"}   # one sex only, or almost only
PANEL_K = [1, 2, 3, 5, 10, 20, 50]   # top-k genes per cancer type


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def score(genes, X_tr, y_tr, X_te, y_te):
    m = make_model().fit(X_tr[genes], y_tr)
    p = m.predict(X_te[genes])
    return accuracy_score(y_te, p), f1_score(y_te, p, average="macro"), m


expr = pd.read_pickle(PROC / "tumour_expression.pkl")
labels = pd.read_csv(PROC / "labels.csv").set_index("sample").loc[expr.index]
surv = pd.read_csv(RAW / "survival.tsv", sep="\t")
sex = labels["patient"].map(surv.drop_duplicates("_PATIENT").set_index("_PATIENT")["gender"])
train = (labels["split"] == "train").values
y_train, y_test = labels.loc[train, "cancer_type"].values, labels.loc[~train, "cancer_type"].values

# (a) Find sex-linked genes (training tumours only, both-sex cancer types only)
m = train & ~labels["cancer_type"].isin(SEX_SPECIFIC).values & sex.isin(["MALE", "FEMALE"]).values
Xs, is_male = expr.values[m], (sex.values[m] == "MALE")
ranks = rankdata(Xs, axis=0)
n1, n0 = is_male.sum(), (~is_male).sum()
auc = (ranks[is_male].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)     # Mann-Whitney AUC per gene
sex_genes = expr.columns[(auc > 0.9) | (auc < 0.1)]
print(f"Sex-linked genes removed ({len(sex_genes)}): {', '.join(sex_genes)}")
(RES / "sex_linked_genes_removed.txt").write_text("\n".join(sex_genes))
expr = expr.drop(columns=sex_genes)

# Same 2,000-gene rule as before, chosen on training data, now without the sex genes
genes = expr[train].var().sort_values(ascending=False).index[:N_GENES]
X_tr, X_te = expr.loc[train, genes], expr.loc[~train, genes]
acc, f1, full_model = score(list(genes), X_tr, y_train, X_te, y_test)
print(f"\nWithout sex-linked genes, 2,000 genes: test accuracy {acc:.3f}, macro-F1 {f1:.3f}"
      f"  (before: 0.963 / 0.949)")

# (b) Rank genes per cancer type by the model's coefficient (fitted on training data only)
coef = pd.DataFrame(full_model[-1].coef_, index=full_model[-1].classes_, columns=genes)
rows = [{"k_per_type": len(genes) // coef.shape[0], "n_genes": len(genes), "test_accuracy": acc, "test_macro_f1": f1}]
for k in PANEL_K:
    panel = list(dict.fromkeys(g for ct in coef.index for g in coef.loc[ct].nlargest(k).index))
    a, f, _ = score(panel, X_tr, y_train, X_te, y_test)
    rows.append({"k_per_type": k, "n_genes": len(panel), "test_accuracy": a, "test_macro_f1": f})
    print(f"Top {k:>2} genes per type -> {len(panel):>4} genes: accuracy {a:.3f}, macro-F1 {f:.3f}")
    if k == 3:
        pd.DataFrame([(ct, g) for ct in coef.index for g in coef.loc[ct].nlargest(3).index],
                     columns=["cancer_type", "gene"]).to_csv(RES / "minimal_panel_genes.csv", index=False)
curve = pd.DataFrame(rows).sort_values("n_genes")
curve.round(4).to_csv(RES / "panel_size_vs_accuracy.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.plot(curve["n_genes"], curve["test_accuracy"], "o-", label="Accuracy", color="#3b6ea8")
plt.plot(curve["n_genes"], curve["test_macro_f1"], "s--", label="Macro-F1", color="#d07a2d")
plt.xscale("log")
plt.xlabel("Number of genes in panel (log scale)")
plt.ylabel("Test score")
plt.ylim(min(0.5, curve["test_macro_f1"].min() - 0.05), 1.0)
plt.title("How many genes does tissue-of-origin need?")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RES / "panel_size_vs_accuracy.png", dpi=150)
print("\nSaved results/panel_size_vs_accuracy.png and results/minimal_panel_genes.csv (3 genes per type)")
