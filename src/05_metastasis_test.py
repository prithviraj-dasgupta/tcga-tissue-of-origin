"""Step 5: does a model trained on primary tumours still work on metastases?

TCGA includes metastatic samples (barcode code 06), almost all of them metastatic melanoma
(lymph node, skin and soft-tissue metastases). We train the same model as step 4
(2,000 genes, sex-linked genes removed) on the primary-tumour training set, then predict
the tissue of origin of each metastasis. Patients whose primary tumour is in the training set are excluded.
Outputs: results/metastasis_predictions.csv, results/metastasis_predictions.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RAW, PROC, RES = Path("data/raw"), Path("data/processed"), Path("results")
N_GENES = 2000

expr = pd.read_pickle(PROC / "tumour_expression.pkl")
labels = pd.read_csv(PROC / "labels.csv").set_index("sample").loc[expr.index]
sex_genes = (RES / "sex_linked_genes_removed.txt").read_text().split()
expr = expr.drop(columns=[g for g in sex_genes if g in expr.columns])
train = (labels["split"] == "train").values
genes = list(expr[train].var().sort_values(ascending=False).index[:N_GENES])

model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
model.fit(expr.loc[train, genes], labels.loc[train, "cancer_type"])
print("Model trained on primary tumours.")

# Load the metastatic samples (code 06) from the raw matrix, only the 2,000 model genes
header = pd.read_csv(RAW / "expression.xena.gz", sep="\t", index_col=0, nrows=0).columns
met_samples = list(header[header.str[13:15] == "06"])
print(f"Metastatic samples in matrix: {len(met_samples)}. Reading them (a few minutes)...")
chunks = []
for chunk in pd.read_csv(RAW / "expression.xena.gz", sep="\t", index_col=0,
                         usecols=["sample", *met_samples], chunksize=2000):
    chunk = chunk[chunk.index.isin(genes)]
    if len(chunk):
        chunks.append(chunk)
met = pd.concat(chunks)
met = met[~met.index.duplicated()].T[genes]

surv = pd.read_csv(RAW / "survival.tsv", sep="\t")
patient_to_type = surv.drop_duplicates("_PATIENT").set_index("_PATIENT")["cancer type abbreviation"]
info = pd.DataFrame({"sample": met.index, "patient": met.index.str[:12]})
info["true_type"] = info["patient"].map(patient_to_type)
train_patients = set(labels.loc[train, "patient"])
keep = info["true_type"].notna() & ~info["patient"].isin(train_patients) & met.notna().all(axis=1).values
print(f"Excluded {(~keep).sum()} metastases (patient in training set, no label, or missing values)")
info, met = info[keep.values], met[keep.values]
info = info.drop_duplicates("patient")
met = met.loc[info["sample"]]

proba = model.predict_proba(met)
info["predicted_type"] = model.classes_[proba.argmax(axis=1)]
info["confidence"] = proba.max(axis=1).round(3)
info["correct"] = info["predicted_type"] == info["true_type"]
info.to_csv(RES / "metastasis_predictions.csv", index=False)

print(f"\nMetastases tested: {len(info)}")
for t, grp in info.groupby("true_type"):
    print(f"  {t}: {grp['correct'].mean():.1%} correct ({grp['correct'].sum()}/{len(grp)})")
wrong = info[~info["correct"]]
if len(wrong):
    print("\nWhat the wrong ones were called:")
    print(wrong["predicted_type"].value_counts().to_string())
print(f"\nMedian confidence: correct {info.loc[info['correct'], 'confidence'].median():.2f}, "
      f"wrong {wrong['confidence'].median() if len(wrong) else float('nan'):.2f}")

# Plot: what the metastases were predicted as
tab = pd.crosstab(info["predicted_type"], info["correct"]).reindex(columns=[True, False], fill_value=0)
tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
plt.figure(figsize=(8, 4.5))
plt.bar(tab.index, tab[True], color="#3b6ea8", label="Correct origin")
plt.bar(tab.index, tab[False], bottom=tab[True], color="#c0504d", label="Wrong origin")
plt.yscale("symlog", linthresh=10)
plt.legend()
plt.ylabel("Number of metastases (log scale above 10)")
plt.xlabel("Predicted tissue of origin")
plt.title(f"Metastases: {info['correct'].mean():.0%} assigned to the correct origin (n = {len(info)})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(RES / "metastasis_predictions.png", dpi=150)
print("Saved results/metastasis_predictions.png and results/metastasis_predictions.csv")
