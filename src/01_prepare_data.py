"""Step 1: build the tumour x gene matrix and the cancer-type labels.

Input  (data/raw/):  expression.xena.gz  - TCGA PanCanAtlas RNA-seq, log2(norm_value + 1), genes x samples
                     survival.tsv        - TCGA-CDR clinical table, used here only for the cancer-type label
Output (data/processed/):
                     tumour_expression.pkl  - samples x genes (float32), primary tumours only
                     labels.csv             - sample, patient, cancer_type, split (train/test)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_SAMPLES_PER_TYPE = 30   # drop cancer types too small to learn/test reliably
MIN_MEAN_LOG2 = 1.0         # drop genes that are barely expressed anywhere

# 1. Which samples do we keep? Primary tumours only: barcode code 01 (solid) or 03 (blood, LAML).
header = pd.read_csv(RAW / "expression.xena.gz", sep="\t", index_col=0, nrows=0).columns
code = header.str[13:15]
keep_samples = header[code.isin(["01", "03"])]
print(f"Samples in matrix: {len(header)}, primary tumours: {len(keep_samples)}")

# 2. Stream the big matrix in chunks of genes, keeping only those samples, as float32 to save memory
chunks = []
for chunk in pd.read_csv(RAW / "expression.xena.gz", sep="\t", index_col=0,
                         usecols=["sample", *keep_samples], chunksize=2000):
    chunks.append(chunk.astype("float32"))
expr = pd.concat(chunks).T           # samples x genes
expr.index.name = "sample"
print(f"Matrix loaded: {expr.shape[0]} samples x {expr.shape[1]} genes")

# 3. Gene filters that do not look at the labels (so they can't leak test information)
expr = expr.loc[:, expr.notna().all()]                  # never impute: drop genes with any missing value
expr = expr.loc[:, expr.mean() >= MIN_MEAN_LOG2]        # drop barely expressed genes
expr = expr.loc[:, ~expr.columns.duplicated()]
print(f"After gene filters: {expr.shape[1]} genes")

# 4. Cancer-type labels from the clinical table, matched by patient ID
surv = pd.read_csv(RAW / "survival.tsv", sep="\t")
patient_to_type = surv.drop_duplicates("_PATIENT").set_index("_PATIENT")["cancer type abbreviation"]
labels = pd.DataFrame({"sample": expr.index, "patient": expr.index.str[:12]})
labels["cancer_type"] = labels["patient"].map(patient_to_type)
labels = labels.dropna(subset=["cancer_type"])
labels = labels.drop_duplicates("patient")              # one tumour per patient

counts = labels["cancer_type"].value_counts()
small = counts[counts < MIN_SAMPLES_PER_TYPE]
if len(small):
    print(f"Dropping small cancer types: {dict(small)}")
labels = labels[labels["cancer_type"].isin(counts[counts >= MIN_SAMPLES_PER_TYPE].index)]

# 5. One fixed, stratified 80/20 train/test split. The test set is not touched until the final evaluation.
train_idx, test_idx = train_test_split(labels.index, test_size=0.2, stratify=labels["cancer_type"],
                                       random_state=SEED)
labels["split"] = "train"
labels.loc[test_idx, "split"] = "test"

expr = expr.loc[labels["sample"]]
expr.to_pickle(OUT / "tumour_expression.pkl")
labels.to_csv(OUT / "labels.csv", index=False)

print(f"\nFinal: {len(labels)} tumours, {labels['cancer_type'].nunique()} cancer types, {expr.shape[1]} genes")
print(labels.groupby("split").size().to_string())
print("\nTumours per cancer type:")
print(labels["cancer_type"].value_counts().to_string())
