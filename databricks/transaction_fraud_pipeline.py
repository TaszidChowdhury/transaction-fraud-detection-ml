# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # Transaction Fraud Pipeline (Databricks Variant)
# MAGIC End-to-end pipeline adapted for Spark/Databricks. Uses pandas for modeling; for very large data, sample or convert to Spark ML.

# COMMAND ----------
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import generate_synthetic_transactions
from src.models import build_full_pipeline, TARGET_COLUMN
from src.metrics import compute_classification_metrics

# COMMAND ----------
# Generate synthetic data (or load from a Delta table)
try:
    n_rows = int(dbutils.widgets.get("n_rows"))  # type: ignore[name-defined]
except Exception:
    n_rows = 200000
seed = 42
pdf = generate_synthetic_transactions(n_rows=n_rows, seed=seed)

# COMMAND ----------
# Train/test split
X = pdf.drop(columns=[TARGET_COLUMN])
y = pdf[TARGET_COLUMN].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# COMMAND ----------
# Train pipelines
plog = build_full_pipeline('logreg')
plog.fit(X_train, y_train)
y_pred = plog.predict(X_test)
y_prob = plog.predict_proba(X_test)[:,1]
metrics = compute_classification_metrics(y_test, y_pred, y_prob)
metrics

# COMMAND ----------
# Save to DBFS path
out_dir = "/dbfs/FileStore/fraud_pipeline"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
pd.concat([
    X_test.reset_index(drop=True),
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})
], axis=1).to_csv(os.path.join(out_dir, "fraud_results.csv"), index=False)

# COMMAND ----------
print("Saved to:", out_dir)
