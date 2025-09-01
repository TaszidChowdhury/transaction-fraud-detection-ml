from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics as skm


@dataclass
class ReportPaths:
    reports_dir: str = "reports"
    figures_dir: str = os.path.join(reports_dir, "figures")

    @property
    def metrics_json(self) -> str:
        return os.path.join(self.reports_dir, "metrics.json")

    @property
    def results_csv(self) -> str:
        return os.path.join(self.reports_dir, "fraud_results.csv")


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    acc = skm.accuracy_score(y_true, y_pred)
    prec = skm.precision_score(y_true, y_pred, zero_division=0)
    rec = skm.recall_score(y_true, y_pred)
    f1 = skm.f1_score(y_true, y_pred)
    roc_auc = skm.roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def ensure_reports_dir(paths: ReportPaths) -> None:
    os.makedirs(paths.reports_dir, exist_ok=True)
    os.makedirs(paths.figures_dir, exist_ok=True)


def export_metrics_json(metrics_dict: Dict[str, float], paths: ReportPaths) -> None:
    ensure_reports_dir(paths)
    with open(paths.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)


def export_results_csv(
    df_with_predictions: pd.DataFrame,
    paths: ReportPaths,
    select_columns: list[str] | None = None,
) -> None:
    ensure_reports_dir(paths)
    if select_columns is not None:
        export_df = df_with_predictions[select_columns]
    else:
        export_df = df_with_predictions
    export_df.to_csv(paths.results_csv, index=False)


