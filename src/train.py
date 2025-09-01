from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cli import build_train_arg_parser
from .data import DataPaths, ensure_dataset, load_dataframe_from_csv
from .metrics import ReportPaths, compute_classification_metrics, export_metrics_json, export_results_csv
from .models import TARGET_COLUMN, build_full_pipeline


def load_or_generate_dataset(csv_path: str, generate: bool, n_rows: int, seed: int) -> pd.DataFrame:
    paths = DataPaths(data_dir=os.path.dirname(csv_path) or ".", file_name=os.path.basename(csv_path))
    ensure_dataset(paths, generate_if_missing=generate, n_rows=n_rows, seed=seed)
    df = load_dataframe_from_csv(paths.csv_path)
    return df


def run_training(model_type: str, df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    pipeline = build_full_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # Handle probability interface for classifiers without predict_proba
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback: decision function -> sigmoid proxy
        if hasattr(pipeline.named_steps["model"], "decision_function"):
            scores = pipeline.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            y_prob = y_pred.astype(float)

    metrics_dict = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    results_df = X_test.copy()
    results_df["y_true"] = y_test
    results_df["y_pred"] = y_pred
    results_df["y_prob"] = y_prob
    return results_df, metrics_dict


def main() -> None:
    parser = build_train_arg_parser()
    args = parser.parse_args()

    df = load_or_generate_dataset(csv_path=args.data_csv, generate=args.generate, n_rows=args.n_rows, seed=args.seed)

    results_df, metrics_dict = run_training(model_type=args.model, df=df, test_size=args.test_size, seed=args.seed)

    report_paths = ReportPaths()
    export_metrics_json(metrics_dict, report_paths)
    export_results_csv(
        results_df,
        report_paths,
        select_columns=[
            "transaction_id",
            "customer_id",
            "amount",
            "merchant_category",
            "time_of_day",
            "location",
            "y_true",
            "y_pred",
            "y_prob",
        ],
    )

    print("Training complete. Metrics saved to:", report_paths.metrics_json)
    print("Results saved to:", report_paths.results_csv)


if __name__ == "__main__":
    main()


