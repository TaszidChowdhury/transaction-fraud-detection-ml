from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import CustomerAggregationTransformer, SimpleCleaner


TARGET_COLUMN = "is_fraud"


@dataclass
class PreprocessConfig:
    numeric_features: Tuple[str, ...] = ("amount", "cust_txn_count", "cust_avg_amount", "cust_std_amount")
    categorical_features: Tuple[str, ...] = ("merchant_category", "time_of_day", "location")


def build_preprocess_pipeline(cfg: PreprocessConfig = PreprocessConfig()) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(cfg.numeric_features)),
            ("cat", categorical_transformer, list(cfg.categorical_features)),
        ]
    )
    return preprocessor


def build_full_pipeline(model_type: str = "logreg") -> Pipeline:
    cleaner = SimpleCleaner(
        numeric_columns=["amount"],
        categorical_columns=["merchant_category", "time_of_day", "location"],
    )
    cust_agg = CustomerAggregationTransformer()
    preprocessor = build_preprocess_pipeline()

    if model_type == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=None)
    elif model_type == "tree":
        model = DecisionTreeClassifier(max_depth=8, random_state=42)
    else:
        raise ValueError("model_type must be either 'logreg' or 'tree'")

    pipeline = Pipeline(
        steps=[
            ("clean", cleaner),
            ("cust_agg", cust_agg),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


