from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomerAggregationTransformer(BaseEstimator, TransformerMixin):
    """Compute per-customer aggregates and join back to rows.

    Adds columns like:
      - cust_txn_count, cust_avg_amount, cust_std_amount
      - cust_cat_ratio_<category> (share of txns in category)
    """

    def __init__(self, category_column: str = "merchant_category", amount_column: str = "amount"):
        self.category_column = category_column
        self.amount_column = amount_column
        self.category_values_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.category_values_ = (
            X[self.category_column].dropna().astype(str).value_counts().index.tolist()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["merchant_category"] = df["merchant_category"].astype("string")

        # Group by customer
        grp = df.groupby("customer_id", dropna=False)
        agg = grp[self.amount_column].agg([
            ("cust_txn_count", "count"),
            ("cust_avg_amount", "mean"),
            ("cust_std_amount", "std"),
        ])
        agg = agg.fillna({"cust_std_amount": 0.0})

        # Category ratios per customer
        cat_counts = (
            df.pivot_table(
                index="customer_id",
                columns=self.category_column,
                values="transaction_id",
                aggfunc="count",
                fill_value=0,
            )
            .astype(float)
        )
        totals = cat_counts.sum(axis=1).replace(0, np.nan)
        cat_ratios = cat_counts.div(totals, axis=0).fillna(0.0)

        # Ensure stable columns
        missing_cols = [c for c in self.category_values_ if c not in cat_ratios.columns]
        for c in missing_cols:
            cat_ratios[c] = 0.0
        cat_ratios = cat_ratios[self.category_values_]
        cat_ratios.columns = [f"cust_cat_ratio_{c}" for c in cat_ratios.columns]

        features = agg.join(cat_ratios, how="left")
        features = features.reset_index()
        df = df.merge(features, on="customer_id", how="left")
        return df


class SimpleCleaner(BaseEstimator, TransformerMixin):
    """Simple cleaning: impute numeric and categorical missing values.

    - amount: median
    - categorical: constant "missing"
    """

    def __init__(self, numeric_columns: List[str], categorical_columns: List[str]):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.medians_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = {c: float(X[c].median()) for c in self.numeric_columns if c in X}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in self.numeric_columns:
            if c in df:
                df[c] = df[c].fillna(self.medians_.get(c, 0.0))
        for c in self.categorical_columns:
            if c in df:
                df[c] = df[c].fillna("missing").astype("string")
        return df


