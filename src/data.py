from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_CATEGORIES = [
    "groceries",
    "gas",
    "restaurants",
    "travel",
    "electronics",
    "fashion",
    "health",
    "digital_goods",
]

DEFAULT_LOCATIONS = [
    "NY",
    "CA",
    "TX",
    "FL",
    "IL",
    "WA",
]

DEFAULT_TIMES = ["morning", "afternoon", "evening", "night"]


@dataclass
class DataPaths:
    data_dir: str = "data"
    file_name: str = "transactions.csv"

    @property
    def csv_path(self) -> str:
        return os.path.join(self.data_dir, self.file_name)


def save_dataframe_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_dataframe_from_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    return pd.read_csv(path)


def generate_synthetic_transactions(
    n_rows: int = 10000,
    seed: int = 42,
    categories: Optional[list[str]] = None,
    locations: Optional[list[str]] = None,
    times: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset.

    Columns: transaction_id, customer_id, amount, merchant_category, time_of_day,
             location, is_fraud (0/1)
    """
    rng = np.random.default_rng(seed)
    categories = categories or DEFAULT_CATEGORIES
    locations = locations or DEFAULT_LOCATIONS
    times = times or DEFAULT_TIMES

    # Customers
    num_customers = max(200, n_rows // 10)
    customer_ids = np.arange(1, num_customers + 1)

    # Base draws
    transaction_id = np.arange(1, n_rows + 1)
    customer_id = rng.choice(customer_ids, size=n_rows, replace=True)
    merchant_category = rng.choice(categories, size=n_rows, replace=True, p=_category_probs(categories))
    time_of_day = rng.choice(times, size=n_rows, replace=True, p=[0.3, 0.35, 0.25, 0.10])
    location = rng.choice(locations, size=n_rows, replace=True)

    # Amounts by category scale
    category_to_scale = {
        "groceries": 30.0,
        "gas": 40.0,
        "restaurants": 50.0,
        "travel": 200.0,
        "electronics": 300.0,
        "fashion": 100.0,
        "health": 80.0,
        "digital_goods": 60.0,
    }
    scales = np.array([category_to_scale.get(cat, 50.0) for cat in merchant_category])
    # Lognormal amounts with occasional outliers
    base = rng.lognormal(mean=np.log(scales), sigma=0.6)
    outlier_mask = rng.random(n_rows) < 0.01
    base[outlier_mask] *= rng.uniform(5, 15, size=outlier_mask.sum())
    amount = np.round(base, 2)

    # Fraud probability model (latent): higher at night/evening, higher for electronics/travel,
    # higher for unusual location-category combos, and very high amounts.
    category_risk = np.isin(merchant_category, ["electronics", "travel", "digital_goods"]).astype(float)
    time_risk = np.isin(time_of_day, ["evening", "night"]).astype(float)
    amount_risk = np.clip((amount - 150) / 200, 0, 1)
    loc_cat_risk = (np.isin(location, ["FL", "TX"]).astype(float) * category_risk)

    logits = -3.0 + 1.2 * category_risk + 1.0 * time_risk + 1.5 * amount_risk + 0.7 * loc_cat_risk
    prob = 1 / (1 + np.exp(-logits))
    is_fraud = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "time_of_day": time_of_day,
            "location": location,
            "is_fraud": is_fraud,
        }
    )

    # Introduce some missingness to simulate real-world dirty data
    _introduce_missing_values_inplace(df, rng=rng, frac=0.01)
    return df


def _category_probs(categories: list[str]) -> np.ndarray:
    # Skew market share slightly towards groceries/restaurants
    base = np.ones(len(categories), dtype=float)
    for i, cat in enumerate(categories):
        if cat in {"groceries", "restaurants"}:
            base[i] = 2.0
        if cat in {"electronics", "travel"}:
            base[i] = 1.3
    base = base / base.sum()
    return base


def _introduce_missing_values_inplace(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.01) -> None:
    n = len(df)
    for col in ["amount", "merchant_category", "time_of_day", "location"]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan


def ensure_dataset(paths: DataPaths, generate_if_missing: bool = True, n_rows: int = 20000, seed: int = 42) -> str:
    os.makedirs(paths.data_dir, exist_ok=True)
    if os.path.exists(paths.csv_path):
        return paths.csv_path
    if not generate_if_missing:
        raise FileNotFoundError(f"Dataset not found at {paths.csv_path} and generation disabled.")
    df = generate_synthetic_transactions(n_rows=n_rows, seed=seed)
    save_dataframe_to_csv(df, paths.csv_path)
    return paths.csv_path


