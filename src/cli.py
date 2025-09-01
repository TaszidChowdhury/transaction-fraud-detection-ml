from __future__ import annotations

import argparse


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transaction Fraud Training CLI")
    parser.add_argument("--model", choices=["logreg", "tree"], default="logreg")
    parser.add_argument("--data_csv", type=str, default="data/transactions.csv")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data if missing")
    parser.add_argument("--n_rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser


