"""Хронологическое разбиение данных: кумулятивный CV по месяцам и
«обучение до месяца X — предсказание месяца Y».

Модуль не знает про LogReg или CatBoost — принимает callable, который
обучает модель на переданных (train, test) и возвращает вероятности для
тестовой выборки.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from utils.metrics import best_f1_threshold

TrainFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def _month_period(df: pd.DataFrame, date_col: str) -> pd.Series:
    return pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")


def cumulative_months_loop(
    df: pd.DataFrame,
    train_fn: TrainFn,
    *,
    date_col: str = "sale_date",
    target_col: str = "buyout_flag",
    verbose: bool = True,
) -> list[dict]:
    """Для каждого месяца M (кроме первого) обучить модель на всех данных
    до него и оценить качество на самом M.

    ``train_fn(train_df, test_df) -> proba_test`` получает срезы датафрейма
    и возвращает вероятности положительного класса по тестовым строкам.
    Возвращает список словарей с метриками по каждому тестовому месяцу.
    """
    months = _month_period(df, date_col)
    unique_months = sorted(m for m in months.dropna().unique())

    from sklearn.metrics import roc_auc_score

    results: list[dict] = []
    for i, test_month in enumerate(unique_months):
        if i == 0:
            continue
        train_mask = months < test_month
        test_mask = months == test_month
        train_df = df[train_mask]
        test_df = df[test_mask]
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            if verbose:
                print(f"[{test_month}] skipped — один класс в train или test")
            continue

        proba = train_fn(train_df, test_df)
        auc = roc_auc_score(y_test, proba)
        f1, thr = best_f1_threshold(y_test, proba)

        entry = {
            "test_month": str(test_month),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_buyout": float(y_train.mean()),
            "test_buyout": float(y_test.mean()),
            "test_auc": float(auc),
            "test_f1": float(f1),
            "best_threshold": float(thr),
        }
        results.append(entry)
        if verbose:
            print(
                f"[{test_month}] train={len(train_df):5d}  test={len(test_df):5d}  "
                f"AUC={auc:.4f}  bestF1={f1:.4f}@{thr:.2f}"
            )
    return results


def split_by_month(
    df: pd.DataFrame,
    *,
    train_through: str,
    test_month: str,
    date_col: str = "sale_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить датафрейм на train (все месяцы ≤ ``train_through``) и test
    (единственный ``test_month``). Формат месяцев — ``'YYYY-MM'``.
    """
    months = _month_period(df, date_col).astype(str)
    train = df[months <= train_through].copy()
    test = df[months == test_month].copy()
    return train, test
