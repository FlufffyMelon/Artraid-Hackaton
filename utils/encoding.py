"""Encoder for the LogReg new-clients model.

Принимает типизированный набор колонок (``FeatureColumns``) и собирает
единую числовую матрицу: one-hot для категориальных, ``StandardScaler`` для
числовых, passthrough для бинарных, сглаженный target encoding для
высококардинальных (гео, менеджеры и т. п.).

При обучении TE-колонки получают out-of-fold значения (StratifiedKFold) — так
модель видит сглаженный сигнал без утечки. При инференсе используются
TE-карты, посчитанные на всём train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils.features import FeatureColumns


class LogRegEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_cols: FeatureColumns,
        te_n_splits: int = 5,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.te_n_splits = te_n_splits
        self.random_state = random_state

    # ------------------------------------------------------------------ public

    def fit(self, X: pd.DataFrame, y) -> "LogRegEncoder":
        self._fit_encoders(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y) -> np.ndarray:  # type: ignore[override]
        self._fit_encoders(X, y)
        return self._apply(X, y=y, oof=True)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._apply(X, y=None, oof=False)

    # ------------------------------------------------------------------ internals

    def _fit_encoders(self, X: pd.DataFrame, y) -> None:
        fc = self.feature_cols
        y = np.asarray(y)
        self.global_mean_ = float(y.mean()) if len(y) else 0.5

        # OHE: per-column drop_first; запоминаем полный список колонок.
        # Принудительно приводим категориальные к строкам — после чтения
        # из CSV числовые коды категорий иначе интерпретируются как float,
        # и ``pd.get_dummies`` не раскрывает их в индикаторы.
        self.onehot_columns_: list[str] = []
        self.onehot_source_: dict[str, str] = {}
        for col in fc.cat_cols:
            drop_first = fc.cat_drop_first.get(col, False)
            oh = pd.get_dummies(self._as_category_frame(X, col), drop_first=drop_first)
            self.onehot_columns_.extend(oh.columns)
            for c in oh.columns:
                self.onehot_source_[c] = col

        # Scaler на числовых.
        if fc.num_cols:
            num = self._numeric_block(X, fc.num_cols)
            self.scaler_ = StandardScaler().fit(num)
        else:
            self.scaler_ = None

        # TE-карты на всех тренировочных данных (для transform).
        te_cols = fc.geo_cols + fc.te_cat_cols
        self.te_maps_: dict[str, pd.Series] = {}
        for col in te_cols:
            alpha = fc.te_alpha.get(col, 10)
            self.te_maps_[col] = self._smoothed_target_map(
                X[col], y, alpha=alpha, global_mean=self.global_mean_
            )

        # Имена выходных колонок и отображение в исходный признак — для
        # feature importance в ноутбуках.
        self.feature_names_ = (
            list(self.onehot_columns_)
            + list(fc.num_cols)
            + list(fc.bin_cols)
            + list(te_cols)
        )
        self.group_map_ = {}
        self.group_map_.update(self.onehot_source_)
        for c in list(fc.num_cols) + list(fc.bin_cols) + list(te_cols):
            self.group_map_[c] = c

    def _apply(self, X: pd.DataFrame, y, oof: bool) -> np.ndarray:
        fc = self.feature_cols
        X = X.reset_index(drop=True)
        parts: list[np.ndarray] = []

        if self.onehot_columns_:
            oh_parts = []
            for col in fc.cat_cols:
                drop_first = fc.cat_drop_first.get(col, False)
                oh_parts.append(
                    pd.get_dummies(self._as_category_frame(X, col), drop_first=drop_first)
                )
            onehot = pd.concat(oh_parts, axis=1)
            onehot = onehot.reindex(columns=self.onehot_columns_, fill_value=0)
            parts.append(onehot.values.astype(float))

        if fc.num_cols:
            num = self._numeric_block(X, fc.num_cols)
            parts.append(self.scaler_.transform(num))

        if fc.bin_cols:
            parts.append(self._numeric_block(X, fc.bin_cols))

        te_cols = fc.geo_cols + fc.te_cat_cols
        if te_cols:
            te_matrix = np.zeros((len(X), len(te_cols)), dtype=float)
            if oof and y is not None and len(X) > 0:
                self._fill_oof_te(X, np.asarray(y), te_cols, te_matrix)
            else:
                for j, col in enumerate(te_cols):
                    te_matrix[:, j] = (
                        X[col]
                        .map(self.te_maps_[col])
                        .fillna(self.global_mean_)
                        .values
                    )
            parts.append(te_matrix)

        return np.hstack(parts) if parts else np.zeros((len(X), 0))

    def _fill_oof_te(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        te_cols: list[str],
        out: np.ndarray,
    ) -> None:
        fc = self.feature_cols
        n_splits = min(self.te_n_splits, int(max(2, y.sum(), len(y) - y.sum())))
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        folds = list(skf.split(np.zeros(len(X)), y))
        for j, col in enumerate(te_cols):
            alpha = fc.te_alpha.get(col, 10)
            series = X[col].reset_index(drop=True)
            oof = np.full(len(X), self.global_mean_, dtype=float)
            for tr_idx, te_idx in folds:
                tr_map = self._smoothed_target_map(
                    series.iloc[tr_idx],
                    y[tr_idx],
                    alpha=alpha,
                    global_mean=self.global_mean_,
                )
                oof[te_idx] = (
                    series.iloc[te_idx].map(tr_map).fillna(self.global_mean_).values
                )
            out[:, j] = oof

    @staticmethod
    def _numeric_block(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return X[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)

    @staticmethod
    def _as_category_frame(X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Строковое представление столбца для one-hot (NaN сохраняется)."""
        series = X[col]
        nan_mask = series.isna()
        out = series.astype("string").astype(object)
        # ``float``-коды CSV пишутся как ``"700242.0"`` — убираем хвост ``.0``.
        if pd.api.types.is_numeric_dtype(series):
            out = out.where(nan_mask, out.str.replace(r"\.0$", "", regex=True))
        out[nan_mask] = np.nan
        return out.to_frame(name=col)

    @staticmethod
    def _smoothed_target_map(
        values: pd.Series, y: np.ndarray, alpha: float, global_mean: float
    ) -> pd.Series:
        stats = (
            pd.DataFrame({"v": values.values, "y": y})
            .groupby("v")["y"]
            .agg(["mean", "count"])
        )
        return (stats["mean"] * stats["count"] + global_mean * alpha) / (
            stats["count"] + alpha
        )
