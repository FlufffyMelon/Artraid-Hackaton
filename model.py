"""Двухголовая модель предсказания выкупа заказа.

Входной датафрейм проходит через тот же feature pipeline, что и при
обучении (``features.yaml``). Затем роутер ``contact_Число сделок`` делит
записи на два потока:

* **повторные клиенты** (``>= 1``) — LogReg на одной переменной;
* **новые клиенты** (``< 1`` / NaN) — LogReg с энкодером по полному набору
  признаков из ``features.yaml``.

Модель не хранит внутри препроцессинг — всё, что нужно для инференса,
лежит в ``data/``:

* ``data/contexts.joblib`` — справочники (города, менеджеры, big_city);
* ``data/logreg_returning.joblib`` — ``Pipeline([StandardScaler, LogReg])``;
* ``data/logreg_new.joblib`` — ``Pipeline([LogRegEncoder, LogReg])``;
* ``data/model_meta.yaml`` — порог решения и мета-информация об обучении.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from utils.data import DEAL_COUNT_COL
from utils.features import build_features, load_feature_config


class BuyoutPredictor:
    def __init__(
        self,
        config_path: str | Path = "features.yaml",
        data_dir: str | Path = "data",
    ):
        data_dir = Path(data_dir)
        self._config = load_feature_config(str(config_path))
        self._contexts = joblib.load(data_dir / "contexts.joblib")
        with open(data_dir / "model_meta.yaml", encoding="utf-8") as f:
            self._meta = yaml.safe_load(f)
        self._returning = joblib.load(data_dir / "logreg_returning.joblib")
        self._new = joblib.load(data_dir / "logreg_new.joblib")

    @property
    def threshold(self) -> float:
        return float(self._meta.get("threshold", 0.5))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # group-missing warnings — нормальная ситуация при инференсе на
        # подвыборке, где редкие категории просто отсутствуют.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*group references missing values.*"
            )
            df, feat_cols = build_features(df, self._config, self._contexts)
        deals = pd.to_numeric(df[DEAL_COUNT_COL], errors="coerce").fillna(0)
        is_new = (deals < 1).values

        proba = np.full(len(df), np.nan, dtype=float)

        if (~is_new).any():
            x_ret = df.loc[~is_new, [DEAL_COUNT_COL]].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0)
            proba[~is_new] = self._returning.predict_proba(x_ret)[:, 1]

        if is_new.any():
            x_new = df.loc[is_new, feat_cols.all_feature_cols]
            proba[is_new] = self._new.predict_proba(x_new)[:, 1]

        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)
