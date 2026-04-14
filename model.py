"""
Двухмодельная система предсказания выкупа заказов.

Повторные клиенты (contact_Число сделок >= 1) → LogReg на одном признаке.
Новые клиенты (contact_Число сделок = NaN/0) → LogReg на ~23 признаках.
"""

import numpy as np
import pandas as pd
import joblib


class BuyoutPredictor:
    """Предсказание вероятности выкупа заказа.

    Использует два LogReg: один для повторных клиентов (единственный признак —
    число прошлых сделок), второй для новых клиентов (категориальные, числовые,
    бинарные и гео-признаки).
    """

    def __init__(self, weights_path: str):
        w = joblib.load(weights_path)

        self._model_new = w["model_new"]
        self._model_ret = w["model_returning"]
        self._scaler_new = w["scaler_new"]
        self._scaler_ret = w["scaler_returning"]
        self._onehot_columns = w["onehot_columns"]
        self._te_maps = w["te_maps"]
        self._global_mean = w["global_mean"]
        self._cat_cols = w["cat_cols"]
        self._num_cols = w["num_cols"]
        self._bin_cols = w["bin_cols"]
        self._geo_cols = w["geo_cols"]
        self._te_cat_cols = w["te_cat_cols"]
        self._threshold = w.get("threshold", 0.5)
        self._manager_deal_count_map = w.get("manager_deal_count_map", {})
        self._manager_deal_count_default = w.get("manager_deal_count_default", 1)
        self._cat_drop_first = w.get("cat_drop_first", {})

        # Bin parameters
        self._cart_bins = w.get("cart_bins", [-1, 8, 12, 16, float("inf")])
        self._cart_labels = w.get("cart_labels", ["1-8", "9-12", "13-16", "17+"])
        self._price_bins = w.get("price_bins", [0, 3000, 5000, 8000, 15000, 25000, float("inf")])
        self._price_labels = w.get("price_labels", ["0-3k", "3-5k", "5-8k", "8-15k", "15-25k", "25k+"])
        self._delta_bins = w.get("delta_bins", [0, 0.5/24, 1/24, 2/24, float("inf")])
        self._delta_labels = w.get("delta_labels", ["<30мин", "30-60мин", "1-2ч", ">2ч"])
        self._manager_bins = w.get("manager_bins", [0, 800, 1800, 3000, float("inf")])
        self._manager_labels = w.get("manager_labels", ["0-800", "800-1.8k", "1.8-3k", "3k+"])

        cities = w["russia_cities"]
        self._sorted_cities = sorted(cities, key=lambda c: len(c["name"]), reverse=True)

    # ------------------------------------------------------------------ public

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Бинарные предсказания (0 — отказ, 1 — выкуп)."""
        return (self.predict_proba(df) >= self._threshold).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Вероятность выкупа P(buyout=1) для каждой строки."""
        df = self._preprocess(df)
        idx_new, idx_ret = self._route(df)

        proba = np.full(len(df), np.nan)

        if len(idx_ret) > 0:
            proba[idx_ret] = self._predict_returning(df.iloc[idx_ret])

        if len(idx_new) > 0:
            proba[idx_new] = self._predict_new(df.iloc[idx_new])

        return proba

    # --------------------------------------------------------------- preprocessing

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Типы
        df["sale_ts"] = pd.to_numeric(df.get("sale_ts"), errors="coerce")
        df["lead_created_at"] = pd.to_numeric(df.get("lead_created_at"), errors="coerce")
        df["sale_date"] = pd.to_datetime(df.get("sale_date"), errors="coerce")
        df["lead_price"] = pd.to_numeric(df.get("lead_price"), errors="coerce")
        df["contact_Число сделок"] = pd.to_numeric(df.get("contact_Число сделок"), errors="coerce")

        for col in ["lead_Скидка", "lead_Стоимость доставки", "lead_Масса (гр)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # NaN → '__NaN__' для категориальных с nan=category.
        # Признаки с nan=drop (lead_Квалификация лида, lead_Категория...,
        # lead_Модель телефона, lead_group_id) оставляют NaN для all-zeros one-hot.
        for col in ["lead_Тариф Доставки", "lead_Служба доставки",
                     "lead_Вид оплаты", "lead_Проблема"]:
            if col in df.columns:
                df[col] = df[col].fillna("__NaN__")

        # Объединение мелких категорий (зеркалит features.yaml)
        if "lead_Квалификация лида" in df.columns:
            df["lead_Квалификация лида"] = df["lead_Квалификация лида"].replace(
                {"Неквал лид": "D/Неквал лид", "D - лид": "D/Неквал лид"})
        if "lead_Вид оплаты" in df.columns:
            df["lead_Вид оплаты"] = df["lead_Вид оплаты"].replace(
                {"__NaN__": "Прочее", "Оплата Золотой Короной": "Прочее"})
        if "lead_Тариф Доставки" in df.columns:
            df["lead_Тариф Доставки"] = df["lead_Тариф Доставки"].replace({
                "Почта Посылка": "Почта",
                "__NaN__": "Почта",
                "Экономичная посылка склад-склад": "Прочее",
                "Почта Посылка 1 класса": "Прочее",
                "Экономичная посылка склад-дверь": "Прочее",
                "Экспресс склад-дверь": "Прочее",
                "Экспресс склад-склад": "Прочее",
                "Посылка склад-постамат": "Прочее",
            })
        if "lead_Служба доставки" in df.columns:
            df["lead_Служба доставки"] = df["lead_Служба доставки"].replace({
                "СДЭК до ПВЗ": "СДЭК",
                "СДЭК до Двери": "СДЭК",
                "Самовывоз": "Прочее",
                "__NaN__": "Прочее",
                "Курьер ЕМС": "Прочее",
            })
        if "lead_Проблема" in df.columns:
            df["lead_Проблема"] = df["lead_Проблема"].replace({
                # first — высокий объём, ~30% buyout
                "Суставы и позвоночник": "first",
                "Варикоз": "first",
                "Сердечно-сосудистые заболевания": "first",
                # second — повышенный buyout (~42-56%)
                "Бессоница": "second",
                "Головные боли": "second",
                "Зрительная система": "second",
                # third — пониженный buyout + __NaN__
                "Отеки": "third",
                "Инсульт": "third",
                "Давление": "third",
                "Боли и тяжесть в ногах": "third",
                "Ушибы. травмы, отеки, ожоги": "third",
                "__NaN__": "third",
                # Прочее — хвост
                "Мочеполовая система": "Прочее",
                "Мужские проблемы": "Прочее",
                "Женские проблемы": "Прочее",
                "Послеоперационная реабилитация": "Прочее",
                "Онкология": "Прочее",
                "Ухо, горло, нос": "Прочее",
                "Сахарный диабет": "Прочее",
                "Проблемная кожа": "Прочее",
                "Желчный пузырь": "Прочее",
                "Щитовидная железа": "Прочее",
                "Склероз, деменция": "Прочее",
                "Дыхательные пути": "Прочее",
                "Волосы": "Прочее",
                "Мозоли, шпоры, шишки": "Прочее",
                "Животные": "Прочее",
                "Грибок ногтей": "Прочее",
                "Детские заболевая": "Прочее",
                "Ковид и ОРВИ": "Прочее",
                "Эпилепсия": "Прочее",
            })

        # sale_weekday
        df["sale_weekday"] = df["sale_date"].dt.dayofweek.astype(str)

        # sale_delta → delta_bin
        df["sale_delta"] = ((df["sale_ts"] - df["lead_created_at"]) / 86400).clip(lower=0).fillna(0)
        df["delta_bin"] = pd.cut(
            df["sale_delta"], bins=self._delta_bins, labels=self._delta_labels,
            include_lowest=True).astype(str)

        # Бинарные
        paid = ["cpc", "cpc__rt_view-yes_lead-no_all", "Bloger", "article_direct", "cpm"]
        df["is_paid_traffic"] = df.get("lead_utm_medium", pd.Series(dtype=str)).isin(paid).astype(int)

        df["has_future_purchase"] = (
            df.get("lead_будущие покупки", pd.Series(dtype=object)).notna().astype(int)
        )

        # cart_n_items → cart_bin
        df["cart_n_items"] = df.get("lead_Состав заказа", pd.Series(dtype=str)).apply(self._count_items)
        df["cart_bin"] = pd.cut(
            df["cart_n_items"], bins=self._cart_bins, labels=self._cart_labels).astype(str)

        # price_bin
        df["price_bin"] = pd.cut(
            df["lead_price"], bins=self._price_bins, labels=self._price_labels).astype(str)
        df.loc[df["lead_price"].isna(), "price_bin"] = "unknown"

        # ID → str (NaN-preserving: nan:drop features keep NaN for all-zeros one-hot)
        if "lead_group_id" in df.columns:
            s = pd.to_numeric(df["lead_group_id"], errors="coerce")
            na_mask = s.isna()
            df["lead_group_id"] = s.astype("Int64").astype(str)
            df.loc[na_mask, "lead_group_id"] = np.nan
        if "lead_responsible_user_id" in df.columns:
            df["lead_responsible_user_id"] = df["lead_responsible_user_id"].astype(str)

        # manager_deal_count → manager_bin
        df["manager_deal_count"] = (
            df["lead_responsible_user_id"]
            .map(self._manager_deal_count_map)
            .fillna(self._manager_deal_count_default)
        )
        df["manager_bin"] = pd.cut(
            df["manager_deal_count"], bins=self._manager_bins,
            labels=self._manager_labels).astype(str)

        # Гео
        cities, regions = self._geo_match(df.get("contact_Город", pd.Series(dtype=str)))
        df["city_clean"] = cities
        df["contact_region"] = regions

        return df

    # --------------------------------------------------------------- routing

    def _route(self, df: pd.DataFrame) -> tuple:
        deals = df["contact_Число сделок"]
        is_new = deals.isna() | (deals < 1)
        idx_new = np.where(is_new)[0].tolist()
        idx_ret = np.where(~is_new)[0].tolist()
        return idx_new, idx_ret

    # --------------------------------------------------------------- returning users

    def _predict_returning(self, df: pd.DataFrame) -> np.ndarray:
        X = df[["contact_Число сделок"]].values
        X_scaled = self._scaler_ret.transform(X)
        return self._model_ret.predict_proba(X_scaled)[:, 1]

    # --------------------------------------------------------------- new users

    def _predict_new(self, df: pd.DataFrame) -> np.ndarray:
        parts = []

        # One-hot (per-column drop_first; NaN → all-zeros via reindex fill_value=0)
        oh_parts = []
        for col in self._cat_cols:
            df_flag = self._cat_drop_first.get(col, False)
            oh_parts.append(pd.get_dummies(df[[col]], drop_first=df_flag))
        onehot = pd.concat(oh_parts, axis=1)
        onehot = onehot.reindex(columns=self._onehot_columns, fill_value=0)
        parts.append(onehot.values)

        # Числовые (опционально — может быть пусто)
        if self._num_cols:
            num = df[self._num_cols].fillna(0).values
            parts.append(self._scaler_new.transform(num))

        # Бинарные
        if self._bin_cols:
            parts.append(df[self._bin_cols].values)

        # Target encoding
        te_cols = self._geo_cols + self._te_cat_cols
        te_vals = np.zeros((len(df), len(te_cols)))
        for j, col in enumerate(te_cols):
            mapping = self._te_maps[col]
            te_vals[:, j] = df[col].map(mapping).fillna(self._global_mean).values
        parts.append(te_vals)

        X = np.hstack(parts)
        return self._model_new.predict_proba(X)[:, 1]

    # --------------------------------------------------------------- helpers

    def _geo_match(self, addresses: pd.Series) -> tuple:
        norm = lambda s: str(s).lower().replace("ё", "е")
        cities_out, regions_out = [], []
        for addr in addresses.values:
            if pd.isna(addr):
                cities_out.append("__unknown__")
                regions_out.append("__unknown__")
                continue
            addr_n = norm(addr)
            matched = False
            for c in self._sorted_cities:
                if norm(c["name"]) in addr_n:
                    cities_out.append(c["name"])
                    regions_out.append(c["region"])
                    matched = True
                    break
            if not matched:
                cities_out.append("__unknown__")
                regions_out.append("__unknown__")
        return cities_out, regions_out

    @staticmethod
    def _count_items(composition) -> int:
        if pd.isna(composition) or not isinstance(composition, str):
            return 0
        lines = [l.strip() for l in composition.replace(";", "\n").split("\n") if l.strip()]
        items = [l for l in lines if "доставк" not in l.lower()]
        return max(len(items), 1) if items else 0
