"""Data loading, cleaning, context building and leakage-column catalogue.

Everything here is intentionally CRM-schema-specific (столбцы датасета хакатона),
unlike `utils.features` which is feature-config driven.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "buyout_flag"
DATE_COL = "sale_date"
CONTACT_COL = "contact_id"
DEAL_COUNT_COL = "contact_Число сделок"


# ============================================================
# Leakage catalogue
# ============================================================

# Колонки, которые заполняются ПОСЛЕ выяснения исхода заказа (выкуп / отказ)
# либо напрямую являются производными от исхода. Использовать их в обучении
# запрещено — приведёт к утечке таргета.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    # Статусы в CRM, меняющиеся после исхода
    "current_status_id",
    "lead_status_id",
    "lead_closed_at",
    "closed_ts",
    # Временные метки исхода
    "received_ts",
    "rejected_ts",
    "returned_ts",
    "handed_to_delivery_ts",
    "issued_or_pvz_ts",
    # Дельты между временными метками исхода
    "days_to_outcome",
    "days_handed_to_issued_pvz",
    "days_sale_to_handed",
    # Явные post-hoc поля
    "lead_Условный отказ",
    "lead_Дата получения денег на Р/С",
    "lead_Дата возврата посылки на склад",
    "lead_Дата перехода Передан в доставку",
    "lead_Дата перехода в Сборку",
    "lead_Ответственный за доставку",
    "lead_LEADQUALIFYCATION",
    # Размерный набор №2 (заполняется при возврате) — см. feature_recommendations.md раздел 6
    "lead_Вес (грамм)*",
    "lead_Длина",
    "lead_Ширина",
    "lead_Высота",
    # Поля-константы, не несут сигнала
    "lifecycle_incomplete",
    "lead_is_deleted",
    "lead_account_id",
)


# ============================================================
# Loading
# ============================================================


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Прочитать датасет AmoCRM, привести типы и отфильтровать записи с
    неизвестным исходом сделки.

    Гарантирует: ``buyout_flag`` есть и приведён к {0, 1}; числовые
    временные метки переведены в numeric; ``sale_date`` — datetime.
    """
    df = pd.read_csv(path, low_memory=False)

    if "outcome_unknown" in df.columns:
        df = df[df["outcome_unknown"] != True].copy()

    df[TARGET] = (
        df[TARGET]
        .map({"True": 1, "true": 1, True: 1, "False": 0, "false": 0, False: 0})
        .astype(int)
    )
    df["sale_ts"] = pd.to_numeric(df.get("sale_ts"), errors="coerce")
    df["lead_created_at"] = pd.to_numeric(df.get("lead_created_at"), errors="coerce")
    df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")
    df["lead_price"] = pd.to_numeric(df.get("lead_price"), errors="coerce")
    df[DEAL_COUNT_COL] = pd.to_numeric(df.get(DEAL_COUNT_COL), errors="coerce").fillna(0)

    drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    return df


# ============================================================
# Historical deal count fix
# ============================================================


def correct_historical_deal_count(df: pd.DataFrame) -> pd.DataFrame:
    """Восстановить ``contact_Число сделок`` как историческое значение на момент
    текущего заказа.

    CSV-выгрузка хранит для каждой строки *текущее* общее число сделок контакта
    (snapshot последнего запроса к CRM), поэтому без коррекции ранние заказы
    повторных клиентов ошибочно попадают в класс «новые». Для каждого ``contact_id``
    заказы сортируются по ``lead_created_at``, и в поле записывается число
    предшествующих заказов этого контакта в исторической последовательности.

    Записи без ``contact_id`` остаются как есть (0 / прежнее значение).
    """
    df = df.copy()
    if CONTACT_COL not in df.columns:
        return df

    has_cid = df[CONTACT_COL].notna()
    w = df.loc[has_cid].sort_values([CONTACT_COL, "lead_created_at"])
    rank = w.groupby(CONTACT_COL).cumcount()
    total_in_dataset = w.groupby(CONTACT_COL)[CONTACT_COL].transform("size")
    last_known_count = w.groupby(CONTACT_COL)[DEAL_COUNT_COL].transform("last")

    # Если CRM-счётчик меньше, чем число наблюдаемых заказов — доверяем порядку.
    # Иначе считаем, что у клиента были более ранние заказы вне датасета.
    offset = last_known_count - total_in_dataset + 1
    corrected = np.where(last_known_count < total_in_dataset, rank, offset + rank)

    df.loc[w.index, DEAL_COUNT_COL] = corrected
    return df


# ============================================================
# Contexts for the feature pipeline
# ============================================================


def build_contexts(df: pd.DataFrame, cities_json_path: str | Path) -> dict:
    """Собрать вспомогательные таблицы, которые нужны операциям в
    ``features.yaml`` (map, substring_lookup).

    ``df`` — набор, из которого считаются статистики по менеджерам. Обычно
    передают подвыборку новых клиентов: она содержит нужное разнообразие и
    не разбавлена старыми клиентами с их особым поведением.
    """
    with open(cities_json_path, encoding="utf-8") as f:
        raw_cities = json.load(f)

    russia_cities = [
        {"name": c["name"], "region": c["region"]["fullname"]}
        for c in raw_cities
        if c.get("name")
    ]
    # Крымские города отсутствуют в открытом справочнике — добавляем руками.
    russia_cities.extend(
        [
            {"name": "Ялта", "region": "Республика Крым"},
            {"name": "Симферополь", "region": "Республика Крым"},
            {"name": "Севастополь", "region": "Город федерального значения Севастополь"},
        ]
    )

    manager_ids = df["lead_responsible_user_id"].astype(str)
    manager_map = manager_ids.groupby(manager_ids).size().to_dict()

    mgr_rates = df.groupby(manager_ids)[TARGET].mean()
    mgr_group_map = {
        mid: ("mgr_high" if rate >= 0.4 else "mgr_mid" if rate >= 0.2 else "mgr_low")
        for mid, rate in mgr_rates.items()
    }

    big_city_map = {
        c["name"]: 1 for c in raw_cities if c.get("population", 0) >= 500_000
    }

    return {
        "russia_cities": russia_cities,
        "manager_map": manager_map,
        "mgr_group_map": mgr_group_map,
        "big_city_map": big_city_map,
    }


# ============================================================
# Routing
# ============================================================


def split_new_returning(
    df: pd.DataFrame, threshold: float = 1.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить заказы на две когорты по числу прошлых сделок контакта.

    Новыми считаются заказы с ``contact_Число сделок < threshold``. Порог =
    1 означает «клиент делает свой первый заказ из наблюдаемых».
    """
    deals = pd.to_numeric(df[DEAL_COUNT_COL], errors="coerce").fillna(0)
    is_new = deals < threshold
    return df[is_new].copy(), df[~is_new].copy()


# ============================================================
# Minimal column selection for clean.csv
# ============================================================


def normalize_string_categoricals(
    df: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Привести перечисленные колонки к строковому представлению без
    хвоста ``.0``, который появляется после ``pd.read_csv`` на числовых
    категориях с пропусками (``546538.0`` → ``546538``). NaN сохраняется.

    Полезно перед подачей в CatBoost или ``pd.get_dummies``.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        na_mask = series.isna()
        if pd.api.types.is_numeric_dtype(series):
            out = (
                pd.to_numeric(series, errors="coerce")
                .astype("Int64")
                .astype("string")
                .astype(object)
            )
        else:
            out = series.astype("string").astype(object)
        out[na_mask] = np.nan
        df[col] = out
    return df


def select_canonical_columns(
    df: pd.DataFrame, feature_cols: list[str], extra: list[str] | None = None
) -> pd.DataFrame:
    """Оставить в датафрейме только те колонки, которые нужны обучению и
    служебным задачам (таргет, время, routing, сам набор признаков).

    ``extra`` можно передать, если нужно сохранить доп. метаданные (например,
    ``contact_id`` для воспроизводимых сплитов).
    """
    meta = [TARGET, DATE_COL, DEAL_COUNT_COL]
    if extra:
        meta = meta + [c for c in extra if c not in meta]
    keep = [c for c in meta + list(feature_cols) if c in df.columns]
    # Сохраняем порядок: сначала meta, потом признаки, без дубликатов.
    seen = set()
    ordered = [c for c in keep if not (c in seen or seen.add(c))]
    return df[ordered].copy()
