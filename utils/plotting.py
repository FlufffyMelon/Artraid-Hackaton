"""Графики для ноутбуков.

Все функции generic — принимают датафрейм и имена колонок, не хардкодят
конкретные признаки домена. Цветовая схема общая: зелёный = выкуп,
красный = отказ, серый = нет данных.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --- цвета ---
GREEN = "#2ecc71"
RED = "#e74c3c"
LIGHT_GRAY = "#ecf0f1"
BLUE = "#3498db"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
DARK_BLUE = "#2980b9"

# --- дефолтные порядки категорий (для визуализации только) ---
LEAD_QUAL_ORDER = [
    "А - лид", "В - лид", "С - лид", "D/Неквал лид", "D - лид", "Е - лид", "Неквал лид",
]

DEFAULT_CATEGORY_ORDERS: dict[str, list[str]] = {
    "price_bin": ["0-5k", "5-8k", "8-15k", "15k+", "unknown"],
    "cart_bin": ["1-8", "9-12", "13+"],
    "manager_bin": ["0-800", "800-1.8k", "1.8-3k", "3k+"],
    "delta_bin": ["<30мин", "30-60мин", "1-2ч", ">2ч"],
    "order_text_bin": ["empty", "1-50", "51-100", "101-200", "201-500", "500+"],
    "sale_weekday": ["0", "1", "2", "3", "4", "5", "6"],
    "lead_Квалификация лида": LEAD_QUAL_ORDER,
}


# ============================================================
# Feature overview — categorical / numeric dispatch
# ============================================================


def _is_numeric_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 15


def _ordered_categories(
    col: str,
    present: Iterable,
    n_tot: pd.Series,
    category_orders: dict[str, list[str]],
) -> list:
    present = set(present)

    def by_total_desc(keys):
        return sorted(keys, key=lambda x: -int(n_tot.loc[x]))

    if col in category_orders:
        known = [c for c in category_orders[col] if c in present]
        rest = by_total_desc([c for c in present if c not in known])
        return known + rest
    return by_total_desc(list(present))


def plot_feature_two_panel(
    df: pd.DataFrame,
    col: str,
    *,
    target_col: str = "buyout_flag",
    title: str | None = None,
    top_n: int | None = None,
    bins: int = 30,
    xlim: tuple[float | None, float | None] | None = None,
    category_orders: dict[str, list[str]] | None = None,
    max_categories_auto: int = 30,
) -> Figure:
    """Обзор признака: верхняя панель — объём (в разбиении на выкупы/отказы),
    нижняя — доля выкупа по значениям / бинам. Числовые колонки идут через
    гистограмму, остальные — через категориальные бары.
    """
    if title is None:
        title = col
    series = df[col]
    if _is_numeric_column(series):
        return _plot_numeric_two_panel(df, col, target_col, title, bins=bins, xlim=xlim)
    orders = category_orders if category_orders is not None else DEFAULT_CATEGORY_ORDERS
    return _plot_categorical_two_panel(
        df, col, target_col, title, top_n=top_n, category_orders=orders,
        max_categories_auto=max_categories_auto,
    )


def _plot_categorical_two_panel(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    title: str,
    *,
    top_n: int | None,
    category_orders: dict[str, list[str]],
    max_categories_auto: int,
) -> Figure:
    work = df
    unique_count = df[col].nunique(dropna=False)
    if top_n is None and unique_count > max_categories_auto:
        top_n = max_categories_auto
    if top_n is not None:
        keep = df[col].value_counts().head(top_n).index
        work = df.loc[df[col].isin(keep)]

    grouped = work.groupby(col, observed=True)[target_col]
    n_buy = grouped.sum()
    n_tot = grouped.count()
    n_cancel = n_tot - n_buy
    rate = (n_buy / n_tot.replace(0, np.nan)).fillna(0.0)

    order = _ordered_categories(col, n_buy.index, n_tot, category_orders)
    n_buy = n_buy.reindex(order, fill_value=0)
    n_cancel = n_cancel.reindex(order, fill_value=0)
    rate = rate.reindex(order, fill_value=0)

    labels = [str(x) for x in order]
    x = np.arange(len(labels))
    w = 0.36

    fw, fh = (14, 9) if (top_n is not None and top_n >= 15) else (11, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fw, fh), sharex=True)
    fig.suptitle(title, fontsize=13)

    ax1.bar(x - w / 2, n_buy.values, width=w, color=GREEN, label="Выкуп")
    ax1.bar(x + w / 2, n_cancel.values, width=w, color=RED, label="Отмена")
    ax1.set_ylabel("Количество")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    n_tot_vals = (n_buy + n_cancel).values
    for i in range(len(x)):
        if n_tot_vals[i] == 0:
            ax2.bar(x[i], 1.0, color=LIGHT_GRAY, edgecolor="#bdc3c7", linewidth=0.5)
        else:
            ax2.bar(x[i], rate.values[i], color=GREEN)
            ax2.bar(x[i], 1.0 - rate.values[i], bottom=rate.values[i], color=RED)

    ax2.set_ylabel("Доля")
    ax2.set_ylim(0, 1)
    legend = [Patch(color=GREEN, label="Доля выкупа"), Patch(color=RED, label="Доля отмен")]
    if 0 in n_tot_vals:
        legend.append(Patch(color=LIGHT_GRAY, label="Нет данных"))
    ax2.legend(handles=legend, loc="upper right")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    return fig


def _plot_numeric_two_panel(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    title: str,
    *,
    bins: int,
    xlim: tuple[float | None, float | None] | None,
) -> Figure:
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    if not mask.any():
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"{col}: нет данных", ha="center", va="center")
        ax.set_axis_off()
        return fig

    vals = s[mask].to_numpy(dtype=float)
    buy = df.loc[mask, target_col].astype(bool).to_numpy()
    hist_range = None
    if xlim is not None:
        lo, hi = xlim
        lo = float(vals.min()) if lo is None else lo
        hi = float(vals.max()) if hi is None else hi
        m = (vals >= lo) & (vals <= hi)
        vals, buy = vals[m], buy[m]
        hist_range = (lo, hi)

    edges = np.histogram_bin_edges(vals, bins=bins, range=hist_range)
    nbins = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nbins - 1)
    n_tot = np.bincount(idx, minlength=nbins)[:nbins].astype(int)
    n_buy = np.bincount(idx[buy], minlength=nbins)[:nbins].astype(int)
    n_cancel = n_tot - n_buy
    rate = np.divide(n_buy, n_tot, out=np.full(nbins, np.nan), where=n_tot > 0)
    left, widths = edges[:-1], np.diff(edges)
    mean_v = float(vals.mean())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25]},
    )
    fig.suptitle(f"{title} (bins={nbins})", fontsize=13)

    ax1.hist(vals[buy], bins=edges, color=GREEN, alpha=0.55, label="Выкуп", edgecolor="none")
    ax1.hist(vals[~buy], bins=edges, color=RED, alpha=0.55, label="Отмена", edgecolor="none")
    ax1.axvline(mean_v, color="black", linestyle="--", linewidth=1, label="Среднее", zorder=5)
    ax1.set_ylabel("Количество")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    for i in range(nbins):
        if n_tot[i] == 0:
            continue
        ax2.bar(left[i], rate[i], width=widths[i], align="edge", color=GREEN)
        ax2.bar(left[i], 1.0 - rate[i], width=widths[i], bottom=rate[i], align="edge", color=RED)
    ax2.set_ylabel("Доля")
    ax2.set_xlabel(col)
    ax2.set_ylim(0, 1)
    ax2.axvline(mean_v, color="black", linestyle="--", linewidth=1, zorder=5)
    ax2.legend(
        handles=[Patch(color=GREEN, label="Доля выкупа"), Patch(color=RED, label="Доля отмен")],
        loc="upper right",
    )
    ax2.grid(axis="y", alpha=0.3)
    if hist_range is not None:
        ax2.set_xlim(*hist_range)

    fig.tight_layout()
    return fig


# ============================================================
# Monthly dynamics (old vs new)
# ============================================================


def plot_old_vs_new_timeline(
    df: pd.DataFrame,
    *,
    date_col: str = "sale_date",
    target_col: str = "buyout_flag",
    deal_count_col: str = "contact_Число сделок",
    deal_threshold: float = 1.0,
) -> Figure:
    """Две панели: объём заказов по месяцам и доля выкупа по месяцам —
    отдельно для новых и повторных клиентов.
    """
    work = df.copy()
    work["_month"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M")
    work = work.dropna(subset=["_month"])
    is_new = pd.to_numeric(work[deal_count_col], errors="coerce").fillna(0) < deal_threshold

    monthly = (
        work.groupby(["_month", is_new.rename("is_new")])
        .agg(orders=(target_col, "size"), buyout_rate=(target_col, "mean"))
        .reset_index()
    )
    new_m = monthly[monthly["is_new"]].set_index("_month").sort_index()
    ret_m = monthly[~monthly["is_new"]].set_index("_month").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(new_m.index.astype(str), new_m["orders"], marker="o", color=RED, label="Новые клиенты")
    ax.plot(ret_m.index.astype(str), ret_m["orders"], marker="s", color=GREEN, label="Повторные клиенты")
    ax.set_title("Число заказов по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Число заказов")
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    ax.plot(new_m.index.astype(str), new_m["buyout_rate"] * 100, marker="o", color=RED, label="Новые клиенты")
    ax.plot(ret_m.index.astype(str), ret_m["buyout_rate"] * 100, marker="s", color=GREEN, label="Повторные клиенты")
    ax.set_title("Buyout rate по месяцам")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Buyout rate, %")
    ax.legend(loc="center right")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


# ============================================================
# Drift analysis (per feature)
# ============================================================


def plot_feature_drift(
    df: pd.DataFrame,
    feature: str,
    *,
    target_col: str = "buyout_flag",
    date_col: str = "sale_date",
    cutoff: str = "2025-10-01",
    title: str | None = None,
    max_categories: int = 8,
    category_orders: dict[str, list[str]] | None = None,
) -> Figure:
    """Один признак, две панели: слева — распределение категорий до/после
    cutoff, справа — доля выкупа в каждой категории до/после.
    """
    if title is None:
        title = feature
    orders = category_orders if category_orders is not None else DEFAULT_CATEGORY_ORDERS

    dates = pd.to_datetime(df[date_col], errors="coerce")
    pre_mask = dates < cutoff
    post_mask = ~pre_mask & dates.notna()

    series = df[feature].astype("string").fillna("__NaN__")
    top = series.value_counts().iloc[:max_categories].index
    labels = _ordered_categories(feature, top, series.value_counts(), orders)
    x = np.arange(len(labels))
    w = 0.36

    pre_share = [(series[pre_mask] == v).mean() for v in labels]
    post_share = [(series[post_mask] == v).mean() for v in labels]
    pre_rate = [df.loc[pre_mask & (series == v), target_col].mean() for v in labels]
    post_rate = [df.loc[post_mask & (series == v), target_col].mean() for v in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    ax.bar(x - w / 2, pre_share, width=w, color=BLUE, label="До октября")
    ax.bar(x + w / 2, post_share, width=w, color=ORANGE, label="После октября")
    ax.set_ylabel("Доля заказов")
    ax.set_title("Распределение категорий")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w / 2, pre_rate, width=w, color=BLUE, label="До октября")
    ax.bar(x + w / 2, post_rate, width=w, color=ORANGE, label="После октября")
    ax.set_ylabel("Доля выкупа")
    ax.set_ylim(0, 1)
    ax.set_title("Buyout rate внутри категории")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def drift_summary_table(
    df: pd.DataFrame,
    spec: list[dict],
    *,
    date_col: str = "sale_date",
    cutoff: str = "2025-10-01",
) -> pd.DataFrame:
    """Табличка «до октября / после октября» для произвольных метрик.

    ``spec`` — список словарей вида
    ``{'label': ..., 'column': ..., 'agg': 'mean' | 'share' | 'rate', 'value': ..., 'format': ...}``.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")
    pre = dates < cutoff
    post = ~pre & dates.notna()

    rows: list[tuple[str, str, str]] = []
    for row in spec:
        label = row["label"]
        agg = row["agg"]
        fmt = row.get("format", "{:.3f}")
        if agg == "mean":
            col = row["column"]
            pre_v = pd.to_numeric(df.loc[pre, col], errors="coerce").mean()
            post_v = pd.to_numeric(df.loc[post, col], errors="coerce").mean()
        elif agg == "share":
            col = row["column"]
            value = str(row["value"])
            s = df[col].astype("string")
            pre_v = (s[pre] == value).mean()
            post_v = (s[post] == value).mean()
        elif agg == "rate":
            col = row.get("column") or row.get("target_col", "buyout_flag")
            if "value" in row:
                m = df[row["column"]].astype("string") == str(row["value"])
                pre_v = df.loc[pre & m, row.get("target_col", "buyout_flag")].mean()
                post_v = df.loc[post & m, row.get("target_col", "buyout_flag")].mean()
            else:
                pre_v = df.loc[pre, col].mean()
                post_v = df.loc[post, col].mean()
        else:
            raise ValueError(f"unknown agg: {agg!r}")
        rows.append((label, fmt.format(pre_v), fmt.format(post_v)))

    return pd.DataFrame(rows, columns=["Показатель", "До октября", "После октября"])


# ============================================================
# Cumulative-month line plot
# ============================================================


def plot_cumulative_metrics(
    results: list[dict],
    *,
    metric_cols: Iterable[str] = ("test_auc", "test_f1"),
    title: str | None = None,
    boundary_month: str | None = "2025-10",
) -> Figure:
    if not results:
        raise ValueError("results is empty")
    months = [r["test_month"] for r in results]
    xs = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        xs,
        [r["test_buyout"] for r in results],
        alpha=0.18, color="gray", width=0.7, label="доля выкупа (факт)",
        zorder=2
    )
    ax.set_ylabel("Доля выкупа", color="gray")
    ax.tick_params(axis="y", labelcolor="gray")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.grid(axis='y', alpha=0.3, zorder=1)

    ax2 = ax.twinx()
    palette = {"test_auc": BLUE, "test_f1": GREEN}
    labels = {"test_auc": "ROC-AUC", "test_f1": "Best F1"}
    for col in metric_cols:
        ax2.plot(
            xs, [r.get(col) for r in results],
            "o-", color=palette.get(col, PURPLE), linewidth=2, markersize=6,
            label=labels.get(col, col),
            zorder=3
        )
    ax2.set_ylabel("Метрика")
    ax2.set_ylim(0, 1)

    if boundary_month and boundary_month in months:
        idx = months.index(boundary_month)
        ax.axvline(idx - 0.5, color=RED, linestyle=":", linewidth=1.5, alpha=0.7)

    if title:
        ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ============================================================
# Sigmoid fit (returning-clients model)
# ============================================================


def plot_sigmoid_fit(
    x_range: np.ndarray,
    proba: np.ndarray,
    actual_stats: pd.DataFrame,
    *,
    xlabel: str,
    title: str,
    x_max: float | None = None,
) -> Figure:
    """Кривая вероятности модели + фактические точки по числу прошлых сделок.

    ``actual_stats`` — датафрейм с колонками ``buyout_rate``, ``n_buyout``,
    ``n_total`` и индексом = integer значение признака.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range, proba, color=BLUE, lw=2.5, label="Модель: P(выкуп)", zorder=3)

    sizes = (actual_stats["n_total"] / actual_stats["n_total"].max()) * 300 + 50
    ax.scatter(
        actual_stats.index, actual_stats["buyout_rate"],
        s=sizes, color=GREEN, edgecolor="black", linewidth=1, zorder=4,
        label="Факт: доля выкупа",
    )
    for deals, row in actual_stats.iterrows():
        offset = -0.07 if row["buyout_rate"] > 0.5 else 0.05
        ax.annotate(
            f"{row['buyout_rate']:.1%}\n({int(row['n_buyout'])}/{int(row['n_total'])})",
            xy=(deals, row["buyout_rate"]),
            xytext=(deals, row["buyout_rate"] + offset),
            ha="center", fontsize=8,
        )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Порог 0.5")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("P(выкуп)", fontsize=12)
    ax.set_title(title, fontsize=13)

    if x_max is None:
        x_max = float(max(actual_stats.index.max(), x_range.max()))
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, int(x_max) + 1))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# Final-model panels (CM + ROC + PR)
# ============================================================


def plot_final_metrics_panels(
    y_test: np.ndarray,
    proba_test: np.ndarray,
    *,
    threshold: float = 0.5,
    y_val: np.ndarray | None = None,
    proba_val: np.ndarray | None = None,
    labels: tuple[str, str] = ("Отказ", "Выкуп"),
    title: str | None = None,
) -> Figure:
    """Три панели: Confusion Matrix (нормированная по строкам) + ROC + PR.
    Если переданы val-данные, они показываются пунктиром рядом с test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    y_pred = (np.asarray(proba_test) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=list(labels),
        normalize="true", values_format=".1%",
        ax=axes[0], cmap="Blues",
    )
    axes[0].set_title(f"Confusion Matrix (порог = {threshold:.2f})")

    fpr_te, tpr_te, _ = roc_curve(y_test, proba_test)
    auc_te = roc_auc_score(y_test, proba_test)
    axes[1].plot(fpr_te, tpr_te, lw=2, color=BLUE, label=f"Test AUC = {auc_te:.4f}")
    if y_val is not None and proba_val is not None:
        fpr_va, tpr_va, _ = roc_curve(y_val, proba_val)
        auc_va = roc_auc_score(y_val, proba_val)
        axes[1].plot(fpr_va, tpr_va, lw=2, linestyle="--", color=ORANGE, label=f"Val AUC = {auc_va:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # prec_te, rec_te, _ = precision_recall_curve(y_test, proba_test)
    # ap_te = average_precision_score(y_test, proba_test)
    # axes[2].plot(rec_te, prec_te, lw=2, color=BLUE, label=f"Test AP = {ap_te:.4f}")
    # if y_val is not None and proba_val is not None:
    #     prec_va, rec_va, _ = precision_recall_curve(y_val, proba_val)
    #     ap_va = average_precision_score(y_val, proba_val)
    #     axes[2].plot(rec_va, prec_va, lw=2, linestyle="--", color=ORANGE, label=f"Val AP = {ap_va:.4f}")
    # axes[2].set_xlabel("Recall")
    # axes[2].set_ylabel("Precision")
    # axes[2].set_title("Precision-Recall")
    # axes[2].legend()
    # axes[2].grid(alpha=0.3)

    if title:
        fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================
# Threshold sweep
# ============================================================


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "Precision / Recall / F1 vs threshold",
) -> tuple[Figure, float, float]:
    """Возвращает фигуру и ``(best_threshold, best_f1)``.

    На графике отрисованы три кривые и горизонтальная линия ROC-AUC; отмечен
    лучший порог по F1 и базовый 0.5.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    ts = np.arange(0.05, 0.96, 0.01)
    precisions, recalls, f1s, valid_t = [], [], [], []
    for t in ts:
        pred = (y_proba >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        valid_t.append(t)
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))

    auc_val = roc_auc_score(y_true, y_proba)
    best_idx = int(np.argmax(f1s))
    best_t = float(valid_t[best_idx])
    best_f1 = float(f1s[best_idx])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valid_t, precisions, color=ORANGE, lw=2, label="Precision")
    ax.plot(valid_t, recalls, color=DARK_BLUE, lw=2, label="Recall")
    ax.plot(valid_t, f1s, color=GREEN, lw=2.5, label="F1")
    ax.axhline(auc_val, color=PURPLE, linestyle="--", lw=1.5, alpha=0.8,
               label=f"ROC-AUC = {auc_val:.4f}")

    ax.axvline(best_t, color="gray", linestyle=":", alpha=0.7)
    ax.scatter([best_t], [best_f1], color=GREEN, s=100, zorder=5, edgecolor="black")
    ax.annotate(
        f"Best F1={best_f1:.3f}\nThreshold={best_t:.2f}",
        xy=(best_t, best_f1),
        xytext=(best_t + 0.08, best_f1 + 0.05),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    # ax.axvline(0.5, color=RED, linestyle="--", alpha=0.5, lw=1)
    # ax.text(0.51, 0.15, "Порог 0.5", color=RED, fontsize=9)

    ax.set_xlabel("Порог вероятности", fontsize=12)
    ax.set_ylabel("Значение метрики", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig, best_t, best_f1


# ============================================================
# Feature importance
# ============================================================


def plot_feature_importance(
    values,
    feature_names: list[str],
    *,
    top_n: int = 25,
    group_map: dict[str, str] | None = None,
    title: str | None = None,
) -> Figure:
    values = np.asarray(values).ravel()
    if group_map is not None:
        grouped: dict[str, float] = {}
        for name, v in zip(feature_names, values):
            key = group_map.get(name, name)
            grouped[key] = grouped.get(key, 0.0) + abs(float(v))
        series = pd.Series(grouped).sort_values(ascending=False)
    else:
        series = pd.Series(np.abs(values), index=feature_names).sort_values(ascending=False)
    series = series.iloc[:top_n].iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(series))))
    ax.barh(series.index, series.values, color=BLUE, alpha=0.85)
    ax.set_xlabel("Важность")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ============================================================
# Metrics table (tabulate)
# ============================================================


_METRIC_LABELS: dict[str, str] = {
    "threshold": "Порог решения",
    "n": "Объём выборки",
    "positive_rate": "Доля выкупа (факт)",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "Matthews corrcoef",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def format_metrics_table(
    metrics: dict,
    *,
    title: str | None = None,
    tablefmt: str = "github",
) -> str:
    """Красивое строковое представление набора метрик через ``tabulate``."""
    from tabulate import tabulate

    rows: list[tuple[str, str]] = []
    for key, label in _METRIC_LABELS.items():
        if key not in metrics or metrics[key] is None:
            continue
        value = metrics[key]
        if key in {"n"}:
            formatted = f"{int(value)}"
        elif key in {"positive_rate", "accuracy", "balanced_accuracy",
                     "precision", "recall", "f1"}:
            formatted = f"{value:.1%}"
        elif key in {"roc_auc", "pr_auc", "mcc"}:
            formatted = f"{value:.4f}"
        else:
            formatted = f"{value:.2f}"
        rows.append((label, formatted))

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        rows.append(("Confusion matrix",
                     f"TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}"))

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(tabulate(rows, headers=["Метрика", "Значение"], tablefmt=tablefmt))
    return "\n".join(lines)


def print_metrics(metrics: dict, *, title: str | None = None) -> None:
    print(format_metrics_table(metrics, title=title))


def format_comparison_table(
    comparison: dict[str, dict],
    *,
    title: str | None = None,
    tablefmt: str = "github",
) -> str:
    """Сравнение нескольких моделей: словарь ``{model_name: metrics_dict}``."""
    from tabulate import tabulate

    model_names = list(comparison.keys())
    rows: list[list[str]] = []
    for key, label in _METRIC_LABELS.items():
        if all(comparison[m].get(key) is None for m in model_names):
            continue
        row = [label]
        for m in model_names:
            v = comparison[m].get(key)
            if v is None:
                row.append("—")
            elif key == "n":
                row.append(f"{int(v)}")
            elif key in {"positive_rate", "accuracy", "balanced_accuracy",
                        "precision", "recall", "f1"}:
                row.append(f"{v:.1%}")
            elif key in {"roc_auc", "pr_auc", "mcc"}:
                row.append(f"{v:.4f}")
            else:
                row.append(f"{v:.2f}")
        rows.append(row)

    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(tabulate(rows, headers=["Метрика"] + model_names, tablefmt=tablefmt))
    return "\n".join(lines)
