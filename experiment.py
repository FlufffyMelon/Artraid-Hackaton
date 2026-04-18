"""
Standalone experiment script for new-user LogReg optimization.

Reads processed_data.pkl (pre-split train/test/val with features already computed),
encodes features, trains LogisticRegression, and reports metrics on TEST and VAL.

Modify the EXPERIMENT PARAMETERS section below to try different configurations.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)

# ============================================================
# EXPERIMENT PARAMETERS — modify this section for each run
# ============================================================

# Regularization
PENALTY = "l1"
C = 4
SOLVER = "liblinear"
L1_RATIO = 0.7  # only used with elasticnet

# Target encoding alpha — read from features.yaml per feature
TE_ALPHA = 10            # fallback if not specified in YAML

# Per-feature drop_first for one-hot encoding (True = drop first category)
# Set to None to use defaults from processed_data.pkl (all False)
DROP_FIRST_OVERRIDE = None
# Example: DROP_FIRST_OVERRIDE = {"lead_Вид оплаты": True, "sale_weekday": True}

# Feature selection: set to None to use all, or list specific features to exclude
EXCLUDE_FEATURES = None
# Example: EXCLUDE_FEATURES = ["delta_bin", "manager_bin"]

# Category remapping (applied on already-grouped data from processed_data.pkl)
# Dict of {column: {old_value: new_value, ...}}
CATEGORY_REMAP = {
    "lead_Проблема": {"second": "high_buyout", "Прочее": "high_buyout"},
    "lead_Тариф Доставки": {"Посылка склад-дверь": "Склад", "Посылка склад-склад": "Склад"},
    "price_bin": {"0-3k": "0-5k", "3-5k": "0-5k", "15-25k": "15k+", "25k+": "15k+"},
    "cart_bin": {"13-16": "13+", "17+": "13+"},
}
# Example: CATEGORY_REMAP = {"lead_Проблема": {"second": "high_buyout", "Прочее": "high_buyout"}}

# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = "processed_data.pkl"
FEATURES_YAML = "features.yaml"

with open(DATA_PATH, "rb") as f:
    art = pickle.load(f)

import yaml
with open(FEATURES_YAML, encoding="utf-8") as f:
    feat_config = yaml.safe_load(f)

cat_cols = art["cat_cols"]
num_cols = art["num_cols"]
bin_cols = art["bin_cols"]
geo_cols = art["geo_cols"]
te_cat_cols = art["te_cat_cols"]
cat_drop_first = art["cat_drop_first"]

# Read per-feature alpha from features.yaml
TE_ALPHA_PER_FEATURE = {}
for fname, fspec in feat_config.get("features", {}).items():
    if "alpha" in fspec:
        TE_ALPHA_PER_FEATURE[fname] = fspec["alpha"]

all_feature_cols = cat_cols + num_cols + bin_cols + geo_cols + te_cat_cols

X_new_train = art["new_train"][all_feature_cols].copy()
y_new_train = art["new_train"]["buyout_flag"].copy()
X_new_test = art["new_test"][all_feature_cols].copy()
y_new_test = art["new_test"]["buyout_flag"].copy()
X_new_val = art["new_val"][all_feature_cols].copy()
y_new_val = art["new_val"]["buyout_flag"].copy()

# ============================================================
# APPLY OVERRIDES
# ============================================================

dfm = dict(cat_drop_first)
if DROP_FIRST_OVERRIDE is not None:
    dfm.update(DROP_FIRST_OVERRIDE)

# Category remapping (merge/rename values in pre-grouped data)
if CATEGORY_REMAP:
    for col, mapping in CATEGORY_REMAP.items():
        if col in X_new_train.columns:
            X_new_train[col] = X_new_train[col].replace(mapping)
            X_new_test[col] = X_new_test[col].replace(mapping)
            X_new_val[col] = X_new_val[col].replace(mapping)

# =============================================================
# DERIVED FEATURES (added directly to train/test/val DataFrames)
# =============================================================

# --- Manager as one-hot categorical instead of TE ---
MGR_AS_ONEHOT = True  # set to False to disable
if MGR_AS_ONEHOT:
    te_cat_cols = [c for c in te_cat_cols if c != "lead_responsible_user_id"]
    cat_cols = cat_cols + ["lead_responsible_user_id"]
    dfm["lead_responsible_user_id"] = False

# --- City-based binary features ---
CITY_FEATURES = False  # set to False to disable
if CITY_FEATURES:
    for X in [X_new_train, X_new_test, X_new_val]:
        X["is_unknown_city"] = (X["city_clean"] == "__unknown__").astype(int)
    bin_cols = bin_cols + ["is_unknown_city"]

# --- Manager buyout-rate group (categorical) ---
MANAGER_GROUPS = True  # set to False to disable
if MANAGER_GROUPS:
    # Group managers by their buyout rate: high (>=0.4), mid (0.2-0.4), low (<0.2)
    high_mgrs = ["MGR_0002", "MGR_0001", "MGR_0004"]  # rates 0.41-0.46
    mid_mgrs = ["MGR_0030", "MGR_0003", "MGR_0005", "MGR_0006"]  # rates 0.23-0.33
    # rest are low (<0.2)
    def mgr_group(uid):
        if uid in high_mgrs: return "mgr_high"
        if uid in mid_mgrs: return "mgr_mid"
        return "mgr_low"
    for X in [X_new_train, X_new_test, X_new_val]:
        X["mgr_group"] = X["lead_responsible_user_id"].apply(mgr_group)
    cat_cols = cat_cols + ["mgr_group"]
    dfm["mgr_group"] = False

# --- City population group (from russia-cities.json) ---
CITY_POP_GROUPS = True  # set to False to disable
if CITY_POP_GROUPS:
    import json
    with open("russia-cities.json", "r") as _f:
        _cities_data = json.load(_f)
    _city_pop = {c["name"]: c.get("population", 0) for c in _cities_data}
    def city_pop_group(city_name):
        if city_name == "__unknown__": return 0
        pop = _city_pop.get(city_name, 0)
        return 1 if pop >= 500_000 else 0
    for X in [X_new_train, X_new_test, X_new_val]:
        X["is_big_city"] = X["city_clean"].apply(city_pop_group)
    bin_cols = bin_cols + ["is_big_city"]

# --- Region group by federal district ---
REGION_GROUPS = False  # set to False to disable
if REGION_GROUPS:
    # Map regions to federal districts (simplified)
    central = ["Москва", "Московская область", "Воронежская область", "Белгородская область",
                "Тульская область", "Ярославская область", "Калужская область", "Владимирская область",
                "Ивановская область", "Смоленская область", "Тверская область", "Липецкая область",
                "Курская область", "Рязанская область", "Брянская область", "Орловская область",
                "Тамбовская область", "Костромская область"]
    northwest = ["Санкт-Петербург", "Ленинградская область", "Калининградская область",
                 "Архангельская область", "Вологодская область", "Мурманская область",
                 "Псковская область", "Республика Коми", "Республика Карелия",
                 "Новгородская область"]
    south = ["Краснодарский край", "Ростовская область", "Волгоградская область",
             "Ставропольский край", "Республика Крым", "Севастополь",
             "Республика Дагестан", "Астраханская область", "Республика Адыгея",
             "Кабардино-Балкарская Республика", "Республика Калмыкия",
             "Карачаево-Черкесская Республика", "Республика Северная Осетия - Алания",
             "Республика Ингушетия", "Чеченская Республика"]
    volga = ["Республика Татарстан", "Республика Башкортостан", "Самарская область",
             "Нижегородская область", "Саратовская область", "Оренбургская область",
             "Пермский край", "Удмуртская Республика", "Ульяновская область",
             "Чувашская Республика", "Кировская область", "Пензенская область",
             "Республика Марий Эл", "Республика Мордовия"]
    ural = ["Свердловская область", "Челябинская область", "Тюменская область",
            "Ханты-Мансийский Автономный округ - Югра", "Курганская область",
            "Ямало-Ненецкий автономный округ"]
    siberia = ["Новосибирская область", "Красноярский край", "Кемеровская область - Кузбасс",
               "Омская область", "Алтайский край", "Иркутская область", "Томская область",
               "Республика Хакасия", "Республика Бурятия", "Республика Тыва",
               "Республика Алтай", "Забайкальский край"]
    far_east = ["Хабаровский край", "Приморский край", "Республика Саха /Якутия/",
                "Сахалинская область", "Амурская область", "Камчатский край",
                "Магаданская область", "Еврейская автономная область",
                "Чукотский автономный округ"]
    def region_to_district(region):
        if region == "__unknown__": return "unknown"
        if region in central: return "central"
        if region in northwest: return "northwest"
        if region in south: return "south"
        if region in volga: return "volga"
        if region in ural: return "ural"
        if region in siberia: return "siberia"
        if region in far_east: return "far_east"
        return "other"
    for X in [X_new_train, X_new_test, X_new_val]:
        X["federal_district"] = X["contact_region"].apply(region_to_district)
    cat_cols = cat_cols + ["federal_district"]
    dfm["federal_district"] = False

# Feature exclusion
if EXCLUDE_FEATURES:
    cat_cols = [c for c in cat_cols if c not in EXCLUDE_FEATURES]
    num_cols = [c for c in num_cols if c not in EXCLUDE_FEATURES]
    bin_cols = [c for c in bin_cols if c not in EXCLUDE_FEATURES]
    geo_cols = [c for c in geo_cols if c not in EXCLUDE_FEATURES]
    te_cat_cols = [c for c in te_cat_cols if c not in EXCLUDE_FEATURES]

# ============================================================
# ENCODING FUNCTIONS
# ============================================================

def kfold_target_encoding(X, y, col, n_splits=5, alpha=10, random_state=42):
    """Out-of-fold target encoding with additive smoothing."""
    global_mean = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    te = np.full(len(X), global_mean)
    for train_idx, val_idx in kf.split(X):
        stats = (X.iloc[train_idx]
                 .join(y.iloc[train_idx].rename("target"))
                 .groupby(col)["target"]
                 .agg(["mean", "count"]))
        stats["te"] = (stats["mean"] * stats["count"] + global_mean * alpha) / (
            stats["count"] + alpha
        )
        te[val_idx] = X.iloc[val_idx][col].map(stats["te"]).fillna(global_mean).values
    return te


def encode_fold(X_tr, y_tr, X_va, cc, nc, bc, gc, tc, alpha, drop_first_map,
                alpha_per_feature=None):
    """Full encoding pipeline: one-hot + scaler + binary + target encoding.

    alpha_per_feature: dict mapping TE column name → alpha override.
    """
    parts_tr, parts_va = [], []

    # One-hot (per-column drop_first; NaN → all-zeros automatically)
    if cc:
        oh_tr, oh_va = [], []
        for col in cc:
            df_flag = drop_first_map.get(col, False)
            oh_tr.append(pd.get_dummies(X_tr[[col]], drop_first=df_flag))
            oh_va.append(pd.get_dummies(X_va[[col]], drop_first=df_flag))
        c_tr = pd.concat(oh_tr, axis=1)
        c_va = pd.concat(oh_va, axis=1)
        c_tr, c_va = c_tr.align(c_va, join="left", axis=1, fill_value=0)
        parts_tr.append(c_tr)
        parts_va.append(c_va)

    # StandardScaler (only if numeric features exist)
    sc = None
    if nc:
        sc = StandardScaler().fit(X_tr[nc])
        parts_tr.append(pd.DataFrame(sc.transform(X_tr[nc]), columns=nc, index=X_tr.index))
        parts_va.append(pd.DataFrame(sc.transform(X_va[nc]), columns=nc, index=X_va.index))

    # Binary passthrough
    if bc:
        parts_tr.append(X_tr[bc].reset_index(drop=True).set_index(X_tr.index))
        parts_va.append(X_va[bc].reset_index(drop=True).set_index(X_va.index))

    # Target encoding (per-feature alpha)
    apf = alpha_per_feature or {}
    gm = float(y_tr.mean())
    te_cols = gc + tc
    for col_te in te_cols:
        a = apf.get(col_te, alpha)
        te_name = col_te + "_te"
        te_tr = kfold_target_encoding(X_tr, y_tr, col_te, alpha=a)
        parts_tr.append(pd.DataFrame({te_name: te_tr}, index=X_tr.index))
        stats = (X_tr.join(y_tr.rename("target"))
                 .groupby(col_te)["target"]
                 .agg(["mean", "count"]))
        te_map = (stats["mean"] * stats["count"] + gm * a) / (stats["count"] + a)
        parts_va.append(
            pd.DataFrame(
                {te_name: X_va[col_te].map(te_map).fillna(gm).values},
                index=X_va.index,
            )
        )

    X_tr_enc = pd.concat(parts_tr, axis=1).fillna(0)
    X_va_enc = pd.concat(parts_va, axis=1).fillna(0)
    return X_tr_enc, X_va_enc


# ============================================================
# CROSS-VALIDATION
# ============================================================

print("=" * 70)
print(f"EXPERIMENT: penalty={PENALTY}, C={C}, solver={SOLVER}, TE_alpha={TE_ALPHA}")
print(f"  cat_cols ({len(cat_cols)}), num_cols ({len(num_cols)}), "
      f"bin_cols ({len(bin_cols)}), geo_cols ({len(geo_cols)}), "
      f"te_cat_cols ({len(te_cat_cols)})")
if DROP_FIRST_OVERRIDE:
    print(f"  drop_first overrides: {DROP_FIRST_OVERRIDE}")
if EXCLUDE_FEATURES:
    print(f"  excluded: {EXCLUDE_FEATURES}")
print("=" * 70)

cv_aucs = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_new_train, y_new_train)):
    X_tr = X_new_train.iloc[tr_idx]
    y_tr = y_new_train.iloc[tr_idx]
    X_va = X_new_train.iloc[va_idx]
    y_va = y_new_train.iloc[va_idx]

    X_tr_enc, X_va_enc = encode_fold(
        X_tr, y_tr, X_va, cat_cols, num_cols, bin_cols, geo_cols, te_cat_cols,
        alpha=TE_ALPHA, drop_first_map=dfm, alpha_per_feature=TE_ALPHA_PER_FEATURE,
    )

    lr_kwargs = dict(C=C, penalty=PENALTY, max_iter=2000,
        class_weight="balanced", random_state=42, solver=SOLVER)
    if PENALTY == "elasticnet":
        lr_kwargs["l1_ratio"] = L1_RATIO
    m = LogisticRegression(**lr_kwargs)
    m.fit(X_tr_enc, y_tr)

    auc = roc_auc_score(y_va, m.predict_proba(X_va_enc)[:, 1])
    cv_aucs.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"\nCV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

# ============================================================
# FINAL TRAIN → TEST + VAL EVALUATION
# ============================================================

X_train_enc, X_test_enc = encode_fold(
    X_new_train, y_new_train, X_new_test,
    cat_cols, num_cols, bin_cols, geo_cols, te_cat_cols,
    alpha=TE_ALPHA, drop_first_map=dfm, alpha_per_feature=TE_ALPHA_PER_FEATURE,
)
_, X_val_enc = encode_fold(
    X_new_train, y_new_train, X_new_val,
    cat_cols, num_cols, bin_cols, geo_cols, te_cat_cols,
    alpha=TE_ALPHA, drop_first_map=dfm, alpha_per_feature=TE_ALPHA_PER_FEATURE,
)

lr_kwargs = dict(C=C, penalty=PENALTY, max_iter=2000,
    class_weight="balanced", random_state=42, solver=SOLVER)
if PENALTY == "elasticnet":
    lr_kwargs["l1_ratio"] = L1_RATIO
model = LogisticRegression(**lr_kwargs)
model.fit(X_train_enc, y_new_train)

# --- Metrics helper ---
def print_metrics(name, y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n{'=' * 60}")
    print(f"Модель новых клиентов — {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision:         {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:            {recall_score(y_true, y_pred):.4f}")
    print(f"  F1:                {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC:           {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  MCC:               {matthews_corrcoef(y_true, y_pred):.4f}")

p_test = model.predict_proba(X_test_enc)[:, 1]
p_val = model.predict_proba(X_val_enc)[:, 1]

print_metrics("TEST", y_new_test, p_test)
print_metrics("VAL", y_new_val, p_val)

# ============================================================
# FEATURE IMPORTANCES (coefficients)
# ============================================================

coefs = pd.Series(model.coef_[0], index=X_train_enc.columns)
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

print(f"\n{'=' * 60}")
print("Feature coefficients (sorted by |coef|)")
print(f"{'=' * 60}")
nonzero = coefs[coefs != 0]
print(f"  Non-zero features: {len(nonzero)} / {len(coefs)}")
print()
for feat, val in nonzero.items():
    print(f"  {val:+.4f}  {feat}")

zero = coefs[coefs == 0]
if len(zero) > 0:
    print(f"\n  Zero-coefficient features ({len(zero)}):")
    for feat in zero.index:
        print(f"    {feat}")

# Machine-readable summary (for autoresearch metric extraction)
val_auc = roc_auc_score(y_new_val, p_val)
val_f1 = f1_score(y_new_val, (p_val >= 0.5).astype(int))
print(f"\nMETRIC_VAL_ROC_AUC={val_auc:.4f}")
print(f"METRIC_VAL_F1={val_f1:.4f}")
