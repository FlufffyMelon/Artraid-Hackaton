"""Generic YAML-driven feature preprocessing.

`utils.py` is a pure library: it knows nothing about specific column names,
domain concepts, or downstream models. Everything column-specific lives in
[features.yaml](features.yaml) and is expressed as a combination of small,
composable operations registered in `_OPS`.

Public API:
    load_feature_config(path)        -> dict
    build_features(df, config, ctx)  -> (DataFrame, FeatureColumns)
    register_op(name)                -> decorator to add new operations
    FeatureColumns                   -> dataclass with the typed column split
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


# ============================================================
# YAML loader with .inf support
# ============================================================


class _SafeLoaderInf(yaml.SafeLoader):
    pass


def _construct_inf(loader, node):  # noqa: ANN001
    val = loader.construct_scalar(node)
    low = val.strip().lower()
    if low in {".inf", "+.inf"}:
        return float("inf")
    if low == "-.inf":
        return float("-inf")
    raise yaml.constructor.ConstructorError(None, None, f"bad float tag: {val!r}")


# PyYAML already resolves `.inf` → float via its default implicit resolver,
# but tag the loader explicitly for robustness across PyYAML versions.
_SafeLoaderInf.add_constructor("tag:yaml.org,2002:float", yaml.SafeLoader.construct_yaml_float)


# ============================================================
# Schema + loading
# ============================================================

_VALID_TYPES = {"categorical", "target_encoding", "numeric", "binary"}
_VALID_NAN = {"category", "fill", "ignore", "drop"}


def load_feature_config(path: str) -> dict:
    """Load YAML and validate schema.

    Structure:
        features:
            <name>:
                type: categorical | target_encoding | numeric | binary
                nan?: category | fill (+nan_value) | ignore | drop
                drop_values?: [val, ...]  # additional values → NaN (e.g., "0")
                cast?: str | int | float
                drop_first?: bool  # for categorical; stored in FeatureColumns
                groups?: {"new": ["old1", "old2"]}
                alpha?: int        # required for target_encoding
                encoding_group?: geo
                hidden?: bool      # computed but excluded from FeatureColumns
                source?: str       # for raw features; defaults to <name>
                op?: {name: ..., ...}   # for engineered features
            ...
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=_SafeLoaderInf)

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config).__name__}")

    features = config.get("features")
    if not isinstance(features, dict):
        raise ValueError("Config must contain a 'features' mapping")

    for name, entry in features.items():
        _validate_entry(name, entry)

    return {"features": features}


def _validate_entry(name: str, entry: dict) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Feature {name!r}: entry must be a mapping")

    t = entry.get("type")
    if t not in _VALID_TYPES:
        raise ValueError(
            f"Feature {name!r}: type must be one of {sorted(_VALID_TYPES)}, got {t!r}"
        )

    nan_policy = entry.get("nan", _default_nan(t))
    if nan_policy not in _VALID_NAN:
        raise ValueError(
            f"Feature {name!r}: nan must be one of {sorted(_VALID_NAN)}, got {nan_policy!r}"
        )
    if nan_policy == "fill" and "nan_value" not in entry:
        raise ValueError(f"Feature {name!r}: nan='fill' requires nan_value")

    if t == "target_encoding" and "alpha" not in entry:
        raise ValueError(f"Feature {name!r}: target_encoding requires 'alpha'")

    if t == "numeric" and "bins" in entry:
        raise ValueError(
            f"Feature {name!r}: bins on a numeric feature is not allowed. "
            "Declare a separate feature with op=bin instead."
        )

    if "op" in entry:
        _validate_op(name, entry["op"])


def _validate_op(feature: str, op: dict) -> None:
    if not isinstance(op, dict):
        raise ValueError(f"Feature {feature!r}: op must be a mapping")
    op_name = op.get("name")
    if op_name not in _OPS:
        raise ValueError(
            f"Feature {feature!r}: unknown op {op_name!r}. Registered: {sorted(_OPS)}"
        )
    if op_name == "pipeline":
        steps = op.get("steps") or []
        if not steps:
            raise ValueError(f"Feature {feature!r}: pipeline requires non-empty 'steps'")
        for step in steps:
            _validate_op(feature, step)


def _default_nan(t: str) -> str:
    return "category" if t in {"categorical", "target_encoding"} else "ignore"


# ============================================================
# FeatureColumns
# ============================================================


@dataclass
class FeatureColumns:
    cat_cols: list[str] = field(default_factory=list)
    num_cols: list[str] = field(default_factory=list)
    bin_cols: list[str] = field(default_factory=list)
    geo_cols: list[str] = field(default_factory=list)
    te_cat_cols: list[str] = field(default_factory=list)
    te_alpha: dict[str, int] = field(default_factory=dict)
    cat_drop_first: dict[str, bool] = field(default_factory=dict)

    @property
    def all_feature_cols(self) -> list[str]:
        return (
            self.cat_cols
            + self.num_cols
            + self.bin_cols
            + self.geo_cols
            + self.te_cat_cols
        )


def _classify(name: str, entry: dict, fc: FeatureColumns) -> None:
    if entry.get("hidden", False):
        return
    t = entry["type"]
    if t == "categorical":
        fc.cat_cols.append(name)
        fc.cat_drop_first[name] = entry.get("drop_first", False)
    elif t == "numeric":
        fc.num_cols.append(name)
    elif t == "binary":
        fc.bin_cols.append(name)
    elif t == "target_encoding":
        if entry.get("encoding_group") == "geo":
            fc.geo_cols.append(name)
        else:
            fc.te_cat_cols.append(name)
        fc.te_alpha[name] = entry["alpha"]


# ============================================================
# Operation registry
# ============================================================

OpFn = Callable[[pd.DataFrame, "pd.Series | None", dict, dict], pd.Series]
_OPS: dict[str, OpFn] = {}


def register_op(name: str) -> Callable[[OpFn], OpFn]:
    def deco(fn: OpFn) -> OpFn:
        _OPS[name] = fn
        return fn
    return deco


def _run_op(
    df: pd.DataFrame, current: pd.Series | None, spec: dict, context: dict
) -> pd.Series:
    fn = _OPS[spec["name"]]
    return fn(df, current, spec, context)


def _resolve_input(df: pd.DataFrame, current: pd.Series | None, spec: dict) -> pd.Series:
    """If we're the first op in a chain and a source is given, pull it from df."""
    if current is not None:
        return current
    src = spec.get("source")
    if src is None:
        raise ValueError(f"op {spec.get('name')!r}: requires 'source' or chained input")
    return df[src]


def _collect_sources(op: dict) -> list[str]:
    """Recursively collect all `source` references used by this op (for missing-cols check)."""
    out: list[str] = []
    if "source" in op:
        out.append(op["source"])
    if "sources" in op:
        out.extend(op["sources"])
    if op.get("name") == "pipeline":
        for step in op.get("steps", []):
            out.extend(_collect_sources(step))
    return out


# ============================================================
# build_features
# ============================================================


def build_features(
    df: pd.DataFrame,
    config: dict,
    context: dict | None = None,
) -> tuple[pd.DataFrame, FeatureColumns]:
    """Apply the YAML-described preprocessing to `df` and return (df_out, feat_cols).

    The input df is not mutated; engineered ops may write their output columns
    (including `hidden` intermediates) to the returned DataFrame.
    """
    df = df.copy()
    context = dict(context or {})
    context.setdefault("_op_cache", {})
    # per-call cache: clear so results don't leak between invocations
    context["_op_cache"] = {}

    features = config["features"]

    # 1. Missing-column check.
    missing: list[str] = []
    for name, entry in features.items():
        if "op" in entry:
            # Engineered: sources from the op itself. Sources that refer to
            # other declared features (produced earlier in this build call)
            # are allowed and checked lazily by df access.
            for src in _collect_sources(entry["op"]):
                if src not in df.columns and src not in features:
                    missing.append(f"{name}: source {src!r}")
        else:
            src = entry.get("source", name)
            if src not in df.columns:
                missing.append(f"{name}: source {src!r}")

    if missing:
        raise ValueError("Missing columns in DataFrame:\n  " + "\n  ".join(missing))

    fc = FeatureColumns()

    # 2. Process features in YAML order.
    for name, entry in features.items():
        if "op" in entry:
            series = _run_op(df, None, entry["op"], context)
        else:
            src = entry.get("source", name)
            series = df[src].copy()

        series = pd.Series(series, index=df.index, name=name)
        df[name] = series
        # Order: fill NaN (while still typed), then cast, then merge groups.
        # Rationale: `nan: fill, nan_value: 0` followed by `cast: str` must see
        # the raw NaN to replace it — once astype(str) runs, NaN becomes the
        # literal string "nan" which fillna can't target.
        _apply_nan_policy(df, name, entry)
        _apply_cast(df, name, entry)
        _apply_groups(df, name, entry)
        _classify(name, entry, fc)

    return df, fc


# ============================================================
# Per-feature post-processing
# ============================================================


def _apply_cast(df: pd.DataFrame, col: str, entry: dict) -> None:
    cast = entry.get("cast")
    if cast is None:
        return
    if cast == "str":
        na_mask = df[col].isna()
        if pd.api.types.is_numeric_dtype(df[col]):
            # Int64 avoids ".0" suffix: 546538.0 → "546538", not "546538.0"
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)
        else:
            df[col] = df[col].astype(str)
        # Restore NaN destroyed by astype(str) → "nan" / "<NA>" string
        if na_mask.any():
            df.loc[na_mask, col] = np.nan
    elif cast == "int":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif cast == "float":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        raise ValueError(f"Feature {col!r}: unknown cast {cast!r}")


def _apply_nan_policy(df: pd.DataFrame, col: str, entry: dict) -> None:
    policy = entry.get("nan", _default_nan(entry["type"]))
    if policy == "category":
        df[col] = df[col].fillna("__NaN__")
    elif policy == "fill":
        df[col] = df[col].fillna(entry["nan_value"])
    # "ignore" / "drop" → do nothing (NaN stays; for "drop" one-hot gives all-zeros)

    # drop_values: additional values to convert to NaN (e.g., "0" for lead_group_id)
    drop_values = entry.get("drop_values")
    if drop_values:
        df.loc[df[col].isin(drop_values), col] = np.nan


def _apply_groups(df: pd.DataFrame, col: str, entry: dict) -> None:
    groups = entry.get("groups")
    if not groups:
        return
    actual = set(df[col].dropna().unique())
    referenced: set = set()
    for vs in groups.values():
        referenced.update(vs)
    unknown = referenced - actual
    if unknown:
        raise ValueError(
            f"Feature {col!r}: unknown category values in groups: {sorted(unknown)}. "
            f"Existing values: {sorted(map(str, actual))}"
        )
    mapping: dict = {}
    for new, olds in groups.items():
        for old in olds:
            mapping[old] = new
    df[col] = df[col].replace(mapping)


# ============================================================
# Bin helper (used by `bin` op)
# ============================================================


def _bin_edges(bins: list) -> list[float]:
    """Accept flat edges [a, b, c] or pair form [[a,b], [b,c], ...]."""
    if not bins:
        raise ValueError("bins is empty")
    if isinstance(bins[0], (list, tuple)):
        edges = [bins[0][0]]
        for i, pair in enumerate(bins):
            if len(pair) != 2:
                raise ValueError(f"Bin range #{i} must have 2 elements, got {pair!r}")
            if i > 0 and pair[0] != bins[i - 1][1]:
                raise ValueError(f"Bin ranges not contiguous: {bins[i-1]} -> {pair}")
            edges.append(pair[1])
    else:
        edges = list(bins)
    return [float(e) for e in edges]


# ============================================================
# Built-in operations
# ============================================================


@register_op("identity")
def _op_identity(df, current, spec, context):
    return _resolve_input(df, current, spec)


@register_op("cast")
def _op_cast(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    to = spec["to"]
    if to == "str":
        return s.astype(str)
    if to == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if to == "float":
        return pd.to_numeric(s, errors="coerce")
    raise ValueError(f"cast: unknown to={to!r}")


@register_op("fillna")
def _op_fillna(df, current, spec, context):
    return _resolve_input(df, current, spec).fillna(spec["value"])


@register_op("clip")
def _op_clip(df, current, spec, context):
    return _resolve_input(df, current, spec).clip(
        lower=spec.get("lower"), upper=spec.get("upper")
    )


@register_op("not_null")
def _op_not_null(df, current, spec, context):
    return _resolve_input(df, current, spec).notna().astype(int)


@register_op("is_in")
def _op_is_in(df, current, spec, context):
    return _resolve_input(df, current, spec).isin(spec["values"]).astype(int)


@register_op("map")
def _op_map(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    table = context[spec["table"]]
    if not isinstance(table, dict):
        raise ValueError(f"map: context[{spec['table']!r}] must be a dict")
    key_cast = spec.get("key_cast")
    if key_cast == "str":
        table = {str(k): v for k, v in table.items()}
        s = s.astype(str)
    elif key_cast == "int":
        table = {int(k): v for k, v in table.items()}
    result = s.map(table)
    if "default" in spec:
        result = result.fillna(spec["default"])
    return result


@register_op("datetime_attr")
def _op_datetime_attr(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, errors="coerce")
    result = getattr(s.dt, spec["attr"])
    cast = spec.get("cast")
    if cast == "str":
        result = result.astype(str)
    elif cast == "int":
        result = result.astype("Int64")
    return result


@register_op("arithmetic")
def _op_arithmetic(df, current, spec, context):
    # df.eval lets the YAML express "(sale_ts - lead_created_at) / 86400" directly.
    result = df.eval(spec["formula"])
    if "clip_lower" in spec or "clip_upper" in spec:
        result = result.clip(lower=spec.get("clip_lower"), upper=spec.get("clip_upper"))
    if "fillna" in spec:
        result = result.fillna(spec["fillna"])
    return result


@register_op("count_tokens")
def _op_count_tokens(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    separators: list[str] = spec.get("separators", ["\n"])
    exclude_contains = spec.get("exclude_contains")
    min_if_any = spec.get("min_if_any")
    default = spec.get("default", 0)

    def count(value):
        if pd.isna(value) or not isinstance(value, str):
            return default
        text = value
        primary, *rest = separators
        for sep in rest:
            text = text.replace(sep, primary)
        tokens = [t.strip() for t in text.split(primary) if t.strip()]
        if exclude_contains is not None:
            needle = exclude_contains.lower()
            tokens = [t for t in tokens if needle not in t.lower()]
        if not tokens:
            return 0 if exclude_contains is None else default
        if min_if_any is not None:
            return max(len(tokens), min_if_any)
        return len(tokens)

    return s.apply(count)


@register_op("substring_lookup")
def _op_substring_lookup(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    table = context[spec["table"]]
    match_field = spec["match_field"]
    return_field = spec["return_field"]
    unknown = spec.get("unknown_label", "__unknown__")
    norm_spec = spec.get("normalize") or {}

    def normalize(text: str) -> str:
        if norm_spec.get("lower", False):
            text = text.lower()
        for a, b in (norm_spec.get("replace") or {}).items():
            text = text.replace(a, b)
        return text

    # Cache key: source column name + table identity + normalize spec.
    src_key = spec.get("source", id(current))
    cache_key = (src_key, spec["table"], match_field, frozenset(norm_spec.get("replace", {}).items()),
                 norm_spec.get("lower", False))
    cache = context["_op_cache"]
    if cache_key not in cache:
        sorted_table = sorted(table, key=lambda r: len(r[match_field]), reverse=True)
        normalized = [(normalize(r[match_field]), r) for r in sorted_table]
        rows_out: list[dict | None] = []
        for value in s.values:
            if pd.isna(value):
                rows_out.append(None)
                continue
            haystack = normalize(str(value))
            matched = None
            for needle, row in normalized:
                if needle in haystack:
                    matched = row
                    break
            rows_out.append(matched)
        cache[cache_key] = rows_out

    rows = cache[cache_key]
    return pd.Series(
        [unknown if r is None else r[return_field] for r in rows],
        index=s.index,
    )


@register_op("bin")
def _op_bin(df, current, spec, context):
    s = _resolve_input(df, current, spec)
    if "fillna" in spec:
        s = s.fillna(spec["fillna"])
    edges = _bin_edges(spec["bins"])
    labels = spec["bin_labels"]
    include_lowest = spec.get("include_lowest", False)
    result = pd.cut(s, bins=edges, labels=labels, include_lowest=include_lowest).astype(str)
    nan_label = spec.get("nan_label")
    if nan_label is not None:
        result = result.where(~s.isna(), nan_label)
    return result


@register_op("pipeline")
def _op_pipeline(df, current, spec, context):
    value = current
    for step in spec["steps"]:
        value = _run_op(df, value, step, context)
    return value
