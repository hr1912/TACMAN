#!/usr/bin/env python3
"""Summarize completed TACMAN outputs into tables and publication figures.

This script reads TACMAN result files only. It does not run TACMAN training.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ConfigError(ValueError):
    """Raised when a YAML configuration is invalid."""


class ValidationError(RuntimeError):
    """Raised when TACMAN result validation fails."""


def default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "input": {
            "tacman_output_directory": None,
            "prediction_table": None,
        },
        "metadata": {
            "species_key": "species",
            "reference_or_query_key": "type",
            "reference_value": "ref",
            "query_value": "que",
            "true_label_key": "true_label",
            "predicted_label_key": "pre_label",
            "model_label_key": "model_label",
            "confidence_key": "max_prob",
            "umap1_key": "UMAP1",
            "umap2_key": "UMAP2",
            "optional_columns": ["dataset", "batch", "original_cluster", "stage"],
        },
        "analysis": {
            "evaluate_query_labels": True,
            "normalize_confusion_matrix": True,
            "allow_umap_recompute": False,
        },
        "figures": {
            "format": "pdf",
            "dpi": 300,
            "width": 7,
            "height": 6,
            "point_size": 5,
        },
        "output": {
            "directory": None,
            "prefix": "tacman_results",
            "overwrite": False,
        },
    }


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_scalar(text: str) -> Any:
    """Parse a small YAML scalar for the fallback reader."""
    text = text.strip()
    if text in {"", "null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+", text
        ):
            return float(text)
    except Exception:
        pass
    return text


def simple_yaml_load(text: str) -> Dict[str, Any]:
    """Load the simple YAML subset used by project templates.

    The fallback supports nested mappings, scalar values, and inline lists. It
    deliberately rejects advanced YAML features so users get explicit errors.
    """
    if re.search(r"(^|\n)\s*[-?]\s+", text):
        raise ConfigError("Simple YAML fallback does not support block lists. Install PyYAML.")
    if any(token in text for token in ["&", "*", "!", "|", ">"]):
        raise ConfigError("Simple YAML fallback does not support anchors, aliases, tags, or multiline blocks.")
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ConfigError(f"Invalid indentation at YAML line {lineno}. Use multiples of two spaces.")
        stripped = line.strip()
        if ":" not in stripped:
            raise ConfigError(f"Invalid YAML line {lineno}: {raw}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML with PyYAML when available, otherwise the simple fallback."""
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = simple_yaml_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root in {path} must be a mapping.")
    return data


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write a readable YAML snapshot."""
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return
    except ModuleNotFoundError:
        pass

    def emit(obj: Any, indent: int = 0) -> List[str]:
        lines: List[str] = []
        pad = " " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    lines.append(f"{pad}{k}:")
                    lines.extend(emit(v, indent + 2))
                else:
                    lines.append(f"{pad}{k}: {format_scalar(v)}")
        return lines

    path.write_text("\n".join(emit(data)) + "\n")


def format_scalar(value: Any) -> str:
    """Format a scalar for fallback YAML writing."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    text = str(value)
    if not text or any(ch in text for ch in [":", "#", "[", "]", "{", "}", ","]):
        return repr(text)
    return text


def resolve_path(value: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve a config path relative to the config file directory."""
    if value in (None, ""):
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load, merge, and resolve a results-summary config."""
    config_path = config_path.resolve()
    cfg = deep_update(default_config(), load_yaml(config_path))
    base = config_path.parent
    cfg["_config_file"] = str(config_path)
    for key in ["tacman_output_directory", "prediction_table"]:
        p = resolve_path(cfg["input"].get(key), base)
        cfg["input"][key] = str(p) if p is not None else None
    out_dir = resolve_path(cfg["output"].get("directory"), base)
    if out_dir is None:
        raise ConfigError("Invalid config field: output.directory is required.")
    cfg["output"]["directory"] = str(out_dir)
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply explicit CLI overrides over YAML values."""
    if args.output_dir is not None:
        cfg["output"]["directory"] = str(Path(args.output_dir).expanduser().resolve())
    if args.prefix is not None:
        cfg["output"]["prefix"] = args.prefix
    if args.prediction_table is not None:
        cfg["input"]["prediction_table"] = str(Path(args.prediction_table).expanduser().resolve())
    if args.overwrite:
        cfg["output"]["overwrite"] = True
    return cfg


def require_pandas():
    """Import pandas with a concise user-facing error."""
    try:
        import pandas as pd  # type: ignore

        return pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas is required. Install with: conda install pandas or pip install pandas") from exc


def require_numpy():
    """Import numpy with a concise user-facing error."""
    try:
        import numpy as np  # type: ignore

        return np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required. Install with: conda install numpy or pip install numpy") from exc


def prepare_output_dir(out_dir: Path, overwrite: bool, validate_only: bool = False) -> Optional[Path]:
    """Create output directory and optionally back up an existing non-empty one."""
    if validate_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        return None
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise ValidationError(
                f"Output directory already exists and is not empty: {out_dir}. "
                "Set output.overwrite=true or choose another output directory/prefix."
            )
        backup = out_dir.with_name(out_dir.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(out_dir), str(backup))
        out_dir.mkdir(parents=True, exist_ok=True)
        return backup
    out_dir.mkdir(parents=True, exist_ok=True)
    return None


def find_result_table(cfg: Dict[str, Any]) -> Path:
    """Resolve the TACMAN final prediction table."""
    explicit = cfg["input"].get("prediction_table")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ValidationError(f"Configured input.prediction_table does not exist: {p}")
        return p
    out_dir = cfg["input"].get("tacman_output_directory")
    if not out_dir:
        raise ValidationError("input.tacman_output_directory is required when input.prediction_table is not set.")
    p_out = Path(out_dir)
    if not p_out.exists():
        raise ValidationError(f"TACMAN output directory does not exist: {p_out}")
    obs = p_out / "obs.csv"
    if obs.exists():
        return obs
    matches = sorted(p_out.glob("obs.csv"))
    if len(matches) == 1:
        return matches[0]
    raise ValidationError(f"Could not find a unique final obs.csv in TACMAN output directory: {p_out}")


def result_table_separator(path: Path) -> str:
    """Choose CSV/TSV separator from a result table suffix."""
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    return ","


def read_result_dataframe(cfg: Dict[str, Any]):
    """Read TACMAN final results from obs.csv or an explicit CSV/TSV table."""
    pd = require_pandas()
    table = find_result_table(cfg)
    sep = result_table_separator(table)
    df = pd.read_csv(table, sep=sep, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "cell_id"
    ensure_model_label(df, cfg)
    return df, table


def ensure_model_label(df, cfg: Dict[str, Any]) -> None:
    """Create TACMAN integrated labels when absent."""
    meta = cfg["metadata"]
    model_key = meta.get("model_label_key")
    true_key = meta.get("true_label_key")
    pred_key = meta.get("predicted_label_key")
    rq_key = meta.get("reference_or_query_key")
    if not model_key or model_key in df.columns:
        return
    if true_key in df.columns and pred_key in df.columns and rq_key in df.columns:
        q = query_mask(df, cfg)
        df[model_key] = df[true_key].where(~q, df[pred_key])


def get_embedding(df, cfg: Dict[str, Any]):
    """Return UMAP coordinates from TACMAN obs columns when present."""
    meta = cfg["metadata"]
    x_key = meta.get("umap1_key") or "UMAP1"
    y_key = meta.get("umap2_key") or "UMAP2"
    if x_key in df.columns and y_key in df.columns:
        return df[[x_key, y_key]].copy(), None
    if not cfg["analysis"].get("allow_umap_recompute", False):
        return None, f"Embedding columns {x_key}/{y_key} were not found; UMAP figures skipped."
    return None, "UMAP recomputation from latent embeddings is not implemented in this summary wrapper."


def validate_metadata(df, cfg: Dict[str, Any], embedding) -> List[str]:
    """Validate configured columns and return warnings."""
    warnings: List[str] = []
    meta = cfg["metadata"]
    required = [meta["species_key"], meta["reference_or_query_key"], meta["predicted_label_key"], meta["confidence_key"]]
    for col in required:
        if col not in df.columns:
            raise ValidationError(f"Configured metadata column is missing from results: {col}")
    if df.index.has_duplicates:
        raise ValidationError("Result cell IDs are not unique.")
    qmask = query_mask(df, cfg)
    if qmask.sum() == 0:
        raise ValidationError("Could not identify query cells using metadata.reference_or_query_key/query_value.")
    qdf = df.loc[qmask]
    pred_key = meta["predicted_label_key"]
    if missing_label_mask(qdf[pred_key]).any():
        raise ValidationError(f"Query predicted label column {pred_key} contains missing values.")
    pd = require_pandas()
    conf = pd.to_numeric(qdf[meta["confidence_key"]], errors="coerce")
    if conf.isna().any():
        raise ValidationError(f"Query confidence column {meta['confidence_key']} must be numeric and non-missing.")
    if ((conf < 0) | (conf > 1)).any():
        raise ValidationError(f"Query confidence column {meta['confidence_key']} contains values outside [0, 1].")
    true_key = meta.get("true_label_key")
    if cfg["analysis"].get("evaluate_query_labels", True):
        if true_key and true_key not in df.columns:
            warnings.append(f"Configured true label column is absent: {true_key}; supervised metrics will be skipped.")
        elif not labels_available(df, cfg):
            warnings.append("Query ground-truth labels are unavailable; supervised metrics and true-vs-predicted plots will be skipped.")
    if embedding is None:
        warnings.append("No UMAP embedding was available; UMAP figures will be skipped.")
    return warnings


def query_mask(df, cfg: Dict[str, Any]):
    """Return boolean mask for query cells."""
    meta = cfg["metadata"]
    return df[meta["reference_or_query_key"]].astype(str) == str(meta["query_value"])


def missing_label_mask(series):
    """Return missing/blank label mask."""
    pd = require_pandas()
    return series.isna() | series.astype(str).isin(["", "NA", "nan", "None"])


def standardize_annotations(df, cfg: Dict[str, Any], query_only: bool = False):
    """Build standardized all-cell annotations or query-only predictions."""
    pd = require_pandas()
    meta = cfg["metadata"]
    src = df.loc[query_mask(df, cfg)].copy() if query_only else df.copy()
    out = pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=src.index)
    out["cell_id"] = src.index.astype(str)
    out["species"] = src[meta["species_key"]].astype(str).values
    true_key = meta.get("true_label_key")
    out["true_label"] = src[true_key].astype(str).values if true_key in src.columns else ""
    out["predicted_label"] = src[meta["predicted_label_key"]].astype(str).values
    out["prediction_confidence"] = pd.to_numeric(src[meta["confidence_key"]], errors="coerce").values
    rq_key = meta.get("reference_or_query_key")
    if rq_key in src.columns:
        out["reference_or_query"] = src[rq_key].astype(str).values
    for col in meta.get("optional_columns", []):
        if col in src.columns:
            out[col] = src[col].values
    return out.reset_index(drop=True)


def labels_available(df, cfg: Dict[str, Any]):
    """Return whether query ground-truth labels are available."""
    if not cfg["analysis"].get("evaluate_query_labels", True):
        return False
    true_key = cfg["metadata"].get("true_label_key")
    if not true_key or true_key not in df.columns:
        return False
    q = df.loc[query_mask(df, cfg), true_key]
    return not missing_label_mask(q).all()


def sorted_labels(values: Iterable[Any]) -> List[str]:
    """Return stable sorted string labels."""
    return sorted({str(v) for v in values if str(v) not in {"", "nan", "None"}})


def compute_metrics(df, cfg: Dict[str, Any]):
    """Compute query annotation metrics and per-class metrics."""
    pd = require_pandas()
    meta = cfg["metadata"]
    qdf = df.loc[query_mask(df, cfg)].copy()
    if not labels_available(df, cfg):
        return None, None, None, None
    true = qdf[meta["true_label_key"]].astype(str)
    pred = qdf[meta["predicted_label_key"]].astype(str)
    labels = sorted_labels(list(true) + list(pred))
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            precision_recall_fscore_support,
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true, pred, labels=labels, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            true, pred, labels=labels, average="weighted", zero_division=0
        )
        per_p, per_r, per_f, support = precision_recall_fscore_support(
            true, pred, labels=labels, average=None, zero_division=0
        )
        metrics = {
            "accuracy": accuracy_score(true, pred),
            "balanced_accuracy": balanced_accuracy_score(true, pred),
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "cohens_kappa": cohen_kappa_score(true, pred),
            "number_of_evaluated_cells": int(len(qdf)),
            "number_of_true_classes": int(true.nunique()),
            "number_of_predicted_classes": int(pred.nunique()),
        }
        per_class = pd.DataFrame(
            {"cell_type": labels, "support": support, "precision": per_p, "recall": per_r, "f1": per_f}
        )
    except ModuleNotFoundError:
        correct = (true == pred)
        metrics = {
            "accuracy": float(correct.mean()) if len(correct) else math.nan,
            "balanced_accuracy": math.nan,
            "macro_precision": math.nan,
            "macro_recall": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "cohens_kappa": math.nan,
            "number_of_evaluated_cells": int(len(qdf)),
            "number_of_true_classes": int(true.nunique()),
            "number_of_predicted_classes": int(pred.nunique()),
        }
        rows = []
        for lab in labels:
            tp = int(((true == lab) & (pred == lab)).sum())
            fp = int(((true != lab) & (pred == lab)).sum())
            fn = int(((true == lab) & (pred != lab)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"cell_type": lab, "support": int((true == lab).sum()), "precision": precision, "recall": recall, "f1": f1})
        per_class = pd.DataFrame(rows)
    metrics_df = pd.DataFrame([metrics])
    raw, norm = confusion_matrices(true, pred, labels)
    return metrics_df, per_class, raw, norm


def confusion_matrices(true, pred, labels: List[str]):
    """Return raw and row-normalized confusion matrices."""
    pd = require_pandas()
    raw = pd.crosstab(true, pred).reindex(index=labels, columns=labels, fill_value=0)
    denom = raw.sum(axis=1).replace(0, float("nan"))
    norm = raw.div(denom, axis=0).fillna(0)
    return raw, norm


def cell_type_counts(df, cfg: Dict[str, Any]):
    """Count cells by dataset role and label."""
    pd = require_pandas()
    meta = cfg["metadata"]
    label = meta.get("true_label_key") if meta.get("true_label_key") in df.columns else meta["predicted_label_key"]
    return (
        df.groupby([meta["reference_or_query_key"], meta["species_key"], label], dropna=False)
        .size()
        .reset_index(name="cell_count")
    )


def confidence_by_predicted_label(df, cfg: Dict[str, Any]):
    """Summarize query prediction confidence by predicted class."""
    pd = require_pandas()
    meta = cfg["metadata"]
    qdf = df.loc[query_mask(df, cfg)].copy()
    qdf["_confidence"] = pd.to_numeric(qdf[meta["confidence_key"]], errors="coerce")
    grouped = qdf.groupby(meta["predicted_label_key"], dropna=False)["_confidence"]
    res = grouped.agg(
        cell_count="count",
        mean_confidence="mean",
        median_confidence="median",
        minimum="min",
        maximum="max",
    ).reset_index()
    q1 = grouped.quantile(0.25).reset_index(name="q1")
    q3 = grouped.quantile(0.75).reset_index(name="q3")
    res = res.merge(q1, on=meta["predicted_label_key"]).merge(q3, on=meta["predicted_label_key"])
    return res.rename(columns={meta["predicted_label_key"]: "predicted_label"})


def confidence_by_true_predicted(df, cfg: Dict[str, Any]):
    """Summarize confidence by query true/predicted label combination."""
    pd = require_pandas()
    meta = cfg["metadata"]
    if not labels_available(df, cfg):
        return None
    qdf = df.loc[query_mask(df, cfg)].copy()
    qdf = qdf.loc[~missing_label_mask(qdf[meta["true_label_key"]])].copy()
    qdf["_confidence"] = pd.to_numeric(qdf[meta["confidence_key"]], errors="coerce")
    grouped = qdf.groupby([meta["true_label_key"], meta["predicted_label_key"]], dropna=False)["_confidence"]
    res = grouped.agg(cell_count="count", mean_confidence="mean", median_confidence="median").reset_index()
    return res.rename(columns={meta["true_label_key"]: "true_label", meta["predicted_label_key"]: "predicted_label"})


def write_tables(df, cfg: Dict[str, Any], out_dir: Path, prefix: str):
    """Write standardized tables and return paths and table summaries."""
    paths: Dict[str, str] = {}
    annotations = standardize_annotations(df, cfg, query_only=False)
    p = out_dir / f"{prefix}.cell_annotations.tsv"
    annotations.to_csv(p, sep="\t", index=False)
    paths["cell_annotations"] = str(p)

    query_predictions = standardize_annotations(df, cfg, query_only=True)
    p = out_dir / f"{prefix}.query_predictions.tsv"
    query_predictions.to_csv(p, sep="\t", index=False)
    paths["query_predictions"] = str(p)

    ccounts = cell_type_counts(df, cfg)
    p = out_dir / f"{prefix}.cell_type_counts.tsv"
    ccounts.to_csv(p, sep="\t", index=False)
    paths["cell_type_counts"] = str(p)

    conf = confidence_by_predicted_label(df, cfg)
    p = out_dir / f"{prefix}.confidence_by_predicted_label.tsv"
    conf.to_csv(p, sep="\t", index=False)
    paths["confidence_by_predicted_label"] = str(p)

    conf_pair = confidence_by_true_predicted(df, cfg)
    if conf_pair is not None:
        p = out_dir / f"{prefix}.confidence_by_true_predicted.tsv"
        conf_pair.to_csv(p, sep="\t", index=False)
        paths["confidence_by_true_predicted"] = str(p)

    metrics_df, per_class, raw, norm = compute_metrics(df, cfg)
    if metrics_df is not None:
        for name, table in [
            ("annotation_metrics", metrics_df),
            ("per_class_metrics", per_class),
            ("confusion_matrix", raw),
            ("confusion_matrix_normalized", norm),
        ]:
            p = out_dir / f"{prefix}.{name}.tsv"
            table.to_csv(p, sep="\t", index=name.startswith("confusion_matrix"))
            paths[name] = str(p)
    return paths


def require_matplotlib():
    """Import matplotlib for plotting."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for figures. Install with: conda install matplotlib") from exc


def plot_categorical_umap(df, coords, color_key: str, cfg: Dict[str, Any], out_path: Path, title: str) -> None:
    """Plot UMAP coordinates colored by a categorical column."""
    plt = require_matplotlib()
    fig = plt.figure(figsize=(cfg["figures"]["width"], cfg["figures"]["height"]))
    ax = fig.add_subplot(111)
    values = df[color_key].astype(str)
    labels = sorted_labels(values)
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    for i, lab in enumerate(labels):
        mask = values == lab
        ax.scatter(
            coords.iloc[mask.values, 0],
            coords.iloc[mask.values, 1],
            s=cfg["figures"]["point_size"],
            label=lab,
            alpha=0.75,
            color=cmap(i),
            linewidths=0,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(table, cfg: Dict[str, Any], out_path: Path) -> None:
    """Plot a confusion matrix heatmap without seaborn."""
    plt = require_matplotlib()
    fig = plt.figure(figsize=(cfg["figures"]["width"], cfg["figures"]["height"]))
    ax = fig.add_subplot(111)
    im = ax.imshow(table.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_prediction_confidence(df, cfg: Dict[str, Any], out_path: Path) -> None:
    """Plot query prediction confidence as a true-label by predicted-label bubble plot."""
    plt = require_matplotlib()
    meta = cfg["metadata"]
    grouped = confidence_by_true_predicted(df, cfg)
    if grouped is None or grouped.empty:
        return
    xlabels = sorted_labels(grouped["predicted_label"])
    ylabels = sorted_labels(grouped["true_label"])
    xmap = {v: i for i, v in enumerate(xlabels)}
    ymap = {v: i for i, v in enumerate(ylabels)}
    fig = plt.figure(figsize=(cfg["figures"]["width"], cfg["figures"]["height"]))
    ax = fig.add_subplot(111)
    sizes = grouped["cell_count"].astype(float)
    if sizes.max() > 0:
        sizes = 30 + 300 * sizes / sizes.max()
    sc = ax.scatter(
        [xmap[str(v)] for v in grouped["predicted_label"]],
        [ymap[str(v)] for v in grouped["true_label"]],
        s=sizes,
        c=grouped["mean_confidence"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_xlabel("Predicted query label")
    ax.set_ylabel("Query true label")
    ax.set_title("Prediction confidence")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean maximum prediction probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def write_figures(df, coords, cfg: Dict[str, Any], out_dir: Path, prefix: str, warnings: List[str]) -> List[str]:
    """Write supported figures and return paths."""
    paths: List[str] = []
    fmt = cfg["figures"].get("format", "pdf")
    meta = cfg["metadata"]
    if coords is not None:
        path = out_dir / f"{prefix}.umap_species.{fmt}"
        plot_categorical_umap(df, coords, meta["species_key"], cfg, path, "Species")
        paths.append(str(path))

        true_key = meta.get("true_label_key")
        if true_key and true_key in df.columns:
            true_mask = ~missing_label_mask(df[true_key])
            if true_mask.any():
                path = out_dir / f"{prefix}.umap_true_labels.{fmt}"
                plot_categorical_umap(df.loc[true_mask], coords.loc[true_mask], true_key, cfg, path, "True labels")
                paths.append(str(path))

        model_key = meta.get("model_label_key")
        if model_key and model_key in df.columns:
            path = out_dir / f"{prefix}.umap_integrated_labels.{fmt}"
            plot_categorical_umap(df, coords, model_key, cfg, path, "Integrated labels")
            paths.append(str(path))
    else:
        warnings.append("UMAP figures skipped because no embedding was available.")
    if labels_available(df, cfg):
        _, _, raw, norm = compute_metrics(df, cfg)
        table = norm if cfg["analysis"].get("normalize_confusion_matrix", True) else raw
        path = out_dir / f"{prefix}.confusion_matrix.{fmt}"
        plot_confusion_matrix(table, cfg, path)
        paths.append(str(path))
        path = out_dir / f"{prefix}.prediction_confidence.{fmt}"
        plot_prediction_confidence(df, cfg, path)
        paths.append(str(path))
    else:
        warnings.append("Supervised confusion and true-vs-predicted confidence bubble plots skipped because query true labels are unavailable.")
    return paths


def write_summary(
    cfg: Dict[str, Any],
    df,
    table_path: Path,
    out_dir: Path,
    prefix: str,
    warnings: List[str],
    backup_dir: Optional[Path],
    generated: Dict[str, Any],
    validate_only: bool,
) -> Path:
    """Write run summary text."""
    meta = cfg["metadata"]
    q = query_mask(df, cfg)
    lines = [
        "TACMAN result summary",
        f"configuration file: {cfg.get('_config_file')}",
        f"TACMAN output directory: {cfg['input'].get('tacman_output_directory')}",
        f"result table: {table_path}",
        f"result table separator: {'tab' if result_table_separator(table_path) == chr(9) else 'comma'}",
        f"output directory: {out_dir}",
        f"output prefix: {prefix}",
        f"backup directory: {backup_dir if backup_dir else 'none'}",
        f"validate only: {validate_only}",
        "",
        "Observed metadata",
        f"row count: {df.shape[0]}",
        f"column count: {df.shape[1]}",
        f"available columns: {', '.join(map(str, df.columns))}",
        f"species column: {meta['species_key']}",
        f"reference/query column: {meta['reference_or_query_key']}",
        f"true label column: {meta.get('true_label_key')}",
        f"predicted label column: {meta['predicted_label_key']}",
        f"confidence column: {meta['confidence_key']}",
        f"query cells: {int(q.sum())}",
        f"reference cells: {int((~q).sum())}",
        f"query true labels available: {labels_available(df, cfg)}",
        "",
        "Generated files",
    ]
    for key, value in generated.items():
        if isinstance(value, list):
            lines.extend([f"{key}: {v}" for v in value])
        else:
            lines.append(f"{key}: {value}")
    if warnings:
        lines.append("")
        lines.append("Warnings")
        lines.extend(f"- {w}" for w in warnings)
    path = out_dir / f"{prefix}.run_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate scalar config fields before reading data."""
    if cfg["figures"]["format"] not in {"pdf", "png"}:
        raise ConfigError("Invalid config field: figures.format. Expected pdf or png.")
    if cfg["figures"]["dpi"] <= 0:
        raise ConfigError("Invalid config field: figures.dpi. Expected a positive integer.")
    if cfg["output"].get("directory") in (None, ""):
        raise ConfigError("Invalid config field: output.directory is required.")
    if not cfg["output"].get("prefix"):
        raise ConfigError("Invalid config field: output.prefix is required.")


def run_summary(cfg: Dict[str, Any], validate_only: bool = False) -> int:
    """Run validation, tables, and figures."""
    validate_config(cfg)
    out_dir = Path(cfg["output"]["directory"])
    prefix = str(cfg["output"]["prefix"])
    backup = prepare_output_dir(out_dir, bool(cfg["output"].get("overwrite")), validate_only=validate_only)
    snapshot = out_dir / f"{prefix}.resolved_config.yaml"
    dump_yaml(cfg, snapshot)
    df, table_path = read_result_dataframe(cfg)
    coords, embed_warning = get_embedding(df, cfg)
    warnings = []
    if embed_warning:
        warnings.append(embed_warning)
    warnings.extend(validate_metadata(df, cfg, coords))
    generated: Dict[str, Any] = {"resolved_config": str(snapshot)}
    if validate_only:
        summary = write_summary(cfg, df, table_path, out_dir, prefix, warnings, backup, generated, True)
        generated["run_summary"] = str(summary)
        print("Result validation completed successfully.")
        print("No figures or summary tables were generated.")
        return 0
    table_paths = write_tables(df, cfg, out_dir, prefix)
    generated.update(table_paths)
    try:
        figure_paths = write_figures(df, coords, cfg, out_dir, prefix, warnings)
        generated["figures"] = figure_paths
    except RuntimeError as exc:
        warnings.append(str(exc))
        generated["figures"] = []
    summary = write_summary(cfg, df, table_path, out_dir, prefix, warnings, backup, generated, False)
    generated["run_summary"] = str(summary)
    log = out_dir / f"{prefix}.log"
    log.write_text("\n".join(["TACMAN result summarization completed.", f"summary: {summary}"]) + "\n")
    print(f"Summary written to: {out_dir}")
    return 0


RESULTS_TEMPLATE = """# TACMAN downstream result summarization template.
input:
  tacman_output_directory: ../output/pancreas;human-corss-mouse;pan_h-map-pan_m;aligned=True
  prediction_table: null
metadata:
  species_key: species
  reference_or_query_key: type
  reference_value: ref
  query_value: que
  true_label_key: true_label
  predicted_label_key: pre_label
  model_label_key: model_label
  confidence_key: max_prob
  umap1_key: UMAP1
  umap2_key: UMAP2
analysis:
  evaluate_query_labels: true
  normalize_confusion_matrix: true
  allow_umap_recompute: false
figures:
  format: pdf
  dpi: 300
  width: 7
  height: 6
  point_size: 5
output:
  directory: ../output_summary/pancreas
  prefix: pancreas_tacman
  overwrite: false
"""


def write_template(path: Path) -> None:
    """Write a commented results config template."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RESULTS_TEMPLATE)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Summarize and visualize completed TACMAN results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  python scripts/summarize_tacman_results.py --config configs/pancreas_results.yaml\n\n"
            "Validation only:\n"
            "  python scripts/summarize_tacman_results.py --config configs/pancreas_results.yaml --validate-only"
        ),
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without generating tables or figures.")
    parser.add_argument("--write-config-template", type=Path, help="Write a results YAML template and exit.")
    parser.add_argument("--output-dir", help="Override output.directory.")
    parser.add_argument("--prefix", help="Override output.prefix.")
    parser.add_argument("--prediction-table", help="Override input.prediction_table.")
    parser.add_argument("--overwrite", action="store_true", help="Override output.overwrite=true.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.write_config_template:
            write_template(args.write_config_template)
            print(f"Wrote config template: {args.write_config_template}")
            return 0
        if not args.config:
            parser.error("--config is required unless --write-config-template is used.")
        cfg = apply_cli_overrides(load_config(args.config), args)
        return run_summary(cfg, validate_only=args.validate_only)
    except (ConfigError, ValidationError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
