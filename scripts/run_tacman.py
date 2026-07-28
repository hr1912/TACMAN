#!/usr/bin/env python3
"""Run TACMAN from a YAML configuration file.

This wrapper validates inputs, resolves the homology file, records run
metadata, and calls the existing TACMAN.run(**params) API. It intentionally does
not copy or modify TACMAN model, graph construction, or training code.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TEMPLATE = """# TACMAN run configuration.
# Relative paths are resolved relative to this YAML file.
reference:
  h5ad: ../data/pancreas_h/counts.h5ad
  species: human
  cell_type_key: cell_type

query:
  h5ad: ../data/pancreas_m/counts.h5ad
  species: mouse

homology:
  info_csv: ../homo/info.csv
  path: null
  is_1v1: false

analysis:
  tissue: pancreas
  aligned: true
  stages: [100, 100, 100]

output:
  directory: ../output
  tag: pan_h-map-pan_m
  overwrite: false

runtime:
  seed: 0
  log_level: INFO
"""


class SimpleYaml:
    """Small YAML reader/writer for simple nested TACMAN config files.

    Supported: nested mappings, scalar values, and inline lists such as
    [100, 100, 100]. Unsupported: anchors, aliases, tags, block scalars,
    list-item blocks, and other advanced YAML features.
    """

    @staticmethod
    def strip_comment(line: str) -> str:
        """Remove comments outside quotes."""
        in_single = False
        in_double = False
        for index, char in enumerate(line):
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                return line[:index]
        return line

    @classmethod
    def parse_scalar(cls, value: str):
        """Parse a YAML scalar used by the config templates."""
        value = value.strip()
        if value == "" or value.lower() in {"null", "none", "~"}:
            return None
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [cls.parse_scalar(part.strip()) for part in inner.split(",")]
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            return value[1:-1]
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
            return float(value)
        except ValueError:
            return value

    @classmethod
    def safe_load(cls, text: str) -> Dict[str, object]:
        """Load a simple nested mapping."""
        root: Dict[str, object] = {}
        stack: List[Tuple[int, Dict[str, object]]] = [(-1, root)]
        for raw_line in text.splitlines():
            line = cls.strip_comment(raw_line).rstrip()
            if not line.strip():
                continue
            stripped_raw = line.strip()
            if stripped_raw.startswith("- "):
                raise ValueError(
                    "Unsupported YAML feature in fallback parser: block-style lists. "
                    "Install PyYAML or use inline lists such as stages: [100, 100, 100]."
                )
            if "|" in stripped_raw or ">" in stripped_raw:
                raise ValueError(
                    "Unsupported YAML feature in fallback parser: multiline block scalars. Install PyYAML."
                )
            tokens = stripped_raw.replace(":", " ").split()
            if any(token.startswith("&") or token.startswith("*") or token.startswith("!") for token in tokens):
                raise ValueError(
                    "Unsupported YAML feature in fallback parser: anchors, aliases, or tags. Install PyYAML."
                )
            indent = len(line) - len(line.lstrip(" "))
            stripped = stripped_raw
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                child: Dict[str, object] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = cls.parse_scalar(value)
        return root

    @classmethod
    def safe_dump(cls, data: object, sort_keys: bool = False) -> str:
        """Dump a small YAML-compatible representation."""
        def scalar(value: object) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, list):
                return "[" + ", ".join(scalar(v) for v in value) + "]"
            text = str(value)
            if text == "" or any(char in text for char in [":", "#", "\n"]):
                return repr(text)
            return text

        def dump_obj(value: object, indent: int = 0) -> List[str]:
            pad = " " * indent
            lines: List[str] = []
            if isinstance(value, dict):
                items = sorted(value.items()) if sort_keys else value.items()
                for key, child in items:
                    if isinstance(child, dict):
                        lines.append(f"{pad}{key}:")
                        lines.extend(dump_obj(child, indent + 2))
                    else:
                        lines.append(f"{pad}{key}: {scalar(child)}")
            else:
                lines.append(f"{pad}{scalar(value)}")
            return lines

        return "\n".join(dump_obj(data)) + "\n"


def get_yaml_module():
    """Return PyYAML when installed, otherwise use the small built-in parser."""
    try:
        import yaml

        return yaml
    except ImportError:
        return SimpleYaml


def load_yaml(path: Path) -> Dict[str, object]:
    """Load a YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = get_yaml_module().safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return data


def write_yaml(path: Path, data: Dict[str, object]) -> None:
    """Write YAML using PyYAML or the fallback serializer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(get_yaml_module().safe_dump(data, sort_keys=False), encoding="utf-8")


def nested_get(data: Dict[str, object], path: Sequence[str], default=None):
    """Read a nested config value."""
    current: object = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def nested_set(data: Dict[str, object], path: Sequence[str], value) -> None:
    """Set a nested config value."""
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def resolve_path(value: Optional[object], config_dir: Path) -> Optional[str]:
    """Resolve a path relative to the YAML directory."""
    if value is None or value == "":
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else (config_dir / path).resolve())


def default_config() -> Dict[str, object]:
    """Return software defaults before YAML and CLI overrides."""
    return {
        "reference": {"h5ad": None, "species": None, "cell_type_key": "cell_type"},
        "query": {"h5ad": None, "species": None},
        "homology": {"info_csv": "homo/info.csv", "path": None, "is_1v1": False},
        "analysis": {"tissue": None, "aligned": True, "stages": [100, 200, 200]},
        "output": {"directory": "output", "tag": "", "overwrite": False},
        "runtime": {"seed": 0, "log_level": "INFO"},
    }


def set_random_seed(seed: Optional[int], tacman_module=None) -> Dict[str, object]:
    """Apply best-effort random seed initialization before TACMAN.run.

    If a TACMAN module exposes a conventional seed helper, use it. Otherwise
    seed Python, NumPy, PyTorch, and DGL when those libraries are importable.
    """
    report: Dict[str, object] = {"seed": seed, "applied": []}
    if seed is None:
        report["note"] = "No seed configured."
        return report
    seed = int(seed)

    if tacman_module is not None:
        for attr in ["set_random_seed", "set_seed", "seed_everything"]:
            func = getattr(tacman_module, attr, None)
            if callable(func):
                func(seed)
                report["applied"].append(f"TACMAN.{attr}")
                report["note"] = "Used TACMAN-provided seed helper."
                return report

    random.seed(seed)
    report["applied"].append("random.seed")

    try:
        import numpy as np

        np.random.seed(seed)
        report["applied"].append("numpy.random.seed")
    except ImportError:
        report["numpy"] = "not installed"

    try:
        import torch

        torch.manual_seed(seed)
        report["applied"].append("torch.manual_seed")
        if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
            torch.cuda.manual_seed_all(seed)
            report["applied"].append("torch.cuda.manual_seed_all")
    except ImportError:
        report["torch"] = "not installed"

    try:
        import dgl

        if hasattr(dgl, "seed"):
            dgl.seed(seed)
            report["applied"].append("dgl.seed")
        if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
            dgl.random.seed(seed)
            report["applied"].append("dgl.random.seed")
    except ImportError:
        report["dgl"] = "not installed"
    return report


def sample_values(values, sample_size: int) -> List[object]:
    """Return at most sample_size values without materializing the full input."""
    try:
        subset = values[:sample_size]
        if hasattr(subset, "tolist"):
            return subset.tolist()
        return list(subset)
    except (TypeError, AttributeError):
        sampled = []
        for value in values:
            sampled.append(value)
            if len(sampled) >= sample_size:
                break
        return sampled


def matrix_storage_and_values(matrix, sample_size: int) -> Tuple[str, str, Tuple[int, ...], List[float]]:
    """Return matrix storage label, dtype, shape, and sampled nonzero values."""
    shape = tuple(getattr(matrix, "shape", ()))
    dtype = str(getattr(matrix, "dtype", type(matrix).__name__))

    if hasattr(matrix, "data") and hasattr(matrix, "shape"):
        storage = "sparse" if hasattr(matrix, "tocoo") or hasattr(matrix, "tocsr") else "data-array"
        values = sample_values(getattr(matrix, "data"), sample_size)
    else:
        storage = "dense"
        try:
            import numpy as np

            arr = np.asarray(matrix)
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
            values = arr.ravel()[:sample_size].tolist()
        except Exception:
            values = []
            for row in matrix:
                if isinstance(row, (list, tuple)):
                    values.extend(row[: max(0, sample_size - len(values))])
                else:
                    values.append(row)
                if len(values) >= sample_size:
                    break

    nonzero: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != 0:
            nonzero.append(numeric)
        if len(nonzero) >= sample_size:
            break
    return storage, dtype, shape, nonzero


def inspect_expression_matrix(adata, sample_size: int = 10000) -> Dict[str, object]:
    """Inspect AnnData .X for count-like values expected by TACMAN."""
    if not hasattr(adata, "X") or adata.X is None:
        raise ValueError("AnnData .X is missing.")
    storage, dtype, shape, values = matrix_storage_and_values(adata.X, sample_size)
    negative = 0
    bad = 0
    finite_values: List[float] = []
    for value in values:
        if math.isnan(value) or math.isinf(value):
            bad += 1
            continue
        if value < 0:
            negative += 1
        finite_values.append(value)

    integer_like = 0
    for value in finite_values:
        if abs(value - round(value)) <= 1e-6:
            integer_like += 1
    n_values = len(finite_values)
    integer_fraction = integer_like / n_values if n_values else 1.0
    if negative > 0:
        assessment = "ERROR: negative values detected"
    elif bad > 0:
        assessment = "ERROR: NaN or Inf values detected"
    elif integer_fraction >= 0.95:
        assessment = "likely raw counts"
    else:
        assessment = "WARNING: matrix may be normalized or log-transformed"

    return {
        "matrix storage": storage,
        "matrix dtype": dtype,
        "matrix shape": shape,
        "sampled nonzero values": len(values),
        "integer-like fraction": round(integer_fraction, 4),
        "minimum": min(finite_values) if finite_values else 0,
        "maximum": max(finite_values) if finite_values else 0,
        "mean": round(sum(finite_values) / n_values, 6) if n_values else 0,
        "negative-value count": negative,
        "NaN/Inf count": bad,
        "raw-count assessment": assessment,
    }


def check_expression_matrix_or_raise(adata, label: str) -> Dict[str, object]:
    """Inspect .X and fail on values TACMAN cannot safely normalize/log."""
    stats = inspect_expression_matrix(adata)
    if stats["negative-value count"] > 0:
        raise ValueError(f"{label}.X contains negative values; TACMAN expects count-like nonnegative .X.")
    if stats["NaN/Inf count"] > 0:
        raise ValueError(f"{label}.X contains NaN or Inf values; clean the matrix before TACMAN.")
    return {f"{label} {key}": value for key, value in stats.items()}


def prepare_output_directory(config: Dict[str, object], validate_only: bool) -> Tuple[Optional[Path], str]:
    """Handle TACMAN final output directory conflicts with safe backup semantics."""
    final_dir = final_output_dir(config)
    overwrite = bool(nested_get(config, ("output", "overwrite")))
    if final_dir.exists() and any(final_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "Output directory already exists and is not empty. "
                f"Set output.overwrite=true or choose another output tag: {final_dir}"
            )
        if validate_only:
            return None, f"would backup existing non-empty output directory before training: {final_dir}"
        backup = final_dir.with_name(final_dir.name + ".backup_" + time.strftime("%Y%m%d_%H%M%S"))
        suffix = 1
        while backup.exists():
            backup = final_dir.with_name(final_dir.name + ".backup_" + time.strftime("%Y%m%d_%H%M%S") + f"_{suffix}")
            suffix += 1
        shutil.move(str(final_dir), str(backup))
        return backup, f"backed up existing output directory to {backup}"
    return None, "no existing non-empty output directory"


def deep_merge(base: Dict[str, object], update: Dict[str, object]) -> Dict[str, object]:
    """Recursively merge dictionaries."""
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_stages(value: str) -> List[int]:
    """Parse CLI --stages value such as '100,100,100'."""
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--stages must be a comma-separated list of integers.") from exc


def parse_bool(value: str) -> bool:
    """Parse a CLI boolean override."""
    if str(value).lower() in {"1", "true", "yes", "y"}:
        return True
    if str(value).lower() in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got: {value}")


def effective_config(args: argparse.Namespace) -> Dict[str, object]:
    """Merge defaults, YAML, and explicit CLI overrides."""
    config_path = Path(args.config).resolve()
    yaml_config = load_yaml(config_path)
    config = deep_merge(default_config(), yaml_config)
    config_dir = config_path.parent

    path_fields = [
        ("reference", "h5ad"),
        ("query", "h5ad"),
        ("homology", "info_csv"),
        ("homology", "path"),
        ("output", "directory"),
    ]
    for field in path_fields:
        value = nested_get(config, field)
        resolved = resolve_path(value, config_dir)
        if resolved is not None:
            nested_set(config, field, resolved)

    if args.output_dir is not None:
        nested_set(config, ("output", "directory"), str(Path(args.output_dir).resolve()))
    if args.tag is not None:
        nested_set(config, ("output", "tag"), args.tag)
    if args.stages is not None:
        nested_set(config, ("analysis", "stages"), parse_stages(args.stages))
    if args.is_1v1 is not None:
        nested_set(config, ("homology", "is_1v1"), parse_bool(args.is_1v1))
    if args.aligned is not None:
        nested_set(config, ("analysis", "aligned"), parse_bool(args.aligned))

    nested_set(config, ("_meta", "config_file"), str(config_path))
    return config


def tacman_output_tag(config: Dict[str, object]) -> str:
    """Mirror TACMAN.run's output tag construction, including its current spelling."""
    tissue = nested_get(config, ("analysis", "tissue"))
    sp_ref = nested_get(config, ("reference", "species"))
    sp_que = nested_get(config, ("query", "species"))
    tag = nested_get(config, ("output", "tag"), "") or ""
    if len(tag) > 0:
        return f"{tissue};{sp_ref}-corss-{sp_que};{tag}"
    return f"{tissue};{sp_ref}-corss-{sp_que}"


def final_output_dir(config: Dict[str, object]) -> Path:
    """Return the directory TACMAN.run will write into."""
    return Path(str(nested_get(config, ("output", "directory")))).joinpath(tacman_output_tag(config))


def get_homo_path(sp_ref: str, sp_que: str, info_csv: Path) -> Path:
    """Resolve a TACMAN homology path from info.csv with unique species matching."""
    if not info_csv.exists():
        raise FileNotFoundError(f"homology.info_csv does not exist: {info_csv}")
    matches: List[Dict[str, str]] = []
    with info_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "name", "sp_ref", "sp_que"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"homology.info_csv must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            if row.get("sp_ref") == sp_ref and row.get("sp_que") == sp_que:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"homology.info_csv species pair is not unique for sp_ref={sp_ref}, sp_que={sp_que}; "
            f"matched {len(matches)} rows in {info_csv}."
        )
    path = Path(matches[0]["path"])
    if not path.is_absolute():
        path = info_csv.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Resolved homology file does not exist: {path}")
    return path.resolve()


def read_homology(path: Path) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """Read and validate TACMAN's three-column comma-separated homology file."""
    if not path.exists():
        raise FileNotFoundError(f"homology.path does not exist: {path}")
    rows: List[Tuple[str, str, str]] = []
    empty_ref = empty_que = empty_type = duplicate = 0
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) != 3:
            raise ValueError(f"Homology file must be comma-separated with exactly three columns: {path}")
        for raw in reader:
            if len(raw) != 3:
                raise ValueError(f"Homology file row is not three columns in {path}: {raw}")
            ref, que, htype = [item.strip() for item in raw]
            if ref == "":
                empty_ref += 1
            if que == "":
                empty_que += 1
            if htype == "":
                empty_type += 1
            if ref == "" or que == "" or htype == "":
                continue
            key = (ref, que, htype)
            if key in seen:
                duplicate += 1
            else:
                seen.add(key)
                rows.append(key)
    stats = {
        "homology pairs": len(rows),
        "empty reference gene rows": empty_ref,
        "empty query gene rows": empty_que,
        "empty homology type rows": empty_type,
        "duplicate rows": duplicate,
        "one-to-one pair count": sum(row[2] == "ortholog_one2one" for row in rows),
        "non-one-to-one pair count": sum(row[2] != "ortholog_one2one" for row in rows),
    }
    return rows, stats


def load_h5ad(path: Path):
    """Load h5ad with scanpy first, then anndata."""
    if not path.exists():
        raise FileNotFoundError(f"h5ad does not exist: {path}")
    try:
        import scanpy as sc

        return sc.read_h5ad(path)
    except ImportError:
        try:
            import anndata as ad

            return ad.read_h5ad(path)
        except ImportError as exc:
            raise RuntimeError(
                "scanpy or anndata is required to read h5ad files. Install scanpy or anndata."
            ) from exc


def adata_shape(adata) -> Tuple[int, int]:
    """Return AnnData cell/gene dimensions."""
    if hasattr(adata, "n_obs") and hasattr(adata, "n_vars"):
        return int(adata.n_obs), int(adata.n_vars)
    return int(adata.shape[0]), int(adata.shape[1])


def var_names(adata) -> List[str]:
    """Return AnnData var_names as strings."""
    if hasattr(adata, "var_names"):
        return [str(x) for x in list(adata.var_names)]
    if hasattr(adata, "var") and hasattr(adata.var, "index"):
        return [str(x) for x in list(adata.var.index)]
    raise ValueError("AnnData object does not expose var_names.")


def obs_has_key(adata, key: str) -> bool:
    """Return whether adata.obs has a metadata column."""
    obs = adata.obs
    if hasattr(obs, "columns"):
        return key in obs.columns
    return key in obs


def obs_values(adata, key: str) -> List[object]:
    """Return metadata values from adata.obs."""
    values = adata.obs[key]
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def has_missing(values: Iterable[object]) -> bool:
    """Return whether values contain NA-like entries."""
    for value in values:
        if value is None:
            return True
        try:
            if value != value:
                return True
        except Exception:
            pass
        if str(value).strip() == "":
            return True
    return False


def validate_anndata(adata, label: str, cell_type_key: Optional[str] = None) -> Dict[str, object]:
    """Validate one AnnData object for TACMAN input."""
    cells, genes = adata_shape(adata)
    names = var_names(adata)
    if len(names) != len(set(names)):
        raise ValueError(f"{label}.h5ad has duplicated gene identifiers in var_names.")
    if not hasattr(adata, "X") or adata.X is None:
        raise ValueError(f"{label}.h5ad must contain an expression matrix in .X.")
    result = {f"{label} cells": cells, f"{label} genes": genes}
    if cell_type_key is not None:
        if not obs_has_key(adata, cell_type_key):
            raise ValueError(f"reference.cell_type_key '{cell_type_key}' is not present in reference .obs.")
        labels = obs_values(adata, cell_type_key)
        if has_missing(labels):
            raise ValueError(f"reference.cell_type_key '{cell_type_key}' contains missing labels.")
        result["reference labels"] = len(set(map(str, labels)))
    return result


def validate_inputs(config: Dict[str, object]) -> Tuple[object, object, Path, Dict[str, object]]:
    """Validate files, AnnData metadata, homology format, and gene overlap."""
    ref_h5ad = Path(str(nested_get(config, ("reference", "h5ad"))))
    que_h5ad = Path(str(nested_get(config, ("query", "h5ad"))))
    sp_ref = nested_get(config, ("reference", "species"))
    sp_que = nested_get(config, ("query", "species"))
    key_cell_type = nested_get(config, ("reference", "cell_type_key"))
    if not sp_ref:
        raise ValueError("reference.species is required.")
    if not sp_que:
        raise ValueError("query.species is required.")
    if not nested_get(config, ("analysis", "tissue")):
        raise ValueError("analysis.tissue is required.")

    adata_ref = load_h5ad(ref_h5ad)
    adata_que = load_h5ad(que_h5ad)
    summary: Dict[str, object] = {}
    summary.update(validate_anndata(adata_ref, "reference", str(key_cell_type)))
    summary.update(validate_anndata(adata_que, "query", None))
    if obs_has_key(adata_que, str(key_cell_type)):
        query_labels = obs_values(adata_que, str(key_cell_type))
        if has_missing(query_labels):
            raise ValueError(f"query metadata column '{key_cell_type}' contains missing labels.")
        summary["query labels"] = len(set(map(str, query_labels)))
    else:
        summary["query labels"] = "not provided; TACMAN.run will fill query labels with 'NA'"
    summary.update(check_expression_matrix_or_raise(adata_ref, "reference"))
    summary.update(check_expression_matrix_or_raise(adata_que, "query"))

    homology_path_value = nested_get(config, ("homology", "path"))
    if homology_path_value:
        homology_path = Path(str(homology_path_value))
        if not homology_path.exists():
            raise FileNotFoundError(f"homology.path does not exist: {homology_path}")
    else:
        homology_path = get_homo_path(
            str(sp_ref),
            str(sp_que),
            Path(str(nested_get(config, ("homology", "info_csv")))),
        )
    homology_rows, homology_stats = read_homology(homology_path)
    if homology_stats["homology pairs"] == 0:
        raise ValueError(f"Homology file contains zero valid pairs: {homology_path}")
    invalid_counts = [
        ("empty reference gene rows", homology_stats["empty reference gene rows"]),
        ("empty query gene rows", homology_stats["empty query gene rows"]),
        ("empty homology type rows", homology_stats["empty homology type rows"]),
        ("duplicate rows", homology_stats["duplicate rows"]),
    ]
    invalid = [f"{name}={count}" for name, count in invalid_counts if count]
    if invalid:
        raise ValueError(f"Homology file contains invalid rows ({', '.join(invalid)}): {homology_path}")

    ref_genes = set(var_names(adata_ref))
    que_genes = set(var_names(adata_que))
    homo_ref = {row[0] for row in homology_rows}
    homo_que = {row[1] for row in homology_rows}
    ref_overlap = len(ref_genes & homo_ref)
    que_overlap = len(que_genes & homo_que)
    if ref_overlap == 0:
        raise ValueError("reference gene overlap with homology file is 0; check gene identifiers.")
    if que_overlap == 0:
        raise ValueError("query gene overlap with homology file is 0; check gene identifiers.")

    summary.update(homology_stats)
    is_1v1 = bool(nested_get(config, ("homology", "is_1v1")))
    usable_pairs = homology_stats["one-to-one pair count"] if is_1v1 else homology_stats["homology pairs"]
    if is_1v1 and usable_pairs == 0:
        raise ValueError("homology.is_1v1 is true but the homology file contains zero ortholog_one2one pairs.")
    summary["is_1v1 expected homology pairs"] = usable_pairs
    summary["homology path"] = str(homology_path)
    summary["reference gene overlap"] = ref_overlap
    summary["query gene overlap"] = que_overlap
    summary["reference gene overlap percent"] = round(ref_overlap / max(1, len(ref_genes)) * 100, 3)
    summary["query gene overlap percent"] = round(que_overlap / max(1, len(que_genes)) * 100, 3)
    if summary["reference gene overlap percent"] < 1 or summary["query gene overlap percent"] < 1:
        summary["overlap warning"] = "Gene overlap is below 1%; verify AnnData.var_names and homology IDs."
    nested_set(config, ("homology", "resolved_path"), str(homology_path))
    return adata_ref, adata_que, homology_path, summary


def write_summary(path: Path, lines: Dict[str, object]) -> None:
    """Write a plain-text run summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key, value in lines.items():
            handle.write(f"{key}: {value}\n")


def import_tacman():
    """Import TACMAN and provide a clear error."""
    try:
        import TACMAN

        return TACMAN
    except ImportError as exc:
        raise RuntimeError(
            "TACMAN package could not be imported. Install TACMAN or run this script from an environment where TACMAN is on PYTHONPATH."
        ) from exc


def build_run_params(config: Dict[str, object], adata_ref, adata_que, homology_path: Path) -> Dict[str, object]:
    """Build keyword arguments for TACMAN.run without changing its API."""
    return {
        "adata_ref": adata_ref,
        "adata_que": adata_que,
        "sp_ref": nested_get(config, ("reference", "species")),
        "sp_que": nested_get(config, ("query", "species")),
        "key_cell_type": nested_get(config, ("reference", "cell_type_key")),
        "tissue": nested_get(config, ("analysis", "tissue")),
        "aligned": bool(nested_get(config, ("analysis", "aligned"))),
        "p_output": Path(str(nested_get(config, ("output", "directory")))),
        "tag_output": nested_get(config, ("output", "tag"), "") or "",
        "p_homo": homology_path,
        "stages": list(nested_get(config, ("analysis", "stages"))),
        "is_1v1": bool(nested_get(config, ("homology", "is_1v1"))),
    }


def check_tacman_signature(tacman, params: Dict[str, object]) -> None:
    """Check that current TACMAN.run can accept the wrapper parameters."""
    try:
        signature = inspect.signature(tacman.run)
    except (TypeError, ValueError):
        return
    try:
        signature.bind_partial(**params)
    except TypeError as exc:
        raise TypeError(f"TACMAN.run parameters do not match this TACMAN version: {exc}") from exc


def run_tacman(config: Dict[str, object], validate_only: bool = False) -> Dict[str, object]:
    """Validate config and optionally call TACMAN.run."""
    adata_ref, adata_que, homology_path, summary = validate_inputs(config)
    output_dir = Path(str(nested_get(config, ("output", "directory"))))
    tag = nested_get(config, ("output", "tag"), "") or "tacman"
    final_dir = final_output_dir(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / f"{tag}.resolved_config.yaml"
    summary_path = output_dir / f"{tag}.run_summary.txt"
    nested_set(config, ("output", "final_directory"), str(final_dir))

    summary["configuration file"] = nested_get(config, ("_meta", "config_file"))
    summary["resolved configuration"] = str(resolved_config_path)
    summary["run summary"] = str(summary_path)
    summary["TACMAN final output directory"] = str(final_dir)
    summary["validate only"] = validate_only
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if validate_only:
        backup_dir, overwrite_message = prepare_output_directory(config, validate_only=True)
        if backup_dir is not None:
            nested_set(config, ("output", "backup_directory"), str(backup_dir))
        summary["output overwrite action"] = overwrite_message
        summary["output backup directory"] = str(backup_dir) if backup_dir is not None else "none"
        write_yaml(resolved_config_path, config)
        seed_report = set_random_seed(nested_get(config, ("runtime", "seed")))
        summary["seed"] = seed_report["seed"]
        summary["seed applied by"] = ", ".join(seed_report.get("applied", []))
        summary["TACMAN.run called"] = False
        summary["validation status"] = "Validation completed successfully. TACMAN model training was not started."
        write_summary(summary_path, summary)
        return summary

    tacman = import_tacman()
    seed_report = set_random_seed(nested_get(config, ("runtime", "seed")), tacman_module=tacman)
    summary["seed"] = seed_report["seed"]
    summary["seed applied by"] = ", ".join(seed_report.get("applied", []))
    params = build_run_params(config, adata_ref, adata_que, homology_path)
    check_tacman_signature(tacman, params)
    backup_dir, overwrite_message = prepare_output_directory(config, validate_only=False)
    if backup_dir is not None:
        nested_set(config, ("output", "backup_directory"), str(backup_dir))
    summary["output overwrite action"] = overwrite_message
    summary["output backup directory"] = str(backup_dir) if backup_dir is not None else "none"
    write_yaml(resolved_config_path, config)
    log_path = output_dir / f"{tag}.run_log.txt"
    summary["run log"] = str(log_path)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("Calling TACMAN.run with wrapper-generated parameters.\n")
            result = tacman.run(**params)
            log.write(f"[finish] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as exc:
        summary["TACMAN.run called"] = True
        summary["TACMAN.run failed"] = repr(exc)
        write_summary(summary_path, summary)
        raise RuntimeError(f"Model run failed while calling TACMAN.run: {exc}") from exc

    summary["TACMAN.run called"] = True
    summary["TACMAN.run return value"] = repr(result)
    write_summary(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run TACMAN from a YAML configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Main command:\n"
            "  python scripts/run_tacman.py --config configs/pancreas_tacman.yaml\n\n"
            "Validation only:\n"
            "  python scripts/run_tacman.py --config configs/pancreas_tacman.yaml --validate-only"
        ),
    )
    parser.add_argument("--config", help="TACMAN YAML configuration file.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without calling TACMAN.run.")
    parser.add_argument("--write-config-template", help="Write a TACMAN run YAML template and exit.")
    parser.add_argument("--output-dir", help="Override output.directory.")
    parser.add_argument("--tag", help="Override output.tag.")
    parser.add_argument("--stages", help="Override analysis.stages, e.g. 100,100,100.")
    parser.add_argument("--is-1v1", help="Override homology.is_1v1: true/false.")
    parser.add_argument("--aligned", help="Override analysis.aligned: true/false.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.write_config_template:
            path = Path(args.write_config_template)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(TEMPLATE, encoding="utf-8")
            print(f"Wrote TACMAN config template: {path}")
            return
        if not args.config:
            parser.error("Specify --config or --write-config-template.")
        config = effective_config(args)
        summary = run_tacman(config, validate_only=args.validate_only)
        print("TACMAN input validation summary")
        for key, value in summary.items():
            print(f"{key}: {value}")
        if args.validate_only:
            print("Validation completed successfully.")
            print("TACMAN model training was not started.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
