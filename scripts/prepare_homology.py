#!/usr/bin/env python3
"""Prepare TACMAN-compatible homology files from external resources."""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

pd = None


DEFAULT_HOMOLOGY_TYPE_MAP: Dict[str, str] = {
    "ortholog_one2one": "ortholog_one2one",
    "ortholog_one2many": "ortholog_one2many",
    "ortholog_many2many": "ortholog_many2many",
    "one2one": "ortholog_one2one",
    "one_to_one": "ortholog_one2one",
    "1:1": "ortholog_one2one",
    "one2many": "ortholog_one2many",
    "one_to_many": "ortholog_one2many",
    "1:n": "ortholog_one2many",
    "many2many": "ortholog_many2many",
    "many_to_many": "ortholog_many2many",
    "m:n": "ortholog_many2many",
}


def require_pandas():
    """Import pandas lazily so --help works before dependencies are installed."""
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as pandas_module
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for homology preparation. Install pandas in "
            "the Python environment used to run this script."
        ) from exc
    pd = pandas_module
    return pd


class SimpleYamlFallback:
    """Tiny YAML reader/writer for the project config templates.

    This fallback supports nested mappings with scalar values. PyYAML is still
    preferred for general YAML support.
    """

    @staticmethod
    def _strip_comment(line: str) -> str:
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

    @staticmethod
    def _parse_scalar(value: str) -> object:
        value = value.strip()
        if value == "" or value.lower() in {"null", "none", "~"}:
            return None
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            return value[1:-1]
        try:
            if re.match(r"^[+-]?\d+$", value):
                return int(value)
            if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", value):
                return float(value)
        except ValueError:
            pass
        return value

    @classmethod
    def safe_load(cls, text: str) -> Dict[str, object]:
        """Parse a small YAML mapping subset used by TACMAN configs."""
        root: Dict[str, object] = {}
        stack: List[Tuple[int, Dict[str, object]]] = [(-1, root)]
        for raw_line in text.splitlines():
            line = cls._strip_comment(raw_line).rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line without ':': {raw_line}")
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
                parent[key] = cls._parse_scalar(value)
        return root

    @classmethod
    def safe_dump(cls, data: object, sort_keys: bool = False) -> str:
        """Serialize simple dict/list/scalar structures for snapshots."""
        def scalar(value: object) -> str:
            if value is None:
                return "null"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return str(value)
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
                    if isinstance(child, (dict, list)):
                        lines.append(f"{pad}{key}:")
                        lines.extend(dump_obj(child, indent + 2))
                    else:
                        lines.append(f"{pad}{key}: {scalar(child)}")
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, (dict, list)):
                        lines.append(f"{pad}-")
                        lines.extend(dump_obj(child, indent + 2))
                    else:
                        lines.append(f"{pad}- {scalar(child)}")
            else:
                lines.append(f"{pad}{scalar(value)}")
            return lines

        return "\n".join(dump_obj(data)) + "\n"


def require_yaml():
    """Import PyYAML lazily, falling back to the bundled simple config parser."""
    try:
        import yaml
    except ImportError as exc:
        print(
            "WARNING: PyYAML is not installed; using the built-in simple YAML "
            "parser for TACMAN config files. Install PyYAML for full YAML support.",
            file=sys.stderr,
        )
        return SimpleYamlFallback
    return yaml


FULL_CONFIG_TEMPLATE = """# TACMAN homology preparation configuration.
# Relative paths are resolved relative to this YAML file.
mode: blast

species:
  reference: C_elegans
  query: D_melanogaster
  query_display_name: Drosophila

input:
  reference_protein: ../raw/C_elegans.protein.fa
  query_protein: ../raw/D_melanogaster.protein.fa
  reference_id_map: ../mapping/C_elegans_protein_to_gene.tsv
  query_id_map: ../mapping/D_melanogaster_protein_to_gene.tsv

columns:
  reference_protein_id: protein_id
  reference_gene: gene_id
  query_protein_id: protein_id
  query_gene: gene_id
  homology_type: null

header_parsing:
  reference_gene_pattern: null
  query_gene_pattern: null

blast:
  evalue: 1e-5
  min_pident: 30
  min_qcov: 50
  min_scov: null
  top_n: 5
  max_target_seqs: 20
  threads: 8
  rbh_level: gene
  best_hit_tie_policy: discard
  blastdb_version: null
  keep_blastdb: false
  makeblastdb_bin: makeblastdb
  blastp_bin: blastp
  strict_id_map: false

output:
  path: ../ortholog/C_elegans_to_D_melanogaster.txt
  primary_output: all_putative
  separator: comma
  register_info: ../ortholog/info.csv
  overwrite_info: false
  skip_register_if_exists: false
  backup_info: true
  qc_report: null

identifiers:
  gene_id_type: stable_id
  allow_non_symbol_ids: false

runtime:
  blast_workdir: null
  keep_intermediate: true
"""


MINIMAL_CONFIG_TEMPLATE = """# Minimal TACMAN homology preparation configuration.
# Relative paths are resolved relative to this YAML file.
mode: blast

species:
  reference: C_elegans
  query: D_melanogaster

input:
  reference_protein: ../raw/C_elegans.protein.fa
  query_protein: ../raw/D_melanogaster.protein.fa

header_parsing:
  reference_gene_pattern: 'gene:([^\\s]+)'
  query_gene_pattern: 'gene:([^\\s]+)'

output:
  path: ../ortholog/C_elegans_to_D_melanogaster.txt
"""


CONFIG_FIELD_TO_DEST: Dict[Tuple[str, ...], str] = {
    ("mode",): "mode",
    ("species", "reference"): "sp_ref",
    ("species", "query"): "sp_que",
    ("species", "query_display_name"): "query_display_name",
    ("input", "path"): "input",
    ("input", "table"): "input",
    ("input", "database_table"): "input",
    ("input", "reference_protein"): "ref_protein",
    ("input", "query_protein"): "que_protein",
    ("input", "reference_id_map"): "ref_id_map",
    ("input", "query_id_map"): "que_id_map",
    ("columns", "reference_protein_id"): "ref_protein_id_column",
    ("columns", "reference_gene"): "ref_gene_column",
    ("columns", "query_protein_id"): "que_protein_id_column",
    ("columns", "query_gene"): "que_gene_column",
    ("columns", "homology_type"): "homology_type_column",
    ("header_parsing", "reference_gene_pattern"): "ref_header_gene_pattern",
    ("header_parsing", "query_gene_pattern"): "que_header_gene_pattern",
    ("blast", "evalue"): "evalue",
    ("blast", "min_pident"): "min_pident",
    ("blast", "min_qcov"): "min_qcov",
    ("blast", "min_scov"): "min_scov",
    ("blast", "top_n"): "top_n",
    ("blast", "max_target_seqs"): "max_target_seqs",
    ("blast", "threads"): "threads",
    ("blast", "rbh_level"): "rbh_level",
    ("blast", "best_hit_tie_policy"): "best_hit_tie_policy",
    ("blast", "blastdb_version"): "blastdb_version",
    ("blast", "keep_blastdb"): "keep_blastdb",
    ("blast", "makeblastdb_bin"): "makeblastdb_bin",
    ("blast", "blastp_bin"): "blastp_bin",
    ("blast", "strict_id_map"): "strict_id_map",
    ("output", "path"): "out",
    ("output", "primary_output"): "primary_output",
    ("output", "separator"): "output_sep",
    ("output", "register_info"): "register_info",
    ("output", "overwrite_info"): "overwrite_info",
    ("output", "skip_register_if_exists"): "skip_register_if_exists",
    ("output", "backup_info"): "backup_info",
    ("output", "qc_report"): "qc_out",
    ("identifiers", "gene_id_type"): "gene_id_type",
    ("identifiers", "allow_non_symbol_ids"): "allow_non_symbol_ids",
    ("runtime", "blast_workdir"): "blast_workdir",
    ("runtime", "keep_intermediate"): "keep_intermediate",
}

CONFIG_PATH_DESTS = {
    "input",
    "ref_protein",
    "que_protein",
    "ref_id_map",
    "que_id_map",
    "out",
    "register_info",
    "qc_out",
    "blast_workdir",
    "homology_type_map",
}


def write_config_template(path: Path, minimal: bool = False) -> None:
    """Write a commented YAML configuration template."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = MINIMAL_CONFIG_TEMPLATE if minimal else FULL_CONFIG_TEMPLATE
    path.write_text(text, encoding="utf-8")


def yaml_path(path_parts: Sequence[str]) -> str:
    """Return a dotted YAML path for error messages."""
    return ".".join(path_parts)


def get_nested(config: Dict[str, object], path_parts: Sequence[str]) -> object:
    """Read a nested config field, returning None when any level is absent."""
    value: object = config
    for part in path_parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def flatten_config(config: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Map supported YAML fields to argparse destination names."""
    flat: Dict[str, object] = {}
    field_paths: Dict[str, str] = {}
    for path_parts, dest in CONFIG_FIELD_TO_DEST.items():
        value = get_nested(config, path_parts)
        if value is not None:
            flat[dest] = value
            field_paths[dest] = yaml_path(path_parts)
    return flat, field_paths


def resolve_config_path(value: object, config_dir: Path) -> object:
    """Resolve a YAML path value relative to the YAML file directory."""
    if value is None or value == "":
        return value
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_config_file(config_path: Path) -> Dict[str, object]:
    """Load a YAML config file with PyYAML."""
    yaml = require_yaml()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level.")
    return data


def collect_explicit_cli_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> Set[str]:
    """Return argparse destination names explicitly provided on the command line."""
    explicit: Set[str] = set()
    option_actions = parser._option_string_actions
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None and action.dest != argparse.SUPPRESS:
            explicit.add(action.dest)
    return explicit


def apply_config_to_args(
    args: argparse.Namespace,
    config_path: Path,
    config: Dict[str, object],
    explicit_dests: Set[str],
) -> Dict[str, str]:
    """Merge YAML values into parsed CLI args without overriding explicit CLI."""
    flat, field_paths = flatten_config(config)
    config_dir = config_path.parent.resolve()
    for dest, value in flat.items():
        if dest in explicit_dests:
            continue
        if dest in CONFIG_PATH_DESTS:
            value = resolve_config_path(value, config_dir)
        setattr(args, dest, value)
    setattr(args, "_config_file", str(config_path.resolve()))
    setattr(args, "_raw_config", config)
    setattr(args, "_config_field_paths", field_paths)
    setattr(args, "_explicit_cli_dests", explicit_dests)
    return field_paths


def config_error(field: str, value: object, expected: str) -> ValueError:
    """Create a standardized YAML validation error."""
    return ValueError(f"Invalid config field: {field}={value}. Expected {expected}.")


def validate_number_range(field: str, value: object, minimum: float, maximum: Optional[float]) -> None:
    """Validate a numeric config field."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise config_error(field, value, "a numeric value.") from exc
    if numeric < minimum or (maximum is not None and numeric > maximum):
        if maximum is None:
            expected = f"a value >= {minimum}."
        else:
            expected = f"a value between {minimum} and {maximum}."
        raise config_error(field, value, expected)


def validate_integer_min(field: str, value: object, minimum: int) -> None:
    """Validate an integer config field."""
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise config_error(field, value, f"an integer >= {minimum}.") from exc
    if numeric < minimum:
        raise config_error(field, value, f"an integer >= {minimum}.")


def coerce_effective_arg_types(args: argparse.Namespace) -> None:
    """Coerce YAML scalar values to the types used by the CLI parser."""
    for name in ["evalue", "min_pident", "min_qcov", "min_scov"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, float(value))
    for name in ["top_n", "threads", "max_target_seqs", "blastdb_version"]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, int(value))


def validate_effective_config(args: argparse.Namespace) -> None:
    """Validate merged config/CLI values before running a workflow."""
    coerce_effective_arg_types(args)
    if args.mode not in {"database", "blast", "blast-postprocess"}:
        raise config_error("mode", args.mode, "database, blast, or blast-postprocess.")
    if not args.sp_ref:
        raise config_error("species.reference", args.sp_ref, "a non-empty reference species name.")
    if not args.sp_que:
        raise config_error("species.query", args.sp_que, "a non-empty query species name.")
    if not args.out:
        raise config_error("output.path", args.out, "a writable output file path.")

    normalize_output_separator(args.output_sep)
    primary_output_key(args.primary_output)
    if args.best_hit_tie_policy not in {"first", "discard", "all"}:
        raise config_error(
            "blast.best_hit_tie_policy",
            args.best_hit_tie_policy,
            "first, discard, or all.",
        )
    if args.rbh_level not in {"protein", "gene"}:
        raise config_error("blast.rbh_level", args.rbh_level, "protein or gene.")
    if args.gene_id_type not in {"auto", "symbol", "stable_id"}:
        raise config_error("identifiers.gene_id_type", args.gene_id_type, "auto, symbol, or stable_id.")

    validate_number_range("blast.evalue", args.evalue, 0, None)
    if float(args.evalue) <= 0:
        raise config_error("blast.evalue", args.evalue, "a value > 0.")
    validate_number_range("blast.min_pident", args.min_pident, 0, 100)
    validate_number_range("blast.min_qcov", args.min_qcov, 0, 100)
    if args.min_scov is not None:
        validate_number_range("blast.min_scov", args.min_scov, 0, 100)
    validate_integer_min("blast.top_n", args.top_n, 1)
    validate_integer_min("blast.threads", args.threads, 1)
    validate_integer_min("blast.max_target_seqs", args.max_target_seqs, 1)
    if args.blastdb_version not in {None, 4, 5}:
        raise config_error("blast.blastdb_version", args.blastdb_version, "4, 5, or null.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.qc_out:
        Path(args.qc_out).parent.mkdir(parents=True, exist_ok=True)
    if args.register_info:
        Path(args.register_info).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "database":
        if not args.input:
            raise config_error("input.path", args.input, "a database homology table path.")
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input file from config does not exist: {args.input}")
        for field, value in [
            ("columns.reference_gene", args.ref_gene_column),
            ("columns.query_gene", args.que_gene_column),
            ("columns.homology_type", args.homology_type_column),
        ]:
            if not value:
                raise config_error(field, value, "a column name.")

    if args.mode == "blast":
        for field, value in [
            ("input.reference_protein", args.ref_protein),
            ("input.query_protein", args.que_protein),
        ]:
            if not value:
                raise config_error(field, value, "a protein FASTA path.")
            if not Path(value).exists():
                raise FileNotFoundError(f"Input file from config does not exist: {value}")
        if not args.ref_id_map and not getattr(args, "ref_header_gene_pattern", None):
            raise config_error(
                "input.reference_id_map",
                args.ref_id_map,
                "a mapping table or header_parsing.reference_gene_pattern.",
            )
        if not args.que_id_map and not getattr(args, "que_header_gene_pattern", None):
            raise config_error(
                "input.query_id_map",
                args.que_id_map,
                "a mapping table or header_parsing.query_gene_pattern.",
            )

    if args.mode in {"blast", "blast-postprocess"}:
        for field, value in [
            ("input.reference_id_map", args.ref_id_map),
            ("input.query_id_map", args.que_id_map),
        ]:
            if not value:
                continue
            if not Path(value).exists():
                raise FileNotFoundError(f"Input file from config does not exist: {value}")
        if args.mode == "blast-postprocess":
            if not args.blast_workdir:
                raise config_error("runtime.blast_workdir", args.blast_workdir, "an existing BLAST workdir.")
            if not Path(args.blast_workdir).exists():
                raise FileNotFoundError(f"BLAST workdir from config does not exist: {args.blast_workdir}")


def open_text_maybe_gzip(path: Path):
    """Open plain-text or gzip-compressed FASTA for reading."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_protein_id_from_header(header: str) -> str:
    """Extract the first-token FASTA protein ID and remove version suffixes."""
    token = header[1:].strip().split()[0]
    return re.sub(r"\.\d+$", "", token)


def generate_id_map_from_fasta_header(
    fasta_path: Path,
    regex_pattern: str,
    output_path: Path,
    side_label: str,
) -> Dict[str, object]:
    """Generate a protein-to-gene map by applying a regex to FASTA headers."""
    pattern = re.compile(regex_pattern)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_entries = 0
    n_mapped = 0
    missing: List[str] = []
    with open_text_maybe_gzip(fasta_path) as handle, output_path.open("w", encoding="utf-8") as out:
        out.write("protein_id\tgene_id\n")
        for line in handle:
            if not line.startswith(">"):
                continue
            n_entries += 1
            protein_id = parse_protein_id_from_header(line)
            match = pattern.search(line)
            if match:
                gene_id = match.group(1).strip()
                if gene_id:
                    n_mapped += 1
                    out.write(f"{protein_id}\t{gene_id}\n")
                    continue
            missing.append(protein_id)

    if n_mapped == 0:
        raise ValueError(
            f"No {side_label} gene IDs were extracted from {fasta_path} using regex: {regex_pattern}"
        )
    return {
        "side": side_label,
        "mapping source": "fasta_header",
        "path": str(output_path),
        "regex": regex_pattern,
        "number of protein entries": n_entries,
        "number successfully mapped": n_mapped,
        "number missing gene ID": len(missing),
    }


def default_blast_workdir_for_output(output_path: Path) -> Path:
    """Return the default BLAST workdir used by --mode blast."""
    return output_path.with_suffix("").with_name(output_path.stem + ".blast_workdir")


def prepare_header_mapping_if_needed(args: argparse.Namespace) -> None:
    """Create protein-to-gene maps from FASTA headers when config omits maps."""
    mapping_metadata: List[Dict[str, object]] = []
    if args.mode == "blast":
        workdir = Path(args.blast_workdir) if args.blast_workdir else default_blast_workdir_for_output(Path(args.out))
        if not args.ref_id_map:
            pattern = getattr(args, "ref_header_gene_pattern", None)
            if not pattern:
                raise config_error(
                    "header_parsing.reference_gene_pattern",
                    pattern,
                    "a regex with one capture group when reference_id_map is null.",
                )
            path = workdir / "generated_reference_protein_to_gene.tsv"
            metadata = generate_id_map_from_fasta_header(Path(args.ref_protein), pattern, path, "reference")
            mapping_metadata.append(metadata)
            args.ref_id_map = str(path)
            args.ref_protein_id_column = "protein_id"
            args.ref_gene_column = "gene_id"
        else:
            mapping_metadata.append(
                {"side": "reference", "mapping source": "external_table", "path": str(args.ref_id_map)}
            )

        if not args.que_id_map:
            pattern = getattr(args, "que_header_gene_pattern", None)
            if not pattern:
                raise config_error(
                    "header_parsing.query_gene_pattern",
                    pattern,
                    "a regex with one capture group when query_id_map is null.",
                )
            path = workdir / "generated_query_protein_to_gene.tsv"
            metadata = generate_id_map_from_fasta_header(Path(args.que_protein), pattern, path, "query")
            mapping_metadata.append(metadata)
            args.que_id_map = str(path)
            args.que_protein_id_column = "protein_id"
            args.que_gene_column = "gene_id"
        else:
            mapping_metadata.append(
                {"side": "query", "mapping source": "external_table", "path": str(args.que_id_map)}
            )

    elif args.mode == "blast-postprocess":
        if args.ref_id_map:
            mapping_metadata.append(
                {"side": "reference", "mapping source": "external_table", "path": str(args.ref_id_map)}
            )
        if args.que_id_map:
            mapping_metadata.append(
                {"side": "query", "mapping source": "external_table", "path": str(args.que_id_map)}
            )

    setattr(args, "_mapping_metadata", mapping_metadata)


def namespace_to_serializable_dict(args: argparse.Namespace) -> Dict[str, object]:
    """Convert a Namespace to a YAML-safe dict, omitting internal bulky values."""
    result: Dict[str, object] = {}
    for key, value in vars(args).items():
        if key.startswith("_raw_config"):
            continue
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, set):
            result[key] = sorted(value)
        else:
            result[key] = value
    return result


def parser_default_values() -> Dict[str, object]:
    """Return argparse defaults for reproducibility snapshots."""
    defaults: Dict[str, object] = {}
    for action in build_parser()._actions:
        if action.dest in {argparse.SUPPRESS, "help"}:
            continue
        if action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    return defaults


def write_resolved_config_snapshot(args: argparse.Namespace) -> Optional[Path]:
    """Write the effective config used for a run to <out_prefix>.resolved_config.yaml."""
    if not getattr(args, "_config_file", None):
        return None
    yaml = require_yaml()
    output_path = Path(args.out)
    snapshot_path = output_prefix_from_out(output_path).with_name(
        output_prefix_from_out(output_path).name + ".resolved_config.yaml"
    )
    snapshot = {
        "configuration_file": getattr(args, "_config_file", None),
        "script_defaults": parser_default_values(),
        "yaml_values": getattr(args, "_raw_config", {}),
        "cli_overrides": {
            dest: getattr(args, dest)
            for dest in sorted(getattr(args, "_explicit_cli_dests", set()))
            if hasattr(args, dest)
        },
        "final_arguments": namespace_to_serializable_dict(args),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    setattr(args, "_resolved_config_snapshot", str(snapshot_path.resolve()))
    return snapshot_path


def config_qc_lines(args: argparse.Namespace) -> List[Tuple[str, object]]:
    """Return QC lines describing config and mapping provenance."""
    lines: List[Tuple[str, object]] = []
    if getattr(args, "_config_file", None):
        lines.append(("configuration file", getattr(args, "_config_file")))
        lines.append(("resolved configuration snapshot", getattr(args, "_resolved_config_snapshot", "not written")))
    for metadata in getattr(args, "_mapping_metadata", []):
        side = metadata.get("side", "unknown")
        lines.append((f"{side} mapping source", metadata.get("mapping source", "unknown")))
        if "path" in metadata:
            lines.append((f"{side} mapping path", metadata["path"]))
        if "regex" in metadata:
            lines.append((f"{side} mapping regex used", metadata["regex"]))
            lines.append((f"{side} mapping number of protein entries", metadata["number of protein entries"]))
            lines.append((f"{side} mapping number successfully mapped", metadata["number successfully mapped"]))
            lines.append((f"{side} mapping number missing gene ID", metadata["number missing gene ID"]))
    return lines


def normalize_separator(sep: Optional[str], input_path: Path) -> str:
    """Return the requested delimiter or infer it from the first input line."""
    if sep is not None and sep != "":
        if sep in {"\\t", "tab", "TAB"}:
            return "\t"
        return sep

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
    return "," if first_line.count(",") > first_line.count("\t") else "\t"


def separator_label(sep: str) -> str:
    """Return a readable separator label for QC reports."""
    if sep == "\t":
        return "\\t"
    if sep == ",":
        return ","
    return sep


def normalize_output_separator(output_sep: str) -> str:
    """Normalize TACMAN-facing output separator names."""
    if output_sep in {",", "comma", "csv"}:
        return ","
    if output_sep in {"\\t", "\t", "tab", "tsv"}:
        return "\t"
    raise ValueError("--output-sep must be one of comma, tab, ',' or '\\t'.")


def species_display_name(sp_que: str, query_display_name: Optional[str]) -> str:
    """Return the species label used in TACMAN homology column names."""
    if query_display_name:
        return query_display_name
    if not sp_que:
        raise ValueError("--sp-que must not be empty")
    return sp_que[:1].upper() + sp_que[1:]


def required_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that all user-selected columns exist in the input table."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Input table is missing required column(s): {}. Available columns: {}".format(
                ", ".join(missing), ", ".join(map(str, df.columns))
            )
        )


def looks_like_external_id(value: str) -> bool:
    """Conservatively detect obvious Ensembl/protein/transcript IDs."""
    value = str(value).strip()
    patterns = [
        r"^ENS[A-Z]*G\d+(\.\d+)?$",
        r"^ENS[A-Z]*P\d+(\.\d+)?$",
        r"^ENSP\d+(\.\d+)?$",
        r"^ENSMUSP\d+(\.\d+)?$",
        r"^ENSRNOP\d+(\.\d+)?$",
        r"^XP_\d+(\.\d+)?$",
        r"^NP_\d+(\.\d+)?$",
        r"^YP_\d+(\.\d+)?$",
    ]
    return any(re.match(pattern, value) for pattern in patterns)


def identifier_like_fraction(values: Iterable[object], sample_size: int = 100) -> float:
    """Return the fraction of sampled values that look like external IDs."""
    values_list = [str(value).strip() for value in values if str(value).strip()]
    if not values_list:
        return 0.0
    sample = values_list[:sample_size]
    return sum(looks_like_external_id(value) for value in sample) / len(sample)


def assert_gene_symbol_columns(df, ref_col: str, que_col: str, allow_non_symbol: bool) -> None:
    """Warn users early when selected columns look like IDs instead of symbols."""
    if allow_non_symbol:
        return

    suspicious = []
    for label, column in [("reference", ref_col), ("query", que_col)]:
        fraction = identifier_like_fraction(df[column].dropna().head(100))
        if fraction > 0.30:
            suspicious.append(f"{label} column '{column}'")

    if suspicious:
        raise ValueError(
            "TACMAN homology files should use gene symbols, but the selected "
            "column(s) look like Ensembl/protein/transcript IDs: {}. Provide "
            "database columns containing gene symbols before running this script, "
            "or rerun with --allow-non-symbol-ids only if these values are truly "
            "the var_names used by your AnnData objects.".format(", ".join(suspicious))
        )


def summarize_identifier_like_gene_values(
    values: Iterable[object],
    label: str,
    allow_non_symbol: bool,
) -> Tuple[List[str], float]:
    """Return warnings and identifier-like value fraction for gene-symbol values."""
    if allow_non_symbol:
        return [], 0.0
    fraction = identifier_like_fraction(values)
    if fraction > 0.30:
        return [
            f"{label} gene values look like Ensembl/protein/transcript IDs. "
            "TACMAN usually expects gene symbols; verify that these values match "
            "adata.var_names or provide a protein-to-gene-symbol mapping."
        ], fraction
    return [], fraction


def load_homology_type_map(path: Optional[Path], sep: Optional[str]) -> Dict[str, str]:
    """Load a two-column homology-type mapping table or return defaults."""
    pandas = require_pandas()
    mapping = DEFAULT_HOMOLOGY_TYPE_MAP.copy()
    if path is None:
        return mapping

    if not path.exists():
        raise FileNotFoundError(f"Homology type map does not exist: {path}")

    map_sep = normalize_separator(sep, path)
    df_map = pandas.read_csv(path, sep=map_sep)
    if df_map.shape[1] < 2:
        raise ValueError(
            f"--homology-type-map must contain at least two columns: {path}"
        )

    source_col, target_col = df_map.columns[:2]
    for _, row in df_map[[source_col, target_col]].dropna().iterrows():
        source = str(row[source_col]).strip().lower()
        target = str(row[target_col]).strip()
        if source and target:
            mapping[source] = target
    return mapping


def map_homology_types(
    values,
    mapping: Dict[str, str],
    strict: bool,
) -> Tuple[object, List[str]]:
    """Map database homology-type labels to TACMAN-style labels."""
    pandas = require_pandas()
    mapped_values: List[str] = []
    unknown: List[str] = []

    for value in values:
        original = str(value).strip()
        if original == "":
            mapped_values.append(original)
            continue
        key = original.lower()
        if key in mapping:
            mapped_values.append(mapping[key])
        else:
            if original not in unknown:
                unknown.append(original)
            mapped_values.append(original)

    if unknown and strict:
        raise ValueError(
            "Unrecognized homology type(s): {}. Provide --homology-type-map or "
            "rerun without --strict-homology-type to keep original values.".format(
                ", ".join(unknown)
            )
        )
    return pandas.Series(mapped_values, index=values.index), unknown


def relative_path_for_info(output_path: Path, info_path: Path) -> str:
    """Return the path string that should be written to TACMAN homo/info.csv."""
    output_resolved = output_path.resolve()
    info_parent_resolved = info_path.parent.resolve()
    try:
        return str(output_resolved.relative_to(info_parent_resolved))
    except ValueError:
        return str(output_resolved)


def read_info_table(info_path: Path) -> pd.DataFrame:
    """Read an existing TACMAN info.csv or create an empty table shape."""
    pandas = require_pandas()
    columns = ["path", "name", "sp_ref", "sp_que"]
    if not info_path.exists():
        return pandas.DataFrame(columns=columns)

    df_info = pandas.read_csv(info_path)
    missing = [column for column in columns if column not in df_info.columns]
    if missing:
        raise ValueError(
            f"info.csv is missing required column(s): {', '.join(missing)}"
        )
    return df_info[columns].copy()


def register_info(
    info_path: Path,
    output_path: Path,
    sp_ref: str,
    sp_que: str,
    overwrite: bool,
    skip_if_exists: bool,
    backup: bool,
) -> Tuple[bool, str]:
    """Add or update one species-pair record in TACMAN homo/info.csv."""
    if overwrite and skip_if_exists:
        raise ValueError(
            "Use only one of --overwrite-info or --skip-register-if-exists."
        )

    info_path.parent.mkdir(parents=True, exist_ok=True)
    df_info = read_info_table(info_path)
    exists = (df_info["sp_ref"] == sp_ref) & (df_info["sp_que"] == sp_que)
    n_existing = int(exists.sum())

    if n_existing > 0 and skip_if_exists:
        return False, (
            f"Registration skipped because {sp_ref}->{sp_que} already exists "
            f"in {info_path}."
        )
    if n_existing > 0 and not overwrite:
        raise ValueError(
            f"info.csv already contains {n_existing} record(s) for "
            f"sp_ref={sp_ref}, sp_que={sp_que}. Use --overwrite-info to replace "
            "them or --skip-register-if-exists to leave info.csv unchanged."
        )

    if backup and info_path.exists():
        shutil.copy2(info_path, info_path.with_suffix(info_path.suffix + ".bak"))

    if n_existing > 0 and overwrite:
        df_info = df_info.loc[~exists].copy()

    record = {
        "path": relative_path_for_info(output_path, info_path),
        "name": output_path.stem,
        "sp_ref": sp_ref,
        "sp_que": sp_que,
    }
    pandas = require_pandas()
    df_info = pandas.concat([df_info, pandas.DataFrame([record])], ignore_index=True)
    df_info.to_csv(info_path, index=False)
    return True, f"Registered {sp_ref}->{sp_que} in {info_path}."


def validate_registration_request(
    info_path: Path,
    sp_ref: str,
    sp_que: str,
    overwrite: bool,
    skip_if_exists: bool,
) -> None:
    """Fail early if info.csv registration would be ambiguous or disallowed."""
    if overwrite and skip_if_exists:
        raise ValueError(
            "Use only one of --overwrite-info or --skip-register-if-exists."
        )

    df_info = read_info_table(info_path)
    exists = (df_info["sp_ref"] == sp_ref) & (df_info["sp_que"] == sp_que)
    n_existing = int(exists.sum())
    if n_existing > 0 and not overwrite and not skip_if_exists:
        raise ValueError(
            f"info.csv already contains {n_existing} record(s) for "
            f"sp_ref={sp_ref}, sp_que={sp_que}. Use --overwrite-info to replace "
            "them or --skip-register-if-exists to leave info.csv unchanged."
        )


def write_qc_report(
    qc_path: Path,
    lines: List[Tuple[str, object]],
    homology_type_counts: pd.Series,
    warnings: List[str],
) -> None:
    """Write a plain-text QC report for STAR Protocols users."""
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with qc_path.open("w", encoding="utf-8") as handle:
        for key, value in lines:
            handle.write(f"{key}: {value}\n")
        handle.write("\nhomology type counts:\n")
        if homology_type_counts.empty:
            handle.write("  NA\n")
        else:
            for homology_type, count in homology_type_counts.items():
                handle.write(f"  {homology_type}: {count}\n")
        if warnings:
            handle.write("\nwarnings:\n")
            for warning in warnings:
                handle.write(f"  - {warning}\n")


def clean_tacman_output_frame(df, ref_col: str, que_col: str, type_col: str) -> Tuple[object, Dict[str, int]]:
    """Remove invalid rows before writing TACMAN-facing homology files."""
    cleaned = df[[ref_col, que_col, type_col]].copy()
    for column in [ref_col, que_col, type_col]:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    missing_ref = cleaned[ref_col] == ""
    rows_missing_ref = int(missing_ref.sum())
    cleaned = cleaned.loc[~missing_ref].copy()

    missing_que = cleaned[que_col] == ""
    rows_missing_que = int(missing_que.sum())
    cleaned = cleaned.loc[~missing_que].copy()

    missing_type = cleaned[type_col] == ""
    rows_missing_type = int(missing_type.sum())
    cleaned = cleaned.loc[~missing_type].copy()

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=[ref_col, que_col, type_col]).copy()
    duplicate_removed = before_dedup - len(cleaned)

    stats = {
        "rows removed for missing reference gene": rows_missing_ref,
        "rows removed for missing query gene": rows_missing_que,
        "rows removed for missing homology type": rows_missing_type,
        "duplicate rows removed": duplicate_removed,
    }
    return cleaned, stats


def tacman_output_columns(sp_que: str, query_display_name: Optional[str]) -> List[str]:
    """Return TACMAN's three output columns for a query species."""
    query_display = species_display_name(sp_que, query_display_name)
    return [
        "Gene name",
        f"{query_display} gene name",
        f"{query_display} homology type",
    ]


def write_tacman_mapping(
    df,
    output_path: Path,
    sp_que: str,
    query_display_name: Optional[str],
    output_sep: str,
    ref_col: str = "ref_gene",
    que_col: str = "que_gene",
    type_col: str = "homology_type",
) -> Tuple[List[str], Dict[str, int], object]:
    """Write a cleaned TACMAN-compatible three-column homology mapping file."""
    sep = normalize_output_separator(output_sep)
    columns = tacman_output_columns(sp_que, query_display_name)
    cleaned, stats = clean_tacman_output_frame(df, ref_col, que_col, type_col)
    df_output = cleaned.rename(
        columns={
            ref_col: columns[0],
            que_col: columns[1],
            type_col: columns[2],
        }
    )[columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, sep=sep, index=False)
    return columns, stats, df_output


def write_support_crosstab(handle, df) -> None:
    """Write support-by-homology-type crosstab to a QC report."""
    if df.empty or "support" not in df.columns:
        handle.write("\nsupport x homology_type:\n  NA\n")
        return
    pandas = require_pandas()
    table = pandas.crosstab(df["support"], df["homology_type"])
    handle.write("\nsupport x homology_type:\n")
    for support, row in table.iterrows():
        for homology_type, count in row.items():
            if int(count) > 0:
                handle.write(f"  {support} + {homology_type}: {int(count)}\n")


def append_blast_qc_tables(qc_path: Path, final_pairs) -> None:
    """Append BLAST support cross-tabulation to an existing QC report."""
    with qc_path.open("a", encoding="utf-8") as handle:
        write_support_crosstab(handle, final_pairs)
        handle.write(
            "\nnotes:\n"
            "  - Protein-level RBH count may exceed unique gene-level RBH count "
            "because multiple protein isoforms can map to the same gene pair.\n"
            "  - Non-RBH one-to-one components are sequence-similarity candidates "
            "and are not RBH-derived orthologs.\n"
            "  - Identifiers in the homology file must match AnnData.var_names.\n"
        )


def output_prefix_from_out(output_path: Path) -> Path:
    """Use --out as a prefix source for fixed BLAST output names."""
    return output_path.with_suffix("") if output_path.suffix else output_path


def fixed_blast_output_paths(output_path: Path) -> Dict[str, Dict[str, Path]]:
    """Return fixed TACMAN/evidence output paths for BLAST-derived results."""
    prefix = output_prefix_from_out(output_path)
    return {
        "all_putative": {
            "txt": prefix.with_name(prefix.name + ".all_putative_homology.txt"),
            "evidence": prefix.with_name(prefix.name + ".all_putative_homology.evidence.tsv"),
        },
        "gene_rbh": {
            "txt": prefix.with_name(prefix.name + ".gene_RBH.txt"),
            "evidence": prefix.with_name(prefix.name + ".gene_RBH.evidence.tsv"),
        },
        "strict_one2one_rbh": {
            "txt": prefix.with_name(prefix.name + ".strict_one2one_RBH.txt"),
            "evidence": prefix.with_name(prefix.name + ".strict_one2one_RBH.evidence.tsv"),
        },
    }


def primary_output_key(value: str) -> str:
    """Normalize the primary output selector."""
    mapping = {
        "all_putative": "all_putative",
        "gene_rbh": "gene_rbh",
        "gene_RBH": "gene_rbh",
        "strict_one2one_rbh": "strict_one2one_rbh",
        "strict_one2one_RBH": "strict_one2one_rbh",
    }
    if value not in mapping:
        raise ValueError(
            "--primary-output must be all_putative, gene_rbh, or strict_one2one_rbh."
        )
    return mapping[value]


def write_blast_output_bundle(
    output_path: Path,
    all_pairs,
    gene_rbh,
    strict_rbh,
    sp_que: str,
    query_display_name: Optional[str],
    output_sep: str,
    primary_output: str,
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Dict[str, int]], Path]:
    """Write all fixed BLAST TACMAN/evidence outputs and a compatibility alias."""
    paths = fixed_blast_output_paths(output_path)
    datasets = {
        "all_putative": all_pairs,
        "gene_rbh": gene_rbh,
        "strict_one2one_rbh": strict_rbh,
    }
    clean_stats: Dict[str, Dict[str, int]] = {}
    for key, df in datasets.items():
        paths[key]["txt"].parent.mkdir(parents=True, exist_ok=True)
        paths[key]["evidence"].parent.mkdir(parents=True, exist_ok=True)
        _, stats, cleaned_tacman = write_tacman_mapping(
            df,
            output_path=paths[key]["txt"],
            sp_que=sp_que,
            query_display_name=query_display_name,
            output_sep=output_sep,
        )
        clean_stats[key] = stats
        evidence = df.copy()
        for column in ["ref_gene", "que_gene", "homology_type"]:
            evidence[column] = evidence[column].fillna("").astype(str).str.strip()
        evidence = evidence[
            (evidence["ref_gene"] != "")
            & (evidence["que_gene"] != "")
            & (evidence["homology_type"] != "")
        ].drop_duplicates(subset=["ref_gene", "que_gene", "homology_type", "support"])
        evidence.to_csv(paths[key]["evidence"], sep="\t", index=False)

        if key == primary_output_key(primary_output):
            # Compatibility alias for existing workflows that expect --out itself.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_tacman.to_csv(
                output_path,
                sep=normalize_output_separator(output_sep),
                index=False,
            )

    primary_path = paths[primary_output_key(primary_output)]["txt"]
    return paths, clean_stats, primary_path


BLAST_COLUMNS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "qlen",
    "slen",
    "qcovs",
    "evalue",
    "bitscore",
]

BLAST_OUTFMT = "6 " + " ".join(BLAST_COLUMNS)

PUTATIVE_BLAST_WARNING = (
    "BLAST-based relationships are putative homologous relationships inferred "
    "from sequence similarity and are not equivalent to curated orthology "
    "annotations."
)


def count_fasta_records(fasta_path: Path) -> int:
    """Count FASTA records using the first token after '>' as the sequence ID."""
    n_records = 0
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                n_records += 1
    return n_records


def require_executable(executable: str, option_name: str) -> str:
    """Return an executable path or raise a clear BLAST+ installation error."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"{Path(executable).name} not found. Install NCBI BLAST+ or provide "
            f"{option_name}."
        )
    return resolved


def command_to_string(command: Sequence[object]) -> str:
    """Format a subprocess command for logs and error messages."""
    return " ".join(str(part) for part in command)


def run_command(command: Sequence[object], log_path: Path) -> subprocess.CompletedProcess:
    """Run a subprocess command and append stdout/stderr to a command log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {command_to_string(command)}\n")
        try:
            result = subprocess.run(
                [str(part) for part in command],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            log_handle.write(exc.stdout or "")
            log_handle.write(exc.stderr or "")
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or "no stdout/stderr captured"
            hint = ""
            if "makeblastdb" in Path(str(command[0])).name:
                hint = (
                    "\nIf makeblastdb failed because of memory or LMDB virtual "
                    "memory limits, try a higher-memory node or rerun with "
                    "--blastdb-version 4 when supported by your BLAST+ version."
                )
            raise RuntimeError(
                "Command failed:\n{}\n\nBLAST+ output:\n{}{}".format(
                    command_to_string(command), details, hint
                )
            ) from exc

        log_handle.write(result.stdout or "")
        log_handle.write(result.stderr or "")
        log_handle.write("\n")
        return result


def load_id_map(
    path: Path,
    protein_id_column: str,
    gene_column: str,
    label: str,
    allow_non_symbol: bool,
) -> Tuple[Dict[str, str], List[str], float]:
    """Read a protein-to-gene-symbol mapping table."""
    pandas = require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"{label} id map does not exist: {path}")

    sep = normalize_separator(None, path)
    df_map = pandas.read_csv(path, sep=sep)
    required_columns_exist(df_map, [protein_id_column, gene_column])
    df_map = df_map[[protein_id_column, gene_column]].dropna().copy()
    df_map[protein_id_column] = df_map[protein_id_column].astype(str).str.strip()
    df_map[gene_column] = df_map[gene_column].astype(str).str.strip()
    df_map = df_map[(df_map[[protein_id_column, gene_column]] != "").all(axis=1)]

    warnings, identifier_fraction = summarize_identifier_like_gene_values(
        df_map[gene_column],
        label=label,
        allow_non_symbol=allow_non_symbol,
    )

    duplicated = df_map[df_map.duplicated(subset=[protein_id_column], keep=False)]
    if not duplicated.empty:
        conflicting = (
            duplicated.groupby(protein_id_column)[gene_column].nunique() > 1
        ).sum()
        warnings.append(
            f"{label} id map contains duplicated protein IDs; keeping the first "
            f"mapping for each protein. Conflicting duplicated IDs: {int(conflicting)}."
        )

    df_map = df_map.drop_duplicates(subset=[protein_id_column], keep="first")
    return dict(zip(df_map[protein_id_column], df_map[gene_column])), warnings, identifier_fraction


def read_blast_table(path: Path) -> object:
    """Read a BLAST outfmt 6 table with TACMAN's expected columns."""
    pandas = require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"BLAST output file does not exist: {path}")
    if path.stat().st_size == 0:
        return pandas.DataFrame(columns=BLAST_COLUMNS)
    df = pandas.read_csv(path, sep="\t", names=BLAST_COLUMNS)
    numeric_columns = ["pident", "length", "qlen", "slen", "qcovs", "evalue", "bitscore"]
    for column in numeric_columns:
        df[column] = pandas.to_numeric(df[column], errors="coerce")
    return add_subject_coverage(df.dropna(subset=numeric_columns).copy())


def read_filtered_blast_table(path: Path) -> object:
    """Read a filtered BLAST table written by this script with a header."""
    pandas = require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"Filtered BLAST table does not exist: {path}")
    if path.stat().st_size == 0:
        return pandas.DataFrame(columns=BLAST_COLUMNS)
    df = pandas.read_csv(path, sep="\t")
    required_columns_exist(df, BLAST_COLUMNS)
    return add_subject_coverage(df.copy())


def add_subject_coverage(df) -> object:
    """Add subject coverage percentage from alignment length and subject length."""
    result = df.copy()
    if "scov" not in result.columns:
        result["scov"] = (result["length"] / result["slen"]) * 100.0
    return result


def filter_and_rank_blast_hits(
    df,
    evalue: float,
    min_pident: float,
    min_qcov: float,
    top_n: int,
    min_scov: Optional[float] = None,
) -> object:
    """Filter BLAST hits and retain top-N hits per query protein."""
    df = add_subject_coverage(df)
    filtered = df[
        (df["pident"] >= min_pident)
        & (df["qcovs"] >= min_qcov)
        & (df["evalue"] <= evalue)
    ].copy()
    if min_scov is not None:
        filtered = filtered[filtered["scov"] >= min_scov].copy()
    if filtered.empty:
        return filtered
    filtered = filtered.sort_values(
        ["qseqid", "bitscore", "evalue", "pident", "qcovs"],
        ascending=[True, False, True, False, False],
    )
    return filtered.groupby("qseqid", group_keys=False).head(top_n).reset_index(drop=True)


def select_best_hits(df, tie_policy: str) -> Tuple[Dict[str, str], Dict[str, object], Set[str]]:
    """Select one best subject per query under an explicit tie policy."""
    if df.empty:
        return {}, {}, set()
    if tie_policy not in {"first", "discard", "all"}:
        raise ValueError("--best-hit-tie-policy must be first, discard, or all.")

    selected: Dict[str, str] = {}
    selected_rows: Dict[str, object] = {}
    tied_queries: Set[str] = set()
    for qseqid, group in df.groupby("qseqid"):
        best_bitscore = group["bitscore"].max()
        best = group[group["bitscore"] == best_bitscore]
        best_evalue = best["evalue"].min()
        best = best[best["evalue"] == best_evalue]
        best_pident = best["pident"].max()
        best = best[best["pident"] == best_pident]
        best_qcovs = best["qcovs"].max()
        best = best[best["qcovs"] == best_qcovs]
        subjects = sorted(set(best["sseqid"].astype(str)))
        is_tied = len(subjects) > 1
        if is_tied:
            tied_queries.add(str(qseqid))
            if tie_policy in {"discard", "all"}:
                continue
        chosen = best.sort_values("sseqid").iloc[0]
        selected[str(qseqid)] = str(chosen["sseqid"])
        selected_rows[str(qseqid)] = chosen
    return selected, selected_rows, tied_queries


def top1_hits(df) -> Dict[str, str]:
    """Return deterministic first-policy top subject per query for compatibility."""
    selected, _, _ = select_best_hits(df, "first")
    return selected


def reciprocal_best_hit_pairs(ref_to_que, que_to_ref, tie_policy: str = "discard") -> Set[Tuple[str, str]]:
    """Identify reciprocal best-hit protein pairs from filtered BLAST hits."""
    forward_top, _, _ = select_best_hits(ref_to_que, tie_policy)
    reverse_top, _, _ = select_best_hits(que_to_ref, tie_policy)
    pairs: Set[Tuple[str, str]] = set()
    for ref_protein, que_protein in forward_top.items():
        if reverse_top.get(que_protein) == ref_protein:
            pairs.add((ref_protein, que_protein))
    return pairs


def reciprocal_best_hit_evidence(
    ref_to_que,
    que_to_ref,
    ref_map: Dict[str, str],
    que_map: Dict[str, str],
    tie_policy: str,
) -> Tuple[object, Dict[str, int]]:
    """Return protein-level RBH evidence mapped to reference/query genes."""
    pandas = require_pandas()
    if ref_to_que.empty or que_to_ref.empty:
        return pandas.DataFrame(
            columns=[
                "ref_gene",
                "que_gene",
                "homology_type",
                "support",
                "relationship_cardinality",
                "classification_method",
                "best_ref_protein",
                "best_que_protein",
                "best_bitscore",
                "best_evalue",
                "best_pident",
                "best_qcovs",
                "best_scov",
                "rbh_level",
                "gene_pair_best_ref_protein",
                "gene_pair_best_que_protein",
                "gene_pair_isoform_hit_count",
                "gene_level_tie_status",
            ]
        ), {
            "number of queries with tied best hits in forward search": 0,
            "number of queries with tied best hits in reverse search": 0,
            "number of candidate RBHs discarded because of ties": 0,
        }

    forward_top, forward_rows, forward_ties = select_best_hits(ref_to_que, tie_policy)
    reverse_top, _, reverse_ties = select_best_hits(que_to_ref, tie_policy)
    legacy_pairs = reciprocal_best_hit_pairs(ref_to_que, que_to_ref, tie_policy="first")
    selected_pairs: Set[Tuple[str, str]] = set()
    records = []
    for ref_protein, que_protein in forward_top.items():
        if reverse_top.get(que_protein) != ref_protein:
            continue
        selected_pairs.add((ref_protein, que_protein))
        if ref_protein not in ref_map or que_protein not in que_map:
            continue
        row = forward_rows[ref_protein]
        records.append(
            {
                "ref_gene": ref_map[ref_protein],
                "que_gene": que_map[que_protein],
                "support": "RBH",
                "best_ref_protein": ref_protein,
                "best_que_protein": que_protein,
                "best_bitscore": row["bitscore"],
                "best_evalue": row["evalue"],
                "best_pident": row["pident"],
                "best_qcovs": row["qcovs"],
                "best_scov": row.get("scov", ""),
                "rbh_level": "protein",
                "gene_pair_best_ref_protein": ref_protein,
                "gene_pair_best_que_protein": que_protein,
                "gene_pair_isoform_hit_count": 1,
                "gene_level_tie_status": "not_evaluated",
            }
        )
    stats = {
        "number of queries with tied best hits in forward search": len(forward_ties),
        "number of queries with tied best hits in reverse search": len(reverse_ties),
        "number of candidate RBHs discarded because of ties": len(legacy_pairs - selected_pairs),
    }
    return pandas.DataFrame(records), stats


def score_columns_with_scov(df) -> List[str]:
    """Return ranking columns, including scov when available."""
    columns = ["bitscore", "evalue", "pident", "qcovs"]
    if "scov" in df.columns:
        columns.append("scov")
    return columns


def best_gene_pair_rows(gene_hits) -> object:
    """Collapse isoform-level hits to one best representative per gene pair."""
    pandas = require_pandas()
    if gene_hits.empty:
        return pandas.DataFrame()
    sort_columns = [
        "ref_gene",
        "que_gene",
        "bitscore",
        "evalue",
        "pident",
        "qcovs",
        "scov",
        "ref_protein",
        "que_protein",
    ]
    ascending = [True, True, False, True, False, False, False, True, True]
    ranked = gene_hits.sort_values(sort_columns, ascending=ascending)
    isoform_counts = (
        gene_hits.groupby(["ref_gene", "que_gene"], as_index=False)
        .size()
        .rename(columns={"size": "gene_pair_isoform_hit_count"})
    )
    best = ranked.groupby(["ref_gene", "que_gene"], as_index=False).first().copy()
    return best.merge(isoform_counts, on=["ref_gene", "que_gene"], how="left")


def gene_pair_candidates_by_direction(gene_hits, direction: str) -> object:
    """Collapse directional protein hits into directional gene-pair candidates."""
    pandas = require_pandas()
    subset = gene_hits[gene_hits["direction"] == direction].copy()
    if subset.empty:
        return pandas.DataFrame()
    sort_columns = [
        "ref_gene",
        "que_gene",
        "bitscore",
        "evalue",
        "pident",
        "qcovs",
        "scov",
        "ref_protein",
        "que_protein",
    ]
    ascending = [True, True, False, True, False, False, False, True, True]
    ranked = subset.sort_values(sort_columns, ascending=ascending)
    counts = (
        subset.groupby(["ref_gene", "que_gene"], as_index=False)
        .size()
        .rename(columns={"size": "gene_pair_isoform_hit_count"})
    )
    best = ranked.groupby(["ref_gene", "que_gene"], as_index=False).first().copy()
    best = best.merge(counts, on=["ref_gene", "que_gene"], how="left")
    best["query_gene"] = best["ref_gene"] if direction == "forward" else best["que_gene"]
    best["partner_gene"] = best["que_gene"] if direction == "forward" else best["ref_gene"]
    return best


def select_best_gene_hits(candidates, tie_policy: str) -> Tuple[Dict[str, str], Dict[str, object], Set[str]]:
    """Select best partner gene per query gene after isoform collapsing."""
    if candidates.empty:
        return {}, {}, set()
    if tie_policy not in {"first", "discard", "all"}:
        raise ValueError("--best-hit-tie-policy must be first, discard, or all.")
    selected: Dict[str, str] = {}
    selected_rows: Dict[str, object] = {}
    tied_queries: Set[str] = set()

    for query_gene, group in candidates.groupby("query_gene"):
        best = group[group["bitscore"] == group["bitscore"].max()]
        best = best[best["evalue"] == best["evalue"].min()]
        best = best[best["pident"] == best["pident"].max()]
        best = best[best["qcovs"] == best["qcovs"].max()]
        if "scov" in best.columns:
            best = best[best["scov"] == best["scov"].max()]
        partner_genes = sorted(set(best["partner_gene"].astype(str)))
        if len(partner_genes) > 1:
            tied_queries.add(str(query_gene))
            if tie_policy in {"discard", "all"}:
                continue
        chosen = best.sort_values(["partner_gene", "ref_protein", "que_protein"]).iloc[0]
        selected[str(query_gene)] = str(chosen["partner_gene"])
        selected_rows[str(query_gene)] = chosen
    return selected, selected_rows, tied_queries


def count_isoform_collapsed_ties(df, query_col: str, partner_gene_col: str) -> int:
    """Count protein-level exact ties whose tied subjects collapse to one gene."""
    count = 0
    if df.empty:
        return count
    for _, group in df.groupby(query_col):
        best = group[group["bitscore"] == group["bitscore"].max()]
        best = best[best["evalue"] == best["evalue"].min()]
        best = best[best["pident"] == best["pident"].max()]
        best = best[best["qcovs"] == best["qcovs"].max()]
        if "scov" in best.columns:
            best = best[best["scov"] == best["scov"].max()]
        if len(set(best[partner_gene_col].astype(str))) == 1 and len(best) > 1:
            count += 1
    return count


def reciprocal_best_gene_evidence(
    gene_hits,
    tie_policy: str,
) -> Tuple[object, Dict[str, int]]:
    """Return gene-level RBH evidence after collapsing isoform protein hits."""
    pandas = require_pandas()
    columns = [
        "ref_gene",
        "que_gene",
        "support",
        "best_ref_protein",
        "best_que_protein",
        "best_bitscore",
        "best_evalue",
        "best_pident",
        "best_qcovs",
        "best_scov",
        "rbh_level",
        "gene_pair_best_ref_protein",
        "gene_pair_best_que_protein",
        "gene_pair_isoform_hit_count",
        "gene_level_tie_status",
    ]
    if gene_hits.empty:
        return pandas.DataFrame(columns=columns), {
            "forward gene queries with tied best gene hits": 0,
            "reverse gene queries with tied best gene hits": 0,
            "gene-level RBH pairs": 0,
            "number of true cross-gene ties discarded": 0,
            "number of protein-level ties resolved by collapsing isoforms to the same gene": 0,
        }

    forward_raw = gene_hits[gene_hits["direction"] == "forward"].copy()
    reverse_raw = gene_hits[gene_hits["direction"] == "reverse"].copy()
    forward = gene_pair_candidates_by_direction(gene_hits, "forward")
    reverse = gene_pair_candidates_by_direction(gene_hits, "reverse")
    forward_top, forward_rows, forward_ties = select_best_gene_hits(forward, tie_policy)
    reverse_top, _, reverse_ties = select_best_gene_hits(reverse, tie_policy)
    forward_first, _, _ = select_best_gene_hits(forward, "first")
    reverse_first, _, _ = select_best_gene_hits(reverse, "first")

    legacy_pairs = {
        (ref_gene, que_gene)
        for ref_gene, que_gene in forward_first.items()
        if reverse_first.get(que_gene) == ref_gene
    }
    selected_pairs: Set[Tuple[str, str]] = set()
    records = []
    for ref_gene, que_gene in forward_top.items():
        if reverse_top.get(que_gene) != ref_gene:
            continue
        selected_pairs.add((ref_gene, que_gene))
        row = forward_rows[ref_gene]
        records.append(
            {
                "ref_gene": ref_gene,
                "que_gene": que_gene,
                "support": "RBH",
                "best_ref_protein": row["ref_protein"],
                "best_que_protein": row["que_protein"],
                "best_bitscore": row["bitscore"],
                "best_evalue": row["evalue"],
                "best_pident": row["pident"],
                "best_qcovs": row["qcovs"],
                "best_scov": row.get("scov", ""),
                "rbh_level": "gene",
                "gene_pair_best_ref_protein": row["ref_protein"],
                "gene_pair_best_que_protein": row["que_protein"],
                "gene_pair_isoform_hit_count": int(row["gene_pair_isoform_hit_count"]),
                "gene_level_tie_status": "unique_best",
            }
        )

    resolved_ties = count_isoform_collapsed_ties(forward_raw, "ref_gene", "que_gene")
    resolved_ties += count_isoform_collapsed_ties(reverse_raw, "que_gene", "ref_gene")
    true_ties_discarded = len(legacy_pairs - selected_pairs) if tie_policy in {"discard", "all"} else 0
    stats = {
        "forward gene queries with tied best gene hits": len(forward_ties),
        "reverse gene queries with tied best gene hits": len(reverse_ties),
        "gene-level RBH pairs": len(records),
        "number of true cross-gene ties discarded": true_ties_discarded,
        "number of protein-level ties resolved by collapsing isoforms to the same gene": resolved_ties,
    }
    return pandas.DataFrame(records, columns=columns), stats


def blast_hits_to_gene_evidence(
    df,
    direction: str,
    ref_map: Dict[str, str],
    que_map: Dict[str, str],
) -> Tuple[object, Set[str], Set[str]]:
    """Map protein-level BLAST hits to oriented reference/query gene evidence."""
    pandas = require_pandas()
    records = []
    missing_ref: Set[str] = set()
    missing_que: Set[str] = set()

    for _, row in df.iterrows():
        if direction == "forward":
            ref_protein = str(row["qseqid"]).strip()
            que_protein = str(row["sseqid"]).strip()
        else:
            ref_protein = str(row["sseqid"]).strip()
            que_protein = str(row["qseqid"]).strip()

        ref_gene = ref_map.get(ref_protein)
        que_gene = que_map.get(que_protein)
        if ref_gene is None:
            missing_ref.add(ref_protein)
        if que_gene is None:
            missing_que.add(que_protein)
        if ref_gene is None or que_gene is None:
            continue

        records.append(
            {
                "ref_gene": ref_gene,
                "que_gene": que_gene,
                "ref_protein": ref_protein,
                "que_protein": que_protein,
                "bitscore": row["bitscore"],
                "evalue": row["evalue"],
                "pident": row["pident"],
                "qcovs": row["qcovs"],
                "scov": row.get("scov", ""),
                "direction": direction,
            }
        )

    return pandas.DataFrame(records), missing_ref, missing_que


def connected_components(edges: Iterable[Tuple[str, str]]) -> List[Tuple[Set[str], Set[str]]]:
    """Return connected components for a bipartite ref/query gene graph."""
    adjacency: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    for ref_gene, que_gene in edges:
        ref_node = ("ref", ref_gene)
        que_node = ("que", que_gene)
        adjacency.setdefault(ref_node, set()).add(que_node)
        adjacency.setdefault(que_node, set()).add(ref_node)

    components: List[Tuple[Set[str], Set[str]]] = []
    seen: Set[Tuple[str, str]] = set()
    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        ref_genes: Set[str] = set()
        que_genes: Set[str] = set()
        while stack:
            current = stack.pop()
            node_type, gene = current
            if node_type == "ref":
                ref_genes.add(gene)
            else:
                que_genes.add(gene)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append((ref_genes, que_genes))
    return components


def component_homology_type(n_ref: int, n_que: int) -> str:
    """Assign a TACMAN/Ensembl-style type from gene component dimensions."""
    if n_ref == 1 and n_que == 1:
        return "ortholog_one2one"
    if n_ref == 1 or n_que == 1:
        return "ortholog_one2many"
    return "ortholog_many2many"


def component_relationship_cardinality(n_ref: int, n_que: int) -> str:
    """Return relationship cardinality from component dimensions."""
    if n_ref == 1 and n_que == 1:
        return "1:1"
    if n_ref == 1 and n_que > 1:
        return "1:n"
    if n_ref > 1 and n_que == 1:
        return "n:1"
    return "n:m"


def classify_pairs_by_components(df, force_one2one: bool = False) -> object:
    """Assign homology_type values from bipartite gene graph cardinality."""
    pandas = require_pandas()
    if df.empty:
        result = df.copy()
        result["homology_type"] = []
        result["relationship_cardinality"] = []
        result["classification_method"] = []
        return result
    component_type_by_pair: Dict[Tuple[str, str], str] = {}
    component_cardinality_by_pair: Dict[Tuple[str, str], str] = {}
    for ref_genes, que_genes in connected_components(zip(df["ref_gene"], df["que_gene"])):
        homology_type = component_homology_type(len(ref_genes), len(que_genes))
        relationship_cardinality = component_relationship_cardinality(len(ref_genes), len(que_genes))
        for ref_gene in ref_genes:
            for que_gene in que_genes:
                component_type_by_pair[(ref_gene, que_gene)] = homology_type
                component_cardinality_by_pair[(ref_gene, que_gene)] = relationship_cardinality
    result = df.copy()
    result["homology_type"] = [
        "ortholog_one2one" if force_one2one else component_type_by_pair[(row["ref_gene"], row["que_gene"])]
        for _, row in result.iterrows()
    ]
    result["relationship_cardinality"] = [
        "1:1" if force_one2one else component_cardinality_by_pair[(row["ref_gene"], row["que_gene"])]
        for _, row in result.iterrows()
    ]
    result["classification_method"] = "gene_component"
    return result


def deduplicate_gene_pair_evidence(df) -> object:
    """Keep the strongest evidence per reference/query gene pair."""
    if df.empty:
        return df.copy()
    ranked = df.sort_values(
        [
            "ref_gene",
            "que_gene",
            "best_bitscore",
            "best_evalue",
            "best_pident",
            "best_qcovs",
            "best_scov",
        ],
        ascending=[True, True, False, True, False, False, False],
    )
    return ranked.groupby(["ref_gene", "que_gene"], as_index=False).first().copy()


def ensure_blast_evidence_columns(df, rbh_level: str, support_default: Optional[str] = None) -> object:
    """Ensure BLAST evidence tables contain the expanded evidence columns."""
    result = df.copy()
    defaults = {
        "support": support_default,
        "best_scov": "",
        "rbh_level": rbh_level,
        "gene_pair_best_ref_protein": result["best_ref_protein"] if "best_ref_protein" in result.columns else "",
        "gene_pair_best_que_protein": result["best_que_protein"] if "best_que_protein" in result.columns else "",
        "gene_pair_isoform_hit_count": 1,
        "gene_level_tie_status": "not_evaluated" if rbh_level == "protein" else "unique_best",
        "relationship_cardinality": "",
        "classification_method": "",
        "homology_type": "",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result


def build_rbh_outputs(rbh_protein_evidence) -> Tuple[object, object, Dict[str, int]]:
    """Create unique gene-level RBH and strict one-to-one RBH outputs."""
    pandas = require_pandas()
    if rbh_protein_evidence.empty:
        empty = pandas.DataFrame(columns=[
            "ref_gene",
            "que_gene",
            "homology_type",
            "support",
            "relationship_cardinality",
            "classification_method",
            "best_ref_protein",
            "best_que_protein",
            "best_bitscore",
            "best_evalue",
            "best_pident",
            "best_qcovs",
            "best_scov",
            "rbh_level",
            "gene_pair_best_ref_protein",
            "gene_pair_best_que_protein",
            "gene_pair_isoform_hit_count",
            "gene_level_tie_status",
        ])
        return empty.copy(), empty.copy(), {
            "unique gene-level RBH pairs": 0,
            "protein RBH pairs collapsed during gene-level deduplication": 0,
            "strict one-to-one gene-level RBH pairs": 0,
        }

    rbh_level = "gene"
    if not rbh_protein_evidence.empty and "rbh_level" in rbh_protein_evidence.columns:
        rbh_level = str(rbh_protein_evidence["rbh_level"].iloc[0])
    gene_rbh = ensure_blast_evidence_columns(rbh_protein_evidence, rbh_level=rbh_level, support_default="RBH")
    gene_rbh = deduplicate_gene_pair_evidence(gene_rbh)
    gene_rbh = classify_pairs_by_components(gene_rbh)
    gene_rbh["support"] = "RBH"
    gene_rbh = ensure_blast_evidence_columns(gene_rbh, rbh_level=rbh_level, support_default="RBH")
    gene_rbh = gene_rbh[
        [
            "ref_gene",
            "que_gene",
            "homology_type",
            "support",
            "relationship_cardinality",
            "classification_method",
            "best_ref_protein",
            "best_que_protein",
            "best_bitscore",
            "best_evalue",
            "best_pident",
            "best_qcovs",
            "best_scov",
            "rbh_level",
            "gene_pair_best_ref_protein",
            "gene_pair_best_que_protein",
            "gene_pair_isoform_hit_count",
            "gene_level_tie_status",
        ]
    ].sort_values(["ref_gene", "que_gene"]).reset_index(drop=True)

    ref_degree = gene_rbh.groupby("ref_gene")["que_gene"].nunique()
    que_degree = gene_rbh.groupby("que_gene")["ref_gene"].nunique()
    strict_mask = [
        ref_degree[row["ref_gene"]] == 1 and que_degree[row["que_gene"]] == 1
        for _, row in gene_rbh.iterrows()
    ]
    strict = gene_rbh.loc[strict_mask].copy()
    strict["homology_type"] = "ortholog_one2one"
    strict["relationship_cardinality"] = "1:1"
    strict["classification_method"] = "gene_component"
    strict = strict.reset_index(drop=True)

    stats = {
        "unique gene-level RBH pairs": len(gene_rbh),
        "protein RBH pairs collapsed during gene-level deduplication": len(rbh_protein_evidence) - len(gene_rbh),
        "strict one-to-one gene-level RBH pairs": len(strict),
    }
    return gene_rbh, strict, stats


def infer_gene_pairs_from_blast(
    ref_to_que_filtered,
    que_to_ref_filtered,
    ref_map: Dict[str, str],
    que_map: Dict[str, str],
    strict_id_map: bool,
    best_hit_tie_policy: str = "discard",
    rbh_level: str = "gene",
) -> Tuple[object, object, object, object, object, object, Dict[str, int]]:
    """Infer final gene-level putative homologs from filtered reciprocal BLASTP."""
    pandas = require_pandas()
    if rbh_level not in {"protein", "gene"}:
        raise ValueError("--rbh-level must be protein or gene.")
    rbh_protein_pairs = reciprocal_best_hit_pairs(
        ref_to_que_filtered, que_to_ref_filtered, tie_policy=best_hit_tie_policy
    )
    rbh_protein_evidence, protein_tie_stats = reciprocal_best_hit_evidence(
        ref_to_que_filtered,
        que_to_ref_filtered,
        ref_map,
        que_map,
        best_hit_tie_policy,
    )

    forward_gene_hits, missing_ref_fwd, missing_que_fwd = blast_hits_to_gene_evidence(
        ref_to_que_filtered, "forward", ref_map, que_map
    )
    reverse_gene_hits, missing_ref_rev, missing_que_rev = blast_hits_to_gene_evidence(
        que_to_ref_filtered, "reverse", ref_map, que_map
    )
    missing_ref = missing_ref_fwd | missing_ref_rev
    missing_que = missing_que_fwd | missing_que_rev
    if strict_id_map and (missing_ref or missing_que):
        raise ValueError(
            "Some BLAST protein IDs could not be mapped to gene symbols "
            f"({len(missing_ref)} reference IDs, {len(missing_que)} query IDs). "
            "Fix the id-map files or rerun without --strict-id-map."
        )

    gene_hits = pandas.concat([forward_gene_hits, reverse_gene_hits], ignore_index=True)
    if gene_hits.empty:
        raise ValueError(
            "Final gene-level homology pairs are zero after protein-to-gene "
            "mapping. Check that FASTA record IDs match the protein_id columns "
            "in the id-map files."
        )

    gene_rbh_evidence, gene_tie_stats = reciprocal_best_gene_evidence(
        gene_hits,
        tie_policy=best_hit_tie_policy,
    )
    if rbh_level == "gene":
        active_rbh_evidence = gene_rbh_evidence
    else:
        active_rbh_evidence = ensure_blast_evidence_columns(
            rbh_protein_evidence,
            rbh_level="protein",
            support_default="RBH",
        )
    rbh_gene_pairs = {
        (row["ref_gene"], row["que_gene"])
        for _, row in active_rbh_evidence.iterrows()
    }

    best_evidence = best_gene_pair_rows(gene_hits)

    best_evidence = classify_pairs_by_components(best_evidence)
    support_values = []
    for _, row in best_evidence.iterrows():
        pair = (row["ref_gene"], row["que_gene"])
        if pair in rbh_gene_pairs:
            support_values.append("RBH")
        else:
            support_values.append("topN_nonreciprocal")

    final_pairs = best_evidence.rename(
        columns={
            "ref_protein": "best_ref_protein",
            "que_protein": "best_que_protein",
            "bitscore": "best_bitscore",
            "evalue": "best_evalue",
            "pident": "best_pident",
            "qcovs": "best_qcovs",
            "scov": "best_scov",
        }
    )
    final_pairs["support"] = support_values
    final_pairs["rbh_level"] = rbh_level
    final_pairs["gene_pair_best_ref_protein"] = final_pairs["best_ref_protein"]
    final_pairs["gene_pair_best_que_protein"] = final_pairs["best_que_protein"]
    final_pairs["gene_level_tie_status"] = "not_best_gene_hit"
    final_pairs.loc[final_pairs["support"] == "RBH", "gene_level_tie_status"] = (
        "unique_best" if rbh_level == "gene" else "not_evaluated"
    )
    final_pairs = ensure_blast_evidence_columns(final_pairs, rbh_level=rbh_level)
    final_pairs = final_pairs[
        [
            "ref_gene",
            "que_gene",
            "homology_type",
            "support",
            "relationship_cardinality",
            "classification_method",
            "rbh_level",
            "best_ref_protein",
            "best_que_protein",
            "best_bitscore",
            "best_evalue",
            "best_pident",
            "best_qcovs",
            "best_scov",
            "gene_pair_best_ref_protein",
            "gene_pair_best_que_protein",
            "gene_pair_isoform_hit_count",
            "gene_level_tie_status",
        ]
    ].sort_values(["ref_gene", "que_gene"]).reset_index(drop=True)

    active_unique_gene_rbh = int(active_rbh_evidence[["ref_gene", "que_gene"]].drop_duplicates().shape[0]) if not active_rbh_evidence.empty else 0
    protein_unique_gene_rbh = int(rbh_protein_evidence[["ref_gene", "que_gene"]].drop_duplicates().shape[0]) if not rbh_protein_evidence.empty else 0
    multi_isoform_gene_pairs = int((best_evidence["gene_pair_isoform_hit_count"] > 1).sum()) if "gene_pair_isoform_hit_count" in best_evidence.columns else 0
    stats = {
        "RBH level": rbh_level,
        "number of protein-level candidate hits": len(gene_hits),
        "number of unique gene-pair candidates": len(best_evidence),
        "number of gene pairs collapsed from multiple isoform hits": multi_isoform_gene_pairs,
        "number of reciprocal best hit protein pairs": len(rbh_protein_pairs),
        "number of unique gene-level RBH pairs": active_unique_gene_rbh,
        "protein-level RBH pairs collapsed to unique gene pairs": protein_unique_gene_rbh,
        "number of protein RBH pairs collapsed during gene-level deduplication": len(rbh_protein_evidence)
        - protein_unique_gene_rbh,
        "number of gene-level pairs before deduplication": len(gene_hits),
        "number of protein IDs missing in ref-id-map": len(missing_ref),
        "number of protein IDs missing in que-id-map": len(missing_que),
    }
    stats.update(protein_tie_stats)
    stats.update(gene_tie_stats)
    gene_rbh, strict_rbh, rbh_stats = build_rbh_outputs(active_rbh_evidence)
    stats.update({
        "number of strict one-to-one gene-level RBH pairs": rbh_stats["strict one-to-one gene-level RBH pairs"],
    })
    return final_pairs, gene_hits, active_rbh_evidence, gene_rbh, strict_rbh, stats


def infer_homology_from_blast_tables(
    ref_to_que_blast: Path,
    que_to_ref_blast: Path,
    ref_map: Dict[str, str],
    que_map: Dict[str, str],
    evalue: float,
    min_pident: float,
    min_qcov: float,
    top_n: int,
    strict_id_map: bool = False,
    min_scov: Optional[float] = None,
    best_hit_tie_policy: str = "discard",
    rbh_level: str = "gene",
) -> Tuple[object, object, object, object, object, object, object, Dict[str, int]]:
    """Parse existing BLAST TSV files and infer final gene-level relationships."""
    ref_raw = read_blast_table(ref_to_que_blast)
    que_raw = read_blast_table(que_to_ref_blast)
    ref_filtered = filter_and_rank_blast_hits(
        ref_raw,
        evalue=evalue,
        min_pident=min_pident,
        min_qcov=min_qcov,
        top_n=top_n,
        min_scov=min_scov,
    )
    que_filtered = filter_and_rank_blast_hits(
        que_raw,
        evalue=evalue,
        min_pident=min_pident,
        min_qcov=min_qcov,
        top_n=top_n,
        min_scov=min_scov,
    )
    if ref_filtered.empty and que_filtered.empty:
        raise ValueError(
            "No BLAST hits passed the filtering thresholds. Try lowering "
            "--min-pident/--min-qcov, increasing --evalue, or checking the FASTA "
            "and id-map files."
        )
    final_pairs, gene_hits, rbh_protein_evidence, gene_rbh, strict_rbh, pair_stats = infer_gene_pairs_from_blast(
        ref_filtered,
        que_filtered,
        ref_map=ref_map,
        que_map=que_map,
        strict_id_map=strict_id_map,
        best_hit_tie_policy=best_hit_tie_policy,
        rbh_level=rbh_level,
    )
    stats = {
        "number of ref_to_que raw BLAST hits": len(ref_raw),
        "number of que_to_ref raw BLAST hits": len(que_raw),
        "number of ref_to_que hits after filtering": len(ref_filtered),
        "number of que_to_ref hits after filtering": len(que_filtered),
    }
    stats.update(pair_stats)
    return final_pairs, ref_filtered, que_filtered, gene_hits, rbh_protein_evidence, gene_rbh, strict_rbh, stats


def write_tacman_homology_from_final_pairs(
    final_pairs,
    output_path: Path,
    sp_que: str,
    query_display_name: Optional[str],
    output_sep: str = "comma",
) -> List[str]:
    """Write final gene pairs in TACMAN's three-column homology txt format."""
    output_columns, _, _ = write_tacman_mapping(
        final_pairs,
        output_path=output_path,
        sp_que=sp_que,
        query_display_name=query_display_name,
        output_sep=output_sep,
    )
    return output_columns


def remove_blast_db_files(prefixes: Iterable[Path]) -> None:
    """Remove BLAST database files created by this script, keeping evidence TSVs."""
    for prefix in prefixes:
        for path in prefix.parent.glob(prefix.name + "*"):
            if path.is_file():
                path.unlink()


def make_blast_db(
    makeblastdb_bin: str,
    fasta_path: Path,
    output_prefix: Path,
    blastdb_version: Optional[int],
    log_path: Path,
) -> None:
    """Run makeblastdb for one protein FASTA file."""
    command: List[object] = [
        makeblastdb_bin,
        "-in",
        fasta_path,
        "-dbtype",
        "prot",
        "-out",
        output_prefix,
    ]
    if blastdb_version is not None:
        command.extend(["-blastdb_version", blastdb_version])
    run_command(command, log_path)


def run_blastp(
    blastp_bin: str,
    query_fasta: Path,
    db_prefix: Path,
    output_path: Path,
    evalue: float,
    max_target_seqs: int,
    threads: int,
    log_path: Path,
) -> None:
    """Run blastp and write an outfmt-6 TSV file."""
    command: List[object] = [
        blastp_bin,
        "-query",
        query_fasta,
        "-db",
        db_prefix,
        "-out",
        output_path,
        "-evalue",
        evalue,
        "-outfmt",
        BLAST_OUTFMT,
        "-max_target_seqs",
        max_target_seqs,
        "-num_threads",
        threads,
    ]
    run_command(command, log_path)


def infer_homology_from_blast_workdir(
    workdir: Path,
    ref_map: Dict[str, str],
    que_map: Dict[str, str],
    evalue: float,
    min_pident: float,
    min_qcov: float,
    top_n: int,
    strict_id_map: bool,
    min_scov: Optional[float],
    best_hit_tie_policy: str,
    rbh_level: str,
) -> Tuple[object, object, object, object, object, object, object, Dict[str, object]]:
    """Infer homology from existing BLAST workdir without running BLAST."""
    ref_raw = workdir / "ref_to_que.blast.tsv"
    que_raw = workdir / "que_to_ref.blast.tsv"
    ref_filtered_path = workdir / "ref_to_que.filtered.tsv"
    que_filtered_path = workdir / "que_to_ref.filtered.tsv"

    if ref_raw.exists() and que_raw.exists():
        return infer_homology_from_blast_tables(
            ref_to_que_blast=ref_raw,
            que_to_ref_blast=que_raw,
            ref_map=ref_map,
            que_map=que_map,
            evalue=evalue,
            min_pident=min_pident,
            min_qcov=min_qcov,
            top_n=top_n,
            strict_id_map=strict_id_map,
            min_scov=min_scov,
            best_hit_tie_policy=best_hit_tie_policy,
            rbh_level=rbh_level,
        )

    if not ref_filtered_path.exists() or not que_filtered_path.exists():
        raise FileNotFoundError(
            "BLAST workdir must contain ref_to_que.blast.tsv and "
            "que_to_ref.blast.tsv, or both filtered tables "
            "ref_to_que.filtered.tsv and que_to_ref.filtered.tsv."
        )

    ref_filtered = read_filtered_blast_table(ref_filtered_path)
    que_filtered = read_filtered_blast_table(que_filtered_path)
    final_pairs, gene_hits, rbh_protein_evidence, gene_rbh, strict_rbh, pair_stats = infer_gene_pairs_from_blast(
        ref_filtered,
        que_filtered,
        ref_map=ref_map,
        que_map=que_map,
        strict_id_map=strict_id_map,
        best_hit_tie_policy=best_hit_tie_policy,
        rbh_level=rbh_level,
    )
    stats: Dict[str, object] = {
        "number of ref_to_que raw BLAST hits": "not evaluated; raw BLAST table not found",
        "number of que_to_ref raw BLAST hits": "not evaluated; raw BLAST table not found",
        "number of ref_to_que hits after filtering": len(ref_filtered),
        "number of que_to_ref hits after filtering": len(que_filtered),
    }
    stats.update(pair_stats)
    return final_pairs, ref_filtered, que_filtered, gene_hits, rbh_protein_evidence, gene_rbh, strict_rbh, stats


def blast_postprocess_common(
    args: argparse.Namespace,
    output_path: Path,
    workdir: Path,
    ref_id_map: Path,
    que_id_map: Path,
    info_path: Optional[Path],
    mode_label: str,
    ref_sequence_count: object = "not evaluated",
    que_sequence_count: object = "not evaluated",
) -> Tuple[Path, Dict[str, object]]:
    """Shared BLAST post-processing, fixed outputs, QC, and registration."""
    if args.gene_id_type == "stable_id":
        allow_non_symbol = True
    else:
        allow_non_symbol = args.allow_non_symbol_ids

    ref_map, ref_map_warnings, ref_identifier_fraction = load_id_map(
        ref_id_map,
        protein_id_column=args.ref_protein_id_column,
        gene_column=args.ref_gene_column or "gene_symbol",
        label="reference",
        allow_non_symbol=allow_non_symbol,
    )
    que_map, que_map_warnings, que_identifier_fraction = load_id_map(
        que_id_map,
        protein_id_column=args.que_protein_id_column,
        gene_column=args.que_gene_column or "gene_symbol",
        label="query",
        allow_non_symbol=allow_non_symbol,
    )

    (
        final_pairs,
        ref_filtered,
        que_filtered,
        gene_hits,
        rbh_protein_evidence,
        gene_rbh,
        strict_rbh,
        blast_stats,
    ) = infer_homology_from_blast_workdir(
        workdir=workdir,
        ref_map=ref_map,
        que_map=que_map,
        evalue=args.evalue,
        min_pident=args.min_pident,
        min_qcov=args.min_qcov,
        top_n=args.top_n,
        strict_id_map=args.strict_id_map,
        min_scov=args.min_scov,
        best_hit_tie_policy=args.best_hit_tie_policy,
        rbh_level=args.rbh_level,
    )

    ref_filtered.to_csv(workdir / "ref_to_que.filtered.tsv", sep="\t", index=False)
    que_filtered.to_csv(workdir / "que_to_ref.filtered.tsv", sep="\t", index=False)
    gene_hits.to_csv(workdir / "blast_gene_pairs.filtered.tsv", sep="\t", index=False)
    final_pairs.to_csv(workdir / "blast_gene_pairs.final.tsv", sep="\t", index=False)
    rbh_protein_evidence.to_csv(workdir / "blast_protein_RBH.evidence.tsv", sep="\t", index=False)

    paths, clean_stats, primary_path = write_blast_output_bundle(
        output_path=output_path,
        all_pairs=final_pairs,
        gene_rbh=gene_rbh,
        strict_rbh=strict_rbh,
        sp_que=args.sp_que,
        query_display_name=args.query_display_name,
        output_sep=args.output_sep,
        primary_output=args.primary_output,
    )

    info_updated = False
    info_message = "Registration was not requested because --register-info was not provided."
    if info_path is not None:
        info_updated, info_message = register_info(
            info_path=info_path,
            output_path=primary_path,
            sp_ref=args.sp_ref,
            sp_que=args.sp_que,
            overwrite=args.overwrite_info,
            skip_if_exists=args.skip_register_if_exists,
            backup=args.backup_info,
        )

    support_one2one = final_pairs[
        (final_pairs["support"] != "RBH")
        & (final_pairs["homology_type"] == "ortholog_one2one")
    ]
    warnings = [
        PUTATIVE_BLAST_WARNING,
        "Protein-level RBH count may exceed unique gene-level RBH count because multiple protein isoforms can map to the same gene pair.",
        "For complete eukaryotic proteomes, rbh-level=gene is recommended because protein isoforms can create exact tied protein-level best hits.",
        "Non-RBH one-to-one components are sequence-similarity candidates and are not RBH-derived orthologs.",
        "Identifiers in the homology file must match AnnData.var_names.",
    ]
    if args.gene_id_type == "stable_id":
        warnings.append("gene-id-type=stable_id: stable identifiers are allowed, but they must match AnnData.var_names.")
    warnings.extend(ref_map_warnings)
    warnings.extend(que_map_warnings)
    if blast_stats["number of protein IDs missing in ref-id-map"]:
        warnings.append(
            "{} reference protein ID(s) from filtered BLAST hits were missing from the reference id map.".format(
                blast_stats["number of protein IDs missing in ref-id-map"]
            )
        )
    if blast_stats["number of protein IDs missing in que-id-map"]:
        warnings.append(
            "{} query protein ID(s) from filtered BLAST hits were missing from the query id map.".format(
                blast_stats["number of protein IDs missing in que-id-map"]
            )
        )

    primary_key = primary_output_key(args.primary_output)
    qc_path = Path(args.qc_out) if args.qc_out else output_path.with_name(
        f"{output_path.stem}.homology_qc.txt"
    )
    write_qc_report(
        qc_path=qc_path,
        lines=[
            ("mode", mode_label),
            *config_qc_lines(args),
            ("sp_ref", args.sp_ref),
            ("sp_que", args.sp_que),
            ("gene identifier type", args.gene_id_type),
            ("ref_id_map path", ref_id_map),
            ("que_id_map path", que_id_map),
            ("suspected identifier-like reference gene value fraction", f"{ref_identifier_fraction:.3f}"),
            ("suspected identifier-like query gene value fraction", f"{que_identifier_fraction:.3f}"),
            ("blast workdir", workdir),
            ("evalue", args.evalue),
            ("min_pident", args.min_pident),
            ("min_qcov", args.min_qcov),
            ("min_scov", args.min_scov if args.min_scov is not None else "not applied"),
            ("subject coverage filtering was applied", args.min_scov is not None),
            ("top_n", args.top_n),
            ("RBH level", blast_stats["RBH level"]),
            ("best-hit tie policy", args.best_hit_tie_policy),
            (
                "number of queries with tied best hits in forward search",
                blast_stats["number of queries with tied best hits in forward search"],
            ),
            (
                "number of queries with tied best hits in reverse search",
                blast_stats["number of queries with tied best hits in reverse search"],
            ),
            (
                "number of candidate RBHs discarded because of ties",
                blast_stats["number of candidate RBHs discarded because of ties"],
            ),
            ("number of protein-level candidate hits", blast_stats["number of protein-level candidate hits"]),
            ("number of unique gene-pair candidates", blast_stats["number of unique gene-pair candidates"]),
            (
                "number of gene pairs collapsed from multiple isoform hits",
                blast_stats["number of gene pairs collapsed from multiple isoform hits"],
            ),
            (
                "forward gene queries with tied best gene hits",
                blast_stats["forward gene queries with tied best gene hits"],
            ),
            (
                "reverse gene queries with tied best gene hits",
                blast_stats["reverse gene queries with tied best gene hits"],
            ),
            ("gene-level RBH pairs", blast_stats["gene-level RBH pairs"]),
            (
                "number of protein-level ties resolved by collapsing isoforms to the same gene",
                blast_stats["number of protein-level ties resolved by collapsing isoforms to the same gene"],
            ),
            (
                "number of true cross-gene ties discarded",
                blast_stats["number of true cross-gene ties discarded"],
            ),
            ("max_target_seqs", args.max_target_seqs),
            ("threads", args.threads),
            ("number of reference protein sequences", ref_sequence_count),
            ("number of query protein sequences", que_sequence_count),
            ("number of ref_to_que raw BLAST hits", blast_stats["number of ref_to_que raw BLAST hits"]),
            ("number of que_to_ref raw BLAST hits", blast_stats["number of que_to_ref raw BLAST hits"]),
            ("number of ref_to_que hits after filtering", blast_stats["number of ref_to_que hits after filtering"]),
            ("number of que_to_ref hits after filtering", blast_stats["number of que_to_ref hits after filtering"]),
            ("protein-level RBH pairs", blast_stats["number of reciprocal best hit protein pairs"]),
            ("unique gene-level RBH pairs", blast_stats["number of unique gene-level RBH pairs"]),
            (
                "protein RBH pairs collapsed during gene-level deduplication",
                blast_stats["number of protein RBH pairs collapsed during gene-level deduplication"],
            ),
            ("strict one-to-one gene-level RBH pairs", len(strict_rbh)),
            ("non-RBH structural one-to-one pairs", len(support_one2one)),
            ("all-putative pair count", len(final_pairs)),
            ("gene-RBH pair count", len(gene_rbh)),
            ("strict-one-to-one RBH count", len(strict_rbh)),
            ("all_putative rows removed for missing reference gene", clean_stats["all_putative"]["rows removed for missing reference gene"]),
            ("all_putative rows removed for missing query gene", clean_stats["all_putative"]["rows removed for missing query gene"]),
            ("all_putative rows removed for missing homology type", clean_stats["all_putative"]["rows removed for missing homology type"]),
            ("all_putative duplicate rows removed", clean_stats["all_putative"]["duplicate rows removed"]),
            ("gene_RBH rows removed for missing reference gene", clean_stats["gene_rbh"]["rows removed for missing reference gene"]),
            ("gene_RBH rows removed for missing query gene", clean_stats["gene_rbh"]["rows removed for missing query gene"]),
            ("gene_RBH rows removed for missing homology type", clean_stats["gene_rbh"]["rows removed for missing homology type"]),
            ("gene_RBH duplicate rows removed", clean_stats["gene_rbh"]["duplicate rows removed"]),
            ("strict_one2one_RBH rows removed for missing reference gene", clean_stats["strict_one2one_rbh"]["rows removed for missing reference gene"]),
            ("strict_one2one_RBH rows removed for missing query gene", clean_stats["strict_one2one_rbh"]["rows removed for missing query gene"]),
            ("strict_one2one_RBH rows removed for missing homology type", clean_stats["strict_one2one_rbh"]["rows removed for missing homology type"]),
            ("strict_one2one_RBH duplicate rows removed", clean_stats["strict_one2one_rbh"]["duplicate rows removed"]),
            ("TACMAN output separator", separator_label(normalize_output_separator(args.output_sep))),
            ("all-putative TACMAN path", paths["all_putative"]["txt"]),
            ("all-putative evidence path", paths["all_putative"]["evidence"]),
            ("gene-RBH TACMAN path", paths["gene_rbh"]["txt"]),
            ("gene-RBH evidence path", paths["gene_rbh"]["evidence"]),
            ("strict-one-to-one RBH TACMAN path", paths["strict_one2one_rbh"]["txt"]),
            ("strict-one-to-one RBH evidence path", paths["strict_one2one_rbh"]["evidence"]),
            ("registered primary output type", primary_key),
            ("registered primary output path", primary_path),
            ("info.csv was updated", info_updated),
            ("info.csv message", info_message),
            ("putative BLAST warning", PUTATIVE_BLAST_WARNING),
        ],
        homology_type_counts=final_pairs["homology_type"].value_counts(),
        warnings=warnings,
    )
    append_blast_qc_tables(qc_path, final_pairs)

    print(f"Wrote primary TACMAN homology alias: {output_path}")
    print(f"Wrote all-putative TACMAN file: {paths['all_putative']['txt']}")
    print(f"Wrote gene-RBH TACMAN file: {paths['gene_rbh']['txt']}")
    print(f"Wrote strict one-to-one RBH TACMAN file: {paths['strict_one2one_rbh']['txt']}")
    print(f"Wrote BLAST evidence files in: {workdir}")
    print(f"Wrote QC report: {qc_path}")
    print(info_message)

    summary = {
        "all_putative": len(final_pairs),
        "gene_rbh": len(gene_rbh),
        "strict_one2one_rbh": len(strict_rbh),
        "protein_rbh": blast_stats["number of reciprocal best hit protein pairs"],
        "unique_gene_rbh": blast_stats["number of unique gene-level RBH pairs"],
        "non_rbh_structural_one2one": len(support_one2one),
        "primary_path": primary_path,
    }
    return primary_path, summary


def run_database_mode(args: argparse.Namespace) -> None:
    """Convert a database homology table into TACMAN homology txt format."""
    pandas = require_pandas()
    required_args = [
        "input",
        "sp_ref",
        "sp_que",
        "ref_gene_column",
        "que_gene_column",
        "homology_type_column",
        "out",
        "register_info",
    ]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(
            "--mode database requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing_args)
        )

    input_path = Path(args.input)
    output_path = Path(args.out)
    info_path = Path(args.register_info)

    if not input_path.exists():
        raise FileNotFoundError(f"Input homology table does not exist: {input_path}")
    validate_registration_request(
        info_path=info_path,
        sp_ref=args.sp_ref,
        sp_que=args.sp_que,
        overwrite=args.overwrite_info,
        skip_if_exists=args.skip_register_if_exists,
    )

    sep = normalize_separator(args.sep, input_path)
    df_input = pandas.read_csv(input_path, sep=sep)
    selected_columns = [
        args.ref_gene_column,
        args.que_gene_column,
        args.homology_type_column,
    ]
    required_columns_exist(df_input, selected_columns)

    input_rows = len(df_input)
    df = df_input[selected_columns].copy()
    assert_gene_symbol_columns(
        df,
        ref_col=args.ref_gene_column,
        que_col=args.que_gene_column,
        allow_non_symbol=args.allow_non_symbol_ids or args.gene_id_type == "stable_id",
    )
    for column in selected_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()
    rows_after_na = int((df[selected_columns] != "").all(axis=1).sum())

    type_map = load_homology_type_map(
        Path(args.homology_type_map) if args.homology_type_map else None,
        args.sep,
    )
    df[args.homology_type_column], unknown_types = map_homology_types(
        df[args.homology_type_column], type_map, args.strict_homology_type
    )

    df_pairs = df.rename(
        columns={
            args.ref_gene_column: "ref_gene",
            args.que_gene_column: "que_gene",
            args.homology_type_column: "homology_type",
        }
    )[["ref_gene", "que_gene", "homology_type"]]
    output_columns, clean_stats, df_output = write_tacman_mapping(
        df_pairs,
        output_path=output_path,
        sp_que=args.sp_que,
        query_display_name=args.query_display_name,
        output_sep=args.output_sep,
    )

    info_updated, info_message = register_info(
        info_path=info_path,
        output_path=output_path,
        sp_ref=args.sp_ref,
        sp_que=args.sp_que,
        overwrite=args.overwrite_info,
        skip_if_exists=args.skip_register_if_exists,
        backup=args.backup_info,
    )

    warnings = []
    if unknown_types:
        warnings.append(
            "Unrecognized homology type(s) were kept unchanged: "
            + ", ".join(unknown_types)
        )

    qc_path = Path(args.qc_out) if args.qc_out else output_path.with_name(
        f"{output_path.stem}.homology_qc.txt"
    )
    write_qc_report(
        qc_path=qc_path,
        lines=[
            *config_qc_lines(args),
            ("input file path", input_path),
            ("detected/input separator", separator_label(sep)),
            ("output file path", output_path),
            ("sp_ref", args.sp_ref),
            ("sp_que", args.sp_que),
            ("number of input rows", input_rows),
            ("number of rows after removing NA", rows_after_na),
            (
                "rows removed for missing reference gene",
                clean_stats["rows removed for missing reference gene"],
            ),
            (
                "rows removed for missing query gene",
                clean_stats["rows removed for missing query gene"],
            ),
            (
                "rows removed for missing homology type",
                clean_stats["rows removed for missing homology type"],
            ),
            ("duplicate rows removed", clean_stats["duplicate rows removed"]),
            ("TACMAN output separator", separator_label(normalize_output_separator(args.output_sep))),
            ("number of final homology pairs", len(df_output)),
            ("number of unique reference genes", df_output[output_columns[0]].nunique()),
            ("number of unique query genes", df_output[output_columns[1]].nunique()),
            ("info.csv was updated", info_updated),
            ("info.csv message", info_message),
        ],
        homology_type_counts=df_output[output_columns[2]].value_counts(),
        warnings=warnings,
    )

    print(f"Wrote TACMAN homology file: {output_path}")
    print(f"Wrote QC report: {qc_path}")
    print(info_message)


def run_blast_mode(args: argparse.Namespace) -> None:
    """Run reciprocal BLASTP and export TACMAN-style putative homology pairs."""
    required_args = [
        "sp_ref",
        "sp_que",
        "ref_protein",
        "que_protein",
        "ref_id_map",
        "que_id_map",
        "out",
    ]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(
            "--mode blast requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing_args)
        )

    ref_protein = Path(args.ref_protein)
    que_protein = Path(args.que_protein)
    ref_id_map = Path(args.ref_id_map)
    que_id_map = Path(args.que_id_map)
    output_path = Path(args.out)
    info_path = Path(args.register_info) if args.register_info else None

    for label, path in [
        ("reference protein FASTA", ref_protein),
        ("query protein FASTA", que_protein),
        ("reference id map", ref_id_map),
        ("query id map", que_id_map),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if args.max_target_seqs < 1:
        raise ValueError("--max-target-seqs must be at least 1.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.min_scov is not None and args.min_scov < 0:
        raise ValueError("--min-scov must be non-negative when provided.")
    if args.blastdb_version not in {None, 4, 5}:
        raise ValueError("--blastdb-version must be 4 or 5 when provided.")
    if args.top_n > args.max_target_seqs:
        raise ValueError("--top-n must be less than or equal to --max-target-seqs.")

    if info_path is not None:
        validate_registration_request(
            info_path=info_path,
            sp_ref=args.sp_ref,
            sp_que=args.sp_que,
            overwrite=args.overwrite_info,
            skip_if_exists=args.skip_register_if_exists,
        )

    makeblastdb_bin = require_executable(args.makeblastdb_bin, "--makeblastdb-bin")
    blastp_bin = require_executable(args.blastp_bin, "--blastp-bin")

    workdir = (
        Path(args.blast_workdir)
        if args.blast_workdir
        else output_path.with_suffix("").with_name(output_path.stem + ".blast_workdir")
    )
    workdir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "blast_commands.log"

    ref_sequence_count = count_fasta_records(ref_protein)
    que_sequence_count = count_fasta_records(que_protein)

    query_db = workdir / "query_db"
    reference_db = workdir / "reference_db"
    ref_to_que_blast = workdir / "ref_to_que.blast.tsv"
    que_to_ref_blast = workdir / "que_to_ref.blast.tsv"
    make_blast_db(
        makeblastdb_bin=makeblastdb_bin,
        fasta_path=que_protein,
        output_prefix=query_db,
        blastdb_version=args.blastdb_version,
        log_path=log_path,
    )
    make_blast_db(
        makeblastdb_bin=makeblastdb_bin,
        fasta_path=ref_protein,
        output_prefix=reference_db,
        blastdb_version=args.blastdb_version,
        log_path=log_path,
    )
    run_blastp(
        blastp_bin=blastp_bin,
        query_fasta=ref_protein,
        db_prefix=query_db,
        output_path=ref_to_que_blast,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        threads=args.threads,
        log_path=log_path,
    )
    run_blastp(
        blastp_bin=blastp_bin,
        query_fasta=que_protein,
        db_prefix=reference_db,
        output_path=que_to_ref_blast,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        threads=args.threads,
        log_path=log_path,
    )
    primary_path, _ = blast_postprocess_common(
        args=args,
        output_path=output_path,
        workdir=workdir,
        ref_id_map=ref_id_map,
        que_id_map=que_id_map,
        info_path=info_path,
        mode_label="blast",
        ref_sequence_count=ref_sequence_count,
        que_sequence_count=que_sequence_count,
    )

    if not args.keep_blastdb:
        remove_blast_db_files([query_db, reference_db])


def run_blast_postprocess_mode(args: argparse.Namespace) -> None:
    """Re-run BLAST homology post-processing from an existing workdir."""
    required_args = [
        "sp_ref",
        "sp_que",
        "ref_id_map",
        "que_id_map",
        "blast_workdir",
        "out",
    ]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(
            "--mode blast-postprocess requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing_args)
        )

    ref_id_map = Path(args.ref_id_map)
    que_id_map = Path(args.que_id_map)
    output_path = Path(args.out)
    workdir = Path(args.blast_workdir)
    info_path = Path(args.register_info) if args.register_info else None

    for label, path in [
        ("reference id map", ref_id_map),
        ("query id map", que_id_map),
        ("BLAST workdir", workdir),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if args.min_scov is not None and args.min_scov < 0:
        raise ValueError("--min-scov must be non-negative when provided.")
    if info_path is not None:
        validate_registration_request(
            info_path=info_path,
            sp_ref=args.sp_ref,
            sp_que=args.sp_que,
            overwrite=args.overwrite_info,
            skip_if_exists=args.skip_register_if_exists,
        )

    blast_postprocess_common(
        args=args,
        output_path=output_path,
        workdir=workdir,
        ref_id_map=ref_id_map,
        que_id_map=que_id_map,
        info_path=info_path,
        mode_label="blast-postprocess",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TACMAN-compatible homology txt files and register them in "
            "TACMAN/homo/info.csv."
        ),
        epilog=(
            "Quick start:\n"
            "  python scripts/prepare_homology.py --config configs/celegans_dmelanogaster.yaml\n\n"
            "Advanced:\n"
            "  python scripts/prepare_homology.py --mode blast [options]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="YAML configuration file for a one-command run.")
    parser.add_argument(
        "--write-config-template",
        help="Write a commented full YAML configuration template and exit.",
    )
    parser.add_argument(
        "--write-minimal-config",
        help="Write a minimal YAML configuration template and exit.",
    )
    parser.add_argument("--mode", choices=["database", "blast", "blast-postprocess"])

    parser.add_argument("--input", help="Database homology table for --mode database.")
    parser.add_argument("--sep", help='Input separator. Use "\\t" for tab or ",".')
    parser.add_argument("--sp-ref", dest="sp_ref", help="Reference species name.")
    parser.add_argument("--sp-que", dest="sp_que", help="Query species name.")
    parser.add_argument(
        "--query-display-name",
        help=(
            "Species display name used in TACMAN column headers. Defaults to "
            "sp_que with the first letter capitalized."
        ),
    )
    parser.add_argument("--ref-gene-column", dest="ref_gene_column")
    parser.add_argument("--que-gene-column", dest="que_gene_column")
    parser.add_argument("--homology-type-column", dest="homology_type_column")
    parser.add_argument(
        "--allow-non-symbol-ids",
        action="store_true",
        help=(
            "Bypass the gene-symbol sanity check. Use only when AnnData var_names "
            "also use these non-symbol IDs."
        ),
    )
    parser.add_argument(
        "--homology-type-map",
        help="Optional two-column table mapping source labels to TACMAN labels.",
    )
    parser.add_argument(
        "--strict-homology-type",
        action="store_true",
        help="Fail if any homology type is not recognized by the mapping.",
    )
    parser.add_argument("--out", help="Output TACMAN homology txt file.")
    parser.add_argument(
        "--output-sep",
        default="comma",
        help="Separator for TACMAN-compatible .txt outputs: comma or tab. Default: comma.",
    )
    parser.add_argument(
        "--primary-output",
        default="all_putative",
        choices=["all_putative", "gene_rbh", "strict_one2one_rbh"],
        help="BLAST output registered in info.csv. Default: all_putative.",
    )
    parser.add_argument(
        "--gene-id-type",
        default="auto",
        choices=["auto", "symbol", "stable_id"],
        help="Identifier type in output homology files. stable_id suppresses gene-symbol warnings.",
    )
    parser.add_argument(
        "--register-info",
        dest="register_info",
        help="Path to TACMAN/homo/info.csv.",
    )
    parser.add_argument(
        "--overwrite-info",
        action="store_true",
        help="Replace existing info.csv records for the same sp_ref/sp_que.",
    )
    parser.add_argument(
        "--skip-register-if-exists",
        action="store_true",
        help="Leave info.csv unchanged if the same sp_ref/sp_que already exists.",
    )
    parser.add_argument(
        "--backup-info",
        dest="backup_info",
        action="store_true",
        default=True,
        help="Back up an existing info.csv to info.csv.bak before updating. Default: on.",
    )
    parser.add_argument(
        "--no-backup-info",
        dest="backup_info",
        action="store_false",
        help="Do not create info.csv.bak before updating info.csv.",
    )
    parser.add_argument("--qc-out", help="Optional QC report path.")

    parser.add_argument("--ref-protein", help="Reference proteome FASTA for --mode blast.")
    parser.add_argument("--que-protein", help="Query proteome FASTA for --mode blast.")
    parser.add_argument("--ref-id-map", help="Reference protein-to-gene map.")
    parser.add_argument("--que-id-map", help="Query protein-to-gene map.")
    parser.add_argument(
        "--ref-header-gene-pattern",
        help="Regex with one capture group for generating a reference id-map from FASTA headers.",
    )
    parser.add_argument(
        "--que-header-gene-pattern",
        help="Regex with one capture group for generating a query id-map from FASTA headers.",
    )
    parser.add_argument(
        "--ref-protein-id-column",
        default="protein_id",
        help="Protein ID column in --ref-id-map. Default: protein_id.",
    )
    parser.add_argument(
        "--que-protein-id-column",
        default="protein_id",
        help="Protein ID column in --que-id-map. Default: protein_id.",
    )
    parser.add_argument("--evalue", type=float, default=1e-5)
    parser.add_argument("--min-pident", type=float, default=30.0)
    parser.add_argument("--min-qcov", type=float, default=50.0)
    parser.add_argument(
        "--min-scov",
        type=float,
        default=None,
        help="Optional minimum subject coverage percentage. Default: not applied.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--best-hit-tie-policy",
        choices=["first", "discard", "all"],
        default="discard",
        help="How to handle exact tied best BLAST hits. Default: discard.",
    )
    parser.add_argument(
        "--rbh-level",
        choices=["protein", "gene"],
        default="gene",
        help=(
            "RBH level for BLAST post-processing. protein preserves legacy "
            "reciprocal best protein matching; gene collapses isoform-level "
            "protein hits before reciprocal best gene matching. Default: gene."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--blast-workdir",
        help="Directory for BLAST databases, raw hits, filtered hits, and evidence.",
    )
    parser.add_argument(
        "--keep-blastdb",
        action="store_true",
        help="Keep makeblastdb intermediate files after BLAST evidence is generated.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        default=True,
        help="Keep intermediate evidence files. Default: on.",
    )
    parser.add_argument(
        "--makeblastdb-bin",
        default="makeblastdb",
        help="makeblastdb executable. Default: makeblastdb.",
    )
    parser.add_argument(
        "--blastp-bin",
        default="blastp",
        help="blastp executable. Default: blastp.",
    )
    parser.add_argument(
        "--max-target-seqs",
        type=int,
        default=20,
        help="BLASTP -max_target_seqs value. Default: 20.",
    )
    parser.add_argument(
        "--blastdb-version",
        type=int,
        choices=[4, 5],
        help="Optional makeblastdb -blastdb_version value, usually 4 or 5.",
    )
    parser.add_argument(
        "--strict-id-map",
        action="store_true",
        help="Fail if filtered BLAST protein IDs are missing from id-map files.",
    )
    return parser


def main() -> None:
    """Run the command-line entry point."""
    parser = build_parser()
    explicit_dests = collect_explicit_cli_dests(parser, sys.argv[1:])
    args = parser.parse_args()

    try:
        if args.write_config_template:
            write_config_template(Path(args.write_config_template), minimal=False)
            print(f"Wrote full config template: {args.write_config_template}")
            return
        if args.write_minimal_config:
            write_config_template(Path(args.write_minimal_config), minimal=True)
            print(f"Wrote minimal config template: {args.write_minimal_config}")
            return

        if args.config:
            config_path = Path(args.config).resolve()
            config = load_config_file(config_path)
            apply_config_to_args(args, config_path, config, explicit_dests)

        if not args.mode:
            parser.error("Specify --config, --write-config-template, --write-minimal-config, or --mode.")

        validate_effective_config(args)
        prepare_header_mapping_if_needed(args)
        validate_effective_config(args)
        write_resolved_config_snapshot(args)

        if args.mode == "database":
            run_database_mode(args)
        elif args.mode == "blast":
            run_blast_mode(args)
        elif args.mode == "blast-postprocess":
            run_blast_postprocess_mode(args)
        else:
            parser.error(f"Unsupported mode: {args.mode}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
