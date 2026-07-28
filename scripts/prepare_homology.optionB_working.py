#!/usr/bin/env python3
"""Prepare TACMAN-compatible homology files from external resources."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
            "pandas is required for --mode database. Install pandas in the "
            "Python environment used to run this script."
        ) from exc
    pd = pandas_module
    return pd


def normalize_separator(sep: Optional[str], input_path: Path) -> str:
    """Return the delimiter requested by the user or inferred from the file."""
    if sep:
        if sep in {"\\t", "tab", "TAB"}:
            return "\t"
        return sep

    if input_path.suffix.lower() == ".csv":
        return ","
    if input_path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return "\t"


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
    """Heuristically detect common non-symbol gene/protein/transcript IDs."""
    value = str(value).strip()
    patterns = [
        r"^ENS[A-Z0-9]*[GTEP]\d+(\.\d+)?$",
        r"^[NX][MP]_\d+(\.\d+)?$",
        r"^[NX]R_\d+(\.\d+)?$",
        r"^XP_\d+(\.\d+)?$",
        r"^YP_\d+(\.\d+)?$",
        r"^FBgn\d+$",
        r"^WBGene\d+$",
    ]
    return any(re.match(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def assert_gene_symbol_columns(df, ref_col: str, que_col: str, allow_non_symbol: bool) -> None:
    """Warn users early when selected columns look like IDs instead of symbols."""
    if allow_non_symbol:
        return

    suspicious = []
    for label, column in [("reference", ref_col), ("query", que_col)]:
        values = [str(value).strip() for value in df[column].dropna().head(100)]
        if not values:
            continue
        n_id_like = sum(looks_like_external_id(value) for value in values)
        if n_id_like / len(values) >= 0.5:
            suspicious.append(f"{label} column '{column}'")

    if suspicious:
        raise ValueError(
            "TACMAN homology files should use gene symbols, but the selected "
            "column(s) look like Ensembl/protein/transcript IDs: {}. Provide "
            "database columns containing gene symbols before running this script, "
            "or rerun with --allow-non-symbol-ids only if these values are truly "
            "the var_names used by your AnnData objects.".format(", ".join(suspicious))
        )


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
        allow_non_symbol=args.allow_non_symbol_ids,
    )
    df = df.dropna(subset=selected_columns)
    for column in selected_columns:
        df[column] = df[column].astype(str).str.strip()
    df = df[(df[selected_columns] != "").all(axis=1)].copy()
    rows_after_na = len(df)

    type_map = load_homology_type_map(
        Path(args.homology_type_map) if args.homology_type_map else None,
        args.sep,
    )
    df[args.homology_type_column], unknown_types = map_homology_types(
        df[args.homology_type_column], type_map, args.strict_homology_type
    )

    before_dedup = len(df)
    df = df.drop_duplicates(subset=selected_columns).copy()
    duplicated_removed = before_dedup - len(df)

    query_display = species_display_name(args.sp_que, args.query_display_name)
    output_columns = [
        "Gene name",
        f"{query_display} gene name",
        f"{query_display} homology type",
    ]
    df_output = df.rename(
        columns={
            args.ref_gene_column: output_columns[0],
            args.que_gene_column: output_columns[1],
            args.homology_type_column: output_columns[2],
        }
    )[output_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, sep="\t", index=False)

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
            ("input file path", input_path),
            ("output file path", output_path),
            ("sp_ref", args.sp_ref),
            ("sp_que", args.sp_que),
            ("number of input rows", input_rows),
            ("number of rows after removing NA", rows_after_na),
            ("number of duplicated rows removed", duplicated_removed),
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
    """Reserve the BLAST workflow interface for reciprocal BLASTP support."""
    raise NotImplementedError(
        "--mode blast is reserved for a future reciprocal BLASTP workflow. "
        "The planned steps are: makeblastdb for both proteomes, forward blastp, "
        "reverse blastp, reciprocal best-hit parsing, one-to-many/many-to-many "
        "retention, TACMAN txt export, QC report writing, and info.csv registration. "
        "For now, prepare a database-style table and use --mode database."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TACMAN-compatible homology txt files and register them in "
            "TACMAN/homo/info.csv."
        )
    )
    parser.add_argument("--mode", choices=["database", "blast"], required=True)

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
    parser.add_argument("--evalue", type=float, default=1e-5)
    parser.add_argument("--min-pident", type=float, default=30.0)
    parser.add_argument("--min-qcov", type=float, default=50.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main() -> None:
    """Run the command-line entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "database":
            run_database_mode(args)
        elif args.mode == "blast":
            run_blast_mode(args)
        else:
            parser.error(f"Unsupported mode: {args.mode}")
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
