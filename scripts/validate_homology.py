#!/usr/bin/env python3
"""Validate TACMAN homology txt files and optional AnnData gene coverage."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

pd = None


CANONICAL_HOMOLOGY_TYPES = {
    "ortholog_one2one",
    "ortholog_one2many",
    "ortholog_many2many",
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
            "pandas is required to validate TACMAN homology txt files. Install "
            "pandas in the Python environment used to run this script."
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
    """Return the species label expected in TACMAN homology column names."""
    if query_display_name:
        return query_display_name
    if not sp_que:
        raise ValueError("--sp-que must not be empty")
    return sp_que[:1].upper() + sp_que[1:]


def expected_columns(sp_que: str, query_display_name: Optional[str]) -> List[str]:
    """Return the exact TACMAN homology columns expected for a query species."""
    query_display = species_display_name(sp_que, query_display_name)
    return [
        "Gene name",
        f"{query_display} gene name",
        f"{query_display} homology type",
    ]


def validate_columns(actual: Sequence[str], expected: Sequence[str]) -> None:
    """Raise a clear error if homology columns do not match TACMAN format."""
    if list(actual) != list(expected):
        raise ValueError(
            "Homology file column names do not match TACMAN expectations.\n"
            f"Expected columns: {list(expected)}\n"
            f"Actual columns: {list(actual)}\n"
            "If the query species display name differs from sp_que capitalization, "
            "rerun with --query-display-name."
        )


def read_homology_table(
    homology_path: Path,
    sp_que: str,
    query_display_name: Optional[str],
    sep: Optional[str],
) -> Tuple[pd.DataFrame, List[str], int]:
    """Read and lightly clean a TACMAN homology txt file."""
    pandas = require_pandas()
    if not homology_path.exists():
        raise FileNotFoundError(f"Homology file does not exist: {homology_path}")

    delimiter = normalize_separator(sep, homology_path)
    df = pandas.read_csv(homology_path, sep=delimiter)
    expected = expected_columns(sp_que, query_display_name)
    validate_columns(df.columns, expected)

    rows_before_cleaning = len(df)
    df = df.dropna(subset=expected).copy()
    for column in expected:
        df[column] = df[column].astype(str).str.strip()
    df = df[(df[expected] != "").all(axis=1)].copy()
    dropped_rows = rows_before_cleaning - len(df)
    return df, expected, dropped_rows


def read_var_names(h5ad_path: Optional[Path]) -> Tuple[Optional[List[str]], List[str]]:
    """Read AnnData var_names with scanpy or anndata when available."""
    warnings: List[str] = []
    if h5ad_path is None:
        return None, warnings
    if not h5ad_path.exists():
        raise FileNotFoundError(f"AnnData file does not exist: {h5ad_path}")

    read_h5ad = None
    try:
        import scanpy as sc  # type: ignore

        read_h5ad = sc.read_h5ad
    except ImportError:
        try:
            import anndata as ad  # type: ignore

            read_h5ad = ad.read_h5ad
        except ImportError:
            warnings.append(
                "scanpy/anndata is not installed; AnnData overlap checks were skipped."
            )
            return None, warnings

    adata = read_h5ad(h5ad_path, backed="r")
    var_names = [str(value) for value in adata.var_names]
    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()
    return var_names, warnings


def write_lines(path: Path, values: Iterable[str]) -> None:
    """Write one value per line."""
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def overlap_summary(
    label: str,
    homology_genes: Iterable[str],
    var_names: Optional[List[str]],
) -> Tuple[List[Tuple[str, object]], List[str]]:
    """Summarize overlap between homology genes and AnnData var_names."""
    homology_set = set(map(str, homology_genes))
    if var_names is None:
        return [
            (f"overlap with {label} AnnData var_names", "not evaluated"),
            (f"percent coverage of {label} AnnData genes", "not evaluated"),
            (f"percent of homology {label} genes found in AnnData", "not evaluated"),
        ], []

    var_set = set(map(str, var_names))
    overlap = homology_set & var_set
    unmatched_var_genes = sorted(var_set - homology_set)
    coverage_var = 100.0 * len(overlap) / len(var_set) if var_set else 0.0
    coverage_homology = (
        100.0 * len(overlap) / len(homology_set) if homology_set else 0.0
    )
    return [
        (f"overlap with {label} AnnData var_names", len(overlap)),
        (
            f"percent coverage of {label} AnnData genes",
            f"{coverage_var:.2f}%",
        ),
        (
            f"percent of homology {label} genes found in AnnData",
            f"{coverage_homology:.2f}%",
        ),
    ], unmatched_var_genes


def write_report(
    report_path: Path,
    lines: List[Tuple[str, object]],
    homology_type_counts: pd.Series,
    top_unmatched_ref: List[str],
    top_unmatched_que: List[str],
    warnings: List[str],
) -> None:
    """Write a plain-text validation report."""
    with report_path.open("w", encoding="utf-8") as handle:
        for key, value in lines:
            handle.write(f"{key}: {value}\n")

        handle.write("\nhomology type counts:\n")
        if homology_type_counts.empty:
            handle.write("  NA\n")
        else:
            for homology_type, count in homology_type_counts.items():
                handle.write(f"  {homology_type}: {count}\n")

        handle.write("\ntop unmatched reference AnnData genes:\n")
        if top_unmatched_ref:
            for gene in top_unmatched_ref[:50]:
                handle.write(f"  {gene}\n")
        else:
            handle.write("  NA\n")

        handle.write("\ntop unmatched query AnnData genes:\n")
        if top_unmatched_que:
            for gene in top_unmatched_que[:50]:
                handle.write(f"  {gene}\n")
        else:
            handle.write("  NA\n")

        if warnings:
            handle.write("\nwarnings:\n")
            for warning in warnings:
                handle.write(f"  - {warning}\n")


def run_validation(args: argparse.Namespace) -> None:
    """Run homology format and optional AnnData overlap validation."""
    homology_path = Path(args.homology)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, columns, dropped_rows = read_homology_table(
        homology_path=homology_path,
        sp_que=args.sp_que,
        query_display_name=args.query_display_name,
        sep=args.sep,
    )
    ref_col, que_col, type_col = columns
    warnings: List[str] = []

    unknown_types = sorted(
        set(df[type_col].astype(str)) - CANONICAL_HOMOLOGY_TYPES
    )
    if unknown_types:
        warnings.append(
            "Homology type value(s) outside TACMAN canonical labels were found: "
            + ", ".join(unknown_types)
        )
    if dropped_rows:
        warnings.append(
            f"{dropped_rows} row(s) with missing or empty values were ignored in statistics."
        )

    ref_var_names, ref_warnings = read_var_names(
        Path(args.ref_h5ad) if args.ref_h5ad else None
    )
    que_var_names, que_warnings = read_var_names(
        Path(args.que_h5ad) if args.que_h5ad else None
    )
    warnings.extend(ref_warnings)
    warnings.extend(que_warnings)

    ref_overlap_lines, unmatched_ref = overlap_summary(
        "reference", df[ref_col], ref_var_names
    )
    que_overlap_lines, unmatched_que = overlap_summary(
        "query", df[que_col], que_var_names
    )

    write_lines(outdir / "unmatched_ref_genes.txt", unmatched_ref)
    write_lines(outdir / "unmatched_query_genes.txt", unmatched_que)

    report_lines: List[Tuple[str, object]] = [
        ("homology file path", homology_path),
        ("sp_ref", args.sp_ref),
        ("sp_que", args.sp_que),
        ("number of homology pairs", len(df)),
        ("number of unique reference genes", df[ref_col].nunique()),
        ("number of unique query genes", df[que_col].nunique()),
        ("rows ignored because of missing or empty values", dropped_rows),
    ]
    report_lines.extend(ref_overlap_lines)
    report_lines.extend(que_overlap_lines)

    report_path = outdir / "homology_validation_report.txt"
    write_report(
        report_path=report_path,
        lines=report_lines,
        homology_type_counts=df[type_col].value_counts(),
        top_unmatched_ref=unmatched_ref,
        top_unmatched_que=unmatched_que,
        warnings=warnings,
    )

    print(f"Wrote validation report: {report_path}")
    print(f"Wrote unmatched reference genes: {outdir / 'unmatched_ref_genes.txt'}")
    print(f"Wrote unmatched query genes: {outdir / 'unmatched_query_genes.txt'}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate a TACMAN homology txt file and optional h5ad overlap."
    )
    parser.add_argument("--homology", required=True, help="TACMAN homology txt file.")
    parser.add_argument("--sp-ref", dest="sp_ref", required=True)
    parser.add_argument("--sp-que", dest="sp_que", required=True)
    parser.add_argument(
        "--query-display-name",
        help=(
            "Species display name expected in TACMAN column headers. Defaults to "
            "sp_que with the first letter capitalized."
        ),
    )
    parser.add_argument("--ref-h5ad", help="Optional reference AnnData h5ad file.")
    parser.add_argument("--que-h5ad", help="Optional query AnnData h5ad file.")
    parser.add_argument("--outdir", required=True, help="Directory for validation QC.")
    parser.add_argument("--sep", help='Homology separator. Use "\\t" for tab or ",".')
    return parser


def main() -> None:
    """Run the command-line entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_validation(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
