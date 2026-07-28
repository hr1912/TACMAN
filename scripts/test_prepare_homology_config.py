#!/usr/bin/env python3
"""Lightweight tests for YAML-style configuration plumbing.

The tests build tiny artificial files and avoid running makeblastdb/blastp.
They exercise config merging, relative-path resolution, FASTA-header mapping,
database mode, and blast-postprocess mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path

from prepare_homology import (
    apply_config_to_args,
    build_parser,
    collect_explicit_cli_dests,
    load_config_file,
    prepare_header_mapping_if_needed,
    run_blast_postprocess_mode,
    run_database_mode,
    validate_effective_config,
    write_resolved_config_snapshot,
)


def write(path: Path, text: str) -> None:
    """Write test text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_blast(path: Path, rows: list[list[object]]) -> None:
    """Write a tiny BLAST outfmt-6 file."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def parse_with_config(config_path: Path, config: dict, extra_cli: list[str] | None = None):
    """Create argparse args from a config dict plus optional CLI overrides."""
    parser = build_parser()
    argv = ["--config", str(config_path)]
    if extra_cli:
        argv.extend(extra_cli)
    explicit = collect_explicit_cli_dests(parser, argv)
    args = parser.parse_args(argv)
    apply_config_to_args(args, config_path, config, explicit)
    return args


def test_merge_and_relative_paths(tmp: Path) -> None:
    """Test minimal/full config merging, CLI override, and relative paths."""
    config_path = tmp / "configs" / "example.yaml"
    raw = tmp / "raw"
    write(raw / "ref.fa", ">R1 gene:RG1\nMPEPTIDE\n")
    write(raw / "que.fa", ">Q1 gene:QG1\nMPEPTIDE\n")

    config = {
        "mode": "blast",
        "species": {"reference": "ref", "query": "que"},
        "input": {
            "reference_protein": "../raw/ref.fa",
            "query_protein": "../raw/que.fa",
        },
        "header_parsing": {
            "reference_gene_pattern": r"gene:([^\s]+)",
            "query_gene_pattern": r"gene:([^\s]+)",
        },
        "blast": {"threads": 4, "best_hit_tie_policy": "discard"},
        "output": {"path": "../out/ref_to_que.txt"},
    }
    args = parse_with_config(config_path, config, extra_cli=["--threads", "32"])
    assert args.mode == "blast"
    assert args.threads == 32
    assert Path(args.ref_protein) == (tmp / "raw" / "ref.fa").resolve()
    assert Path(args.out) == (tmp / "out" / "ref_to_que.txt").resolve()
    validate_effective_config(args)
    prepare_header_mapping_if_needed(args)
    assert Path(args.ref_id_map).exists()
    assert Path(args.que_id_map).exists()
    assert args.ref_gene_column == "gene_id"
    assert args.que_gene_column == "gene_id"
    snapshot = write_resolved_config_snapshot(args)
    assert snapshot is not None
    assert snapshot.exists()
    assert "script_defaults" in snapshot.read_text(encoding="utf-8")


def test_database_config(tmp: Path) -> None:
    """Run database mode from config-derived args."""
    config_path = tmp / "configs" / "database.yaml"
    write(
        tmp / "data" / "rat_mouse.csv",
        "rat_gene_symbol,mouse_gene_symbol,mouse_homology_type\nTp53,Trp53,one2one\n",
    )
    config = {
        "mode": "database",
        "species": {"reference": "rat", "query": "mouse", "query_display_name": "Mouse"},
        "input": {"path": "../data/rat_mouse.csv"},
        "columns": {
            "reference_gene": "rat_gene_symbol",
            "query_gene": "mouse_gene_symbol",
            "homology_type": "mouse_homology_type",
        },
        "output": {
            "path": "../out/rat_to_mouse.txt",
            "register_info": "../out/info.csv",
            "overwrite_info": True,
            "separator": "comma",
        },
    }
    args = parse_with_config(config_path, config)
    validate_effective_config(args)
    run_database_mode(args)
    out = tmp / "out" / "rat_to_mouse.txt"
    assert out.exists()
    assert out.read_text(encoding="utf-8").splitlines()[0] == (
        "Gene name,Mouse gene name,Mouse homology type"
    )
    assert (tmp / "out" / "info.csv").exists()


def test_blast_postprocess_config(tmp: Path) -> None:
    """Run blast-postprocess mode from tiny precomputed BLAST tables."""
    config_path = tmp / "configs" / "postprocess.yaml"
    workdir = tmp / "work"
    workdir.mkdir(parents=True)
    write_blast(
        workdir / "ref_to_que.blast.tsv",
        [["Rp1", "Qp1", 95, 100, 100, 100, 95, "1e-50", 200]],
    )
    write_blast(
        workdir / "que_to_ref.blast.tsv",
        [["Qp1", "Rp1", 95, 100, 100, 100, 95, "1e-50", 200]],
    )
    write(tmp / "maps" / "ref.tsv", "protein_id\tgene_id\nRp1\tRG1\n")
    write(tmp / "maps" / "que.tsv", "protein_id\tgene_id\nQp1\tQG1\n")

    config = {
        "mode": "blast-postprocess",
        "species": {"reference": "ref", "query": "que", "query_display_name": "Que"},
        "input": {
            "reference_id_map": "../maps/ref.tsv",
            "query_id_map": "../maps/que.tsv",
        },
        "columns": {
            "reference_protein_id": "protein_id",
            "reference_gene": "gene_id",
            "query_protein_id": "protein_id",
            "query_gene": "gene_id",
        },
        "blast": {"best_hit_tie_policy": "discard"},
        "output": {
            "path": "../out/ref_to_que.txt",
            "primary_output": "all_putative",
            "separator": "comma",
            "register_info": "../out/info.csv",
            "overwrite_info": True,
        },
        "identifiers": {"gene_id_type": "stable_id"},
        "runtime": {"blast_workdir": "../work"},
    }
    args = parse_with_config(config_path, config)
    validate_effective_config(args)
    prepare_header_mapping_if_needed(args)
    validate_effective_config(args)
    try:
        snapshot = write_resolved_config_snapshot(args)
    except RuntimeError:
        snapshot = None
    run_blast_postprocess_mode(args)
    assert (tmp / "out" / "ref_to_que.all_putative_homology.txt").exists()
    evidence = tmp / "out" / "ref_to_que.all_putative_homology.evidence.tsv"
    text = evidence.read_text(encoding="utf-8")
    assert "relationship_cardinality" in text
    assert "classification_method" in text
    if snapshot is not None:
        assert snapshot.exists()


def test_invalid_fields(tmp: Path) -> None:
    """Invalid numeric fields should produce field-path errors."""
    config_path = tmp / "configs" / "bad.yaml"
    config = {
        "mode": "blast",
        "species": {"reference": "ref", "query": "que"},
        "input": {
            "reference_protein": "../raw/ref.fa",
            "query_protein": "../raw/que.fa",
        },
        "header_parsing": {
            "reference_gene_pattern": r"gene:([^\s]+)",
            "query_gene_pattern": r"gene:([^\s]+)",
        },
        "blast": {"min_pident": 130},
        "output": {"path": "../out/ref_to_que.txt"},
    }
    write(tmp / "raw" / "ref.fa", ">R gene:RG\nM\n")
    write(tmp / "raw" / "que.fa", ">Q gene:QG\nM\n")
    args = parse_with_config(config_path, config)
    try:
        validate_effective_config(args)
    except ValueError as exc:
        assert "blast.min_pident" in str(exc)
    else:
        raise AssertionError("Expected invalid min_pident to fail")


def test_no_pyyaml_message(tmp: Path) -> None:
    """The command should enter config handling even when dependencies are incomplete."""
    config_path = tmp / "minimal.yaml"
    write(
        config_path,
        """mode: blast
species:
  reference: ref
  query: que
input:
  reference_protein: missing_ref.fa
  query_protein: missing_que.fa
header_parsing:
  reference_gene_pattern: 'gene:([^\\s]+)'
  query_gene_pattern: 'gene:([^\\s]+)'
output:
  path: out/ref_to_que.txt
""",
    )
    script = Path(__file__).with_name("prepare_homology.py")
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "Input file from config does not exist" in result.stderr


def main() -> None:
    """Run all configuration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_merge_and_relative_paths(tmp / "merge")
        if importlib.util.find_spec("pandas") is not None:
            test_database_config(tmp / "database")
            test_blast_postprocess_config(tmp / "postprocess")
        test_invalid_fields(tmp / "invalid")
        try:
            load_config_file(tmp / "not_found.yaml")
        except (FileNotFoundError, RuntimeError):
            pass
        test_no_pyyaml_message(tmp / "nopyyaml")
    print("PASS")


if __name__ == "__main__":
    main()
