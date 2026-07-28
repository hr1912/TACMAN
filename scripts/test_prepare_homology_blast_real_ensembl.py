#!/usr/bin/env python3
"""Real-data smoke test for TACMAN BLAST homology preparation.

This script downloads Ensembl protein FASTA files, extracts a small subset of
proteins, builds protein-to-gene mapping tables from FASTA headers, and runs
``prepare_homology.py --mode blast``. It intentionally uses protein FASTA from
Ensembl ``pep`` directories, not FASTQ.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class SpeciesFasta:
    """Downloaded and extracted FASTA paths for one species."""

    label: str
    ensembl_name: str
    ftp_url: str
    downloaded_fasta_gz: Path
    small_fasta: Path
    id_map: Path
    extracted_count: int
    mapped_to_symbols: int
    fallback_count: int


def strip_version(identifier: str) -> str:
    """Remove a trailing numeric Ensembl-style version from an identifier."""
    return re.sub(r"\.\d+$", "", identifier.strip())


def require_blast_plus(makeblastdb_bin: str, blastp_bin: str) -> None:
    """Fail early if NCBI BLAST+ executables are not available."""
    missing = [
        binary
        for binary in [makeblastdb_bin, blastp_bin]
        if shutil.which(binary) is None
    ]
    if missing:
        raise RuntimeError(
            "NCBI BLAST+ is required. Install with: "
            "conda install -c bioconda blast -y. Missing executable(s): "
            + ", ".join(missing)
        )


def read_url_text(url: str, timeout: int = 60) -> str:
    """Read text from a URL with a helpful user agent."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TACMAN-STAR-Protocols-real-blast-smoke-test"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def find_pep_all_filename(directory_url: str) -> str:
    """Find the Ensembl pep.all.fa.gz file from an FTP/HTTPS directory listing."""
    listing = read_url_text(directory_url)
    href_matches = re.findall(r'href=["\']([^"\']*pep\.all\.fa\.gz)["\']', listing)
    text_matches = re.findall(r"([A-Za-z0-9_.-]+pep\.all\.fa\.gz)", listing)
    candidates = sorted({html.unescape(match) for match in href_matches + text_matches})
    candidates = [candidate for candidate in candidates if not candidate.endswith(".md5")]
    if not candidates:
        raise RuntimeError(f"No pep.all.fa.gz file found in Ensembl directory: {directory_url}")
    return candidates[0]


def cached_pep_all(proteome_dir: Path, ensembl_name: str) -> Optional[Path]:
    """Return a manually downloaded or previously cached pep.all.fa.gz file."""
    candidates = sorted(proteome_dir.glob(f"{ensembl_name}*.pep.all.fa.gz"))
    if candidates:
        return candidates[0]
    candidates = sorted(proteome_dir.glob("*pep.all.fa.gz"))
    for candidate in candidates:
        if ensembl_name.lower() in candidate.name.lower():
            return candidate
    return None


def download_file(url: str, output_path: Path, timeout: int = 120) -> None:
    """Download a URL to a local file without third-party dependencies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TACMAN-STAR-Protocols-real-blast-smoke-test"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def download_or_use_cached_pep_all(
    proteome_dir: Path,
    directory_url: str,
    ensembl_name: str,
) -> Path:
    """Download an Ensembl pep.all.fa.gz file, or reuse a local cached copy."""
    cached = cached_pep_all(proteome_dir, ensembl_name)
    if cached is not None:
        return cached

    try:
        filename = find_pep_all_filename(directory_url)
        fasta_url = urllib.parse.urljoin(directory_url, filename)
        output_path = proteome_dir / Path(filename).name
        print(f"Downloading {fasta_url}")
        download_file(fasta_url, output_path)
        return output_path
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(
            "Failed to download Ensembl protein FASTA. You can manually download "
            f"the pep.all.fa.gz file from {directory_url} into {proteome_dir} "
            "and rerun this script. Original error: "
            f"{exc}"
        ) from exc


def parse_gene_symbol(header_without_gt: str, protein_id: str) -> Tuple[str, bool]:
    """Parse gene_symbol from an Ensembl FASTA header, falling back as needed."""
    gene_symbol_match = re.search(r"(?:^|\s)gene_symbol:([^\s]+)", header_without_gt)
    if gene_symbol_match:
        return gene_symbol_match.group(1), True

    gene_match = re.search(r"(?:^|\s)gene:([^\s]+)", header_without_gt)
    if gene_match:
        return strip_version(gene_match.group(1)), True

    return protein_id, False


def wrap_sequence(sequence: str, width: int = 60) -> str:
    """Wrap a FASTA sequence to a fixed line width."""
    return "\n".join(textwrap.wrap(sequence, width))


def iter_fasta_records_from_gzip(fasta_gz: Path) -> Iterable[Tuple[str, str]]:
    """Yield ``(header_without_gt, sequence)`` records from a gzipped FASTA."""
    header: Optional[str] = None
    chunks: List[str] = []
    with gzip.open(fasta_gz, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def extract_small_fasta_and_map(
    fasta_gz: Path,
    small_fasta: Path,
    id_map: Path,
    n_proteins: int,
) -> Tuple[int, int, int]:
    """Extract the first N proteins and write a matching protein-to-gene map."""
    small_fasta.parent.mkdir(parents=True, exist_ok=True)
    id_map.parent.mkdir(parents=True, exist_ok=True)

    extracted = 0
    mapped_to_symbols = 0
    fallback_count = 0
    with small_fasta.open("w", encoding="utf-8") as fasta_handle:
        with id_map.open("w", encoding="utf-8", newline="") as map_handle:
            writer = csv.DictWriter(
                map_handle,
                fieldnames=["protein_id", "gene_symbol"],
                delimiter="\t",
            )
            writer.writeheader()

            for header, sequence in iter_fasta_records_from_gzip(fasta_gz):
                if extracted >= n_proteins:
                    break
                if not sequence:
                    continue

                raw_protein_id = header.split()[0]
                protein_id = strip_version(raw_protein_id)
                gene_symbol, parsed_symbol = parse_gene_symbol(header, protein_id)
                if parsed_symbol:
                    mapped_to_symbols += 1
                else:
                    fallback_count += 1

                fasta_handle.write(f">{protein_id}\n")
                fasta_handle.write(wrap_sequence(sequence) + "\n")
                writer.writerow({"protein_id": protein_id, "gene_symbol": gene_symbol})
                extracted += 1

    if extracted == 0:
        raise RuntimeError(f"No protein records were extracted from {fasta_gz}")
    return extracted, mapped_to_symbols, fallback_count


def prepare_species_fasta(
    label: str,
    ensembl_name: str,
    directory_url: str,
    proteome_dir: Path,
    small_dir: Path,
    n_proteins: int,
) -> SpeciesFasta:
    """Download/cache and extract a small Ensembl protein FASTA for one species."""
    fasta_gz = download_or_use_cached_pep_all(
        proteome_dir=proteome_dir,
        directory_url=directory_url,
        ensembl_name=ensembl_name,
    )
    small_fasta = small_dir / f"{label}_{n_proteins}.pep.fa"
    id_map = small_dir / f"{label}_{n_proteins}_protein_to_gene.tsv"
    extracted_count, mapped_to_symbols, fallback_count = extract_small_fasta_and_map(
        fasta_gz=fasta_gz,
        small_fasta=small_fasta,
        id_map=id_map,
        n_proteins=n_proteins,
    )
    return SpeciesFasta(
        label=label,
        ensembl_name=ensembl_name,
        ftp_url=directory_url,
        downloaded_fasta_gz=fasta_gz,
        small_fasta=small_fasta,
        id_map=id_map,
        extracted_count=extracted_count,
        mapped_to_symbols=mapped_to_symbols,
        fallback_count=fallback_count,
    )


def run_prepare_homology(
    python_executable: str,
    repo_root: Path,
    rat: SpeciesFasta,
    mouse: SpeciesFasta,
    output_dir: Path,
    threads: int,
) -> None:
    """Run prepare_homology.py --mode blast on the small real FASTA files."""
    command = [
        python_executable,
        str(repo_root / "scripts" / "prepare_homology.py"),
        "--mode",
        "blast",
        "--sp-ref",
        "rat",
        "--sp-que",
        "mouse",
        "--ref-protein",
        str(rat.small_fasta),
        "--que-protein",
        str(mouse.small_fasta),
        "--ref-id-map",
        str(rat.id_map),
        "--que-id-map",
        str(mouse.id_map),
        "--ref-protein-id-column",
        "protein_id",
        "--ref-gene-column",
        "gene_symbol",
        "--que-protein-id-column",
        "protein_id",
        "--que-gene-column",
        "gene_symbol",
        "--evalue",
        "1e-5",
        "--min-pident",
        "30",
        "--min-qcov",
        "50",
        "--top-n",
        "5",
        "--threads",
        str(threads),
        "--out",
        str(output_dir / "rat_to_mouse.txt"),
        "--register-info",
        str(output_dir / "info.csv"),
        "--overwrite-info",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "prepare_homology.py --mode blast failed. Command:\n"
            + " ".join(command)
        )
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def count_tsv_rows(path: Path) -> Optional[int]:
    """Count data rows in a TSV file, returning None when the file is unreadable."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            return sum(1 for _ in reader)
    except OSError:
        return None


def check_expected_outputs(output_dir: Path) -> None:
    """Ensure the smoke test produced the expected TACMAN and evidence files."""
    expected_files = [
        output_dir / "rat_to_mouse.txt",
        output_dir / "rat_to_mouse.homology_qc.txt",
        output_dir / "info.csv",
        output_dir / "rat_to_mouse.blast_workdir" / "blast_gene_pairs.final.tsv",
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        raise RuntimeError(
            "Smoke test finished but expected output file(s) are missing: "
            + ", ".join(map(str, missing))
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download small Ensembl protein FASTA subsets and smoke-test "
            "prepare_homology.py --mode blast. This test uses protein FASTA, not FASTQ."
        )
    )
    parser.add_argument("--release", default="115", help="Ensembl release. Default: 115.")
    parser.add_argument(
        "--n-proteins",
        type=int,
        default=100,
        help="Number of proteins to extract from each species. Default: 100.",
    )
    parser.add_argument(
        "--base-dir",
        default="test_real_blast",
        help="Directory for downloaded data and outputs. Default: test_real_blast.",
    )
    parser.add_argument("--threads", type=int, default=4, help="BLAST threads. Default: 4.")
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
    return parser


def main() -> None:
    """Run the real Ensembl small BLAST smoke test."""
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / args.base_dir
    proteome_dir = base_dir / "proteome"
    small_dir = base_dir / "small"
    output_dir = base_dir / "output_small"

    try:
        require_blast_plus(args.makeblastdb_bin, args.blastp_bin)

        release_base = f"https://ftp.ensembl.org/pub/release-{args.release}/fasta"
        rat = prepare_species_fasta(
            label="rat",
            ensembl_name="Rattus_norvegicus",
            directory_url=f"{release_base}/rattus_norvegicus/pep/",
            proteome_dir=proteome_dir,
            small_dir=small_dir,
            n_proteins=args.n_proteins,
        )
        mouse = prepare_species_fasta(
            label="mouse",
            ensembl_name="Mus_musculus",
            directory_url=f"{release_base}/mus_musculus/pep/",
            proteome_dir=proteome_dir,
            small_dir=small_dir,
            n_proteins=args.n_proteins,
        )

        run_prepare_homology(
            python_executable=sys.executable,
            repo_root=repo_root,
            rat=rat,
            mouse=mouse,
            output_dir=output_dir,
            threads=args.threads,
        )
        check_expected_outputs(output_dir)

        final_pairs = output_dir / "rat_to_mouse.blast_workdir" / "blast_gene_pairs.final.tsv"
        final_pair_count = count_tsv_rows(final_pairs)

        print("\nReal Ensembl small BLAST smoke test summary")
        print(f"downloaded rat FASTA: {rat.downloaded_fasta_gz}")
        print(f"downloaded mouse FASTA: {mouse.downloaded_fasta_gz}")
        print(f"number of extracted rat proteins: {rat.extracted_count}")
        print(f"number of extracted mouse proteins: {mouse.extracted_count}")
        print(
            "number of rat protein IDs mapped to gene symbols: "
            f"{rat.mapped_to_symbols} (fallback to protein ID: {rat.fallback_count})"
        )
        print(
            "number of mouse protein IDs mapped to gene symbols: "
            f"{mouse.mapped_to_symbols} (fallback to protein ID: {mouse.fallback_count})"
        )
        print(f"output homology file path: {output_dir / 'rat_to_mouse.txt'}")
        if final_pair_count is not None:
            print(f"final homology pair count: {final_pair_count}")
        else:
            print("final homology pair count: not readable")

        if rat.fallback_count or mouse.fallback_count:
            print(
                "WARNING: Some FASTA headers did not contain gene_symbol or gene; "
                "their protein IDs were used as gene_symbol fallback values."
            )

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
