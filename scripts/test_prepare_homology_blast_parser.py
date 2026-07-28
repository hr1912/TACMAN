#!/usr/bin/env python3
"""Small parser test for BLAST-based TACMAN homology inference.

This test does not require BLAST+. It uses artificial outfmt-6 TSV files to
exercise reciprocal-best-hit detection, component-based classification, and
TACMAN-style txt export.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from prepare_homology import (
    infer_homology_from_blast_tables,
    write_tacman_homology_from_final_pairs,
)


def write_blast(path: Path, rows: list[list[object]]) -> None:
    """Write a tiny BLAST outfmt-6 table without a header."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def main() -> None:
    """Run the no-BLAST parser test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_to_que = tmp / "ref_to_que.blast.tsv"
        que_to_ref = tmp / "que_to_ref.blast.tsv"
        output = tmp / "speciesA_to_speciesB.txt"

        write_blast(
            ref_to_que,
            [
                ["A1p", "B1p", 95, 100, 100, 100, 95, "1e-50", 200],
                ["A2p", "B2p", 88, 100, 100, 100, 90, "1e-40", 180],
                ["A2p", "B3p", 78, 90, 100, 100, 82, "1e-25", 150],
                ["A3p", "B2p", 74, 88, 100, 100, 80, "1e-20", 140],
            ],
        )
        write_blast(
            que_to_ref,
            [
                ["B1p", "A1p", 95, 100, 100, 100, 95, "1e-50", 200],
                ["B2p", "A2p", 88, 100, 100, 100, 90, "1e-40", 180],
                ["B2p", "A3p", 74, 88, 100, 100, 80, "1e-20", 140],
                ["B3p", "A2p", 78, 90, 100, 100, 82, "1e-25", 150],
            ],
        )

        final_pairs, _, _, _, _, gene_rbh, strict_rbh, stats = infer_homology_from_blast_tables(
            ref_to_que_blast=ref_to_que,
            que_to_ref_blast=que_to_ref,
            ref_map={"A1p": "GeneA1", "A2p": "GeneA2", "A3p": "GeneA3"},
            que_map={"B1p": "GeneB1", "B2p": "GeneB2", "B3p": "GeneB3"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
        )
        write_tacman_homology_from_final_pairs(
            final_pairs,
            output_path=output,
            sp_que="speciesB",
            query_display_name="SpeciesB",
        )

        assert len(final_pairs) == 4, final_pairs
        assert len(gene_rbh) == 2, gene_rbh
        assert len(strict_rbh) == 2, strict_rbh
        assert stats["number of reciprocal best hit protein pairs"] == 2, stats
        assert set(final_pairs["homology_type"]) == {
            "ortholog_one2one",
            "ortholog_many2many",
        }
        assert set(final_pairs["support"]) == {"RBH", "topN_nonreciprocal"}
        assert "component_inferred" not in set(final_pairs["support"])
        assert {"relationship_cardinality", "classification_method"}.issubset(final_pairs.columns)
        assert output.read_text(encoding="utf-8").splitlines()[0] == (
            "Gene name,SpeciesB gene name,SpeciesB homology type"
        )

        tied_ref_to_que = tmp / "tied_ref_to_que.blast.tsv"
        tied_que_to_ref = tmp / "tied_que_to_ref.blast.tsv"
        write_blast(
            tied_ref_to_que,
            [
                ["A4p", "B4p", 90, 100, 100, 100, 90, "1e-20", 100],
                ["A4p", "B5p", 90, 100, 100, 100, 90, "1e-20", 100],
            ],
        )
        write_blast(
            tied_que_to_ref,
            [
                ["B4p", "A4p", 90, 100, 100, 100, 90, "1e-20", 100],
                ["B5p", "A4p", 90, 100, 100, 100, 90, "1e-20", 100],
            ],
        )
        tied_default, *_rest, tied_default_stats = infer_homology_from_blast_tables(
            ref_to_que_blast=tied_ref_to_que,
            que_to_ref_blast=tied_que_to_ref,
            ref_map={"A4p": "GeneA4"},
            que_map={"B4p": "GeneB4", "B5p": "GeneB5"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
        )
        tied_first, *_rest, tied_first_stats = infer_homology_from_blast_tables(
            ref_to_que_blast=tied_ref_to_que,
            que_to_ref_blast=tied_que_to_ref,
            ref_map={"A4p": "GeneA4"},
            que_map={"B4p": "GeneB4", "B5p": "GeneB5"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
            best_hit_tie_policy="first",
        )
        assert tied_default_stats["number of queries with tied best hits in forward search"] == 1
        assert tied_default_stats["number of candidate RBHs discarded because of ties"] == 1
        assert tied_default_stats["forward gene queries with tied best gene hits"] == 1
        assert tied_default_stats["number of true cross-gene ties discarded"] == 1
        assert tied_default["support"].tolist() == ["topN_nonreciprocal", "topN_nonreciprocal"]
        assert tied_first_stats["number of reciprocal best hit protein pairs"] == 1
        assert "RBH" in set(tied_first["support"])

        one_to_many_ref_to_que = tmp / "one_to_many_ref_to_que.blast.tsv"
        one_to_many_que_to_ref = tmp / "one_to_many_que_to_ref.blast.tsv"
        write_blast(
            one_to_many_ref_to_que,
            [
                ["A6p1", "B6p", 91, 100, 100, 100, 91, "1e-30", 210],
                ["A6p2", "B7p", 89, 100, 100, 100, 89, "1e-28", 205],
            ],
        )
        write_blast(
            one_to_many_que_to_ref,
            [
                ["B6p", "A6p1", 91, 100, 100, 100, 91, "1e-30", 210],
                ["B7p", "A6p2", 89, 100, 100, 100, 89, "1e-28", 205],
            ],
        )
        (
            one_to_many_final,
            _ref_filtered,
            _que_filtered,
            _gene_hits,
            _protein_rbh,
            one_to_many_gene_rbh,
            one_to_many_strict,
            one_to_many_stats,
        ) = infer_homology_from_blast_tables(
            ref_to_que_blast=one_to_many_ref_to_que,
            que_to_ref_blast=one_to_many_que_to_ref,
            ref_map={"A6p1": "GeneA6", "A6p2": "GeneA6"},
            que_map={"B6p": "GeneB6", "B7p": "GeneB7"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
        )
        assert one_to_many_stats["number of reciprocal best hit protein pairs"] == 2
        assert one_to_many_stats["RBH level"] == "gene"
        assert len(one_to_many_gene_rbh) == 1
        assert len(one_to_many_strict) == 1
        assert set(one_to_many_final["support"]) == {"RBH", "topN_nonreciprocal"}
        assert set(one_to_many_gene_rbh["relationship_cardinality"]) == {"1:1"}

        (
            protein_level_final,
            _ref_filtered,
            _que_filtered,
            _gene_hits,
            _protein_rbh,
            protein_level_gene_rbh,
            protein_level_strict,
            protein_level_stats,
        ) = infer_homology_from_blast_tables(
            ref_to_que_blast=one_to_many_ref_to_que,
            que_to_ref_blast=one_to_many_que_to_ref,
            ref_map={"A6p1": "GeneA6", "A6p2": "GeneA6"},
            que_map={"B6p": "GeneB6", "B7p": "GeneB7"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
            rbh_level="protein",
        )
        assert protein_level_stats["RBH level"] == "protein"
        assert len(protein_level_gene_rbh) == 2
        assert len(protein_level_strict) == 0
        assert set(protein_level_final["support"]) == {"RBH"}
        assert set(protein_level_gene_rbh["relationship_cardinality"]) == {"1:n"}

        same_gene_tie_ref_to_que = tmp / "same_gene_tie_ref_to_que.blast.tsv"
        same_gene_tie_que_to_ref = tmp / "same_gene_tie_que_to_ref.blast.tsv"
        write_blast(
            same_gene_tie_ref_to_que,
            [
                ["A7p", "B7p1", 92, 100, 100, 100, 92, "1e-35", 220],
                ["A7p", "B7p2", 92, 100, 100, 100, 92, "1e-35", 220],
            ],
        )
        write_blast(
            same_gene_tie_que_to_ref,
            [
                ["B7p1", "A7p", 92, 100, 100, 100, 92, "1e-35", 220],
                ["B7p2", "A7p", 92, 100, 100, 100, 92, "1e-35", 220],
            ],
        )
        (
            same_gene_final,
            _ref_filtered,
            _que_filtered,
            _gene_hits,
            _rbh_evidence,
            same_gene_rbh,
            same_gene_strict,
            same_gene_stats,
        ) = infer_homology_from_blast_tables(
            ref_to_que_blast=same_gene_tie_ref_to_que,
            que_to_ref_blast=same_gene_tie_que_to_ref,
            ref_map={"A7p": "GeneA7"},
            que_map={"B7p1": "GeneB7", "B7p2": "GeneB7"},
            evalue=1e-5,
            min_pident=30,
            min_qcov=50,
            top_n=5,
            rbh_level="gene",
            best_hit_tie_policy="discard",
        )
        assert same_gene_stats["forward gene queries with tied best gene hits"] == 0
        assert same_gene_stats["number of protein-level ties resolved by collapsing isoforms to the same gene"] >= 1
        assert len(same_gene_rbh) == 1
        assert len(same_gene_strict) == 1
        assert same_gene_final["gene_pair_isoform_hit_count"].iloc[0] == 4
        assert {"rbh_level", "gene_pair_best_ref_protein", "gene_pair_best_que_protein", "gene_pair_isoform_hit_count", "gene_level_tie_status"}.issubset(same_gene_final.columns)
        print("BLAST parser test passed")
        print(final_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
