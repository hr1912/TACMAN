#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_homology.py \
  --mode blast \
  --sp-ref C_elegans \
  --sp-que D_melanogaster \
  --query-display-name Drosophila \
  --ref-protein examples/homology/celegans_dmelanogaster/raw/Caenorhabditis_elegans.WBcel235.pep.all.fa \
  --que-protein examples/homology/celegans_dmelanogaster/raw/Drosophila_melanogaster.BDGP6.46.pep.all.fa \
  --ref-id-map examples/homology/celegans_dmelanogaster/mapping/C_elegans_protein_to_gene.tsv \
  --que-id-map examples/homology/celegans_dmelanogaster/mapping/D_melanogaster_protein_to_gene.tsv \
  --ref-protein-id-column protein_id \
  --que-protein-id-column protein_id \
  --evalue 1e-5 \
  --min-pident 30 \
  --min-qcov 50 \
  --top-n 5 \
  --max-target-seqs 20 \
  --threads 16 \
  --rbh-level gene \
  --best-hit-tie-policy discard \
  --gene-id-type stable_id \
  --primary-output all_putative \
  --out examples/homology/celegans_dmelanogaster/ortholog/C_elegans_to_D_melanogaster.txt \
  --blast-workdir examples/homology/celegans_dmelanogaster/ortholog/C_elegans_to_D_melanogaster.blast_workdir \
  --register-info examples/homology/celegans_dmelanogaster/ortholog/info.csv \
  --overwrite-info
