#!/usr/bin/env bash
set -euo pipefail

mkdir -p examples/homology/celegans_dmelanogaster/raw

cd examples/homology/celegans_dmelanogaster/raw

wget -c https://ftp.ensembl.org/pub/release-115/fasta/caenorhabditis_elegans/pep/Caenorhabditis_elegans.WBcel235.pep.all.fa.gz

wget -c https://ftp.ensembl.org/pub/release-115/fasta/drosophila_melanogaster/pep/Drosophila_melanogaster.BDGP6.46.pep.all.fa.gz

gunzip -f Caenorhabditis_elegans.WBcel235.pep.all.fa.gz
gunzip -f Drosophila_melanogaster.BDGP6.46.pep.all.fa.gz
