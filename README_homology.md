# TACMAN Homology Preparation

TACMAN reads species-pair homology files through `TACMAN/homo/info.csv`.
Adding a new species pair does not require changing TACMAN core source code:
create a TACMAN-compatible three-column homology `.txt` file, place it under
`TACMAN/homo/` or another registered directory, and add one row to `info.csv`.

## Quick start

Run the complete homology preparation workflow using the provided configuration
file:

```bash
python scripts/prepare_homology.py --config configs/celegans_dmelanogaster.yaml
```

The YAML configuration file defines the species names, input protein FASTA
files, protein-to-gene mapping method, BLAST filtering thresholds,
computational resources, and output paths. Relative paths in YAML files are
resolved relative to the YAML file location.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate tacman-homology
```

NCBI BLAST+ is required for `mode: blast`:

```bash
conda install -c bioconda blast -y
```

## Configuration file

The recommended entry point is:

```bash
python scripts/prepare_homology.py --config configs/celegans_dmelanogaster.yaml
```

Configuration precedence is:

```text
script defaults < YAML configuration < explicit command-line options
```

For example, this uses all values from the YAML file but overrides the number
of threads:

```bash
python scripts/prepare_homology.py \
  --config configs/celegans_dmelanogaster.yaml \
  --threads 32
```

## CLI overrides

Explicit command-line values override YAML values. This is useful for resource
settings such as `threads` without editing the shared configuration file.

## Resolved config snapshot

Each config-mode run writes a reproducibility snapshot:

```text
<output_prefix>.resolved_config.yaml
```

The snapshot records the configuration file path, YAML values, explicit CLI
overrides, and final effective arguments.

## Minimal Configuration

The smallest BLAST-mode config can generate protein-to-gene maps directly from
FASTA headers:

```yaml
mode: blast
species:
  reference: C_elegans
  query: D_melanogaster
input:
  reference_protein: ../raw/C_elegans.protein.fa
  query_protein: ../raw/D_melanogaster.protein.fa
header_parsing:
  reference_gene_pattern: 'gene:([^\s]+)'
  query_gene_pattern: 'gene:([^\s]+)'
output:
  path: ../ortholog/C_elegans_to_D_melanogaster.txt
```

When `reference_id_map` or `query_id_map` is omitted, the script extracts the
protein ID from the first token after `>` and applies the corresponding regex
to the FASTA header. Generated maps are written to the BLAST workdir as:

```text
generated_reference_protein_to_gene.tsv
generated_query_protein_to_gene.tsv
```

These generated tables contain `protein_id` and `gene_id` columns. QC reports
record whether each map came from an external table or FASTA header parsing,
the regex used, the number of protein entries, mapped entries, and missing
gene IDs.

Generate templates with:

```bash
python scripts/prepare_homology.py --write-config-template configs/homology_template.yaml
python scripts/prepare_homology.py --write-minimal-config configs/homology_minimal.yaml
```

## Full Configuration

The formal C. elegans to D. melanogaster example is provided in
`configs/celegans_dmelanogaster.yaml`. It uses:

```yaml
mode: blast
species:
  reference: C_elegans
  query: D_melanogaster
  query_display_name: Drosophila
blast:
  evalue: 1e-5
  min_pident: 30
  min_qcov: 50
  min_scov: null
  top_n: 5
  max_target_seqs: 20
  threads: 16
  rbh_level: gene
  best_hit_tie_policy: discard
output:
  primary_output: all_putative
  separator: comma
identifiers:
  gene_id_type: stable_id
```

The software default for `threads` is 8. The cross-phylum example sets
`threads: 16` only for that case.

## Advanced command-line usage

Long CLI options remain available for advanced users.

Database mode converts a curated Ensembl, OMA, OrthoDB, or BioMart table:

```bash
python scripts/prepare_homology.py \
  --mode database \
  --input data/homology/rat_mouse_database.csv \
  --sp-ref rat \
  --sp-que mouse \
  --ref-gene-column rat_gene_symbol \
  --que-gene-column mouse_gene_symbol \
  --homology-type-column mouse_homology_type \
  --out TACMAN/homo/rat_to_mouse.txt \
  --register-info TACMAN/homo/info.csv
```

BLAST mode runs makeblastdb, reciprocal BLASTP, gene-level post-processing,
three-output export, QC reporting, and optional `info.csv` registration:

```bash
python scripts/prepare_homology.py \
  --mode blast \
  --sp-ref <reference_species> \
  --sp-que <query_species> \
  --ref-protein data/proteome/<reference_species>.pep.fa \
  --que-protein data/proteome/<query_species>.pep.fa \
  --ref-id-map data/proteome/<reference_species>_protein_to_gene.tsv \
  --que-id-map data/proteome/<query_species>_protein_to_gene.tsv \
  --ref-protein-id-column protein_id \
  --ref-gene-column gene_symbol \
  --que-protein-id-column protein_id \
  --que-gene-column gene_symbol \
  --evalue 1e-5 \
  --min-pident 30 \
  --min-qcov 50 \
  --top-n 5 \
  --threads 8 \
  --rbh-level gene \
  --best-hit-tie-policy discard \
  --out TACMAN/homo/<reference_species>_to_<query_species>.txt \
  --register-info TACMAN/homo/info.csv
```

Postprocess-only mode reuses an existing BLAST workdir without rerunning BLAST:

```bash
python scripts/prepare_homology.py \
  --mode blast-postprocess \
  --sp-ref C_elegans \
  --sp-que D_melanogaster \
  --ref-id-map mapping/C_elegans_protein_to_gene.tsv \
  --que-id-map mapping/D_melanogaster_protein_to_gene.tsv \
  --ref-protein-id-column protein_id \
  --ref-gene-column gene_id \
  --que-protein-id-column protein_id \
  --que-gene-column gene_id \
  --query-display-name Drosophila \
  --gene-id-type stable_id \
  --blast-workdir ortholog/C_elegans_to_D_melanogaster.blast_workdir \
  --out ortholog/C_elegans_to_D_melanogaster.txt
```

## Output Files

TACMAN-compatible `.txt` mapping files are comma-separated by default, matching
TACMAN's bundled `homo/*.txt` files and `pandas.read_csv()` default behavior.
They contain exactly three columns:

```text
Gene name
<QueryDisplayName> gene name
<QueryDisplayName> homology type
```

Evidence tables are tab-separated `.tsv` files and are not the files TACMAN
loads through `info.csv`.

BLAST mode writes three fixed output sets:

- `<prefix>.all_putative_homology.txt` and
  `<prefix>.all_putative_homology.evidence.tsv`: all filtered gene-level
  sequence-similarity candidates.
- `<prefix>.gene_RBH.txt` and `<prefix>.gene_RBH.evidence.tsv`: unique
  gene-level pairs supported by at least one protein-level reciprocal best hit.
- `<prefix>.strict_one2one_RBH.txt` and
  `<prefix>.strict_one2one_RBH.evidence.tsv`: RBH-derived putative one-to-one
  relationships where both sides have degree 1 in the RBH-only gene graph.

The `--primary-output` or `output.primary_output` setting controls which `.txt`
file is registered in `info.csv`. The default is `all_putative`, preserving
TACMAN support for one-to-many and many-to-many relationships.

Evidence TSVs include:

```text
ref_gene
que_gene
homology_type
support
relationship_cardinality
classification_method
rbh_level
best_ref_protein
best_que_protein
best_bitscore
best_evalue
best_pident
best_qcovs
best_scov
gene_pair_best_ref_protein
gene_pair_best_que_protein
gene_pair_isoform_hit_count
gene_level_tie_status
```

`homology_type` is the TACMAN-compatible relationship label:

- `ortholog_one2one`
- `ortholog_one2many`
- `ortholog_many2many`

`support` is independent evidence support and is only:

- `RBH`
- `topN_nonreciprocal`

`classification_method` is `gene_component`. `relationship_cardinality` maps to
TACMAN labels as:

- `1:1` -> `ortholog_one2one`
- `1:n` -> `ortholog_one2many`
- `n:1` -> `ortholog_one2many`
- `n:m` -> `ortholog_many2many`

TACMAN does not use a separate `ortholog_many2one` label, so `n:1` is mapped to
`ortholog_one2many` for compatibility.

## Method Notes

Protein FASTA files are used for BLASTP. FASTQ files are not required for this
step.

The candidate BLAST hit filters are `evalue <= 1e-5`, `pident >= 30`, and
`qcovs >= 50` by default. These are hit-filtering thresholds, not definitions
of RBH or one-to-one relationships. Subject coverage filtering is optional via
`--min-scov` or `blast.min_scov`; it is not applied by default.

Two RBH levels are supported:

- `--rbh-level protein`: protein-level RBH, defined as reciprocal best protein
  matches. This preserves the earlier behavior and is useful for comparison.
- `--rbh-level gene`: gene-level RBH, defined as reciprocal best gene matches
  after collapsing isoform-level protein hits to gene-pair candidates. This is
  the default and is recommended for complete eukaryotic proteomes because
  multiple protein isoforms often create exact tied protein-level best hits.

Protein-level RBH is defined after candidate filtering:

- the two proteins pass the candidate thresholds;
- they are reciprocal unique best hits in the two BLAST directions;
- best hits are ranked by highest bitscore, lowest evalue, highest pident, then
  highest qcovs;
- exact ties across all ranking fields are controlled by
  `--best-hit-tie-policy` or `blast.best_hit_tie_policy`.

Gene-level RBH first maps protein IDs to genes, collapses all protein-protein
hits for a `ref_gene` and `que_gene` pair, keeps the best representative
protein evidence for that gene pair, and then applies reciprocal best-hit
selection between genes. Gene-level ties are evaluated between different
partner genes. Multiple tied protein isoforms that map to the same partner gene
do not create a gene-level tie.

The default tie policy is `discard`, which excludes exact tied best hits from
RBH evidence. In gene mode, this discards true cross-gene ties but keeps
isoform ties that collapse to the same gene. Use `first` only when
deterministic tie breaking by partner ID is acceptable.

`ortholog_one2one` is a TACMAN-compatible relationship label, not an RBH label.
Broad all-putative output can contain `RBH + ortholog_one2many` or
`RBH + ortholog_many2many` when an RBH-supported gene pair belongs to a larger
gene component. It can also contain nonreciprocal `ortholog_one2one` sequence
similarity candidates; these should not be described as RBH orthologs. Only the
strict file should be described as RBH-derived putative one-to-one
relationships.

BLAST-derived relationships are putative sequence-similarity relationships and
are not curated phylogeny-based orthology annotations.

Stable gene IDs such as `WBGene...` and `FBgn...` can be used with
`gene_id_type: stable_id` or `--gene-id-type stable_id`, but identifiers in the
homology file must match `AnnData.var_names`.

## Validate The Generated Homology File

```bash
python scripts/validate_homology.py \
  --homology TACMAN/homo/<reference_species>_to_<query_species>.txt \
  --sp-ref <reference_species> \
  --sp-que <query_species> \
  --ref-h5ad data/<reference_species>/counts.h5ad \
  --que-h5ad data/<query_species>/counts.h5ad \
  --outdir TACMAN/homo/qc_<reference_species>_to_<query_species>
```

The validator checks TACMAN column names and homology-type counts. When h5ad
files are supplied and `scanpy` or `anndata` is installed, it also reports the
overlap between homology identifiers and `adata.var_names`.

## Tests

The parser-only test does not require BLAST+:

```bash
python scripts/test_prepare_homology_blast_parser.py
```

The configuration tests create tiny artificial input files and do not run the
complete C. elegans to D. melanogaster BLAST job:

```bash
python scripts/test_prepare_homology_config.py
```

The real Ensembl small BLAST smoke test uses Ensembl protein FASTA files, not
FASTQ files:

```bash
python scripts/test_prepare_homology_blast_real_ensembl.py
```

For final analyses, users are recommended to provide curated
protein-to-gene-symbol mapping tables, for example exported from Ensembl
BioMart, and verify consistency with AnnData `var_names`.

## Troubleshooting

If `--config` fails with a PyYAML error, install PyYAML:

```bash
pip install pyyaml
```

or:

```bash
conda install pyyaml
```

If BLAST+ executables are not found, install NCBI BLAST+ or provide executable
paths with `--makeblastdb-bin` and `--blastp-bin`.

If `makeblastdb` fails because of BLAST+ version-5 LMDB virtual-memory limits,
try a higher-memory node or set `blast.blastdb_version: 4` when supported by
your BLAST+ installation.

If many protein IDs cannot be mapped, confirm that FASTA IDs use the first token
after `>` and match the mapping table's `protein_id` column. For header parsing,
verify that the regex has one capture group and matches the FASTA headers.
