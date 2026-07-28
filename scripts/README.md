# TACMAN homology file preparation

The helper scripts in `scripts/` prepare external homology tables for TACMAN
without modifying TACMAN core source code. TACMAN discovers homology files from
`TACMAN/homo/info.csv`, so adding a new species pair only requires a
TACMAN-style three-column txt file and one `info.csv` row.

## Prepare a database-derived homology file

```bash
python scripts/prepare_homology.py \
  --mode database \
  --input data/homology/rat_mouse_database.tsv \
  --sp-ref rat \
  --sp-que mouse \
  --ref-gene-column rat_gene_symbol \
  --que-gene-column mouse_gene_symbol \
  --homology-type-column mouse_homology_type \
  --out TACMAN/homo/rat_to_mouse.txt \
  --register-info TACMAN/homo/info.csv
```

This writes `TACMAN/homo/rat_to_mouse.txt` with TACMAN column names:
`Gene name`, `Mouse gene name`, and `Mouse homology type`. It removes missing
values and duplicate rows, maps common database labels such as `one2one` or
`1:1` to TACMAN-style labels such as `ortholog_one2one`, writes a QC report
beside the output file, and registers `rat_to_mouse.txt,rat_to_mouse,rat,mouse`
in `TACMAN/homo/info.csv`.

By default, the script stops if `info.csv` already contains the same
`sp_ref`/`sp_que` pair. Use `--overwrite-info` to replace the existing record,
or `--skip-register-if-exists` to keep `info.csv` unchanged while still writing
the output file. Existing `info.csv` files are backed up to `info.csv.bak`
before updates unless `--no-backup-info` is supplied.

If the query species label in TACMAN's column names needs manual control, add
`--query-display-name Mouse`. If the input homology-type labels are custom,
provide a two-column mapping table with `--homology-type-map`.

TACMAN's existing homology files use gene symbols rather than Ensembl,
protein, or transcript IDs. The preparation script performs a basic sanity
check for common ID-like values and asks for symbol columns if they are found.
Use `--allow-non-symbol-ids` only when those IDs are also the exact
`adata.var_names` values used in TACMAN.

## Validate a TACMAN homology file

```bash
python scripts/validate_homology.py \
  --homology TACMAN/homo/rat_to_mouse.txt \
  --sp-ref rat \
  --sp-que mouse \
  --ref-h5ad data/rat/counts.h5ad \
  --que-h5ad data/mouse/counts.h5ad \
  --outdir TACMAN/homo/qc_rat_to_mouse
```

The validator checks the expected TACMAN columns, reports homology type counts,
and, when `scanpy` or `anndata` is installed and h5ad files are supplied,
summarizes overlap between homology gene symbols and `adata.var_names`. It
writes `homology_validation_report.txt`, `unmatched_ref_genes.txt`, and
`unmatched_query_genes.txt` in the requested output directory. Without h5ad
inputs or AnnData readers, it still performs the basic format and homology-type
checks.
