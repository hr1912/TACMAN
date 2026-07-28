#!/usr/bin/env python3
"""Unit tests for summarize_tacman_results.py.

The tests use small synthetic TACMAN-like obs.csv files and never train TACMAN.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional


SCRIPT = Path(__file__).resolve().parent / "summarize_tacman_results.py"
spec = importlib.util.spec_from_file_location("summarize_tacman_results", SCRIPT)
summ = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summ)

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


def make_obs_dataframe(with_true: bool = True, with_umap: bool = True):
    """Create a tiny TACMAN-like result table."""
    rows = [
        {"cell": "h1", "cell_type": "alpha", "type": "ref", "species": "human", "true_label": "alpha", "pre_label": "", "max_prob": "", "is_right": ""},
        {"cell": "h2", "cell_type": "beta", "type": "ref", "species": "human", "true_label": "beta", "pre_label": "", "max_prob": "", "is_right": ""},
        {"cell": "m1", "cell_type": "alpha", "type": "que", "species": "mouse", "true_label": "alpha", "pre_label": "alpha", "max_prob": 0.91, "is_right": True},
        {"cell": "m2", "cell_type": "beta", "type": "que", "species": "mouse", "true_label": "beta", "pre_label": "alpha", "max_prob": 0.62, "is_right": False},
        {"cell": "m3", "cell_type": "gamma", "type": "que", "species": "mouse", "true_label": "gamma", "pre_label": "delta", "max_prob": 0.55, "is_right": False},
    ]
    df = pd.DataFrame(rows).set_index("cell")
    if not with_true:
        df["true_label"] = ""
    if with_umap:
        df["UMAP1"] = [0.0, 1.0, 0.1, 1.1, 2.0]
        df["UMAP2"] = [0.0, 0.5, 0.2, 0.7, 1.5]
    return df


def write_obs(path: Path, with_true: bool = True, with_umap: bool = True) -> None:
    """Write a tiny TACMAN-like obs.csv."""
    make_obs_dataframe(with_true=with_true, with_umap=with_umap).to_csv(path)


def write_config(
    path: Path,
    tacman_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    prediction_table: Optional[Path] = None,
    evaluate_query_labels: bool = True,
) -> None:
    """Write a minimal config using paths relative to the config file."""
    rel_in = os.path.relpath(tacman_dir, path.parent)
    rel_out = os.path.relpath(output_dir, path.parent)
    pred = "null" if prediction_table is None else os.path.relpath(prediction_table, path.parent)
    path.write_text(
        f"""input:
  tacman_output_directory: {rel_in}
  prediction_table: {pred}
metadata:
  species_key: species
  reference_or_query_key: type
  reference_value: ref
  query_value: que
  true_label_key: true_label
  predicted_label_key: pre_label
  model_label_key: model_label
  confidence_key: max_prob
  umap1_key: UMAP1
  umap2_key: UMAP2
analysis:
  evaluate_query_labels: {'true' if evaluate_query_labels else 'false'}
  normalize_confusion_matrix: true
  allow_umap_recompute: false
figures:
  format: pdf
  dpi: 72
  width: 4
  height: 3
  point_size: 8
output:
  directory: {rel_out}
  prefix: pancreas_tacman
  overwrite: {'true' if overwrite else 'false'}
"""
    )


@unittest.skipIf(pd is None, "pandas is required for summary tests")
class SummarizeTacmanResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tacman_summary_test_"))
        self.config_dir = self.tmp / "configs"
        self.config_dir.mkdir()
        self.tacman_dir = self.tmp / "output" / "pancreas;human-corss-mouse;pan_h-map-pan_m;aligned=True"
        self.tacman_dir.mkdir(parents=True)
        self.out_dir = self.tmp / "summary"
        write_obs(self.tacman_dir / "obs.csv")
        self.config = self.config_dir / "pancreas_results.yaml"
        write_config(self.config, self.tacman_dir, self.out_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def load_df(self):
        cfg = summ.load_config(self.config)
        df, table = summ.read_result_dataframe(cfg)
        return cfg, df, table

    def test_yaml_reading_and_relative_paths_without_removed_fields(self) -> None:
        cfg = summ.load_config(self.config)
        self.assertEqual(Path(cfg["input"]["tacman_output_directory"]), self.tacman_dir.resolve())
        self.assertEqual(Path(cfg["output"]["directory"]), self.out_dir.resolve())
        self.assertNotIn("result_h5ad", cfg["input"])
        self.assertNotIn("reference_species", cfg["metadata"])
        self.assertNotIn("query_species", cfg["metadata"])
        self.assertNotIn("embedding_key", cfg["metadata"])
        self.assertNotIn("confidence_threshold", cfg["analysis"])
        self.assertNotIn("min_cells_per_class", cfg["analysis"])

    def test_explicit_csv_prediction_table(self) -> None:
        table = self.tmp / "manual_predictions.csv"
        make_obs_dataframe().to_csv(table)
        write_config(self.config, self.tacman_dir, self.out_dir, prediction_table=table)
        cfg, df, path = self.load_df()
        self.assertEqual(path, table.resolve())
        self.assertEqual(df.shape[0], 5)

    def test_explicit_tsv_prediction_table(self) -> None:
        table = self.tmp / "manual_predictions.tsv"
        make_obs_dataframe().to_csv(table, sep="\t")
        write_config(self.config, self.tacman_dir, self.out_dir, prediction_table=table)
        cfg, df, path = self.load_df()
        self.assertEqual(path, table.resolve())
        self.assertEqual(df.shape[0], 5)
        self.assertEqual(summ.result_table_separator(path), "\t")

    def test_output_directory_conflict_and_backup_for_full_run(self) -> None:
        self.out_dir.mkdir()
        (self.out_dir / "old.txt").write_text("old")
        cfg = summ.load_config(self.config)
        with self.assertRaises(summ.ValidationError):
            summ.prepare_output_dir(Path(cfg["output"]["directory"]), False)
        backup = summ.prepare_output_dir(Path(cfg["output"]["directory"]), True)
        self.assertTrue(backup is not None and backup.exists())
        self.assertTrue(self.out_dir.exists())

    def test_validate_only_does_not_backup_existing_output(self) -> None:
        self.out_dir.mkdir()
        old = self.out_dir / "old.txt"
        old.write_text("old")
        cfg = summ.load_config(self.config)
        code = summ.run_summary(cfg, validate_only=True)
        self.assertEqual(code, 0)
        self.assertTrue(old.exists())
        self.assertEqual(list(self.tmp.glob("summary.backup_*")), [])

    def test_reference_confidence_and_prediction_missing_are_allowed(self) -> None:
        cfg, df, _ = self.load_df()
        warnings = summ.validate_metadata(df, cfg, summ.get_embedding(df, cfg)[0])
        self.assertIsInstance(warnings, list)

    def test_query_confidence_missing_is_error(self) -> None:
        df = make_obs_dataframe()
        df.loc["m2", "max_prob"] = ""
        df.to_csv(self.tacman_dir / "obs.csv")
        cfg, df, _ = self.load_df()
        with self.assertRaises(summ.ValidationError):
            summ.validate_metadata(df, cfg, summ.get_embedding(df, cfg)[0])

    def test_query_predicted_label_missing_is_error(self) -> None:
        df = make_obs_dataframe()
        df.loc["m2", "pre_label"] = ""
        df.to_csv(self.tacman_dir / "obs.csv")
        cfg, df, _ = self.load_df()
        with self.assertRaises(summ.ValidationError):
            summ.validate_metadata(df, cfg, summ.get_embedding(df, cfg)[0])

    def test_integrated_model_label_generated_correctly(self) -> None:
        cfg, df, _ = self.load_df()
        self.assertEqual(df.loc["h1", "model_label"], "alpha")
        self.assertEqual(df.loc["m2", "model_label"], "alpha")
        self.assertEqual(df.loc["m3", "model_label"], "delta")

    def test_cell_annotations_and_query_predictions(self) -> None:
        cfg, df, _ = self.load_df()
        annotations = summ.standardize_annotations(df, cfg, query_only=False)
        query = summ.standardize_annotations(df, cfg, query_only=True)
        self.assertEqual(annotations.shape[0], 5)
        self.assertEqual(query.shape[0], 3)
        self.assertTrue((query["reference_or_query"] == "que").all())

    def test_metrics_with_query_true_labels_and_unpredicted_class(self) -> None:
        cfg, df, _ = self.load_df()
        metrics, per_class, raw, norm = summ.compute_metrics(df, cfg)
        self.assertIsNotNone(metrics)
        self.assertIn("accuracy", metrics.columns)
        self.assertIn("gamma", raw.index)
        self.assertIn("delta", raw.columns)
        self.assertTrue((norm.sum(axis=1).round(6) <= 1.0).all())

    def test_no_query_true_labels_skips_supervised_outputs_but_keeps_confidence(self) -> None:
        write_obs(self.tacman_dir / "obs.csv", with_true=False)
        cfg = summ.load_config(self.config)
        code = summ.run_summary(cfg, validate_only=False)
        self.assertEqual(code, 0)
        self.assertTrue((self.out_dir / "pancreas_tacman.cell_annotations.tsv").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.query_predictions.tsv").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.confidence_by_predicted_label.tsv").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.annotation_metrics.tsv").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.confusion_matrix.tsv").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.confidence_by_true_predicted.tsv").exists())

    def test_confidence_summaries(self) -> None:
        cfg, df, _ = self.load_df()
        conf = summ.confidence_by_predicted_label(df, cfg)
        self.assertIn("predicted_label", conf.columns)
        self.assertIn("mean_confidence", conf.columns)
        self.assertIn("minimum", conf.columns)
        pair = summ.confidence_by_true_predicted(df, cfg)
        self.assertIn("true_label", pair.columns)
        self.assertIn("median_confidence", pair.columns)

    def test_embedding_missing_skips_umap(self) -> None:
        write_obs(self.tacman_dir / "obs.csv", with_umap=False)
        cfg, df, _ = self.load_df()
        coords, warning = summ.get_embedding(df, cfg)
        self.assertIsNone(coords)
        self.assertIn("UMAP figures skipped", warning)

    def test_validate_only_writes_no_analysis_tables(self) -> None:
        cfg = summ.load_config(self.config)
        code = summ.run_summary(cfg, validate_only=True)
        self.assertEqual(code, 0)
        self.assertTrue((self.out_dir / "pancreas_tacman.resolved_config.yaml").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.run_summary.txt").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.cell_annotations.tsv").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.query_predictions.tsv").exists())

    def test_full_run_tables_and_stable_names(self) -> None:
        cfg = summ.load_config(self.config)
        code = summ.run_summary(cfg, validate_only=False)
        self.assertEqual(code, 0)
        expected = [
            "pancreas_tacman.cell_annotations.tsv",
            "pancreas_tacman.query_predictions.tsv",
            "pancreas_tacman.annotation_metrics.tsv",
            "pancreas_tacman.per_class_metrics.tsv",
            "pancreas_tacman.confusion_matrix.tsv",
            "pancreas_tacman.confusion_matrix_normalized.tsv",
            "pancreas_tacman.cell_type_counts.tsv",
            "pancreas_tacman.confidence_by_predicted_label.tsv",
            "pancreas_tacman.confidence_by_true_predicted.tsv",
            "pancreas_tacman.resolved_config.yaml",
            "pancreas_tacman.run_summary.txt",
            "pancreas_tacman.log",
        ]
        for name in expected:
            self.assertTrue((self.out_dir / name).exists(), name)
        self.assertFalse((self.out_dir / "pancreas_tacman.predictions.tsv").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.confidence_summary.tsv").exists())

    def test_figure_generation_when_matplotlib_available(self) -> None:
        try:
            summ.require_matplotlib()
        except RuntimeError:
            self.skipTest("matplotlib is not installed")
        cfg = summ.load_config(self.config)
        summ.run_summary(cfg, validate_only=False)
        self.assertTrue((self.out_dir / "pancreas_tacman.umap_species.pdf").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.umap_true_labels.pdf").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.umap_integrated_labels.pdf").exists())
        self.assertFalse((self.out_dir / "pancreas_tacman.umap_predictions.pdf").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.confusion_matrix.pdf").exists())
        self.assertTrue((self.out_dir / "pancreas_tacman.prediction_confidence.pdf").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
