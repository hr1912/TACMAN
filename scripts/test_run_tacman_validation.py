#!/usr/bin/env python3
"""Validation-focused tests for scripts/run_tacman.py.

These tests do not train TACMAN. They use fake AnnData-like objects and a mock
TACMAN module where a run call is required.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import run_tacman as rt


class FakeObs:
    """Minimal obs object."""

    def __init__(self, data):
        self.data = data
        self.columns = list(data)

    def __getitem__(self, key):
        return self.data[key]


class FakeSparse:
    """Sparse-like matrix exposing .data, .shape, and .dtype."""

    def __init__(self, data, shape, dtype="int64"):
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tocsr(self):
        return self


class TrackingSparseData:
    """Large data-like object that records slicing without full materialization."""

    def __init__(self, size):
        self.size = size
        self.last_slice = None
        self.iterated = False

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise AssertionError("Sparse data should be sliced, not indexed one value at a time.")
        self.last_slice = item
        stop = min(item.stop or self.size, self.size)
        start = item.start or 0
        return range(start, stop)

    def __iter__(self):
        self.iterated = True
        raise AssertionError("Sparse data should not be fully iterated before sampling.")


class FakeAdata:
    """Minimal AnnData-like object."""

    def __init__(self, genes, labels=None, x=None):
        labels = labels or ["A", "B"]
        self.var_names = genes
        self.n_obs = len(labels)
        self.n_vars = len(genes)
        self.shape = (self.n_obs, self.n_vars)
        self.obs = FakeObs({"cell_type": labels})
        self.X = x if x is not None else [[1, 0], [0, 2]]


def write(path: Path, text: str) -> None:
    """Write text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def base_fixture(tmp: Path, overwrite: bool = False) -> Path:
    """Create a tiny config fixture."""
    write(tmp / "data" / "ref.h5ad", "placeholder")
    write(tmp / "data" / "que.h5ad", "placeholder")
    write(
        tmp / "homo" / "human_to_mouse.txt",
        "Gene name,Mouse gene name,Mouse homology type\nG1,M1,ortholog_one2one\nG2,M2,ortholog_one2many\n",
    )
    write(tmp / "homo" / "info.csv", "path,name,sp_ref,sp_que\nhuman_to_mouse.txt,human_to_mouse,human,mouse\n")
    config = tmp / "configs" / "pancreas_tacman.yaml"
    write(
        config,
        f"""reference:
  h5ad: ../data/ref.h5ad
  species: human
  cell_type_key: cell_type
query:
  h5ad: ../data/que.h5ad
  species: mouse
homology:
  info_csv: ../homo/info.csv
  path: null
  is_1v1: false
analysis:
  tissue: pancreas
  aligned: true
  stages: [100, 100, 100]
output:
  directory: ../output
  tag: pan_h-map-pan_m
  overwrite: {'true' if overwrite else 'false'}
runtime:
  seed: 7
  log_level: INFO
""",
    )
    return config


def parse_config(path: Path):
    """Parse config."""
    args = rt.build_parser().parse_args(["--config", str(path)])
    return rt.effective_config(args)


def patch_loader(ref=None, que=None):
    """Patch h5ad loader."""
    ref = ref or FakeAdata(["G1", "G2"], ["alpha", "beta"])
    que = que or FakeAdata(["M1", "M2"], ["alpha", "beta"])

    def fake_load(path: Path):
        return ref if "ref" in str(path) else que

    rt.load_h5ad = fake_load


def install_mock_tacman(captured=None):
    """Install a mock TACMAN module."""
    captured = captured if captured is not None else {}
    module = types.ModuleType("TACMAN")

    def fake_run(**params):
        captured.update(params)
        return None

    module.run = fake_run
    sys.modules["TACMAN"] = module
    return captured


def test_seed_called(tmp: Path) -> None:
    """set_random_seed is called and summary records seed."""
    patch_loader()
    calls = []
    original = rt.set_random_seed

    def fake_seed(seed, tacman_module=None):
        calls.append((seed, tacman_module is not None))
        return {"seed": seed, "applied": ["mock_seed"]}

    rt.set_random_seed = fake_seed
    try:
        summary = rt.run_tacman(parse_config(base_fixture(tmp)), validate_only=True)
    finally:
        rt.set_random_seed = original
    assert calls == [(7, False)]
    assert summary["seed"] == 7
    assert summary["seed applied by"] == "mock_seed"


def test_overwrite_behaviour(tmp: Path) -> None:
    """Output conflict handling and backup semantics."""
    patch_loader()
    cfg = parse_config(base_fixture(tmp / "fresh"))
    summary = rt.run_tacman(cfg, validate_only=True)
    assert summary["output overwrite action"] == "no existing non-empty output directory"

    conflict = tmp / "conflict"
    cfg = parse_config(base_fixture(conflict))
    final_dir = rt.final_output_dir(cfg)
    write(final_dir / "old.txt", "old")
    try:
        rt.run_tacman(cfg, validate_only=True)
    except FileExistsError as exc:
        assert "already exists and is not empty" in str(exc)
    else:
        raise AssertionError("Expected output conflict error")

    backup_case = tmp / "backup"
    cfg = parse_config(base_fixture(backup_case, overwrite=True))
    final_dir = rt.final_output_dir(cfg)
    write(final_dir / "old.txt", "old")
    install_mock_tacman()
    summary = rt.run_tacman(cfg, validate_only=False)
    backup = Path(summary["output backup directory"])
    assert backup.exists()
    assert backup.name.startswith(final_dir.name + ".backup_")


def test_matrix_inspection() -> None:
    """Inspect sparse, dense integer, log-like, negative, and NaN matrices."""
    sparse_stats = rt.inspect_expression_matrix(FakeAdata(["G1", "G2"], x=FakeSparse([1, 2, 3], (2, 2))))
    assert sparse_stats["matrix storage"] == "sparse"
    assert sparse_stats["raw-count assessment"] == "likely raw counts"

    tracking_data = TrackingSparseData(1_000_000)
    storage, dtype, shape, sampled = rt.matrix_storage_and_values(
        FakeSparse(tracking_data, (1000, 1000)),
        sample_size=17,
    )
    assert storage == "sparse"
    assert tracking_data.last_slice.stop == 17
    assert tracking_data.iterated is False
    assert len(sampled) == 16

    dense_float = rt.inspect_expression_matrix(FakeAdata(["G1", "G2"], x=[[1.0, 0.0], [2.0, 3.0]]))
    assert dense_float["integer-like fraction"] == 1.0
    assert dense_float["raw-count assessment"] == "likely raw counts"

    log_like = rt.inspect_expression_matrix(FakeAdata(["G1", "G2"], x=[[0.1, 0.0], [1.7, 2.2]]))
    assert "WARNING" in log_like["raw-count assessment"]

    try:
        rt.check_expression_matrix_or_raise(FakeAdata(["G1"], x=[[-1.0]]), "reference")
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("Expected negative matrix error")

    try:
        rt.check_expression_matrix_or_raise(FakeAdata(["G1"], x=[[float("nan")]]), "reference")
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("Expected NaN matrix error")


def test_validate_only_summary(tmp: Path) -> None:
    """Validate-only writes summary and does not call TACMAN.run."""
    patch_loader()
    summary = rt.run_tacman(parse_config(base_fixture(tmp)), validate_only=True)
    assert summary["TACMAN.run called"] is False
    assert "Validation completed successfully" in summary["validation status"]
    assert "reference integer-like fraction" in summary
    assert "is_1v1 expected homology pairs" in summary


def test_yaml_fallback(tmp: Path) -> None:
    """Fallback parser handles the basic template and rejects advanced YAML."""
    basic = rt.SimpleYaml.safe_load("analysis:\n  stages: [100, 100, 100]\n")
    assert basic["analysis"]["stages"] == [100, 100, 100]
    try:
        rt.SimpleYaml.safe_load("a: &anchor 1\nb: *anchor\n")
    except ValueError as exc:
        assert "anchors" in str(exc)
    else:
        raise AssertionError("Expected unsupported YAML feature error")
    if importlib.util.find_spec("yaml") is not None:
        loaded = rt.load_yaml(base_fixture(tmp))
        assert loaded["reference"]["species"] == "human"


def test_mock_run_param_regression(tmp: Path) -> None:
    """Mock run still receives legacy TACMAN.run parameters."""
    patch_loader()
    captured = install_mock_tacman()
    summary = rt.run_tacman(parse_config(base_fixture(tmp, overwrite=True)), validate_only=False)
    assert summary["TACMAN.run called"] is True
    for key in ["adata_ref", "adata_que", "key_cell_type", "sp_ref", "sp_que", "p_homo", "tissue", "aligned", "p_output", "tag_output", "stages", "is_1v1"]:
        assert key in captured


def test_optional_real_validate_only_integration(tmp: Path) -> None:
    """If anndata and TACMAN are importable, run validate-only on tiny h5ad files."""
    try:
        tacman_spec = importlib.util.find_spec("TACMAN")
    except ValueError:
        tacman_spec = None
    if importlib.util.find_spec("anndata") is None or tacman_spec is None:
        print("SKIP real validate-only integration: anndata or TACMAN not importable")
        return
    import anndata as ad
    import numpy as np
    import pandas as pd

    root = tmp / "real"
    config = base_fixture(root)
    ref = ad.AnnData(np.array([[1, 0], [0, 2]], dtype=float), obs=pd.DataFrame({"cell_type": ["A", "B"]}), var=pd.DataFrame(index=["G1", "G2"]))
    que = ad.AnnData(np.array([[1, 0], [0, 2]], dtype=float), obs=pd.DataFrame({"cell_type": ["A", "B"]}), var=pd.DataFrame(index=["M1", "M2"]))
    ref.write_h5ad(root / "data" / "ref.h5ad")
    que.write_h5ad(root / "data" / "que.h5ad")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_tacman.py")), "--config", str(config), "--validate-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Validation completed successfully." in result.stdout


def main() -> None:
    """Run validation tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_seed_called(root / "seed")
        test_overwrite_behaviour(root / "overwrite")
        test_matrix_inspection()
        test_validate_only_summary(root / "validate")
        test_yaml_fallback(root / "yaml")
        test_mock_run_param_regression(root / "mock")
        test_optional_real_validate_only_integration(root / "integration")
    print("PASS")


if __name__ == "__main__":
    main()
