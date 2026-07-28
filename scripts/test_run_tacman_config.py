#!/usr/bin/env python3
"""Tests for the YAML-driven TACMAN runner.

These tests use fake AnnData-like objects and a mock TACMAN module. They do not
train a model.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import run_tacman as rt


class FakeObs:
    """Small obs-like object with columns and __getitem__."""

    def __init__(self, data):
        self.data = data
        self.columns = list(data)

    def __getitem__(self, key):
        return self.data[key]


class FakeAdata:
    """Small AnnData-like object for validation tests."""

    def __init__(self, genes, labels=None):
        labels = labels or ["A", "B"]
        self.var_names = genes
        self.n_obs = len(labels)
        self.n_vars = len(genes)
        self.shape = (self.n_obs, self.n_vars)
        self.obs = FakeObs({"cell_type": labels})
        self.X = [[1 for _ in genes] for _ in labels]


def write(path: Path, text: str) -> None:
    """Write text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def base_fixture(tmp: Path) -> Path:
    """Create a minimal TACMAN run fixture and return config path."""
    write(tmp / "data" / "ref.h5ad", "placeholder")
    write(tmp / "data" / "que.h5ad", "placeholder")
    write(
        tmp / "homo" / "human_to_mouse.txt",
        "Gene name,Mouse gene name,Mouse homology type\nG1,M1,ortholog_one2one\nG2,M2,ortholog_one2many\n",
    )
    write(
        tmp / "homo" / "info.csv",
        "path,name,sp_ref,sp_que\nhuman_to_mouse.txt,human_to_mouse,human,mouse\n",
    )
    config = tmp / "configs" / "pancreas_tacman.yaml"
    write(
        config,
        """reference:
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
  overwrite: false
runtime:
  seed: 0
  log_level: INFO
""",
    )
    return config


def patch_loader(ref=None, que=None):
    """Patch run_tacman.load_h5ad."""
    ref = ref or FakeAdata(["G1", "G2", "G3"], ["alpha", "beta"])
    que = que or FakeAdata(["M1", "M2", "M3"], ["unknown", "unknown"])

    def fake_load(path: Path):
        return ref if "ref" in str(path) else que

    rt.load_h5ad = fake_load


def parse_config(path: Path, extra=None):
    """Parse config with optional CLI-like overrides."""
    parser = rt.build_parser()
    argv = ["--config", str(path)]
    if extra:
        argv.extend(extra)
    args = parser.parse_args(argv)
    return rt.effective_config(args)


def test_yaml_relative_and_override(tmp: Path) -> None:
    """YAML loading, relative paths, and CLI override."""
    defaults = rt.default_config()
    assert list(defaults).count("analysis") == 1
    assert defaults["analysis"] == {"tissue": None, "aligned": True, "stages": [100, 200, 200]}
    config_path = base_fixture(tmp)
    cfg = parse_config(config_path, ["--tag", "override", "--stages", "1,2,3", "--aligned", "false"])
    assert Path(rt.nested_get(cfg, ("reference", "h5ad"))) == (tmp / "data" / "ref.h5ad").resolve()
    assert rt.nested_get(cfg, ("output", "tag")) == "override"
    assert rt.nested_get(cfg, ("analysis", "stages")) == [1, 2, 3]
    assert rt.nested_get(cfg, ("analysis", "aligned")) is False


def test_get_homo_path(tmp: Path) -> None:
    """Unique, zero, and multiple homology matches."""
    config_path = base_fixture(tmp)
    info = tmp / "homo" / "info.csv"
    assert rt.get_homo_path("human", "mouse", info).name == "human_to_mouse.txt"
    try:
        rt.get_homo_path("rat", "mouse", info)
    except ValueError as exc:
        assert "matched 0 rows" in str(exc)
    else:
        raise AssertionError("Expected zero-match error")

    write(
        info,
        "path,name,sp_ref,sp_que\nhuman_to_mouse.txt,a,human,mouse\nhuman_to_mouse.txt,b,human,mouse\n",
    )
    try:
        rt.get_homo_path("human", "mouse", info)
    except ValueError as exc:
        assert "matched 2 rows" in str(exc)
    else:
        raise AssertionError("Expected multi-match error")
    assert config_path.exists()


def test_validation_and_validate_only(tmp: Path) -> None:
    """Validation summary and validate-only path."""
    patch_loader()
    cfg = parse_config(base_fixture(tmp))
    summary = rt.run_tacman(cfg, validate_only=True)
    assert summary["reference cells"] == 2
    assert summary["query cells"] == 2
    assert summary["homology pairs"] == 2
    assert summary["reference gene overlap"] == 2
    assert summary["query gene overlap"] == 2
    assert summary["one-to-one pair count"] == 1
    assert summary["TACMAN.run called"] is False
    assert (tmp / "output" / "pan_h-map-pan_m.run_summary.txt").exists()


def test_metadata_missing(tmp: Path) -> None:
    """Missing reference metadata key should fail before model run."""
    ref = FakeAdata(["G1", "G2"], ["A", "B"])
    ref.obs = FakeObs({"wrong": ["A", "B"]})
    patch_loader(ref=ref)
    cfg = parse_config(base_fixture(tmp))
    try:
        rt.run_tacman(cfg, validate_only=True)
    except ValueError as exc:
        assert "reference.cell_type_key" in str(exc)
    else:
        raise AssertionError("Expected missing metadata error")


def test_mock_tacman_run_params(tmp: Path) -> None:
    """Mock TACMAN.run and confirm wrapper parameter passing."""
    patch_loader()
    cfg = parse_config(base_fixture(tmp), ["--is-1v1", "true", "--output-dir", str(tmp / "custom_out")])
    captured = {}

    module = types.ModuleType("TACMAN")

    def fake_run(**params):
        captured.update(params)
        return None

    module.run = fake_run
    sys.modules["TACMAN"] = module
    summary = rt.run_tacman(cfg, validate_only=False)
    assert summary["TACMAN.run called"] is True
    assert captured["sp_ref"] == "human"
    assert captured["sp_que"] == "mouse"
    assert captured["key_cell_type"] == "cell_type"
    assert captured["stages"] == [100, 100, 100]
    assert captured["is_1v1"] is True
    assert Path(captured["p_homo"]).name == "human_to_mouse.txt"
    assert Path(captured["p_output"]) == (tmp / "custom_out").resolve()


def main() -> None:
    """Run tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_yaml_relative_and_override(root / "relative")
        test_get_homo_path(root / "homo")
        test_validation_and_validate_only(root / "validate")
        test_metadata_missing(root / "metadata")
        test_mock_tacman_run_params(root / "mock")
    print("PASS")


if __name__ == "__main__":
    main()
