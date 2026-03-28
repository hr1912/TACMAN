import zipfile
import tarfile
import shutil
import tempfile
import argparse
from pathlib import Path

import scipy
import pandas as pd
import scanpy as sc


def unzip(p_data: Path, p_tempdir: Path):
    with zipfile.ZipFile(p_data, "r") as zip_ref:
        zip_ref.extractall(p_tempdir)


def untar(p_data: Path, p_tempdir: Path):
    with tarfile.open(p_data, "r") as tar_ref:
        tar_ref.extractall(p_tempdir)


def load_adata_MCA(row):
    adata = sc.read_csv(row.dge).T
    df_gene = pd.read_csv(row.gene, index_col=0).set_index("x")

    df_obs = pd.read_csv(row.barcodes_anno, index_col=0)
    df_obs.columns = ["source_cell_type", "source_cluster"]

    adata.var = df_gene
    adata.obs = df_obs
    return adata


def load_adata_HCL(row):
    adata = sc.read_csv(row.dge).T
    df_obs = pd.read_csv(row.barcodes_anno, index_col=0)
    df_obs.columns = ["source_cluster", "tissue", "source_cell_type"]
    df_obs = df_obs.drop(columns="tissue")
    adata.obs = df_obs
    return adata


def save_adata(adata, p_save_dir):
    p_save_dir.mkdir(exist_ok=True, parents=True)
    ##################################################
    # mtx
    ##################################################

    X = adata.X
    if not scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(X)
    X = X.T.tocoo()
    scipy.io.mmwrite(p_save_dir.joinpath("matrix.mtx"), X)
    ##################################################
    # genes
    ##################################################
    df_genes = pd.DataFrame(
        {
            "gene_ids": (
                adata.var_names if adata.var_names is not None else adata.var.index
            ),
            "gene_names": (
                adata.var_names if adata.var_names is not None else adata.var.index
            ),
        }
    )
    df_genes.to_csv(
        p_save_dir.joinpath("genes.tsv"), sep="\t", header=False, index=False
    )

    ##################################################
    # barcodes
    ##################################################
    df_barcodes = pd.DataFrame(
        {
            "barcodes": (
                adata.obs_names if adata.obs_names is not None else adata.obs.index
            )
        }
    )
    df_barcodes.to_csv(
        p_save_dir.joinpath("barcodes.tsv"), sep="\t", header=False, index=False
    )

    if not adata.obs.empty:
        adata.obs.to_csv(p_save_dir.joinpath("obs.tsv"), sep="\t", index=True)

    print("[save mtx] {}".format(p_save_dir))


MAP_DB2FUN = dict(
    MCA=unzip,
    HCL=untar,
)
MAP_DB2FUN_LOAD_ADATA = dict(
    MCA=load_adata_MCA,
    HCL=load_adata_HCL,
)


def merge_adatas_to_mtx(p_data_dir: Path, db: str, tissue: str, p_out_dir: Path):

    p_datas = [
        p
        for p in p_data_dir.iterdir()
        if p.match("*{}*".format(tissue)) and p.suffix in [".zip", ".tar"]
    ]

    p_tempdir = Path(tempfile.mkdtemp(prefix="scRNA_decompress"))
    for p_data in p_datas:
        MAP_DB2FUN[db](p_data, p_tempdir)
    df_path = pd.DataFrame({"path": [p for p in p_tempdir.iterdir()]})
    df_path["stem"] = df_path["path"].apply(lambda x: x.stem)
    df_path = df_path.join(
        df_path["stem"].str.extract(
            "(?P<name>.+)_(?P<type>markers|dge|gene|barcodes_anno|Anno)"
        )
    )
    df_path["type"] = df_path["type"].map(
        lambda k: {"Anno": "barcodes_anno"}.setdefault(k, k)
    )
    df_path = df_path.pivot(index="name", columns="type", values="path")
    adatas = dict()
    for row in df_path.itertuples():
        ad = MAP_DB2FUN_LOAD_ADATA[db](row)
        ad.obs["batch"] = row.Index
        adatas[row.Index] = ad

    adata = sc.concat(adatas, index_unique="|")
    shutil.rmtree(p_tempdir)

    p_save_dir = p_out_dir.joinpath("{}_{}".format(db, tissue))
    save_adata(adata, p_save_dir)


pass
Parser = argparse.ArgumentParser(description="merge_adatas")
Parser.add_argument("--data", help="path dir of data", required=True)
Parser.add_argument("-t", "--tissue", required=True, help="specify tissue")
Parser.add_argument(
    "-d", "--db", help="database type, must be one of HCL or MCA", required=True
)
Parser.add_argument(
    "-o", "--out", required=False, default=None, help="path dir of output"
)

if __name__ == "__main__":

    args = Parser.parse_args()
    p_data_dir = Path(args.data)
    db = args.db
    tissue = args.tissue
    p_out_dir = Path(args.out)

    # p_data_dir = Path("data/deg/HCL")
    # tissue = "Spleen"
    # db = "HCL"
    # p_out_dir = Path("data/mtx")
    merge_adatas_to_mtx(p_data_dir, db, tissue, p_out_dir)
    print("\n[finish]\n".center(100, "-"))
pass
