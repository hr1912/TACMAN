import json
import argparse
from pathlib import Path


import pandas as pd

MAP_DB2URL = dict(
    HCL="https://bis.zju.edu.cn/HCL/data/DGE/USE_TAR/{}.tar",
    MCA="https://bis.zju.edu.cn/MCA/data/dge/{}.zip",
)


def load_df_HCL(p_info: Path):
    data = json.loads(p_info.read_text())
    data = data["tissueinfo"]
    df = pd.DataFrame(list(data.values()))
    names = pd.Series(list(data.keys()))
    df.columns = [
        "name",
        "tissue",
        "count",
        "donor_id",
        "age",
        "sex",
        "source",
        "sample_loction",
        "database",
    ]
    assert names.is_unique
    df.index = names
    df["name"] = names
    return df


def load_df_MCA(p_info: Path):
    data = json.loads(p_info.read_text())
    data = data["tissueinfo"]
    df = pd.DataFrame(list(data.values()))
    names = pd.Series(list(data.keys()))
    df = df.iloc[:, :4]
    df.columns = [
        "name",
        "tissue",
        "count",
        "age",
    ]

    assert names.is_unique
    df.index = names
    df["name"] = names
    return df


def extract_urls(p_info: str, db: str, tissue: str, p_out=None) -> None:
    # processing path
    p_info = Path(p_info)
    assert p_info.exists()
    if p_out is None:
        p_out = Path("./urls")
    else:
        p_out = Path(p_out)

    df = None
    if db == "HCL":
        df = load_df_HCL(p_info)
    elif db == "MCA":
        df = load_df_MCA(p_info)
    else:
        raise RuntimeError("can not laod df with db = '{}'".format(db))
    df = df[df["tissue"] == tissue]

    assert df.shape[0], "can not find tissue = '{}'".format(tissue)

    url_formater = MAP_DB2URL[db]
    p_out.write_text("\n".join([url_formater.format(name) for name in df.index]))
    print("[out] {}".format(p_out))


pass

Parser = argparse.ArgumentParser(description="extract urls")
Parser.add_argument("-i", "--info", help="path of info json", required=True)
Parser.add_argument(
    "-d", "--db", help="database type, must be one of HCL or MCA", required=True
)
Parser.add_argument("-t", "--tissue", required=True, help="specify tissue")
Parser.add_argument("-o", "--out", required=False, default=None, help="path of urls")

if __name__ == "__main__":
    args = Parser.parse_args()
    extract_urls(p_info=args.info, db=args.db, tissue=args.tissue, p_out=args.out)
    print("\n[finish]\n".center(100, "-"))
