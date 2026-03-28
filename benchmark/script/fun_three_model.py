from pathlib import Path
import os
import sys
from zipfile import ZipFile
import tarfile
from collections import namedtuple
import time
import re
import warnings
from json import loads, dumps
from io import StringIO
from itertools import zip_longest
from collections.abc import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.plotting.palettes as palettes
from scipy import stats
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from IPython.display import display

import PyPDF2
from io import BytesIO

import came

__PATH_ROOT_TACMAN__ = str(Path(__file__).parent.parent.parent)
sys.path.append(__PATH_ROOT_TACMAN__)
# None if  __PATH_ROOT_TACMAN__ in sys.path else

import TACMAN

# # parameters


# p_root = Path('[you path]')
p_root = Path(Path(__file__).parent.parent.parent)

p_df_varmap = p_root.joinpath("homo/info.csv")
assert p_df_varmap.exists(), "[not exists] {}".format(p_df_varmap)
p_maps_SAMap = p_root.joinpath("homo/SAMap/maps_gene_name")
assert p_maps_SAMap.exists(), "[not exists] {}".format(p_maps_SAMap)


map_sp = {
    k: v
    for k, v in zip(
        "h,m,z,ma,c,x,r,p".split(","),
        "human,mouse,zebrafish,macaque,chicken,xenopus,rat,pig".split(","),
    )
}

map_sp_reverse = {v: k for k, v in map_sp.items()}
map_sp.update(
    {v: v for v in "human,mouse,zebrafish,macaque,chicken,xenopus,rat,pig".split(",")}
)


map_sp.update({k: v for k, v in zip("hs,mm".split(","), "human,mouse".split(","))})

rng = np.random.default_rng()


# # SAMap package and new functions

import samap
from samalg import SAM
from samap.mapping import SAMAP
from samap import analysis as sana
import scipy as sp
import typing

map_sp_SAMap = {
    "human": "hu",
    "mouse": "mm",
    "zebrafish": "zf",
    "chicken": "ch",
    "macaque": "ma",
    "xenopus": "xe",
}


def get_alignment_score_for_each_cell(sm, keys, n_top=0):
    "n_top 无效 但保留"

    def customize_compute_csim(samap, key, X=None, prepend=True, n_top=0):
        splabels = sana.q(samap.adata.obs["species"])
        skeys = splabels[np.sort(np.unique(splabels, return_index=True)[1])]

        cl = []
        clu = []
        for sid in skeys:
            if prepend:
                cl.append(
                    sid
                    + "_"
                    + sana.q(samap.adata.obs[key])[samap.adata.obs["species"] == sid]
                    .astype("str")
                    .astype("object")
                )
            else:
                cl.append(
                    sana.q(samap.adata.obs[key])[samap.adata.obs["species"] == sid]
                )
            clu.append(np.unique(cl[-1]))

        clu = np.concatenate(clu)
        cl = np.concatenate(cl)

        CSIM = np.zeros((clu.size, clu.size))
        if X is None:
            X = samap.adata.obsp["connectivities"].copy()

        xi, yi = X.nonzero()
        spxi = splabels[xi]
        spyi = splabels[yi]

        filt = spxi != spyi
        di = X.data[filt]
        xi = xi[filt]
        yi = yi[filt]

        px, py = xi, cl[yi]
        p = px.astype("str").astype("object") + ";" + py.astype("object")

        A = pd.DataFrame(data=np.vstack((p, di)).T, columns=["x", "y"])
        valdict = sana.df_to_dict(A, key_key="x", val_key="y")
        cell_scores = [valdict[k].sum() for k in valdict.keys()]
        ixer = pd.Series(data=np.arange(clu.size), index=clu)
        if len(valdict.keys()) > 0:
            xc, yc = sana.substr(list(valdict.keys()), ";")
            xc = xc.astype("int")
            yc = ixer[yc].values
            cell_cluster_scores = sp.sparse.coo_matrix(
                (cell_scores, (xc, yc)), shape=(X.shape[0], clu.size)
            ).A
            return pd.DataFrame(
                cell_cluster_scores, index=samap.adata.obs.index, columns=clu
            )
        else:
            raise Exception("[Error]")
            # return np.zeros((clu.size, clu.size)), clu

    if len(list(keys.keys())) < len(list(sm.sams.keys())):
        samap = SAM(
            counts=sm.samap.adata[
                np.in1d(sm.samap.adata.obs["species"], list(keys.keys()))
            ]
        )
    else:
        samap = sm.samap

    clusters = []
    ix = np.unique(samap.adata.obs["species"], return_index=True)[1]
    skeys = sana.q(samap.adata.obs["species"])[np.sort(ix)]

    for sid in skeys:
        clusters.append(
            sana.q([sid + "_" + str(x) for x in sm.sams[sid].adata.obs[keys[sid]]])
        )

    cl = np.concatenate(clusters)
    l = "{}_mapping_scores".format(";".join([keys[sid] for sid in skeys]))
    samap.adata.obs[l] = pd.Categorical(cl)

    # CSIMth, clu = _compute_csim(samap, l, n_top = n_top, prepend = False)
    cell_cluster_scores = customize_compute_csim(samap, l, n_top=n_top, prepend=False)
    return cell_cluster_scores


# cell_cluster_scores = get_alignment_score_for_each_cell(sm, keys, n_top = 0)
# cell_cluster_scores


# 重新定义SAMAP 的初始化函数__init__ 和 samap.mapping._calculate_blast_graph ,使得可以进行1v1的运算
# gnnm, gns, gns_dict = _calculate_blast_graph(
#     ids, f_maps=f_maps, reciprocate=True, eval_thr=eval_thr)
# 修改为 [追加参数is_1v1]
# gnnm, gns, gns_dict = _calculate_blast_graph(
#     ids, f_maps=f_maps, reciprocate=True, eval_thr=eval_thr,
#     is_1v1=is_1v1)
def customize__init__for_SAMAP(
    self,
    sams: dict,
    f_maps: typing.Optional[str] = "maps/",
    names: typing.Optional[dict] = None,
    keys: typing.Optional[dict] = None,
    resolutions: typing.Optional[dict] = None,
    gnnm: typing.Optional[tuple] = None,
    save_processed: typing.Optional[bool] = True,
    eval_thr: typing.Optional[float] = 1e-6,
    is_1v1=False,
):
    """Initializes and preprocess data structures for SAMap algorithm.

    Parameters
    ----------
    sams : dict of string OR SAM
        Dictionary of (indexed by species IDs):
        The path to an unprocessed '.h5ad' `AnnData` object for organisms.
        OR
        A processed and already-run SAM object.

    f_maps : string, optional, default 'maps/'
        Path to the `maps` directory output by `map_genes.sh`.
        By default assumes it is in the local directory.

    names : dict of list of 2D tuples or Nx2 numpy.ndarray, optional, default None
        If BLAST was run on a transcriptome with Fasta headers that do not match
        the gene symbols used in the dataset, you can pass a list of tuples mapping
        the Fasta header name to the Dataset gene symbol:
        (Fasta header name , Dataset gene symbol). Transcripts with the same gene
        symbol will be collapsed into a single node in the gene homology graph.
        By default, the Fasta header IDs are assumed to be equivalent to the
        gene symbols used in the dataset.

        The above mapping should be contained in a dicitonary keyed by the corresponding species.
        For example, if we have `hu` and `mo` species and the `hu` BLAST results need to be translated,
        then `names = {'hu' : mapping}, where `mapping = [(Fasta header 1, Gene symbol 1), ... , (Fasta header n, Gene symbol n)]`.

    keys : dict, optional, default None
        Dictionary of obs keys indexed by species to use for determining maximum
        neighborhood size of each cell.

    resolutions : dict, optional, default None
        Dictionary of leiden clustering resolutions indexed by species. This parameter is ignored if
        `keys` is set.

    gnnm : tuple(scipy.sparse.csr_matrix,numpy array, dict[numpy array])
        If the homology graph was already computed, you can pass it here in the form of a tuple:
        (sparse adjacency matrix, numpy array of genes, dictionary of species-specific genes).
        Note that all genes must be prefixed with their species IDs, e.g. `hu_SOX2` instead of `SOX2`.

        This is the tuple returned by `_calculate_blast_graph(...)` or `_coarsen_eggnog_graph(...)`.

    save_processed : bool, optional, default False
        If True saves the processed SAM objects corresponding to each species to an `.h5ad` file.
        This argument is unused if preloaded SAM objects are passed in to SAMAP.

    eval_thr : float, optional, default 1e-6
        E-value threshold above which BLAST results will be filtered out.
    """
    print("[a new __init__ for SAMAP]")
    for key, data in zip(sams.keys(), sams.values()):
        if not (isinstance(data, str) or isinstance(data, SAM)):
            raise TypeError(f"Input data {key} must be either a path or a SAM object.")

    ids = list(sams.keys())

    if keys is None:
        keys = {}
        for sid in ids:
            keys[sid] = "leiden_clusters"

    if resolutions is None:
        resolutions = {}
        for sid in ids:
            resolutions[sid] = 3

    for sid in ids:
        data = sams[sid]
        key = keys[sid]
        res = resolutions[sid]

        if isinstance(data, str):
            print("Processing data {} from:\n{}".format(sid, data))
            sam = SAM()
            sam.load_data(data)
            sam.preprocess_data(
                sum_norm="cell_median",
                norm="log",
                thresh_low=0.0,
                thresh_high=0.96,
                min_expression=1,
            )
            sam.run(
                preprocessing="StandardScaler",
                npcs=100,
                weight_PCs=False,
                k=20,
                n_genes=3000,
                weight_mode="rms",
            )
        else:
            sam = data

        if key == "leiden_clusters":
            sam.leiden_clustering(res=res)

        if "PCs_SAMap" not in sam.adata.varm.keys():
            samap.mapping.prepare_SAMap_loadings(sam)

        if save_processed and isinstance(data, str):
            sam.save_anndata(data.split(".h5ad")[0] + "_pr.h5ad")

        sams[sid] = sam

    if gnnm is None:
        gnnm, gns, gns_dict = samap.mapping._calculate_blast_graph(
            ids, f_maps=f_maps, reciprocate=True, eval_thr=eval_thr, is_1v1=is_1v1
        )
        if names is not None:
            gnnm, gns_dict, gns = _coarsen_blast_graph(gnnm, gns, names)

        gnnm = samap.mapping._filter_gnnm(gnnm, thr=0.25)
    else:
        gnnm, gns, gns_dict = gnnm

    gns_list = []
    ges_list = []
    for sid in ids:
        samap.utils.prepend_var_prefix(sams[sid], sid)
        ge = sana.q(sams[sid].adata.var_names)
        gn = gns_dict[sid]
        gns_list.append(gn[np.in1d(gn, ge)])
        ges_list.append(ge)

    f = np.in1d(gns, np.concatenate(gns_list))
    gns = gns[f]
    gnnm = gnnm[f][:, f]
    A = pd.DataFrame(data=np.arange(gns.size)[None, :], columns=gns)
    ges = np.concatenate(ges_list)
    ges = ges[np.in1d(ges, gns)]
    ix = A[ges].values.flatten()
    gnnm = gnnm[ix][:, ix]
    gns = ges

    gns_dict = {}
    for i, sid in enumerate(ids):
        gns_dict[sid] = ges[np.in1d(ges, gns_list[i])]

        print(
            "{} `{}` gene symbols match between the datasets and the BLAST graph.".format(
                gns_dict[sid].size, sid
            )
        )

    for sid in sams:
        if not sp.sparse.issparse(sams[sid].adata.X):
            sams[sid].adata.X = sp.sparse.csr_matrix(sams[sid].adata.X)

    smap = samap.mapping._Samap_Iter(sams, gnnm, gns_dict, keys=keys)
    self.sams = sams
    self.gnnm = gnnm
    self.gns_dict = gns_dict
    self.gns = gns
    self.ids = ids
    self.smap = smap


def customize_calculate_blast_graph(
    ids, f_maps="maps/", eval_thr=1e-6, reciprocate=False, is_1v1=False
):
    print("[new_calculate_blast_graph]")

    gns = []
    Xs = []
    Ys = []
    Vs = []

    for i in range(len(ids)):
        id1 = ids[i]
        for j in range(i, len(ids)):
            id2 = ids[j]
            if i != j:
                if os.path.exists(f_maps + "{}{}".format(id1, id2)):
                    fA = f_maps + "{}{}/{}_to_{}.txt.gz".format(id1, id2, id1, id2)
                    fB = f_maps + "{}{}/{}_to_{}.txt.gz".format(id1, id2, id2, id1)
                elif os.path.exists(f_maps + "{}{}".format(id2, id1)):
                    fA = f_maps + "{}{}/{}_to_{}.txt.gz".format(id2, id1, id1, id2)
                    fB = f_maps + "{}{}/{}_to_{}.txt.gz".format(id2, id1, id2, id1)
                else:
                    raise FileExistsError(
                        "BLAST mapping tables with the input IDs ({} and {}) not found in the specified path.".format(
                            id1, id2
                        )
                    )
                if is_1v1:
                    fA = fA.replace(".txt", "_1v1.txt")
                    fB = fB.replace(".txt", "_1v1.txt")
                print("{} {}".format(os.path.basename(fA), os.path.basename(fB)))
                A = pd.read_csv(fA, sep="\t", header=None, index_col=0)
                B = pd.read_csv(fB, sep="\t", header=None, index_col=0)

                A.columns = A.columns.astype("<U100")
                B.columns = B.columns.astype("<U100")

                A = A[A.index.astype("str") != "nan"]
                A = A[A.iloc[:, 0].astype("str") != "nan"]
                B = B[B.index.astype("str") != "nan"]
                B = B[B.iloc[:, 0].astype("str") != "nan"]

                A.index = samap.mapping._prepend_blast_prefix(A.index, id1)
                B[B.columns[0]] = samap.mapping._prepend_blast_prefix(
                    B.iloc[:, 0].values.flatten(), id1
                )

                B.index = samap.mapping._prepend_blast_prefix(B.index, id2)
                A[A.columns[0]] = samap.mapping._prepend_blast_prefix(
                    A.iloc[:, 0].values.flatten(), id2
                )

                i1 = np.where(A.columns == "10")[0][0]
                i3 = np.where(A.columns == "11")[0][0]

                inA = sana.q(A.index)
                inB = sana.q(B.index)

                inA2 = sana.q(A.iloc[:, 0])
                inB2 = sana.q(B.iloc[:, 0])
                gn1 = np.unique(np.append(inB2, inA))
                gn2 = np.unique(np.append(inA2, inB))
                gn = np.append(gn1, gn2)
                gnind = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)

                A.index = pd.Index(gnind[A.index].values.flatten())
                B.index = pd.Index(gnind[B.index].values.flatten())
                A[A.columns[0]] = gnind[A.iloc[:, 0].values.flatten()].values.flatten()
                B[B.columns[0]] = gnind[B.iloc[:, 0].values.flatten()].values.flatten()

                Arows = np.vstack((A.index, A.iloc[:, 0], A.iloc[:, i3])).T
                Arows = Arows[A.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm1 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm1[Arows[:, 0].astype("int32"), Arows[:, 1].astype("int32")] = Arows[
                    :, 2
                ]  # -np.log10(Arows[:,2]+1e-200)

                Brows = np.vstack((B.index, B.iloc[:, 0], B.iloc[:, i3])).T
                Brows = Brows[B.iloc[:, i1].values.flatten() <= eval_thr, :]
                gnnm2 = sp.sparse.lil_matrix((gn.size,) * 2)
                gnnm2[Brows[:, 0].astype("int32"), Brows[:, 1].astype("int32")] = Brows[
                    :, 2
                ]  # -np.log10(Brows[:,2]+1e-200)

                gnnm = (gnnm1 + gnnm2).tocsr()
                gnnms = (gnnm + gnnm.T) / 2
                if reciprocate:
                    gnnm.data[:] = 1
                    gnnms = gnnms.multiply(gnnm).multiply(gnnm.T).tocsr()
                gnnm = gnnms

                f1 = np.where(np.in1d(gn, gn1))[0]
                f2 = np.where(np.in1d(gn, gn2))[0]
                f = np.append(f1, f2)
                gn = gn[f]
                gnnm = gnnm[f, :][:, f]

                V = gnnm.data
                X, Y = gnnm.nonzero()

                Xs.extend(gn[X])
                Ys.extend(gn[Y])
                Vs.extend(V)
                gns.extend(gn)

    gns = np.unique(gns)
    gns_sp = np.array([x.split("_")[0] for x in gns])
    gns2 = []
    gns_dict = {}
    for sid in ids:
        gns2.append(gns[gns_sp == sid])
        gns_dict[sid] = gns2[-1]
    gns = np.concatenate(gns2)
    indexer = pd.Series(index=gns, data=np.arange(gns.size))

    X = indexer[Xs].values
    Y = indexer[Ys].values
    gnnm = sp.sparse.coo_matrix((Vs, (X, Y)), shape=(gns.size, gns.size)).tocsr()

    return gnnm, gns, gns_dict


SAMAP.__init__ = customize__init__for_SAMAP
samap.mapping._calculate_blast_graph = customize_calculate_blast_graph


# # statistics


# F1 score [start]
def calculate_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.sort(np.union1d(np.unique(y_pred), np.unique(y_true)))
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    matrix.update(
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        .groupby(["y_true", "y_pred"])["y_pred"]
        .count()
        .to_frame("count")
        .reset_index()
        .pivot(index="y_true", columns="y_pred", values="count")
        .fillna(0)
    )
    matrix.index.name = "true"
    matrix.columns.name = "pred"
    return matrix


def calculate_more_with_confusion_matrix(data):
    df = pd.DataFrame(
        {
            "TP": np.diag(data),
            "FP": data.sum(axis=0) - np.diag(data),
            "FN": data.sum(axis=1) - np.diag(data),
        }
    )
    df["Precision"] = df.eval("TP /(TP + FP)").fillna(0)
    df["Recall"] = df.eval("TP /(TP + FN)").fillna(0)
    df["F1 Score"] = df.eval("2*(Precision * Recall) / (Precision + Recall)").fillna(0)
    return df


def calculate_accuracy_with_confusion_matrix(data):
    data = data.loc[:, data.columns.isin(data.index)]
    return np.diag(data).sum() / np.sum(data.values)


def calculate_F1Score_with_confusion_matrix(data, average="weighted"):
    """
    average:
        macro (default): 算数均值
        weighted : 加权均值，以实际为真值的数量(TP+FN) 为权重
        micro: ？？？？Sum statistics over all labels
            分别对各类的TP , FP, FN 求和,在计算TP / (1/2(FP + FN))
            咦,这个不需要分别计算各类的F1-score
    """

    res = None
    if average == "macro":
        res = data["F1 Score"].mean()
    elif average == "weighted":
        res = np.average(data["F1 Score"], weights=data.eval("TP+FN"))
    elif average == "micro":
        res = data["TP"].sum() / (
            data["TP"].sum() + 1 / 2 * (data["FP"].sum() + data["FN"].sum())
        )
    else:
        raise Exception("[Error] average= {}".format(average))
    return res


def calculate_res_stat(row, q, update=False):
    p_res_stats = row["dir"].joinpath("res_stats.json")
    res_stats = {}
    if p_res_stats.exists() and (not update):
        res_stats = loads(p_res_stats.read_text())
    else:
        df_obs = get_res_obs(row).query(q)
        df_obs.index = df_obs.index.astype(str)
        cm = calculate_confusion_matrix(df_obs["true_label"], df_obs["pre_label"])
        acc = calculate_accuracy_with_confusion_matrix(cm)
        f1 = calculate_F1Score_with_confusion_matrix(
            calculate_more_with_confusion_matrix(cm)
        )
        res_stats.update(
            {
                q: {
                    "confusion_matrix": cm.to_json(orient="columns"),
                    "confusion_matrix_more": calculate_more_with_confusion_matrix(
                        cm
                    ).to_json(orient="columns"),
                    "Accuracy": calculate_accuracy_with_confusion_matrix(cm),
                    "F1-score": calculate_F1Score_with_confusion_matrix(
                        calculate_more_with_confusion_matrix(cm)
                    ),
                }
            }
        )
        p_res_stats.write_text(dumps(res_stats))

    for k, v in res_stats.items():
        res_stats[k]["confusion_matrix"] = pd.read_json(
            StringIO(v["confusion_matrix"]), orient="columns"
        )
        res_stats[k]["confusion_matrix_more"] = pd.read_json(
            StringIO(v["confusion_matrix_more"]), orient="columns"
        )
    return res_stats[q]


# F1 score [end]


def get_res_stat(row, q, key, update=False):
    res_stat = calculate_res_stat(row, q, update)
    return res_stat.setdefault(key, None)


# # time_tag


def time_tag_get(time_format="%y%m%d-%H%M"):
    return time.strftime(time_format, time.localtime())


def time_tag_detect(p):
    return True if re.match(".+;\\d{6}-\\d{4}$", Path(p).name) else False


def time_tag_toggle(p):
    p = Path(p)
    assert p.exists(), "[not exists]\n{}".format(p)
    p_res = p
    if time_tag_detect(p):
        # raise Exception('[time tag has existsed]\n{}'.format(p.name))
        p_res = p.with_name(re.sub(";\\d{6}-\\d{4}$", "", p_res.name))
    else:
        p_res = p.with_name("{};{}".format(p.name, time_tag_get()))
    assert not p_res.exists(), "[target has existed]\n{}".format(p_res)
    p.rename(p_res)
    return


# # other functions


def show_str_arr(data, n_cols=4, order="F", return_str=False):
    """
    Examples
    --------

    import numpy as np
    rng = np.random.default_rng()

    show_str_arr(['12345678910'[:_]
        for _ in rng.integers(1,10,13)],n_cols=3)

    # | 12345678| 1234567  | 123456|
    # | 1234    | 12345678 | 1234  |
    # | 123     | 123456789| 1234  |
    # | 123     | 1        |       |
    # | 123456  | 1        |       |
    """
    data = np.array(data)
    data = pd.Series(np.concatenate([data, np.repeat("", n_cols - data.size % n_cols)]))
    data = pd.DataFrame(np.reshape(data, (data.size // n_cols, n_cols), order=order))
    for k in data.columns:
        data[k] = data[k].str.ljust(data[k].str.len().max())
    res = "\n".join(
        data.apply(lambda row: "| {}|".format("| ".join(row.values)), axis=1)
    )
    if return_str:
        return res
    print(res)


def show_obj_attr(
    obj,
    regex_filter=["^_"],
    regex_select=[],
    show_n_cols=3,
    group=False,
    extract_pat="^([^_]+)_",
    return_series=False,
):
    from functools import reduce

    if isinstance(regex_filter, str):
        regex_filter = [regex_filter]
    if isinstance(regex_select, str):
        regex_select = [regex_select]

    attr = pd.Series([_ for _ in dir(obj)])
    # filter
    if len(regex_filter) == 1:
        attr = attr[~attr.str.match(regex_filter[0])]
    elif len(regex_filter) == 0:
        pass
    else:
        attr = attr[
            reduce(lambda x, y: x & y, [~attr.str.match(_) for _ in regex_filter])
        ]
    # select
    if len(regex_select) == 1:
        attr = attr[attr.str.match(regex_select[0])]
    elif len(regex_select) == 0:
        pass
    else:
        attr = attr[
            reduce(lambda x, y: x | y, [attr.str.match(_) for _ in regex_select])
        ]
    if show_n_cols:
        # print(*['\t'.join(_) for _ in np.array_split(
        # attr.to_numpy(),attr.size//4)],sep='\n')
        show_str_arr(attr, show_n_cols)
    if group:
        print("[group]".ljust(15, "-"))
        print(attr.str.extract(extract_pat, expand=False).value_counts(dropna=False))

    if return_series:
        attr.index = np
        return attr


def archive_gzip(
    p_source, decompress=False, out_dir=None, remove_source=True, show=True
):
    """gzip 的压缩和解压缩"""

    def archive_gzip_default(p_source, decompress, out_dir, remove_source, show):
        import gzip
        from shutil import copyfileobj

        func_open_source, func_open_target = open, gzip.open
        if decompress:
            func_open_source, func_open_target = func_open_target, func_open_source

        with func_open_source(p_source, "rb") as f_source:
            with func_open_target(p_target, "wb") as f_target:
                copyfileobj(f_source, f_target)

    def archive_gzip_Linux(p_source, decompress, out_dir, remove_source, show):
        import os

        os.system(
            "gzip -{}c {} > {}".format("d" if decompress else "", p_source, p_target)
        )

    import platform

    handel_func = {"Linux": archive_gzip_Linux}.setdefault(
        platform.system(), archive_gzip_default
    )

    p_source = Path(p_source)
    assert p_source.exists(), "[not exitst] {}".format(p_source)

    out_dir = p_source.parent if out_dir is None else Path(out_dir)
    if decompress:
        assert p_source.match(
            "*.gz"
        ), "[Error] p_source must end with .gz when decompress=True"
        p_target = out_dir.joinpath(p_source.name[:-3])
    else:
        if p_source.name.endswith(".gz"):
            print("[archive_gzip] has commpress {}".format(p_source)) if show else None
            return
        p_target = out_dir.joinpath("{}.gz".format(p_source.name))

    handel_func(p_source, decompress, out_dir, remove_source, show)
    (
        print(
            "[archive_gzip][{}compress] {} -> {}".format(
                "de" if decompress else "", p_source.name, p_target.name
            )
        )
        if show
        else None
    )
    p_source.unlink() if remove_source else None


def group_agg(
    df,
    groupby_list,
    agg_dict=None,
    dropna=True,
    reindex=True,
    recolumn=True,
    rename_dict=None,
):
    if None is agg_dict:
        agg_dict = {groupby_list[-1]: ["count"]}
    res = df.groupby(groupby_list, dropna=dropna, observed=False).agg(agg_dict)
    if recolumn:
        res.columns = ["_".join(i) for i in res.columns]
    if reindex:
        res = res.index.to_frame().join(res)
        res.index = np.arange(res.shape[0])
    if isinstance(rename_dict, dict):
        res = res.rename(columns=lambda k: rename_dict.setdefault(k, k))
    return res


def h5ad_to_mtx(adata, p_dir, prefixes="", as_int=True):
    """
    将adata对象保存为mtx
    p_dir ： 输出路径
    as_int : 是否将矩阵转换int类型
        default True
    """
    assert adata.obs.index.is_unique, "[Error] obs index is not unique"
    assert adata.var.index.is_unique, "[Error] var index is not unique"

    p_dir = Path(p_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # [out] genes.tsv
    adata.var["gene_names"] = adata.var_names.to_numpy()
    if "gene_ids" not in adata.var_keys():
        adata.var["gene_ids"] = adata.var["gene_names"]
    df_genes = adata.var.loc[:, ["gene_ids", "gene_names"]]
    df_genes.to_csv(
        p_dir.joinpath("{}genes.tsv".format(prefixes)),
        header=False,
        index=False,
        sep="\t",
    )

    # [out] barcodes.tsv obs.csv
    adata.obs.loc[:, []].to_csv(
        p_dir.joinpath("{}barcodes.tsv".format(prefixes)),
        header=False,
        index=True,
        sep="\t",
    )

    if len(adata.obs_keys()) > 0:
        adata.obs.to_csv(p_dir.joinpath("{}obs.csv".format(prefixes)), index=True)

    # [out] matrix.mtx
    adata.X = csr_matrix(adata.X)
    if as_int:
        adata.X = adata.X.astype(int)
    nonzero_index = [i[:10] for i in adata.X.nonzero()]
    print(
        "frist 10 adata.X nonzero elements:\n",
        adata.X[nonzero_index[0], nonzero_index[1]],
    )
    mmwrite(p_dir.joinpath("{}matrix.mtx".format(prefixes)), adata.X.getH())
    archive_gzip(
        p_dir.joinpath("{}matrix.mtx".format(prefixes)),
        decompress=False,
        remove_source=True,
        show=False,
    )
    print("[out] {}".format(p_dir))


def get_path_varmap(
    sp_ref, sp_que, p_df_varmap=p_df_varmap, p_maps_SAMap=p_maps_SAMap, model="TACMAN"
):
    """
    通过sp_ref 和 sp_que获取path_varmap
    ./homo/df_varmap.csv 存储了
    path_varmap路径及信息
    """
    p_df_varmap = Path(p_df_varmap)
    if model in "TACMAN,CAME,came,TACMAN_before_custom_trainer".split(","):
        df_varmap = pd.read_csv(p_df_varmap)
        index_ = df_varmap.query(
            "sp_ref == '{}' & sp_que == '{}'".format(sp_ref, sp_que)
        ).index
        assert (
            index_.size == 1
        ), "[get {} path]can not get speicifed and unique path\nsp_ref\tsp_que\n{}\t{}".format(
            index_.size, sp_ref, sp_que
        )
        res = Path(df_varmap.loc[index_[0], "path"])
        if not res.is_absolute():
            res = p_df_varmap.parent.joinpath(res)
        assert res.exists(), "[not exists] {}".format(res)
        return res

    elif model == "SAMap":
        return p_maps_SAMap
    else:
        raise Exception(
            "[Error] can not find path_varmap with model '{}'".format(model)
        )


def exchange_gn(sp_ref, sp_que, gns, return_type="array"):
    gns = pd.Series(gns)
    df_varmap = pd.read_csv(
        get_path_varmap(sp_ref, sp_que),
        skiprows=[0],
        names="gn_ref,gn_que,gn_type".split(","),
    ).dropna(axis=0)
    df_varmap = df_varmap[df_varmap["gn_ref"].isin(gns)].copy()
    res = df_varmap["gn_que"].to_numpy()

    if return_type == "array":
        res = df_varmap["gn_que"].to_numpy()
    elif return_type == "df":
        res = df_varmap
    elif return_type == "array_with_not_match":
        res = np.concatenate([gns[~gns.isin(df_varmap["gn_ref"])], df_varmap["gn_que"]])
    else:
        Warning("[Waring] return_type exchange to array")

    return res


def get_test_result_df(p):
    p = Path(p)
    df = pd.DataFrame({"dir": [i for i in p.iterdir() if i.is_dir()]})

    df = df[df["dir"].apply(lambda x: x.joinpath("finish").exists())]
    df["name"] = df["dir"].apply(lambda x: x.name)
    df["split"] = df["name"].str.split(";")
    df = df.join(
        df["split"]
        .apply(lambda x: x[0])
        .str.extract("^(?P<tissue>.+)_(?P<sp_ref>.+)-corss-(?P<sp_que>.+)")
    )
    df["model"] = df["split"].apply(lambda x: x[1])
    df = df.join(
        df["split"]
        .apply(lambda x: x[2])
        .str.extract("^(?P<name_ref>.+)-map-(?P<name_que>.+)")
    )
    df["resdir_tag"] = df["split"].apply(lambda x: ";".join(x[3:]))
    df = df.drop(columns="split")
    return df


def get_source_obs(row, dataset_type="ref"):
    """
    Parameters
    ----------
    row : dict
        由 get_test_result_df 获得的row
    dataset_type : str
        ref/que
    """
    return pd.read_csv(
        Path(row["dir"]).joinpath("obs_{}.csv".format(dataset_type)), index_col=0
    )


def is_1v1(row):
    res = None
    if row["model"] == "seurat":
        res = True
    elif row["model"] in ["came", "TACMAN"]:
        if "is_1v1=True" in row["resdir_tag"]:
            res = True
        if "is_1v1=False" in row["resdir_tag"]:
            res = False
    return res


def get_res_obs(row):
    """
    row 为来自get_test_result_df的行
    dataset,cell_type,true_label,pre_label,max_prob,is_right,UMAP1,UMAP2
    """

    def get_res_obs_TACMAN(row):
        df = pd.read_csv(
            row["dir"].joinpath("res_2", "pre_out_2.csv"), index_col=0
        ).loc[:, "true_label,pre_label,pre_label_NUM,max_prob".split(",")]

        p_am = row["dir"].joinpath("adata_meta")
        if not p_am.exists():
            print("[not exists] {}/{} ".format(row["dir"].name, p_am.name))
            # return df
        df_obs = pd.read_csv(
            row["dir"].joinpath("adata_meta", "obs.csv"), index_col=0
        ).loc[:, ["dataset", "cell_type"]]

        df_umap = pd.read_csv(p_am.joinpath("obsm.csv"), index_col=None).rename(
            columns={"X_umap1": "UMAP1", "X_umap2": "UMAP2"}
        )
        df_umap.index = df_obs.index

        if not df_obs.index.is_unique:
            if df_obs.apply(
                lambda row: "{};{}".format(row.name, row["dataset"]), axis=1
            ).is_unique:
                df_obs.index = df_obs.apply(
                    lambda row: "{};{}".format(row.name, row["dataset"]), axis=1
                ).to_numpy()
                df.index = df_obs.index
                df_umap.index = df_obs.index
                print("[index] add dataset")
            else:
                raise Exception("[index not unique]")
        df = df.join(df_obs).join(df_umap)

        df["is_right"] = df.eval("true_label == pre_label")
        df = df.loc[
            :,
            np.intersect1d(
                "dataset,cell_type,true_label,pre_label,max_prob,is_right,UMAP1,UMAP2".split(
                    ","
                ),
                df.columns,
            ),
        ]
        return df

    def get_res_obs_came(row):
        df = pd.read_csv(
            row["dir"].joinpath("obs.csv"),
            index_col=0,
            usecols="original_name,dataset,REF,celltype,predicted,max_probs,is_right".split(
                ","
            ),
        )

        df.index = df.index.to_numpy()
        df = df.rename(
            columns={
                # "REF": "",
                "predicted": "pre_label",
                "max_probs": "max_prob",
                "celltype": "cell_type",
            }
        )
        df["true_label"] = df["cell_type"]

        df = df.loc[
            :,
            np.intersect1d(
                "dataset,cell_type,true_label,pre_label,max_prob,is_right,UMAP1,UMAP2".split(
                    ","
                ),
                df.columns,
            ),
        ]
        p_am = row["dir"].joinpath("adata_meta")
        if not p_am.exists():
            print("[not exists] {}/{} ".format(row["dir"].name, p_am.name))
            return df

        # get UMAP1,UMAP2
        temp_df = pd.read_csv(p_am.joinpath("obsm.csv"), index_col=None).rename(
            columns={"X_umap1": "UMAP1", "X_umap2": "UMAP2"}
        )
        assert (
            temp_df.shape[0] == df.shape[0]
        ), "[Error][length not equal] {} {}".format(temp_df.shape[0], df.shape[0])
        temp_df.index = df.index
        df = df.join(temp_df)

        df["is_right"] = df.eval("true_label == pre_label")
        df = df.loc[
            :,
            np.intersect1d(
                "dataset,cell_type,true_label,pre_label,max_prob,is_right,UMAP1,UMAP2".split(
                    ","
                ),
                df.columns,
            ),
        ]
        return df

    def get_res_obs_seurat(row):
        return pd.read_csv(
            row["dir"].joinpath("obs.csv"), index_col=0, low_memory=False
        ).loc[
            :,
            "UMAP1,UMAP2,cell_type,dataset,is_right,max_prob,pre_label,true_label".split(
                ","
            ),
        ]

    map_func = {
        "TACMAN": get_res_obs_TACMAN,
        "TACMAN": get_res_obs_TACMAN,
        "came": get_res_obs_came,
        "CAME": get_res_obs_came,
        "Seurat": get_res_obs_seurat,
        "SAMap": get_res_obs_seurat,
        "TACTiCS": get_res_obs_seurat,
        "TOSICA": get_res_obs_seurat,
    }

    assert row["model"] in map_func.keys(), "can note get res obs. model = {}".format(
        row["model"]
    )

    res = map_func[row["model"]](row)
    res["dataset_type"] = res["dataset"].map(
        {
            "{tissue}_{sp_ref}".format(**row): "ref",
            "{tissue}_{sp_que}".format(**row): "que",
        }
    )
    res["sp"] = res["dataset"].map(
        {
            "{tissue}_{sp_ref}".format(**row): row["sp_ref"],
            "{tissue}_{sp_que}".format(**row): row["sp_que"],
        }
    )

    return res


def get_adata_umap(row, intersect_pre_field=True):

    df_obs_ref = pd.read_csv(row["dir"].joinpath("obs_ref.csv"), index_col=0)
    df_obs_que = pd.read_csv(row["dir"].joinpath("obs_que.csv"), index_col=0)
    if intersect_pre_field:
        df_obs_ref, df_obs_que = [
            _.loc[:, np.intersect1d(df_obs_ref.columns, df_obs_que.columns)]
            for _ in [df_obs_ref, df_obs_que]
        ]
    df_obs = pd.concat([df_obs_ref, df_obs_que]).rename(
        columns=lambda x: "preobs_{}".format(x)
    )

    df_res = get_res_obs(row).join(df_obs)
    adata_umap = sc.AnnData(obs=df_res)
    adata_umap.obsm["X_umap"] = df_res.loc[:, "UMAP1,UMAP2".split(",")].to_numpy()
    return adata_umap


def get_matrix_max_prob_median(row, dataset_type="que"):
    data = get_adata_umap(row).obs.query("dataset_type == '{}' ".format(dataset_type))
    data = (
        data.groupby("true_label,pre_label".split(","))["max_prob"]
        .median()
        .reset_index()
        .pivot(index="true_label", columns="pre_label", values="max_prob")
        .fillna(0)
    )
    return data


def get_matrix_count(row, dataset_type="que"):
    data = get_adata_umap(row).obs.query("dataset_type == '{}' ".format(dataset_type))
    data = (
        data.groupby("true_label,pre_label".split(","))
        .agg({"pre_label": "count"})
        .rename(columns={"pre_label": "pre_label_count"})
        .reset_index()
        .pivot(index="true_label", columns="pre_label", values="pre_label_count")
        .fillna(0)
    )
    return data


def get_matrix_count_ratio(row, dataset_type="que"):
    data = get_matrix_count(row, dataset_type=dataset_type)
    data = (data.transpose() / data.sum(axis=1)).transpose()
    return data


def subset_adata(adata, *args):
    def _process_values(values):
        if isinstance(values, Iterable):
            if isinstance(values, str):
                values = [values]
        else:
            values = [values]
        return values

    assert len(args) % 2 == 0, "[Error][{}] length of args must be 2*n".format(
        len(args)
    )

    for key, values in zip(args[0::2], args[1::2]):
        values = _process_values(values)
        adata = adata[adata.obs[key].isin(values), :]
    return adata


def show_umap(row):
    """
    row 为来自get_test_result_df的行

    show test result umap
    row
        umap_dataset      :  png path
        umap_umap         :  png path
        p_cell_type_table :  csv path
    """
    p_umap_dataset = Path(row["dir"]).joinpath("figs", "umap_dataset.png")
    p_umap_umap = Path(row["dir"]).joinpath("figs", "umap_umap.png")
    p_cell_type_table = Path(row["dir"]).joinpath("group_counts.csv")
    print(row["name"].ljust(75, "-"))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(mpimg.imread(p_umap_dataset))
    ax[0].set_axis_off()
    ax[1].imshow(mpimg.imread(p_umap_umap))
    ax[1].set_axis_off()
    display(fig, pd.read_csv(p_cell_type_table, index_col=0))
    fig.clear()


def find_path_from_para(df_para, name):
    s = pd.Series(
        np.concatenate((df_para["path_ref"], df_para["path_que"])),
        index=np.concatenate((df_para["name_ref"], df_para["name_que"])),
    )
    s = s.drop_duplicates(ignore_index=False)
    assert s.index.is_unique, "[Error] df_para name_ref and name_que is not unique"
    return s[name]


def df_apply_merge_field(df, str_format):
    return df.apply(lambda row: str_format.format(**row), axis=1)


def df_varmap_query_exists(df_varmap, list_gn_ref=[], list_gn_que=[], model="both"):
    df_varmap = df_varmap.copy()
    df_varmap["gn_ref_exists"] = df_varmap["gn_ref"].isin(list_gn_ref)
    df_varmap["gn_que_exists"] = df_varmap["gn_que"].isin(list_gn_que)
    if model == "both":
        df_varmap = df_varmap.query("gn_ref_exists & gn_que_exists")
    elif model == "ref":
        df_varmap = df_varmap.query("gn_ref_exists")
    elif model == "que":
        df_varmap = df_varmap.query("gn_que_exists")
    else:
        raise Exception("[Error] model must be one of both, ref, que")
    df_varmap = df_varmap.drop(columns="gn_ref_exists,gn_que_exists".split(","))
    return df_varmap


# # cross species Model


def load_adata(p):
    def load_h5ad_from_mtx(p):
        p = Path(p)
        if p.joinpath("matrix.mtx.gz").exists():
            archive_gzip(
                p.joinpath("matrix.mtx.gz"),
                decompress=True,
                remove_source=False,
                show=False,
            )
        adata = sc.read_10x_mtx(p)

        # if p.joinpath('matrix.mtx.gz').exists():
        #     p.joinpath('matrix.mtx').unlink()
        # else:
        #     archive_gzip(p.joinpath("matrix.mtx"),
        #                  decompress=False, remove_source=True, show=False)

        if p.joinpath("obs.tsv").exists():
            adata.obs = pd.read_csv(p.joinpath("obs.tsv"), sep="\t", index_col=0)
            adata.obs.to_numpy()
            # adata.obs = adata.obs.loc[:, []].join(
            #     pd.read_csv(p.joinpath("obs.csv"), index_col=0))
        else:
            print("[not exists]obs.csv\nin {}".format(p))
        return adata

    p = Path(p)
    if p.is_dir() and p.joinpath("count.h5ad").exists():
        p = p.joinpath("count.h5ad")

    if p.match("*.h5ad"):
        return sc.read_h5ad(p)
    elif p.is_dir() and (
        p.joinpath("matrix.mtx").exists() or p.joinpath("matrix.mtx.gz").exists()
    ):
        return load_h5ad_from_mtx(p)
    else:
        raise Exception("[can not load adata] {}".format(p))


def load_normalized_adata(p, obs=None, update=False):
    """adata.X 为 normalize 即 CMP
    layers: counts log1p
    """

    def _process_adata(adata, p_out):
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        adata.layers["normalize"] = adata.X.copy()
        sc.pp.log1p(adata, base=np.e)
        adata.layers["log1p"] = adata.X.copy()
        sc.pp.scale(adata)
        adata.layers["scaled"] = adata.X.copy()
        adata.X = adata.layers.pop("normalize")
        adata.write_h5ad(p_out)
        return adata

    p = Path(p)
    assert p.is_dir, "[Error] please get a path of dir"
    p_h5ad = p.joinpath("normalize.h5ad")
    adata = sc.read_h5ad(p_h5ad) if p_h5ad.exists() else None

    # 判断是否需要重新运行_process_adata
    if adata is None:
        update = True
    elif isinstance(adata, sc.AnnData) and update == False:
        update = not (
            "counts" in adata.layers.keys()
            and "log1p" in adata.layers.keys()
            and "scaled" in adata.layers.keys()
        )

    if update:
        adata = load_adata(p)
        print("[normalize adata]\n{}\n".format(p_h5ad))
        adata = _process_adata(adata, p_h5ad)
    if isinstance(obs, pd.DataFrame):
        adata.obs = adata.obs.loc[:, []].join(obs)
    return adata


def get_1v1_matches(
    df_match, key_homology_type="homology_type", value_homology_type="ortholog_one2one"
):
    """
    from came.pp.take_1v_matches
    """
    l, r = df_match.columns[:2]
    l_unique = df_match[l].value_counts().to_frame().query("count == 1").index
    r_unique = df_match[r].value_counts().to_frame().query("count == 1").index
    keep = pd.DataFrame(
        {
            "l_is_unique": df_match[l].isin(l_unique),
            "r_is_unique": df_match[r].isin(r_unique),
        }
    ).min(axis=1)
    df_match = df_match[keep]
    df_match = df_match.query(
        "{} == '{}'".format(key_homology_type, value_homology_type)
    )
    return df_match


def get_homology_parameters(adata1, adata2, df_varmap):
    res = {}
    df_homo_paras = pd.DataFrame({"gn_ref": adata1.var_names}).merge(
        df_varmap, on="gn_ref", how="left"
    )
    res["homology_one2one_find"] = df_homo_paras["gn_que"].notna().sum()

    df_homo_paras["gn_que"] = df_homo_paras.apply(
        lambda row: (
            "not_o2o_" + row["gn_ref"] if pd.isna(row["gn_que"]) else row["gn_que"]
        ),
        axis=1,
    )

    assert (
        df_homo_paras["gn_ref"] == adata1.var_names
    ).all(), "df_homo_paras['gn_ref'] not equal adata1.var_names"
    assert (
        df_homo_paras["gn_ref"].is_unique & df_homo_paras["gn_que"].is_unique
    ), "df_homo_paras gn_ref or gn_que is not unique"
    # came 和 TACMAN不做替换
    # adata1.var.index = df_homo_paras['gn_que'].to_numpy()
    res["homology_one2one_use"] = np.intersect1d(
        df_homo_paras["gn_que"], adata2.var.index
    ).size
    res = {k: int(v) for k, v in res.items()}
    return res


def get_type_counts_info(adatas, key_class, dsnames):
    type_counts_list = []
    for i in range(len(adatas)):
        type_counts_list.append(pd.value_counts(adatas[i].obs[key_class]))
    counts_info = pd.concat(type_counts_list, axis=1, keys=dsnames)
    return counts_info


def aligned_type(adatas, key_calss):
    adata1 = adatas[0].copy()
    adata2 = adatas[1].copy()
    counts_info = get_type_counts_info(
        adatas, key_calss, dsnames=["reference", "query"]
    )
    print("----raw----")
    print(counts_info)
    counts_info = counts_info.dropna(how="any")
    print("----new----")
    print(counts_info)

    com_type = counts_info.index.tolist()
    adata1 = adata1[adata1.obs[key_calss].isin(com_type)]
    adata2 = adata2[adata2.obs[key_calss].isin(com_type)]
    return adata1, adata2


def unify_group_counts_index_name(resdir):
    """
    不知为何，group_counts.index.name参差不齐
    故将group_counts_unalign.index.name 赋给 group_counts.index.name
    """
    p_group_counts = resdir.joinpath("group_counts.csv")
    p_group_counts_unalign = resdir.joinpath("group_counts_unalign.csv")

    lines = p_group_counts.read_text().split("\n")
    lines[0] = ",".join(
        [p_group_counts_unalign.read_text().split("\n")[0].split(",")[0]]
        + lines[0].split(",")[1:]
    )
    p_group_counts.write_text("\n".join(lines))


# ## came


def precess_after_came(resdir, tissue_name, sp_ref, sp_que, is_display=False):
    unify_group_counts_index_name(resdir)
    resdir = Path(resdir)
    figdir = resdir.joinpath("figs")
    sc.settings.figdir = figdir

    (
        display(pd.read_csv(resdir.joinpath("group_counts.csv"), index_col=0))
        if is_display
        else None
    )
    obs = pd.read_csv(resdir.joinpath("obs.csv"), index_col=0)

    # umap
    # the last layer of hidden states
    h_dict = came.load_hidden_states(resdir.joinpath("hidden_list.h5"))[-1]
    adt = came.pp.make_adata(h_dict["cell"], obs=obs, assparse=False, ignore_index=True)
    sc.pp.neighbors(adt, n_neighbors=15, metric="cosine", use_rep="X")
    sc.tl.umap(adt)

    sc.pl.umap(adt, color="dataset", save="_dataset.png")
    sc.pl.umap(adt, color="celltype", save="_umap.png")
    adt.write_csvs(resdir.joinpath("adata_meta"))
    # # umap.csv umap坐标存储
    # pd.DataFrame(
    #     adt.obsm["X_umap"],
    #     columns=["umap_1", "umap_2"],
    #     index=sc.get.obs_df(adt, "original_name"),
    # ).reset_index().to_csv(resdir.joinpath("umap.csv"), index=False)

    # test umap.csv
    # temp_obs = pd.read_csv(resdir.joinpath("obs.csv"))
    # temp_obs = temp_obs.merge(pd.read_csv(resdir.joinpath("umap.csv")),on="original_name")
    # temp_adata = sc.AnnData(obs=temp_obs)
    # sc.pl.scatter(temp_adata,"umap_1", "umap_2",color="dataset")
    # sc.pl.scatter(temp_adata,"umap_1", "umap_2",color="celltype")
    # del temp_adata,temp_obs

    # ratio
    display(obs["is_right"].sum() / obs["is_right"].size) if is_display else None

    obs["name"] = obs["original_name"].str.extract(";(.+)", expand=False)
    obs["sp"] = obs["dataset"].str.extract(
        "{}_(\\w+)".format(tissue_name), expand=False
    )
    # 物种
    df_sp = (
        group_agg(obs, ["sp"], {"is_right": ["sum", "count"]})
        .eval("ratio = is_right_sum/is_right_count")
        .sort_values(["sp"])
    )
    df_sp["type"] = "species"
    # 各个dataset
    df_dataset = (
        group_agg(obs, ["sp", "name"], {"is_right": ["sum", "count"]})
        .eval("ratio = is_right_sum/is_right_count")
        .sort_values(["sp", "name"])
    )
    df_dataset["type"] = "dataset"
    df_ratio = pd.concat([df_sp, df_dataset], axis=0)
    df_ratio["tissue"] = tissue_name
    df_ratio = df_ratio.loc[
        :,
        "tissue,type,sp,name,is_right_sum,is_right_count,ratio".split(","),
    ]
    display(df_ratio) if is_display else None
    df_ratio.to_csv(resdir.joinpath("ratio.csv"), index=False)
    del df_sp, df_dataset, df_ratio

    # heatmap
    # all,reference,query
    for q, sp in zip(
        (
            "{col} == {col}".format(col=obs.columns[0]),
            # all, both sp_ref and sp_que
            "sp == '{sp}'".format(sp=sp_ref),
            "sp == '{sp}'".format(sp=sp_que),
        ),
        ("all", sp_ref, sp_que),
    ):
        res = (
            group_agg(
                obs.query(q),
                ["celltype", "predicted"],
                {"predicted": ["count"]},
            )
            .pivot(
                index="celltype",
                columns="predicted",
                values="predicted_count",
            )
            .fillna(0)
        )
        display(res) if is_display else None
        res.to_csv(resdir.joinpath("predicted_count_{}.csv".format(sp)), index=True)
        ax = sns.heatmap(data=stats.zscore(res, axis=1), cmap=plt.get_cmap("Greens"))

        ax.set_title("{}-{}".format(tissue_name, sp))
        ax.figure.savefig(
            figdir.joinpath("heatmap_ratio_{}.pdf".format(ax.get_title())),
            bbox_inches="tight",
            dpi=120,
        )
        ax.figure.clear()


def run_came(
    path_adata_ref,
    path_adata_que,
    key_class_ref,
    key_class_que,
    sp_ref,
    sp_que,
    tissue_name,
    path_varmap,
    aligned=False,
    resdir_tag=".",
    resdir=Path("."),
    limite_func=lambda adata1, adata2: (adata1, adata2),
    **kvargs,
):
    """
    version:0.0.5
    kvargs:
        n_epochs: int
            default,500,但是为了与TACMAN统一
            n_epochs = sum(kvargs.setdefault("n_epochs",[100,200,200]))

        is_1v1: bool
            default,False
        n_degs:
            default,50
            ntop_deg = n_degs
            ntop_deg_nodes = n_degs

    """
    # Parameter settings
    # n_epochs = 500
    n_epochs = sum(kvargs.setdefault("n_epochs", [100, 200, 200]))
    batch_size = None
    n_pass = 100
    use_scnets = True
    # n_hvgs = kvargs.setdefault('n_hvgs', 2000)
    n_degs = kvargs.setdefault("n_degs", 50)
    ntop_deg = n_degs  # 50
    ntop_deg_nodes = n_degs  # 50
    node_source = "deg,hvg"
    # keep_non1v1_feats = True
    keep_non1v1_feats = not kvargs.setdefault("is_1v1", False)

    # setting directory for results
    if len(resdir_tag) > 0:

        resdir_tag = "{}_{}-corss-{};{}".format(tissue_name, sp_ref, sp_que, resdir_tag)
    else:
        resdir_tag = "{}_{}-corss-{}".format(tissue_name, sp_ref, sp_que)

    resdir = Path(resdir).joinpath(resdir_tag)

    # 终止 判断
    p_finish = resdir.joinpath("finish")
    if p_finish.exists():
        # precess_after_came(resdir,tissue_name,sp_ref, sp_que)
        print(
            "[has finish]{} {}".format(
                time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
            )
        )
        return
    print(
        "[start]{} {}".format(
            time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
        )
    )
    # return

    figdir = resdir.joinpath("figs")
    sc.settings.figdir = figdir
    resdir.mkdir(parents=True, exist_ok=True)

    finish_content = ["[strat] {}".format(time.time())]

    # # setting

    dsnames = ("{}_{}".format(tissue_name, sp_ref), "{}_{}".format(tissue_name, sp_que))
    dsn1, dsn2 = dsnames
    homo_method = "biomart"

    # load data
    adata_raw1 = load_adata(path_adata_ref)
    adata_raw2 = load_adata(path_adata_que)
    key_class = key_class_ref
    if key_class not in adata_raw2.obs.columns:
        adata_raw2.obs[key_class] = ""

    # limite 进一步对adata进行限制，默认不操作直接返回
    adata_raw1, adata_raw2 = limite_func(adata_raw1, adata_raw2)

    # group_counts_unalign.csv
    pd.concat(
        [
            adata_raw1.obs[key_class_ref].value_counts(),
            adata_raw2.obs[key_class_que].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    ).to_csv(resdir.joinpath("group_counts_unalign.csv"), index=True)
    # align
    if aligned:
        adata_raw1, adata_raw2 = aligned_type(
            [adata_raw1, adata_raw2], key_calss=key_class_ref
        )

    # 保存obs ,即真正测试的细胞的mata
    adata_raw1.obs.to_csv(resdir.joinpath("obs_ref.csv"), index=True)
    adata_raw2.obs.to_csv(resdir.joinpath("obs_que.csv"), index=True)

    # group_counts.csv
    temp = pd.concat(
        [
            adata_raw1.obs[key_class_ref].value_counts(),
            adata_raw2.obs[key_class_que].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    )
    # came会自行导出
    # temp.to_csv(resdir.joinpath("group_counts.csv"), index=True)
    # if temp.shape[0] < 2:
    #     # 错误标记
    #     print("[Error][group_counts no any item]")
    #     finish_content.append(
    #         "[Error][group_counts no any item] %f" % time.time()
    #     )
    #     p_finish.with_name("error").write_text("\n".join(finish_content))
    #     return

    adatas = [adata_raw1, adata_raw2]
    print("cell count --> {}".format(sum([i.shape[0] for i in adatas])))

    df_varmap = pd.read_csv(path_varmap, usecols=range(3))
    df_varmap.columns = ["gn_ref", "gn_que", "homology_type"]
    if kvargs.setdefault("is_1v1", False):
        df_varmap = get_1v1_matches(df_varmap)
        homology_parameter = get_homology_parameters(adata_raw1, adata_raw2, df_varmap)
        print(
            """
[homology one2one]find {homology_one2one_find} genes
[homology one2one]use {homology_one2one_use} genes""".format(
                **homology_parameter
            )
        )
        kvargs.update(homology_parameter)

    df_varmap_1v1 = came.pp.take_1v1_matches(df_varmap)

    kvargs.update(
        {
            "path_adata_ref": str(path_adata_ref),
            "path_adata_que": str(path_adata_que),
            "key_class_ref": key_class_ref,
            "key_class_que": key_class_que,
            "sp_ref": sp_ref,
            "sp_que": sp_que,
            "tissue_name": tissue_name,
            "path_varmap": str(path_varmap),
            "aligned": aligned,
            "resdir_tag": resdir_tag,
            "resdir": str(resdir),
            "n_degs": n_degs,
        }
    )
    resdir.joinpath("kvargs.json").write_text(dumps(kvargs))

    finish_content.append("[finish before run] {}".format(time.time()))
    warnings.filterwarnings("ignore")

    came_inputs, (adata1, adata2) = came.pipeline.preprocess_unaligned(
        adatas,
        key_class=key_class_ref,
        use_scnets=use_scnets,
        ntop_deg=ntop_deg,
        ntop_deg_nodes=ntop_deg_nodes,
        node_source=node_source,
    )

    outputs = came.pipeline.main_for_unaligned(
        **came_inputs,
        df_varmap=df_varmap,
        df_varmap_1v1=df_varmap_1v1,
        dataset_names=dsnames,
        key_class1=key_class_ref,
        key_class2=key_class_que,
        do_normalize=True,
        keep_non1v1_feats=keep_non1v1_feats,
        n_epochs=n_epochs,
        resdir=resdir,
        n_pass=n_pass,
        batch_size=batch_size,
        plot_results=True,
    )

    finish_content.append("[finish run] {}".format(time.time()))

    dpair = outputs["dpair"]
    trainer = outputs["trainer"]
    h_dict = outputs["h_dict"]
    out_cell = outputs["out_cell"]
    predictor = outputs["predictor"]

    obs_ids1, obs_ids2 = dpair.obs_ids1, dpair.obs_ids2
    obs = dpair.obs
    classes = predictor.classes
    # 后处理
    precess_after_came(resdir, tissue_name, sp_ref, sp_que)
    finish_content.append("[finish after run] {}".format(time.time()))

    # 完成标记

    finish_content.append("[end] {}".format(time.time()))
    p_finish.write_text("\n".join(finish_content))


# ## TACMAN


def precess_after_TACMAN(
    resdir, tissue_name, sp_ref, sp_que, is_display=False, **kvargs
):
    """
    kvargs:
        adt
    """
    unify_group_counts_index_name(resdir)

    resdir = Path(resdir)
    assert resdir.joinpath("res_2").exists(), "[not exists] res_2"
    figdir = resdir.joinpath("figs")
    sc.settings.figdir = figdir

    (
        display(pd.read_csv(resdir / "group_counts.csv", index_col=0))
        if is_display
        else None
    )

    # 为obs添加dataset
    obs = pd.read_csv(resdir.joinpath("res_2", "pre_out_2.csv"), index_col=0)
    if not obs.index.is_unique:
        print("[obs index is not unique]")

    pre_obs = {
        "{}_{}".format(tissue_name, k): pd.read_csv(
            resdir.joinpath(file_name), index_col=0
        )
        for k, file_name in zip([sp_ref, sp_que], ["obs_ref.csv", "obs_que.csv"])
    }
    for k, value in pre_obs.items():
        value["dataset"] = k
        pre_obs[k] = value.loc[:, ["dataset"]]
    obs = pd.concat(pre_obs.values()).join(obs)
    obs["is_right"] = obs["true_label"] == obs["pre_label"]
    del pre_obs

    # umap
    adt = kvargs.setdefault("adt", None)
    assert adt is not None
    sc.pp.neighbors(adt, n_neighbors=15, metric="cosine", use_rep="X")
    sc.tl.umap(adt)
    sc.pl.umap(adt, color="dataset", save="_dataset.png")
    sc.pl.umap(adt, color="cell_type", save="_umap.png")

    adt.write_csvs(resdir.joinpath("adata_meta"))
    # # 存储adt.X 和 umap坐标
    # pd.DataFrame(adt.X, index=adt.obs.index).to_orc(resdir.joinpath("adt.X.orc"))
    # pd.DataFrame(
    #     adt.obsm["X_umap"], columns=["umap_1", "umap_2"], index=adt.obs.index
    # ).assign(**{k: adt.obs[k] for k in adt.obs.keys()}).reset_index().to_csv(
    #     resdir.joinpath("umap.csv"), index=False
    # )

    # # test umap.csv
    # temp_obs = pd.read_csv(resdir.joinpath("res_2", "pre_out_2.csv"), index_col=0)
    # temp_obs = temp_obs.join(pd.read_csv(resdir.joinpath("umap.csv"),index_col=0))
    # temp_adata = sc.AnnData(obs=temp_obs)
    # sc.pl.scatter(temp_adata,"umap_1", "umap_2",color="dataset")
    # sc.pl.scatter(temp_adata,"umap_1", "umap_2",color="cell_type")
    # del temp_adata,temp_obs

    # ratio
    obs["name"] = obs.index.to_series().str.extract(";([^;]+)$", expand=False)
    obs["sp"] = obs["dataset"].str.extract(
        "{}_(\\w+)".format(tissue_name), expand=False
    )
    display(obs["is_right"].sum() / obs["is_right"].size) if is_display else None
    # 物种
    df_sp = group_agg(obs, ["sp"], {"is_right": ["count", "sum"]})
    df_sp["type"] = "species"
    # 各个dataset
    df_dataset = group_agg(obs, ["sp", "name"], {"is_right": ["count", "sum"]})
    df_dataset["type"] = "dataset"

    df_ratio = pd.concat([df_sp, df_dataset]).eval(
        "ratio = is_right_sum/is_right_count"
    )
    df_ratio["tissue"] = tissue_name
    df_ratio = df_ratio.loc[
        :,
        "tissue,type,sp,name,is_right_sum,is_right_count,ratio".split(","),
    ]
    df_ratio.to_csv(resdir / "ratio.csv", index=False)
    del df_sp, df_dataset, df_ratio

    # heatmap
    # all,reference,query
    for q, sp in zip(
        (
            "{col} == {col}".format(col=obs.columns[0]),
            # all, both sp_ref and sp_que
            "sp == '{sp}'".format(sp=sp_ref),
            "sp == '{sp}'".format(sp=sp_que),
        ),
        ("all", sp_ref, sp_que),
    ):
        res = (
            group_agg(
                obs.query(q),
                ["true_label", "pre_label"],
                {"pre_label": ["count"]},
                dropna=False,
            )
            .pivot(
                index="true_label",
                columns="pre_label",
                values="pre_label_count",
            )
            .fillna(0)
        )
        display(res) if is_display else None

        res.to_csv(resdir.joinpath("predicted_count_{}.csv".format(sp)), index=True)

        ax = sns.heatmap(data=stats.zscore(res, axis=1), cmap=plt.get_cmap("Greens"))

        ax.set_title("{}-{}".format(tissue_name, sp))
        ax.figure.savefig(
            figdir.joinpath("heatmap_ratio_{}.pdf".format(ax.get_title())),
            bbox_inches="tight",
            dpi=120,
        )
        ax.figure.clear()


def run_TACMAN(
    path_adata_ref,
    path_adata_que,
    key_class_ref,
    key_class_que,
    sp_ref,
    sp_que,
    tissue_name,
    path_varmap,
    aligned=False,
    resdir_tag=".",
    resdir=Path("."),
    limite_func=lambda adata1, adata2: (adata1, adata2),
    **kvargs,
):
    """
    version = 0.0.9
    kvargs:
        n_epochs:
            default,[100, 200, 300]
            stages,即res_0,res_1，res_2 的 epochs
            累加制，res_0,res_1，res_2,实际epochs分别为100,300,600
            故最终epochs为stages之和
            stages = kvargs.setdefault("n_epochs",[100, 200, 300])

        is_1v1: bool
            default,False
        n_hvgs:
            default,2000
        n_degs:
            default,50
    """

    homo_method = "biomart"
    n_hvgs = kvargs.setdefault("n_hvgs", 2000)
    n_degs = kvargs.setdefault("n_degs", 50)
    seed = 123
    stages = kvargs.setdefault("n_epochs", [100, 200, 200])  # [200, 200, 200]
    nfeats = kvargs.setdefault("nfeats", 64)  # 64  # embedding size #128
    hidden = kvargs.setdefault("hidden", 64)  # 64  # 128
    input_drop = 0.2
    att_drop = 0.2
    residual = True

    threshold = 0.9  # 0.8
    lr = 0.01  # lr = 0.01
    weight_decay = 0.001
    patience = 100
    enhance_gama = 10
    simi_gama = 0.1

    dsnames = ("{}_{}".format(tissue_name, sp_ref), "{}_{}".format(tissue_name, sp_que))
    assert key_class_ref == key_class_que, "key_class is not equal"
    key_class = key_class_ref

    # make file to save
    resdir_tag = (
        "{}_{}-corss-{};{}".format(tissue_name, sp_ref, sp_que, resdir_tag)
        if len(resdir_tag) > 0
        else "{}_{}-corss-{}".format(tissue_name, sp_ref, sp_que)
    )
    # curdir = os.path.join()
    resdir = Path(resdir).joinpath(resdir_tag)
    model_dir = resdir.joinpath("model_")
    figdir = resdir.joinpath("figs")
    [_.mkdir(exist_ok=True, parents=True) for _ in [resdir, model_dir, figdir]]
    [
        resdir.joinpath("res_{}".format(i)).mkdir(exist_ok=True, parents=True)
        for i in range(len(stages))
    ]

    checkpt_file = model_dir.joinpath("mutistages")

    # is finish
    p_finish = Path(resdir).joinpath("finish")
    if p_finish.exists():
        print(
            "[has finish]{} {}".format(
                time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
            )
        )
        return
    print(
        "[start]{} {}".format(
            time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
        )
    )

    finish_content = ["[strat] {}".format(time.time())]
    print("[path_varmap] {}".format(path_varmap))
    adata_raw1 = load_adata(path_adata_ref)
    adata_raw2 = load_adata(path_adata_que)
    if key_class not in adata_raw2.obs.columns:
        adata_raw2.obs[key_class] = ""
    adata_raw1.obs[key_class] = adata_raw1.obs[key_class].astype(str)
    adata_raw2.obs[key_class] = adata_raw2.obs[key_class].astype(str)
    # limite 进一步对adata进行限制，默认不操作直接返回
    adata_raw1, adata_raw2 = limite_func(adata_raw1, adata_raw2)
    # group_counts_unalign.csv
    pd.concat(
        [
            adata_raw1.obs[key_class].value_counts(),
            adata_raw2.obs[key_class].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    ).to_csv(resdir.joinpath("group_counts_unalign.csv"), index=True)
    # 仅保留公共细胞类群
    if aligned:
        adata_raw1, adata_raw2 = TACMAN.pp.aligned_type(
            [adata_raw1, adata_raw2], key_class
        )

    # group_counts.csv
    temp = pd.concat(
        [
            adata_raw1.obs[key_class].value_counts(),
            adata_raw2.obs[key_class].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    )
    print(temp)
    temp.to_csv(resdir.joinpath("group_counts.csv"), index=True)
    adata_raw1.obs.to_csv(resdir.joinpath("obs_ref.csv"), index=True)
    adata_raw2.obs.to_csv(resdir.joinpath("obs_que.csv"), index=True)

    # homo = pd.read_csv(path_varmap)
    homo = pd.read_csv(path_varmap, usecols=range(3))
    homo.columns = ["gn_ref", "gn_que", "homology_type"]
    if kvargs.setdefault("is_1v1", False):
        homo = get_1v1_matches(homo)
        homology_parameter = get_homology_parameters(adata_raw1, adata_raw2, homo)
        print(
            """
[homology one2one]find {homology_one2one_find} genes
[homology one2one]use {homology_one2one_use} genes""".format(
                **homology_parameter
            )
        )
        kvargs.update(homology_parameter)

    kvargs.update(
        {
            "path_adata_ref": str(path_adata_ref),
            "path_adata_que": str(path_adata_que),
            "key_class_ref": key_class_ref,
            "key_class_que": key_class_que,
            "sp_ref": sp_ref,
            "sp_que": sp_que,
            "tissue_name": tissue_name,
            "path_varmap": str(path_varmap),
            "aligned": aligned,
            "resdir_tag": resdir_tag,
            "resdir": str(resdir),
            "n_hvgs": n_hvgs,
            "n_degs": n_degs,
            "nfeats": nfeats,
            "hidden": hidden,
        }
    )
    resdir.joinpath("kvargs.json").write_text(dumps(kvargs))
    print(
        """Task: refernece:{} {} cells x {} gene -> query:{} {} cells x {} gene in {}""".format(
            dsnames[0],
            adata_raw1.shape[0],
            adata_raw1.shape[1],
            dsnames[1],
            adata_raw2.shape[0],
            adata_raw2.shape[1],
            tissue_name,
        )
    )

    start = time.time()
    finish_content.append("[finish before run] {}".format(time.time()))
    # knn时间较长
    print("\n[process_for_graph]\n".center(100, "-"))

    adatas, features_genes, nodes_genes, scnets, one2one, n2n = (
        TACMAN.pp.process_for_graph(
            [adata_raw1, adata_raw2],
            homo,
            key_class,
            "leiden",
            n_hvgs=n_hvgs,
            n_degs=n_degs,
        )
    )

    g, inter_net, one2one_gene_nodes_net, cell_label, n_classes, list_idx = (
        TACMAN.pp.make_graph(
            adatas,
            aligned,
            key_class,
            features_genes,
            nodes_genes,
            scnets,
            one2one,
            n2n,
            has_mnn=True,
            seed=seed,
        )
    )

    end = time.time()

    # 包括预处理时间
    print("Times preprocess for graph:{:.2f}".format(end - start))
    print("\n[Trainer]\n".center(100, "-"))

    trainer = TACMAN.Trainer(
        adatas,
        g,
        inter_net,
        list_idx,
        cell_label,
        n_classes,
        threshold=threshold,
        key_class=key_class,
    )
    print("\n[train]\n".center(100, "-"))
    trainer.train(
        curdir=str(resdir),
        checkpt_file=str(checkpt_file),
        nfeats=nfeats,
        hidden=hidden,
        enhance_gama=enhance_gama,
        simi_gama=simi_gama,
    )

    finish_content.append("[finish run] {}".format(time.time()))
    adt = sc.AnnData(
        trainer.embedding_hidden.detach().numpy(),
        obs=pd.concat(
            [
                adatas[0].obs.loc[:, [key_class]].assign(dataset=dsnames[0]),
                adatas[1].obs.loc[:, [key_class]].assign(dataset=dsnames[1]),
            ]
        ).rename(columns={key_class: "cell_type"}),
    )
    # plot_umap(trainer.embedding_hidden, adatas, dsnames, figdir)
    adt.write_h5ad(resdir.joinpath("adt.h5ad"))
    precess_after_TACMAN(resdir, tissue_name, sp_ref, sp_que, is_display=False, adt=adt)

    # 完成标记
    finish_content.append("[end] {}".format(time.time()))
    p_finish.write_text("\n".join(finish_content))

    # return trainer, adatas


def run_TACMAN_before_custom_trainer(
    path_adata_ref,
    path_adata_que,
    key_class_ref,
    key_class_que,
    sp_ref,
    sp_que,
    tissue_name,
    path_varmap,
    aligned=False,
    resdir_tag=".",
    resdir=Path("."),
    limite_func=lambda adata1, adata2: (adata1, adata2),
    **kvargs,
):
    """
    version = 0.0.9
    kvargs:
        n_epochs:
            default,[100, 200, 300]
            stages,即res_0,res_1，res_2 的 epochs
            累加制，res_0,res_1，res_2,实际epochs分别为100,300,600
            故最终epochs为stages之和
            stages = kvargs.setdefault("n_epochs",[100, 200, 300])

        is_1v1: bool
            default,False
        n_hvgs:
            default,2000
        n_degs:
            default,50
    """
    homo_method = "biomart"
    n_hvgs = kvargs.setdefault("n_hvgs", 2000)
    n_degs = kvargs.setdefault("n_degs", 50)
    seed = 123
    stages = kvargs.setdefault("n_epochs", [100, 200, 200])  # [200, 200, 200]
    nfeats = kvargs.setdefault("nfeats", 64)  # 64  # enbedding size #128
    hidden = kvargs.setdefault("hidden", 64)  # 64  # 128
    input_drop = 0.2
    att_drop = 0.2
    residual = True

    threshold = 0.9  # 0.8
    lr = 0.01  # lr = 0.01
    weight_decay = 0.001
    patience = 100
    enhance_gama = 10
    simi_gama = 0.1

    dsnames = ("{}_{}".format(tissue_name, sp_ref), "{}_{}".format(tissue_name, sp_que))
    assert key_class_ref == key_class_que, "key_class is not equal"
    key_class = key_class_ref

    # make file to save
    resdir_tag = (
        "{}_{}-corss-{};{}".format(tissue_name, sp_ref, sp_que, resdir_tag)
        if len(resdir_tag) > 0
        else "{}_{}-corss-{}".format(tissue_name, sp_ref, sp_que)
    )
    # curdir = os.path.join()
    resdir = Path(resdir).joinpath(resdir_tag)
    model_dir = resdir.joinpath("model_")
    figdir = resdir.joinpath("figs")
    [_.mkdir(exist_ok=True, parents=True) for _ in [resdir, model_dir, figdir]]
    [
        resdir.joinpath("res_{}".format(i)).mkdir(exist_ok=True, parents=True)
        for i in range(len(stages))
    ]

    checkpt_file = model_dir.joinpath("mutistages")

    # # is finish
    # p_finish = Path(resdir).joinpath("finish")
    # if p_finish.exists():
    #     print(
    #         "[has finish]{} {}".format(
    #             time.strftime('%y%m%d-%H%M', time.localtime()), resdir.name
    #         ))
    #     return
    # print(
    #     "[start]{} {}".format(
    #         time.strftime('%y%m%d-%H%M', time.localtime()),
    #         resdir.name
    #     ))

    finish_content = ["[strat] {}".format(time.time())]
    print("[path_varmap] {}".format(path_varmap))
    adata_raw1 = load_adata(path_adata_ref)
    adata_raw2 = load_adata(path_adata_que)
    if key_class not in adata_raw2.obs.columns:
        adata_raw2.obs[key_class] = ""

    # limite 进一步对adata进行限制，默认不操作直接返回
    adata_raw1, adata_raw2 = limite_func(adata_raw1, adata_raw2)
    # # group_counts_unalign.csv
    # pd.concat([adata_raw1.obs[key_class].value_counts(),
    #            adata_raw2.obs[key_class].value_counts(),],
    #           axis=1, keys=dsnames,).to_csv(
    #     resdir.joinpath("group_counts_unalign.csv"), index=True
    # )
    # 仅保留公共细胞类群
    if aligned:
        adata_raw1, adata_raw2 = TACMAN.pp.aligned_type(
            [adata_raw1, adata_raw2], key_class
        )

    # group_counts.csv
    temp = pd.concat(
        [
            adata_raw1.obs[key_class].value_counts(),
            adata_raw2.obs[key_class].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    )
    print(temp)
    # temp.to_csv(resdir.joinpath("group_counts.csv"), index=True)
    # adata_raw1.obs.to_csv(resdir.joinpath("obs_ref.csv"), index=True)
    # adata_raw2.obs.to_csv(resdir.joinpath("obs_que.csv"), index=True)

    # homo = pd.read_csv(path_varmap)
    homo = pd.read_csv(path_varmap, usecols=range(3))
    homo.columns = ["gn_ref", "gn_que", "homology_type"]
    if kvargs.setdefault("is_1v1", False):
        homo = get_1v1_matches(homo)
        homology_parameter = get_homology_parameters(adata_raw1, adata_raw2, homo)
        print(
            """
[homology one2one]find {homology_one2one_find} genes
[homology one2one]use {homology_one2one_use} genes""".format(
                **homology_parameter
            )
        )
        kvargs.update(homology_parameter)

    kvargs.update(
        {
            "path_adata_ref": str(path_adata_ref),
            "path_adata_que": str(path_adata_que),
            "key_class_ref": key_class_ref,
            "key_class_que": key_class_que,
            "sp_ref": sp_ref,
            "sp_que": sp_que,
            "tissue_name": tissue_name,
            "path_varmap": str(path_varmap),
            "aligned": aligned,
            "resdir_tag": resdir_tag,
            "resdir": str(resdir),
            "n_hvgs": n_hvgs,
            "n_degs": n_degs,
            "nfeats": nfeats,
            "hidden": hidden,
        }
    )
    # resdir.joinpath("kvargs.json").write_text(dumps(kvargs))
    print(
        """Task: refernece:{} {} cells x {} gene -> query:{} {} cells x {} gene in {}""".format(
            dsnames[0],
            adata_raw1.shape[0],
            adata_raw1.shape[1],
            dsnames[1],
            adata_raw2.shape[0],
            adata_raw2.shape[1],
            tissue_name,
        )
    )

    start = time.time()
    finish_content.append("[finish before run] {}".format(time.time()))
    # knn时间较长
    print("\n[process_for_graph]\n".center(100, "-"))
    adatas, features_genes, nodes_genes, scnets, one2one, n2n = (
        TACMAN.pp.process_for_graph(
            [adata_raw1, adata_raw2],
            homo,
            key_class,
            "leiden",
            n_hvgs=n_hvgs,
            n_degs=n_degs,
        )
    )
    g, inter_net, one2one_gene_nodes_net, cell_label, n_classes, list_idx = (
        TACMAN.pp.make_graph(
            adatas,
            aligned,
            key_class,
            features_genes,
            nodes_genes,
            scnets,
            one2one,
            n2n,
            has_mnn=True,
            seed=seed,
        )
    )
    end = time.time()
    # 包括预处理时间
    print("Times preprocess for graph:{:.2f}".format(end - start))
    return {
        "process_for_graph": {
            "adatas": adatas,
            "features_genes": features_genes,
            "nodes_genes": nodes_genes,
            "scnets": scnets,
            "one2one": one2one,
            "n2n": n2n,
        },
        "make_graph": {
            "g": g,
            "inter_net": inter_net,
            "one2one_gene_nodes_net": one2one_gene_nodes_net,
            "cell_label": cell_label,
            "n_classes": n_classes,
            "list_idx": list_idx,
        },
    }


# ## SAMap


def run_SAMap(
    path_adata_ref,
    path_adata_que,
    key_class_ref,
    key_class_que,
    sp_ref,
    sp_que,
    tissue_name,
    path_varmap,
    aligned=False,
    resdir_tag=".",
    resdir=Path("."),
    limite_func=lambda adata1, adata2: (adata1, adata2),
    **kvargs,
):

    path_specie_1 = path_adata_ref
    path_specie_2 = path_adata_que
    tissue = tissue_name
    species = [sp_ref, sp_que]
    dsnames = (f"{tissue_name}_{sp_ref}", f"{tissue_name}_{sp_que}")
    assert key_class_ref == key_class_que, "key_class is not equal"
    key_class = key_class_ref

    # make file to save
    resdir_tag = (
        f"{tissue_name}_{sp_ref}-corss-{sp_que};{resdir_tag}"
        if len(resdir_tag) > 0
        else f"{tissue_name}_{sp_ref}-corss-{sp_que}"
    )
    curdir = os.path.join(resdir, resdir_tag)
    resdir = Path(curdir)
    figdir = os.path.join(curdir, "figs")
    [Path(_).mkdir(exist_ok=True, parents=True) for _ in [curdir, figdir]]

    # is finish
    p_finish = Path(resdir).joinpath("finish")
    if p_finish.exists():
        print(
            "[has finish]{} {}".format(
                time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
            )
        )
        return
    print(
        "[start]{} {}".format(
            time.strftime("%y%m%d-%H%M", time.localtime()), resdir.name
        )
    )

    finish_content = ["[strat] {}".format(time.time())]
    print("[path_varmap] {}".format(path_varmap))
    adata_1 = load_adata(path_specie_1)
    adata_2 = load_adata(path_specie_2)
    assert pd.Series(
        np.concatenate((adata_1.obs.index, adata_2.obs.index))
    ).is_unique, "[Error] index is not unique"

    # limite 进一步对adata进行限制，默认不操作直接返回
    adata_1, adata_2 = limite_func(adata_1, adata_2)
    if key_class not in adata_2.obs.columns:
        adata_2.obs[key_class] = ""
    # group_counts_unalign.csv
    pd.concat(
        [
            adata_1.obs[key_class].value_counts(),
            adata_2.obs[key_class].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    ).to_csv(resdir.joinpath("group_counts_unalign.csv"), index=True)
    # 仅保留公共细胞类群
    if aligned:
        adata_1, adata_2 = TACMAN.pp.aligned_type([adata_1, adata_2], key_class)

    # group_counts.csv
    temp = pd.concat(
        [
            adata_1.obs[key_class].value_counts(),
            adata_2.obs[key_class].value_counts(),
        ],
        axis=1,
        keys=dsnames,
    )
    print(temp)
    temp.to_csv(resdir.joinpath("group_counts.csv"), index=True)
    # if temp.shape[0] < 2:
    #     # 错误标记
    #     print("[Error][group_counts no any item]")
    #     finish_content.append(
    #         "[Error][group_counts no any item] %f" % time.time()
    #     )
    #     p_finish.with_name("error").write_text("\n".join(finish_content))
    #     return

    adata_1.obs.to_csv(resdir.joinpath("obs_ref.csv"), index=True)
    adata_2.obs.to_csv(resdir.joinpath("obs_que.csv"), index=True)

    kvargs.update(
        {
            "path_adata_ref": str(path_adata_ref),
            "path_adata_que": str(path_adata_que),
            "key_class_ref": key_class_ref,
            "key_class_que": key_class_que,
            "sp_ref": sp_ref,
            "sp_que": sp_que,
            "tissue_name": tissue_name,
            "path_varmap": str(path_varmap),
            "aligned": aligned,
            "resdir_tag": resdir_tag,
            "resdir": str(resdir),
        }
    )
    resdir.joinpath("kvargs.json").write_text(dumps(kvargs))
    start = time.time()
    finish_content.append("[finish before run] {}".format(time.time()))
    # SAMap -----------------------------------------------------
    sq_ref_SAMap = map_sp_SAMap[map_sp[sp_ref]]
    sq_que_SAMap = map_sp_SAMap[map_sp[sp_que]]
    _SAMP = {sq_ref_SAMap: SAM(counts=adata_1), sq_que_SAMap: SAM(counts=adata_2)}
    [v.preprocess_data() for k, v in _SAMP.items()]
    [v.run() for k, v in _SAMP.items()]
    keys = {sq_ref_SAMap: key_class_ref, sq_que_SAMap: key_class_que}
    sm = SAMAP(
        _SAMP,
        # f_maps + "{}{}/{}_to_{}.txt".format(id2, id1, id2, id1)
        # 。。。昂,不能传Path,得传str,还得加个 os.sep
        f_maps=(
            str(path_varmap)
            if str(path_varmap).endswith(os.sep)
            else str(path_varmap) + os.sep
        ),
        keys=keys,
        is_1v1=kvargs.setdefault("is_1v1", False),
    )

    # run SAMap
    sm.run(pairwise=False)
    samap = sm.samap  # SAM object with three species stitched together

    finish_content.append("[finish run] {}".format(time.time()))

    alignment_score = samap.adata.obs.loc[:, ["species"]].join(
        get_alignment_score_for_each_cell(sm, keys)
    )
    alignment_score.to_csv(
        resdir.joinpath("alignment_score_for_each_cell.csv"), index=True
    )
    alignment_score.head(2)

    alignment_score_query = alignment_score.query(
        "species == '{}'".format(sq_que_SAMap)
    ).filter(regex="^{}".format(sq_ref_SAMap))
    # 取每个细胞 在各类型上alignment score 最大值对应的类型
    alignment_score_query["pre_label"] = [
        alignment_score_query.columns[i]
        for i in np.argmax(alignment_score_query, axis=1)
    ]
    alignment_score_query["pre_label"] = (
        alignment_score_query["pre_label"]
        .str.replace("{}_".format(sq_ref_SAMap), "")
        .str.replace("{}_".format(sq_que_SAMap), "")
    )
    alignment_score_query["max_prob"] = alignment_score_query.filter(
        regex="^{}".format(sq_ref_SAMap)
    ).max(axis=1)
    alignment_score_query.head(2)

    # constrect the res_obs
    res_obs = samap.adata.obs.loc[:, ["species"]].copy()
    res_obs = res_obs.join(
        pd.DataFrame(
            samap.adata.obsm["X_umap"],
            columns="UMAP1,UMAP2".split(","),
            index=samap.adata.obs.index,
        )
    )
    res_obs["dataset"] = res_obs["species"].map(
        {k: v for k, v in zip([sq_ref_SAMap, sq_que_SAMap], dsnames)}
    )
    res_obs["cell_type"] = (
        samap.adata.obs["{};{}_mapping_scores".format(key_class_ref, key_class_que)]
        .str.replace("^{}_".format(sq_ref_SAMap), "", regex=True)
        .str.replace("^{}_".format(sq_que_SAMap), "", regex=True)
    )
    res_obs["true_label"] = res_obs["cell_type"]
    res_obs = res_obs.drop(columns="species")

    res_obs = res_obs.join(
        alignment_score_query.loc[:, "pre_label,max_prob".split(",")]
    )
    res_obs["is_right"] = res_obs.eval("true_label == pre_label")
    res_obs.to_csv(resdir.joinpath("obs.csv"), index=True)

    # df_ratio
    df_ratio = group_agg(
        res_obs, "dataset,is_right".split(","), {"is_right": ["sum"]}
    ).merge(
        res_obs["dataset"].value_counts().to_frame(name="dataset_count"), on="dataset"
    )
    df_ratio = (
        df_ratio.query("is_right")
        .drop(columns="is_right")
        .rename(columns={"dataset_count": "is_right_count"})
    )
    df_ratio = df_ratio.join(
        df_ratio["dataset"].str.extract("(?P<tissue>[^_]+)_(?P<sp>[^_]+)")
    )
    assert df_ratio["sp"].is_unique, "[Error] not unique"
    df_ratio.index = df_ratio["sp"].to_numpy()
    df_ratio.loc[sp_ref, "is_right_sum"] = np.nan

    df_ratio["ratio"] = df_ratio.eval("is_right_sum/is_right_count")

    df_ratio["type"] = "species"
    df_ratio["name"] = ""
    df_ratio = df_ratio.loc[
        :, "tissue,type,sp,name,is_right_sum,is_right_count,ratio".split(",")
    ]
    df_ratio.to_csv(resdir.joinpath("ratio.csv"), index=False)

    # plot umap
    samap.adata.obs["cell_type"] = (
        samap.adata.obs["{};{}_mapping_scores".format(key_class_ref, key_class_que)]
        .str.replace("^{}_".format(sq_ref_SAMap), "", regex=True)
        .str.replace("^{}_".format(sq_que_SAMap), "", regex=True)
    )
    samap.adata.obs["dataset"] = samap.adata.obs["species"].map(
        {k: v for k, v in zip([sq_ref_SAMap, sq_que_SAMap], dsnames)}
    )
    display(samap.adata.obs.head(2))

    ax = sc.pl.umap(samap.adata, color="dataset", show=False)
    ax.figure.savefig(
        Path(figdir).joinpath("umap_dataset.png"), bbox_inches="tight", dpi=120
    )
    ax = sc.pl.umap(samap.adata, color="cell_type", show=False)
    ax.figure.savefig(
        Path(figdir).joinpath("umap_umap.png"), bbox_inches="tight", dpi=120
    )

    # 完成标记
    finish_content.append("[end] {}".format(time.time()))
    p_finish.write_text("\n".join(finish_content))


# ## run_cross_species_models


map_func_run_cross_species_models = {
    "came": run_came,
    "CAME": run_came,
    "TACMAN": run_TACMAN,
    "TACMAN": run_TACMAN,
    "SAMap": run_SAMap,
    "TACMAN_before_custom_trainer": run_TACMAN_before_custom_trainer,
}

# del run_came, run_TACMAN, run_SAMap, run_TACMAN_before_custom_trainer


def run_cross_species_models(
    path_adata_ref,
    path_adata_que,
    key_class_ref,
    key_class_que,
    sp_ref,
    sp_que,
    tissue_name,
    resdir,
    resdir_tag="",
    aligned=False,
    limite_func=lambda adata1, adata2: (adata1, adata2),
    models="".split(","),
    **kvargs,
):
    """
    models: came,TACMAN,SAMap,TACMAN_before_custom_trainer
    kvargs:
    n_epochs:
        default,[100, 200, 300]
        stages,即res_0,res_1，res_2 的 epochs
        累加制，res_0,res_1，res_2,实际epochs分别为100,300,600
        故最终epochs为stages之和
        stages = kvargs.setdefault("n_epochs",[100, 200, 300])


    is_1v1: bool
        default,False
    """

    assert pd.Series(
        [model in map_func_run_cross_species_models.keys() for model in models]
    ).all(), "[Error] not all models in {}\nmodels {}".format(
        ",".join(map_func.keys()),
        ",".join(models),
    )
    res = {}
    for model in models:
        path_varmap = get_path_varmap(map_sp[sp_ref], map_sp[sp_que], model=model)
        print(path_varmap)
        print("[path_varmap] {}\t{}".format(model, Path(path_varmap).name))
        _res = map_func_run_cross_species_models[model](
            path_adata_ref,
            path_adata_que,
            key_class_ref,
            key_class_que,
            sp_ref,
            sp_que,
            tissue_name,
            path_varmap,
            aligned=aligned,
            resdir_tag=";".join([model, resdir_tag]),
            resdir=resdir,
            limite_func=limite_func,
            **kvargs,
        )
        res.update({model: _res})
    if len(res.keys()) == 1:
        res = list(res.values())[0]
    return res


# # help


__all__ = [
    i
    for i in """
p_root
p_run
p_plot
p_res
p_cache
p_pdf
p_data_process

map_sp
map_sp_reverse
map_sp_SAMap
run_cross_species_models

display

h5ad_to_mtx
load_adata
load_normalized_adata

pdf2_merge
show_umap

time_tag_detect
time_tag_get
time_tag_toggle

get_test_result_df
get_adata_umap
get_res_obs
get_source_obs
get_matrix_max_prob_median
get_matrix_count
get_path_varmap
get_res_stat
find_path_from_para

func_help
run_came
run_TACMAN
TACMAN
run_SAMap
""".split(
        "\n"
    )
    if len(i) > 0
]


dir_context = show_str_arr(
    [i for i in dir() if not i.startswith("_")], n_cols=2, return_str=True
)


def func_help(show_absolute=False):
    text = """{}
> parameter
    p_root\t{}
        p_run, p_plot, p_res, p_cache, p_pdf
    p_df_varmap
    map_sp_reverse
    rng
{}
{}

{}
{}
""".format(
        "[func help]".ljust(75, "-"),
        p_root.absolute() if show_absolute else "[name] {}".format(p_root.name),
        "[from func import * ]".ljust(75, "-"),
        show_str_arr(__all__, n_cols=3, return_str=True),
        "[func all name ]".ljust(75, "-"),
        dir_context,
    )

    print(text)


class Info:
    def __init__(self, **kvargs):
        self.__dict__["_data"] = kvargs

    def __getattr__(self, name):
        if name in self._data.keys():
            return self._data[name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        if name in self._data.keys():
            self._data[name] = value
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __repr__(self):
        return "{}\n{}".format(
            self.__class__.__name__,
            "\n".join(["  {}:{}".format(k, v) for k, v in self._data.items()]),
        )

    def __deepcopy__(self, memo):
        return self.__class__(**self._data)

    def drop(self, *keys):
        for k in keys:
            if k in self._data.keys():
                self._data.pop(k)

    def update(self, **kvargs):
        self._data.update(kvargs)

    def keys(self):
        return self._data.keys()


if __name__ != "__main__":
    func_help()
