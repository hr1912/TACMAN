#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from pathlib import Path


def sys_path_show():
    print(*sys.path, sep='\n')


def sys_path_append(p):
    p = Path(p)
    assert p.exists()
    p = str(p)
    None if p in sys.path else sys.path.append(p)


sys_path_append(Path("~/link/other_model/learn_scGCN/scGCN").expanduser())


# In[ ]:


import numpy as np
import pandas as pd
import scanpy as sc
import scipy


# In[1]:


def adata_matrix_to_df(adata, layer=None):
    X = adata.X if layer is None else adata.layers[layer]
    if isinstance(X, (np.ndarray, pd.DataFrame)):
        X = X
    elif scipy.sparse.issparse(X):
        X = X.todense()
    return pd.DataFrame(X, index=adata.obs.index, columns=adata.var.index)


def select_feature(adata, key_cell_type, key_pval='pvalue_oneway', nf=2000):
    if not key_pval in adata.var.columns:
        # 用不着多线程
        # adata_var_oneway(adata,key_cell_type,key_pval)
        adata_split = {k: adata[adata.obs[key_cell_type] == k, :]
                       for k in adata.obs[key_cell_type].unique()}
        adata.var[key_pval] = scipy.stats.f_oneway(
            *[sc.get.obs_df(ad, adata.var.index.to_list())
              for k, ad in adata_split.items()]).pvalue

    return adata.var.sort_values([key_pval]).head(nf).index


def adata_norm_scale_for_scGCN(adata):
    adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.layers['count_data'] = adata.X.copy()
    # Seurat:::NormalizeData.default的线性标准化scale.factor默认为10000
    # 并且对数化
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    adata.layers['norm_data'] = adata.X.copy()
    # Seurat:::ScaleData.default 的 scale.max = 10
    sc.pp.scale(adata, max_value=10)
    adata.layers['scale_data'] = adata.X.copy()
    return adata


def df_group_agg(df, groupby_list, agg_dict=None, dropna=True,
                 reindex=True, recolumn=True, rename_dict=None):
    groupby_list = handle_type_to_list(groupby_list)
    if None is agg_dict:
        agg_dict = {groupby_list[-1]: ['count']}

    res = df.groupby(
        groupby_list,
        dropna=dropna,
        observed=False).agg(agg_dict)
    if recolumn:
        res.columns = ["_".join(i) for i in res.columns]
    if reindex:
        res = res.index.to_frame().join(res)
        res.index = np.arange(res.shape[0])
    if isinstance(rename_dict, dict):
        res = res.rename(columns=lambda k: rename_dict.setdefault(k, k))
    return res


def df_to_dict(df, key_key, key_value, check_key_unique=False):
    if check_key_unique:
        assert df[key_key].is_unique, "[Error] col '{}' is not unique".format(
            key_key)
    res = {k: v for k, v in zip(df[key_key], df[key_value])}
    return res


def df_reindex_with_unique_col(df, key, drop=False):
    assert df[key].is_unique, "[Error][not unique] {} column".format(key)
    df.index = df[key].to_numpy()
    if drop:
        df = df.drop(columns=[key])
    return df


def handle_type_to_list(data, t=str):
    """若 data 为 t 则返回 [data]
若 data 为list 则返回 data
"""
    assert isinstance(data, (list, t)), '[Error] data is not a list or {}'.format(t)
    return [data] if isinstance(data, t) else data


# # 对于scGCN本地化的修改
# 
# > 追加文件`scGCN/__init__.py`
# 
# 将scGCN/scGCN视为包
# 
# ```python
# from . import utility
# from . import utils
# from . import graph
# from . import data
# ```
# > 修改文件`scGCN/data.py`
# 
# ```python
# # from graph import *
# from .graph import *
# ```
# 
# > 修改文件`scGCN/graph.py`
# 
# ```python
# # from utility import *
# from .utility import *
# ```
# 
# > 修改文件`scGCN/layers.py`
# 
# ```python
# # from utils import *
# from .utils import *
# ```
# 
# 
# > 修改文件`scGCN/models.py`
# 
# ```python
# # from layers import *
# # from utils import *
# from .layers import *
# from .utils import *
# 
# ```
# 
# > 修改文件`scGCN/train.py`
# ```python
# 
# ```
# 
# > 修改文件`scGCN/utils.py`
# ```python
# # from data import *
# from .data import *
# ```
# 
# 
# > 其他 见下文
# 
# |file|function|note||
# |:-|:-|:-|:-|
# |utility|runCCA|`.loc`|key error 取个交集|
# |utility|generate_graph|dataframe从slice实例化|提示在dataframe的slice上进行赋值操作|
# |graph|graph_construct|更改函数签名,数据的获取方式|直接将数据传入,而非从文件中读取|
# |||更改输出路径||
# |||添加完成判定||
# |data|input_data|更改函数签名,数据的获取方式|直接将数据传入,而非从文件中读取|
# |utils|load_data|不调用data.input_data||
# |||在load_data执行前|在外部手动调用data.input_data|
# |||更改inter_graph.csv和intra_graph的读入路径||
# |||更改类型转换写法|`.astype('Float64')`-> `.astype(np.float64)`|
# |utility|svd1|大数据量时`np.linalg.svd`奇异值分解时出现了||
# |||Segmentation fault (core dumped)||
# |||段错误...以前只在c/cpp里见过,这回倒是在python里见到了|使用`tf.linalg.svd`|
# ||||哎,一核有难,八核围观...这是八八六十四-1核围观|
# 

# In[ ]:


import scGCN


def customize_scGCN_utility_runCCA(data_use1, data_use2, features, count_names, num_cc):
    from scGCN.utility import checkFeature, runcca
    features = checkFeature(data_use1, features)
    features = checkFeature(data_use2, features)
    data1 = data_use1.loc[features, ]
    data2 = data_use2.loc[features, ]
    cca_results = runcca(data1=data1, data2=data2, num_cc=num_cc)
    cell_embeddings = np.matrix(cca_results[0])
    combined_data = data1.merge(data2,
                                left_index=True,
                                right_index=True,
                                how='inner')
    # new_data1 = combined_data.loc[count_names, ].dropna()
    # key error 取个交集
    new_data1 = combined_data.loc[np.intersect1d(combined_data.index,
                                                 count_names),].dropna()
    # loadings=loadingDim(new.data1,cell.embeddings)
    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return cca_results, loadings


scGCN.utility.runCCA = customize_scGCN_utility_runCCA
del customize_scGCN_utility_runCCA


def customize_scGCN_utility_generate_graph(
        count_list, norm_list, scale_list, features, combine, k_filter=200, k_neighbor=5):
    from scGCN.utility import runCCA, l2norm, findNN, findMNN, TopGenes, filterPair
    all_pairs = []
    for row in combine:
        i = row[0]
        j = row[1]
        counts1 = count_list[i]
        counts2 = count_list[j]
        norm_data1 = norm_list[i]
        norm_data2 = norm_list[j]
        scale_data1 = scale_list[i]
        scale_data2 = scale_list[j]
        rowname = counts1.index
        # ' @param data_use1 pandas data frame
        # ' @param data_use2 pandas data frame
        # ' @export feature loadings and embeddings (pandas data frame)
        cell_embedding, loading = runCCA(data_use1=scale_data1,
                                         data_use2=scale_data2,
                                         features=features,
                                         count_names=rowname,
                                         num_cc=30)
        norm_embedding = l2norm(mat=cell_embedding[0])
        # ' identify nearest neighbor
        cells1 = counts1.columns
        cells2 = counts2.columns
        neighbor = findNN(cell_embedding=norm_embedding,
                          cells1=cells1,
                          cells2=cells2,
                          k=30)
        # ' identify mutual nearest neighbors
        # ' @param neighbors,colnames
        # ' @export mnn_pairs
        mnn_pairs = findMNN(neighbors=neighbor,
                            colnames=cell_embedding[0].index,
                            num=k_neighbor)
        select_genes = TopGenes(Loadings=loading,
                                dims=range(30),
                                DimGenes=100,
                                maxGenes=200)
        Mat = pd.concat([norm_data1, norm_data2], axis=1)
        final_pairs = filterPair(pairs=mnn_pairs,
                                 neighbors=neighbor,
                                 mats=Mat,
                                 features=select_genes,
                                 k_filter=k_filter)
        # 从slice实例化
        final_pairs = final_pairs.copy()
        final_pairs['Dataset1'] = [i + 1] * final_pairs.shape[0]
        final_pairs['Dataset2'] = [j + 1] * final_pairs.shape[0]
        all_pairs.append(final_pairs)
    return all_pairs


scGCN.utility.generate_graph = customize_scGCN_utility_generate_graph
del customize_scGCN_utility_generate_graph


def customize_scGCN_graph_graph_construct(
    adata_1, adata_2, key_label, p_out
):
    """adata_1 和 adata_2 的layers中存储有count_data, norm_data, scale_data
"""
    from scGCN.utility import generate_graph

    # 完成判断
    p_out = Path(p_out).joinpath('input')
    p_out.mkdir(exist_ok=True, parents=True)
    if p_out.joinpath('inter_graph.csv').exists() and \
            p_out.joinpath('intra_graph.csv').exists():
        print('[msg][has finish] graph_construct')
        return
    print('\n[load data]\n'.center(100, '-'))
    # 由传入的adataadata_1,adadata_2 获得数据，而非读取文件
    features = np.array(select_feature(adata_1, key_label))
    count_list = [adata_matrix_to_df(ad, 'count_data').transpose() for ad in [adata_1, adata_2]]
    norm_list = [adata_matrix_to_df(ad, 'norm_data').transpose() for ad in [adata_1, adata_2]]
    scale_list = [adata_matrix_to_df(ad, 'scale_data').transpose() for ad in [adata_1, adata_2]]
    label_list = [sc.get.obs_df(ad, [key_label]) for ad in [adata_1, adata_2]]

    # ' graph construction
    import itertools

    N = len(count_list)
    if (N == 1):
        combine = pd.Series([(0, 0)])
    else:
        combin = list(itertools.product(list(range(N)), list(range(N))))
        index = [i for i, x in enumerate([i[0] < i[1] for i in combin]) if x]
        combine = pd.Series(combin)[index]

    print('\n[generate_graph] pairss1\n'.center(100, '-'))
    if not p_out.joinpath('inter_graph.csv').exists():
        pairss1 = generate_graph(count_list=count_list,
                                 norm_list=norm_list,
                                 scale_list=scale_list,
                                 features=features,
                                 combine=combine, k_neighbor=10)
        pairss1[0].iloc[:, 0:2].reset_index().to_csv(p_out.joinpath('inter_graph.csv'))

    count_list2 = [count_list[1], count_list[1]]
    norm_list2 = [norm_list[1], norm_list[1]]
    scale_list2 = [scale_list[1], scale_list[1]]

    print('\n[generate_graph] pairss2\n'.center(100, '-'))
    if not p_out.joinpath('intra_graph.csv').exists():
        pairss2 = generate_graph(count_list=count_list2,
                                 norm_list=norm_list2,
                                 scale_list=scale_list2,
                                 features=features,
                                 combine=combine, k_neighbor=10)
        pairss2[0].iloc[:, 0:2].reset_index().to_csv(p_out.joinpath('intra_graph.csv'))


scGCN.graph.graph_construct = customize_scGCN_graph_graph_construct
del customize_scGCN_graph_graph_construct


def customize_scGCN_data_input_data(p_out, adata_1, adata_2, key_label):
    from sklearn.model_selection import train_test_split
    import random
    import pickle as pkl

    lab_data1 = adata_matrix_to_df(adata_1).reset_index(drop=True)
    lab_data2 = adata_matrix_to_df(adata_2).reset_index(drop=True)
    lab_label1 = sc.get.obs_df(adata_1, [key_label]).reset_index(drop=True)
    lab_label2 = sc.get.obs_df(adata_2, [key_label]).reset_index(drop=True)

    lab_label1.columns = ['type']
    lab_label2.columns = ['type']

    types = np.unique(lab_label1['type']).tolist()

    random.seed(123)
    p_data = []
    p_label = []
    for i in types:
        tem_index = lab_label1[lab_label1['type'] == i].index
        tem_label = lab_label1[lab_label1['type'] == i]
        tem_data = lab_data1.iloc[tem_index]
        num_to_select = len(tem_data)
        random_items = random.sample(range(0, len(tem_index)), num_to_select)
        # print(random_items)
        sub_data = tem_data.iloc[random_items]
        sub_label = tem_label.iloc[random_items]
        # print((sub_data.index == sub_label.index).all())
        p_data.append(sub_data)
        p_label.append(sub_label)

    # ' split data to training, test, valdiaton sets

    data_train = []
    data_test = []
    data_val = []
    label_train = []
    label_test = []
    label_val = []

    for i in range(0, len(p_data)):
        temD_train, temd_test, temL_train, teml_test = train_test_split(
            p_data[i], p_label[i], test_size=0.1, random_state=1)
        temd_train, temd_val, teml_train, teml_val = train_test_split(
            temD_train, temL_train, test_size=0.1, random_state=1)
        # print((temd_train.index == teml_train.index).all())
        # print((temd_test.index == teml_test.index).all())
        # print((temd_val.index == teml_val.index).all())
        assert (temd_train.index == teml_train.index).all()
        assert (temd_test.index == teml_test.index).all()
        assert (temd_val.index == teml_val.index).all()
        data_train.append(temd_train)
        label_train.append(teml_train)
        data_test.append(temd_test)
        label_test.append(teml_test)
        data_val.append(temd_val)
        label_val.append(teml_val)

    data_train1 = pd.concat(data_train)
    data_test1 = pd.concat(data_test)
    data_val1 = pd.concat(data_val)
    label_train1 = pd.concat(label_train)
    label_test1 = pd.concat(label_test)
    label_val1 = pd.concat(label_val)

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    # ' save objects

    PIK = "{}/datasets.dat".format(p_out)
    res = [
        data_train1, data_test1, data_val1, label_train1, label_test1,
        label_val1, lab_data2, lab_label2, types
    ]

    with open(PIK, "wb") as f:
        pkl.dump(res, f)
    print('load data succesfully....')


scGCN.data.input_data = customize_scGCN_data_input_data
del customize_scGCN_data_input_data


def customize_scGCN_utils_load_data(datadir, rgraph=True):
    # 这哥们怎么导了两回scipy.sparse
    import pickle as pkl
    from scipy import sparse as sp
    import scipy.sparse
    import networkx as nx
    from scGCN.utils import preprocess_features, sample_mask, graph

    # input_data(datadir,Rgraph=rgraph)
    PIK = "{}/datasets.dat".format(datadir)
    with open(PIK, "rb") as f:
        objects = pkl.load(f)

    data_train1, data_test1, data_val1, label_train1, label_test1, label_val1, lab_data2, lab_label2, types = tuple(
        objects)

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    datas_train = np.array(train2)
    datas_test = np.array(data_test1)
    datas_val = np.array(data_val1)

    index_guide = np.concatenate(
        (label_train1.index, lab_label2.index * (-1) - 1, label_val1.index,
         label_test1.index))

    labels_train = np.array(lab_train2).flatten()
    labels_test = np.array(label_test1).flatten()
    labels_val = np.array(label_val1).flatten()

    # ' convert pandas data frame to csr_matrix format
    # datas_tr = scipy.sparse.csr_matrix(datas_train.astype('Float64'))
    # datas_va = scipy.sparse.csr_matrix(datas_val.astype('Float64'))
    # datas_te = scipy.sparse.csr_matrix(datas_test.astype('Float64'))
    datas_tr = scipy.sparse.csr_matrix(datas_train.astype(np.float64))
    datas_va = scipy.sparse.csr_matrix(datas_val.astype(np.float64))
    datas_te = scipy.sparse.csr_matrix(datas_test.astype(np.float64))

    # ' 3) set the unlabeled data in training set

    # ' @param N; the number of labeled samples in training set
    M = len(data_train1)

    # ' 4) get the feature object by combining training, test, valiation sets

    features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
    features = preprocess_features(features)

    # ' 5) Given cell type, generate three sets of labels with the same dimension
    labels_tr = labels_train.flatten()
    labels_va = labels_val.flatten()
    labels_te = labels_test.flatten()

    labels = np.concatenate(
        [np.concatenate([labels_tr, labels_va]), labels_te])
    Labels = pd.DataFrame(labels)

    true_label = Labels
    # ' convert list to binary matrix
    uniq = np.unique(Labels.values)

    rename = {}

    for line in range(0, len(types)):
        key = types[line]
        rename[key] = int(line)

    Label1 = Labels.replace(rename)
    indices = np.array(Label1.values, dtype='int').tolist()

    indice = [item for sublist in indices for item in sublist]

    # ' convert list to binary matrix
    indptr = range(len(indice) + 1)
    dat = np.ones(len(indice))
    binary_label = scipy.sparse.csr_matrix((dat, indice, indptr))

    # ' new label with binary values
    new_label = np.array(binary_label.todense())
    idx_train = range(M)
    idx_pred = range(M, len(labels_tr))
    idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
    idx_test = range(
        len(labels_tr) + len(labels_va),
        len(labels_tr) + len(labels_va) + len(labels_te))

    train_mask = sample_mask(idx_train, new_label.shape[0])
    pred_mask = sample_mask(idx_pred, new_label.shape[0])
    val_mask = sample_mask(idx_val, new_label.shape[0])
    test_mask = sample_mask(idx_test, new_label.shape[0])

    labels_binary_train = np.zeros(new_label.shape)
    labels_binary_val = np.zeros(new_label.shape)
    labels_binary_test = np.zeros(new_label.shape)
    labels_binary_train[train_mask, :] = new_label[train_mask, :]
    labels_binary_val[val_mask, :] = new_label[val_mask, :]
    labels_binary_test[test_mask, :] = new_label[test_mask, :]

    # ' ----- construct adjacent matrix ---------

    id_graph1 = pd.read_csv('{}/input/inter_graph.csv'.format(datadir),
                            index_col=0,
                            sep=',')
    id_graph2 = pd.read_csv('{}/input/intra_graph.csv'.format(datadir),
                            sep=',',
                            index_col=0)

    # ' --- map index ----
    fake1 = np.array([-1] * len(lab_data2.index))
    index1 = np.concatenate((data_train1.index, fake1, data_val1.index,
                             data_test1.index)).flatten()
    # ' (feature_data.index==index1).all()
    fake2 = np.array([-1] * len(data_train1))
    fake3 = np.array([-1] * (len(data_val1) + len(data_test1)))
    find1 = np.concatenate((fake2, np.array(lab_data2.index), fake3)).flatten()

    # ' ---------------------------------------------
    # '  intra-graph
    # ' ---------------------------------------------
    id_grp1 = np.array([
        np.concatenate((np.where(find1 == id_graph2.iloc[i, 1])[0],
                        np.where(find1 == id_graph2.iloc[i, 0])[0]))
        for i in range(len(id_graph2))
    ])

    id_grp2 = np.array([
        np.concatenate((np.where(find1 == id_graph2.iloc[i, 0])[0],
                        np.where(find1 == id_graph2.iloc[i, 1])[0]))
        for i in range(len(id_graph2))
    ])

    # ' ---------------------------------------------
    # '  inter-graph
    # ' ---------------------------------------------
    id_gp1 = np.array([
        np.concatenate((np.where(find1 == id_graph1.iloc[i, 1])[0],
                        np.where(index1 == id_graph1.iloc[i, 0])[0]))
        for i in range(len(id_graph1))
    ])

    id_gp2 = np.array([
        np.concatenate((np.where(index1 == id_graph1.iloc[i, 0])[0],
                        np.where(find1 == id_graph1.iloc[i, 1])[0]))
        for i in range(len(id_graph1))
    ])

    matrix = np.identity(len(labels))
    matrix[tuple(id_grp1.T)] = 1
    matrix[tuple(id_grp2.T)] = 1
    matrix[tuple(id_gp1.T)] = 1
    matrix[tuple(id_gp2.T)] = 1

    adj = graph(matrix)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))

    print("assign input coordinatly....")
    return adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label, index_guide


scGCN.utils.load_data = customize_scGCN_utils_load_data
del customize_scGCN_utils_load_data


def customize_scGCN_utility_svd1(mat, num_cc):

    import tensorflow as tf
    print("\n\n[msg] scGCN.utility.svd1 exchange np.linalg.svd to tf.linalg.svd\n\n".center(200, '-'))
    # U, s, V = np.linalg.svd(mat)
    # mat_tf = tf.constant(mat, dtype=tf.float32)
    # tensor 超过2G时使用  无法封装为函数 return时是复制了一份.....6
    # from stackoverflow
    # [link](https://stackoverflow.com/questions/51470991/create-a-tensor-proto-whose-content-is-larger-than-2gb)
    tf.reset_default_graph()  # 清除所有的tensor
    plhdr = tf.placeholder(dtype=tf.float32, shape=mat.shape)
    mat_tf = tf.get_variable('mat_tf', mat.shape)
    # 占用全部CPU
    with tf.device("/cpu:0"):
        with tf.Session(config=tf.ConfigProto(
            device_count={"CPU": 64},
            intra_op_parallelism_threads=16,
            inter_op_parallelism_threads=16
        )) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(mat_tf.assign(plhdr), {plhdr: mat})
            s, U, V = sess.run(tf.linalg.svd(mat_tf, full_matrices=True,
                                             compute_uv=True))

    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d


scGCN.utility.svd1 = customize_scGCN_utility_svd1
del customize_scGCN_utility_svd1


# # train

# In[3]:


from collections import namedtuple
setting_FLAGS = namedtuple(
    'setting_FLAGS',
    'dataset,output,graph,model,learning_rate,epochs,hidden1,dropout,weight_decay,early_stopping,max_degree'.split(
        ',')
)


# In[ ]:


from scGCN.utils import *
from scGCN.models import scGCN as scGCN_model

import os
import sys
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import warnings

# 全局变量仅设置一次
__flags__ = tf.app.flags
__FLAGS__ = __flags__.FLAGS
# flags.DEFINE_string('dataset', 'input', 'data dir')
# flags.DEFINE_string('output', 'results', 'predicted results')
# flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
__flags__.DEFINE_bool('graph', True, 'select the optional graph.')
__flags__.DEFINE_string('model', 'scGCN', 'Model string.')
__flags__.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
__flags__.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
__flags__.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
__flags__.DEFINE_float('weight_decay', 0,
                       'Weight for L2 loss on embedding matrix.')
__flags__.DEFINE_integer('early_stopping', 10,
                         'Tolerance for early stopping (# of epochs).')
__flags__.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


# 源自scGCN/train.py
def scGCN_train(p_input, p_output, **kvarg):
    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                            placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    p_input, p_output = str(p_input), str(p_output)
    assert os.path.exists(p_input)

    warnings.filterwarnings("ignore")
    # ' del_all_flags(FLAGS)

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    # tf.set_random_seed(seed)

    # Settings
    # tf.app.flags 是个全局变量,还不能重复定义期内的属性，改用namedtuple
    FLAGS = setting_FLAGS(
        dataset=p_input,
        output=p_output,
        epochs=kvarg.setdefault('epochs', 200),
        graph=__FLAGS__.graph,
        model=__FLAGS__.model,
        learning_rate=__FLAGS__.learning_rate,
        hidden1=__FLAGS__.hidden1,
        dropout=__FLAGS__.dropout,
        weight_decay=__FLAGS__.weight_decay,
        early_stopping=__FLAGS__.early_stopping,
        max_degree=__FLAGS__.max_degree
    )

    # flags = tf.app.flags
    # FLAGS = flags.FLAGS
    # # flags.DEFINE_string('dataset', 'input', 'data dir')
    # # flags.DEFINE_string('output', 'results', 'predicted results')
    # flags.DEFINE_string('dataset', p_input, 'data dir')
    # flags.DEFINE_string('output', p_output, 'predicted results')
    # flags.DEFINE_bool('graph', True, 'select the optional graph.')
    # flags.DEFINE_string('model', 'scGCN', 'Model string.')
    # flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    # flags.DEFINE_integer('epochs', kvarg.setdefault('epochs', 200), 'Number of epochs to train.')
    # flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    # # flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    # flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
    # flags.DEFINE_float('weight_decay', 0,
    #                    'Weight for L2 loss on embedding matrix.')
    # flags.DEFINE_integer('early_stopping', 10,
    #                      'Tolerance for early stopping (# of epochs).')
    # flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    print('\n[Load data]\n'.center(100, '-'))
    # Load data
    adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label, index_guide = load_data(
        FLAGS.dataset, rgraph=FLAGS.graph)

    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = scGCN_model

    # Define placeholders
    placeholders = {
        'support':
        # [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features':
        # tf.sparse_placeholder(tf.float32,
        #                       shape=tf.constant(features[2], dtype=tf.int64)),
        tf.compat.v1.sparse_placeholder(tf.float32,
                                        shape=tf.constant(features[2], dtype=tf.int64)),
        'labels':
        tf.placeholder(tf.float32, shape=(None, labels_binary_train.shape[1])),
        'labels_mask':
        tf.placeholder(tf.int32),
        'dropout':
        # tf.placeholder_with_default(0., shape=()),
        tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
        tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    print('\n[Create model]\n'.center(100, '-'))
    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    test_accuracy = []
    test_loss = []
    # Train model

    # configurate checkpoint directory to save intermediate model training weights
    saver = tf.train.Saver()

    save_dir = os.path.join(p_output, 'checkpoints/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'best_validation')

    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, labels_binary_train,
                                        train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy],
                        feed_dict=feed_dict)
        train_accuracy.append(outs[2])
        train_loss.append(outs[1])
        # Validation
        cost, acc, duration = evaluate(features, support, labels_binary_val,
                                       val_mask, placeholders)
        val_loss.append(cost)
        val_accuracy.append(acc)
        test_cost, test_acc, test_duration = evaluate(features, support,
                                                      labels_binary_test,
                                                      test_mask, placeholders)
        test_accuracy.append(test_acc)
        test_loss.append(test_cost)
        saver.save(sess=sess, save_path=save_path)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(outs[1]), "train_acc=", "{:.5f}".format(outs[2]),
              "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),
              "time=", "{:.5f}".format(time.time() - t))
        if epoch > FLAGS.early_stopping and val_loss[-1] > np.mean(
                val_loss[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Finished Training....")

    # '  outputs
    all_mask = np.array([True] * len(train_mask))
    labels_binary_all = new_label

    feed_dict_all = construct_feed_dict(features, support, labels_binary_all,
                                        all_mask, placeholders)
    feed_dict_all.update({placeholders['dropout']: FLAGS.dropout})

    activation_output = sess.run(model.activations, feed_dict=feed_dict_all)[1]
    predict_output = sess.run(model.outputs, feed_dict=feed_dict_all)

    # ' accuracy on all masks
    ab = sess.run(tf.nn.softmax(predict_output))
    all_prediction = sess.run(
        tf.equal(sess.run(tf.argmax(ab, 1)),
                 sess.run(tf.argmax(labels_binary_all, 1))))

    # ' accuracy on prediction masks
    acc_train = np.sum(all_prediction[train_mask]) / np.sum(train_mask)
    acc_test = np.sum(all_prediction[test_mask]) / np.sum(test_mask)
    acc_val = np.sum(all_prediction[val_mask]) / np.sum(val_mask)
    acc_pred = np.sum(all_prediction[pred_mask]) / np.sum(pred_mask)
    print('Checking train/test/val set accuracy: {}, {}, {}'.format(
        acc_train, acc_test, acc_val))
    print('Checking pred set accuracy: {}'.format(acc_pred))

    # ' save the predicted labels of query data
    if not os.path.exists(FLAGS.output):
        os.mkdir(FLAGS.output)
    scGCN_all_labels = true_label.values.flatten()  # ' ground truth
    np.savetxt(
        FLAGS.output +
        '/scGCN_all_input_labels.csv',
        scGCN_all_labels,
        delimiter=',',
        comments='',
        fmt='%s')
    np.savetxt(
        FLAGS.output +
        '/scGCN_query_mask.csv',
        pred_mask,
        delimiter=',',
        comments='',
        fmt='%s')
    ab = sess.run(tf.nn.softmax(predict_output))
    all_binary_prediction = sess.run(tf.argmax(ab, 1))  # ' predict catogrized labels
    all_binary_labels = sess.run(tf.argmax(labels_binary_all, 1))  # ' true catogrized labels
    np.savetxt(
        FLAGS.output +
        '/scGCN_all_binary_predicted_labels.csv',
        all_binary_prediction,
        delimiter=',',
        comments='',
        fmt='%f')
    np.savetxt(
        FLAGS.output +
        '/scGCN_index_guide.csv',
        index_guide,
        delimiter=',',
        comments='',
        fmt='%f')
    np.savetxt(
        FLAGS.output +
        '/scGCN_all_binary_input_labels.csv',
        all_binary_labels,
        delimiter=',',
        comments='',
        fmt='%f')


# # scGCN_get_res

# In[ ]:


def scGCN_get_res(p_data, key_class1, key_class2, dsn1, dsn2):
    # read obs_ref.csv and obs_que.csv
    df_res = pd.concat([pd.read_csv(p_data.joinpath('obs_ref.csv'), index_col=0)
                        .loc[:, [key_class1]]
                        .rename(columns={key_class1: 'cell_type'})
                        .assign(dataset=dsn1),
                        pd.read_csv(p_data.joinpath('obs_que.csv'), index_col=0)
                        .loc[:, [key_class2]]
                        .rename(columns={key_class2: 'cell_type'})
                        .assign(dataset=dsn2)
                        ])
    # df_res = df_res.reset_index(names='cell_name')
    # 不支持 names 参数
    df_res['cell_name'] = df_res.index
    df_res.index = np.arange(df_res.shape[0])

    # read scGCN results
    data = pd.concat([
        pd.read_csv(p_data.joinpath('scGCN_all_input_labels.csv'),
                    header=None, names=['label']),
        pd.read_csv(p_data.joinpath('scGCN_all_binary_input_labels.csv'),
                    header=None, names=['label_binary']),
        pd.read_csv(p_data.joinpath('scGCN_all_binary_predicted_labels.csv'),
                    header=None, names=['label_pre_binary']),
        pd.read_csv(p_data.joinpath('scGCN_index_guide.csv'),
                    header=None, names=['index']),
        pd.read_csv(p_data.joinpath('scGCN_query_mask.csv'),
                    header=None, names=['mask_que'])
    ], axis=1)
    data['label_binary'] = data['label_binary'].astype(int).astype(str)
    data['label_pre_binary'] = data['label_pre_binary'].astype(int).astype(str)
    data['index'] = data['index'].astype(int)

    data_group_count = df_group_agg(data, 'label,label_binary'.split(','))
    assert data_group_count['label'].is_unique
    assert data_group_count['label_binary'].is_unique
    map_label = df_to_dict(data_group_count, 'label_binary', 'label')
    del data_group_count

    data['label_pre'] = data['label_pre_binary'].map(map_label)

    data['index_abs'] = data['index'].abs()
    data = data.sort_values('mask_que,index_abs'.split(','), ascending=True)
    data = data.reset_index(drop=True)

    # modification
    assert (df_res['cell_type'].to_numpy() == data['label'].to_numpy()).all()
    df_res = df_res.join(data.loc[:, 'label,label_pre'.split(',')]).rename(columns={
        'label': 'true_label',
        'label_pre': 'pre_label'})
    df_res['max_prob'] = -1
    df_res['is_right'] = df_res.eval('true_label == pre_label')
    df_res = df_reindex_with_unique_col(df_res, 'cell_name', True)
    return df_res

