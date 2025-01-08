import time

import dgl
import scanpy as sc
from pandas import DataFrame
from scipy import sparse as sp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .MNN import mnn_from_counts
from .utility import *

"""
    process wrapper
"""


def zscore(X, with_mean=True, scale=True, ):
    """ For each column of X, do centering (z-scoring)
    """
    X = np.asarray(X)
    # code borrowed from `scanpy.pp._simple`
    scaler = StandardScaler(with_mean=with_mean, copy=True).partial_fit(X)
    if scale:
        # user R convention (unbiased estimator)
        e_adjust = np.sqrt(X.shape[0] / (X.shape[0] - 1))
        scaler.scale_ *= e_adjust
    else:
        scaler.scale_ = np.array([1] * X.shape[1])
    X_new = scaler.transform(X)
    if isinstance(X, pd.DataFrame):
        X_new = pd.DataFrame(X_new, index=X.index, columns=X.columns)
    return X_new


def process_for_graph(adatas,
                      homo,
                      key_class,
                      key_clust,
                      relation='many',
                      add_no1v1=True,
                      n_pcs=30,
                      n_hvgs=2000,
                      n_degs=50,
                      use_scnets=True,
                      n_neighbors_scnet=5,
                      n_neighbors_clust=20,
                      reso=0.4,
                      degs_method="wilcoxon",
                      ):
    # process homo relation
    one2one, n2n = biomart_process(homo)

    # process for adata
    params_preproc = dict(
        copy=True,
        target_sum=None,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors_scnet,
    )
    time1 = time.time()
    adata1 = preprocess_for_adata(adatas[0], **params_preproc)
    adata2 = preprocess_for_adata(adatas[1], **params_preproc)
    time2 = time.time()
    print('the time2 of processing adatas is {:.2f} s'.format(time2-time1))
    if use_scnets:
        scnets = [get_intra_net_from_adata(adata1), get_intra_net_from_adata(adata2)]
        scnets[0].data[:] = 1
        scnets[1].data[:] = 1
    else:
        scnets = None

    reference_hvgs, query_hvgs = get_hvgs(adata1), get_hvgs(adata2)

    time1 = time.time()
    # 或许需要重新knn
    clust_lbs2 = get_leiden_labels(adata2,
                                   n_neighbors=n_neighbors_clust,
                                   reso=reso,
                                   neighbors_key='clust',
                                   key_added='leiden',
                                   copy=False)
    time2 = time.time()
    print('the time of leiden is {:.2f} s'.format(time2 - time1))


    adatas[1].obs[key_clust] = clust_lbs2

    reference_degs = get_degs(adata1, groupby=key_class, n_degs=n_degs, method=degs_method)
    query_degs = get_degs(adata2, groupby=key_clust, n_degs=n_degs, method=degs_method)
    time3 = time.time()
    print('the time of degs is {:.2f} s'.format(time3 - time2))
    reference_higs = np.union1d(reference_hvgs, reference_degs)
    query_higs = np.union1d(query_hvgs, query_degs)
    _adatas = [adata1, adata2]

    
    print("[hvgs degs info]".ljust(50,'-'))
    print("""\t[hvgs degs] reference_hvgs\t{}
\t[hvgs degs] reference_degs\t{}
\t[hvgs degs] reference_higs\t{}
\t[hvgs degs] query_hvgs\t\t{}
\t[hvgs degs] query_degs\t\t{}
\t[hvgs degs] query_higs\t\t{}""".format(
        len(reference_hvgs),len(reference_degs),len(reference_higs),
        len(query_hvgs),len(query_degs),len(query_higs)))
    # print("num of reference_hvgs,reference_degs,reference_higs are {0},{1},{2}".format(len(reference_hvgs),
    #                                                                                    len(reference_degs),
    #                                                                                    len(reference_higs)))
    # print("num of query_hvgs,query_degs,query_higs are {0},{1},{2}".format(len(query_hvgs), len(query_degs),
    #                                                                        len(query_higs)))
    # nodes gene
    # 使用one2one,n2n选择加入的同源关系类型
    gene_relation = n2n if relation == 'many' else one2one
    reference_all_gene, query_all_gene = adata1.raw.var.index.tolist(), adata2.raw.var.index.tolist()
    reference_gene_nodes, query_gene_nodes = select_gene_nodes(reference_all_gene, query_all_gene,
                                                               reference_higs, query_higs, gene_relation)
    nodes_genes = [reference_gene_nodes, query_gene_nodes]
    # feature gene
    features_genes = get_feature_genes(reference_all_gene, query_all_gene, reference_degs, query_degs, one2one , add_no1v1=add_no1v1, n2n=n2n)

    return _adatas, features_genes, nodes_genes, scnets, one2one, n2n


def make_graph(adatas,
               aligned,
               key_class,
               features_genes,
               nodes_genes,
               scnets,
               one2one, n2n,
               gene_embedding=None,
               has_mnn=False,
               graph_type='dgl',
               seed=123):
    """
    Load cell expression matrix ,cell label, homo information to camputa cell hvgs degs
    adata should be normalized
    """

    # normalized and log
    norm_counts = get_counts_from_adatas(adatas)
    # scale

    # def _zscore(adata):
    #     """
    #     Adata has been normalized
    #     """
    #     _adata = adata.copy()
    #     sc.pp.scale(_adata)
    #     return pd.DataFrame(_adata.X, columns=_adata.var.index)
    #
    # scale_counts = [_zscore(adata) for adata in adatas]
    reference_labels, query_labels = get_labels_from_adatas(adatas, key_class)

    cell_feature = get_feature_counts(norm_counts, features_genes)
    cell_label = get_labelEncoder(reference_labels, query_labels)
    if aligned:
        n_classes = max(cell_label) + 1
    else:
        n_classes = len(np.unique(adatas[0].obs[key_class]))
    train_idx, val_idx, pred_idx = get_idx_cross_classes(reference_labels, query_labels)
    list_idx = [train_idx, val_idx, pred_idx]

    def _get_nums_gene(data):
        return len(data)

    def _get_nums_feature_from_cell(data: Tensor):
        return data.shape[1]

    all_gene_nums = _get_nums_gene(nodes_genes[0]) + _get_nums_gene(nodes_genes[1])
    cell_feature_nums = _get_nums_feature_from_cell(cell_feature)
    gene_feature = torch.Tensor(get_gene_feature(all_gene_nums, cell_feature_nums, gene_embedding))

    cell_gene_adj, gene_cell_adj = get_adj_cell_gene(norm_counts[0], norm_counts[1], nodes_genes[0], nodes_genes[1])
    gene_adj = get_adj_gene(nodes_genes, n2n)
    one2one_gene_nodes_net = get_adj_gene(nodes_genes, one2one)

    inter_net = get_pair_from_adatas(norm_counts, features_genes, N1=10, N2=10, N=5, n_jobs=32)
    acc_inter_net = (cell_label[inter_net[0]] == cell_label[inter_net[1]]).sum() / len(cell_label[inter_net[0]])
    print('Inter-net pairs ACC: {}'.format(acc_inter_net))
    cell_adj = get_adj_cell(scnets, inter_net) if has_mnn else get_adj_cell(scnets)

    # cell_adj = get_adj_cell
    
    # print graph info
    print("[nodes info]".ljust(50,'-'))
    print('\t[nodes] {}\t{}'.format('cell_feats',cell_feature_nums))
    print('\t[nodes] {}\t{}'.format('cell_nodes',cell_feature.shape[0]))
    print('\t[nodes] {}\t{}'.format('gene_nodes',all_gene_nums))
    # print(f'the num of cell feats is {cell_feature_nums}')
    # print(f'the num of cell nodes is {cell_feature.shape[0]}')
    # print(f'the num of gene nodes is {all_gene_nums}')
    dct = dict(
        cell_feature=cell_feature,
        cell_label=cell_label,
        list_idx=list_idx,
        gene_feature=gene_feature,

        adj_gg=gene_adj,
        adj_cc=cell_adj,
        adj_gc=gene_cell_adj,
        adj_cg=cell_gene_adj,
    )
    # if graph_type == 'pyg':
    #     g = get_pyg(dct)
    # elif graph_type == 'dgl':
    #     g = get_dgl(dct)
    g = get_dgl(dct)
    return g, inter_net, one2one_gene_nodes_net, cell_label, n_classes, list_idx


# def get_pyg(dct):
#     g = HeteroData()
#     # cell node
#     g['C'].x = dct['cell_feature']
#     g['C'].train_idx, g['C'].val_idx, g['C'].pred_idx = dct['list_idx']
#     g['C'].y_label = dct['cell_label']
#     # gene node
#     g['G'].x = dct['gene_feature']
#     # edge
#     g['C', 'similarity', 'C'].edge_index = dct['cell_adj']
#     g['C', 'expression', 'G'].edge_index = dct['cell_gene_adj']
#     g['G', 'expressed', 'C'].edge_index = dct['gene_cell_adj']
#     g['G', 'homology', 'G'].edge_index = dct['gene_adj']
#     return g


def get_dgl(dct):
    # Graph construction
    # edge
    gene_adj = dct['adj_gg']
    cell_adj = dct['adj_cc']
    gene_cell_adj = dct['adj_gc']
    cell_gene_adj = dct['adj_cg']
    edge_dict = {
        ('C', 'similarity', 'C'): (cell_adj[0], cell_adj[1]),
        ('C', 'expression', 'G'): (cell_gene_adj[0], cell_gene_adj[1]),
        ('G', 'expressed', 'C'): (gene_cell_adj[0], gene_cell_adj[1]),
        ('G', 'homology', 'G'): (gene_adj[0], gene_adj[1])
    }
    g = dgl.heterograph(edge_dict)
    # node feature
    g.nodes['C'].data['C'] = dct['cell_feature']
    g.nodes['G'].data['G'] = dct['gene_feature']
    return g


"""
    get adjacent matrixs and nodes feature
"""


def get_adj_cell_from_adatas(scnet):
    """
    Outputs a cells sparse adjacent matrix
    """
    cell_1 = scnet[0].todense()
    cell_2 = scnet[1].todense()
    a = np.zeros((cell_1.shape[0], cell_2.shape[0]))
    b = np.zeros((cell_2.shape[0], cell_1.shape[0]))
    cell_adj = np.block([
        [cell_1, a],
        [b, cell_2]
    ])
    # sparse transformer
    cell_adj = sp.coo_matrix(cell_adj)
    indice = np.vstack((cell_adj.row, cell_adj.col))
    cell_adj = torch.LongTensor(indice)
    return cell_adj


def get_adj_cell(scnet, extra_net=None):
    """
    Outputs a cells sparse adjacent matrix include inter-species neighbors
    """
    cell_1 = scnet[0].todense()
    cell_2 = scnet[1].todense()
    a = np.zeros((cell_1.shape[0], cell_2.shape[0]))
    b = np.zeros((cell_2.shape[0], cell_1.shape[0]))
    cell_adj = np.block([
        [cell_1, a],
        [b, cell_2]
    ])
    # sparse transformer
    cell_adj = sp.coo_matrix(cell_adj)
    indice0 = np.vstack((cell_adj.row, cell_adj.col))
    if extra_net is not None:
        indice1 = np.vstack((extra_net[0], extra_net[1]))
        indice2 = np.vstack((extra_net[1], extra_net[0]))
        indice = np.concatenate((indice0, indice1, indice2), axis=1)
    else:
        indice = indice0
    return torch.LongTensor(indice)


def match_bigraph(left: list, right: list, bigraph):
    select_bigraph = bigraph[bigraph.iloc[:, 0].isin(left)]
    select_bigraph = select_bigraph[select_bigraph.iloc[:, 1].isin(right)]
    return select_bigraph


def get_adj_gene(nodes_genes, homo):
    """
    Outputs a gene aparse adjacent matrix
    :param homo: genes homology relation (select one2one or many2many to control)
    """
    reference_gene_nodes = nodes_genes[0]
    query_gene_nodes = nodes_genes[1]
    adj = match_bigraph(reference_gene_nodes, query_gene_nodes, homo)
    print('[homo edges]'.ljust(50,'-'))
    [print('\t[homo edges] {}\t{}'.format(i,row['counts']))
        for i,row in adj.iloc[:, 2].value_counts().to_frame(name='counts').iterrows()]

    gn = np.append(reference_gene_nodes, query_gene_nodes)
    gnind_1 = pd.DataFrame(data=np.arange(reference_gene_nodes.size)[None, :], columns=reference_gene_nodes)
    gnind_2 = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    gnind_2 = gnind_2.iloc[:, len(reference_gene_nodes):]
    a = np.array(adj.iloc[:, 0]).tolist()
    b = np.array(adj.iloc[:, 1]).tolist()
    reference_indic = pd.Index(gnind_1[a].values.flatten())
    query_indic = pd.Index(gnind_2[b].values.flatten())
    indice1 = np.vstack((reference_indic, query_indic))
    indice2 = np.vstack((query_indic, reference_indic))
    gene_adj = np.concatenate((indice1, indice2), axis=1)

    return torch.LongTensor(gene_adj)


def get_adj_cell_gene(reference_raw_counts, query_raw_counts, reference_gene_nodes, query_gene_nodes):
    """
    :param query_raw_counts: query species gene expression matrix
    :param reference_raw_counts: reference species gene expression matrix
    :param query_gene_nodes: query species selected genes
    :param reference_gene_nodes: reference species selected genes
    """
    a = pd.DataFrame(data=None, columns=reference_gene_nodes)
    a_1 = a.T
    b = pd.DataFrame(data=None, columns=query_gene_nodes)
    b_1 = b.T
    reference_1 = reference_raw_counts.T
    query_1 = query_raw_counts.T

    cell_gene_1 = a_1.join(reference_1, how='left').fillna(0).T
    cell_gene_2 = b_1.join(query_1, how='left').fillna(0).T

    cell_gene = np.block([
        [cell_gene_1, np.zeros((cell_gene_1.shape[0], cell_gene_2.shape[1]))],
        [np.zeros((cell_gene_2.shape[0], cell_gene_1.shape[1])), cell_gene_2]
    ])

    cell_gene = sp.coo_matrix(cell_gene)
    indice_cell_gene = np.vstack((cell_gene.row, cell_gene.col))
    indice_gene_cell = np.vstack((cell_gene.col, cell_gene.row))
    cell_gene = torch.LongTensor(indice_cell_gene)
    gene_cell = torch.LongTensor(indice_gene_cell)
    return cell_gene, gene_cell


# TODO Rename this here and in `get_adj_cell_gene`


def get_feature_genes(reference_all_gene, query_all_gene, reference_degs, query_degs, one2one, add_no1v1=False, n2n=None):
    df_tmp = match_df(reference_all_gene, query_all_gene, one2one)

    one2one = match_df(reference_degs, query_degs, df_tmp, union=True)
    if add_no1v1:
        m2m = match_df( [g for g in reference_degs if g not in one2one.values[:, 0]],
                        [g for g in query_degs if g not in one2one.values[:, 1]],
                        n2n,
                        union=False)
        map = pd.concat([one2one, m2m], axis=0)
    else:
        map = one2one



    return map.iloc[:, 0].tolist(), map.iloc[:, 1].tolist()

# def get_feature_genes(reference_all_gene, query_all_gene, reference_degs, query_degs, one2one):
#     df_tmp = match_df(reference_all_gene, query_all_gene, one2one)

#     res = match_df(reference_degs, query_degs, df_tmp, union=True)

#     return res.iloc[:, 0].tolist(), res.iloc[:, 1].tolist()


def match_df(left, right, df_match, union=False):
    c1, c2 = df_match.columns[:2]
    tmp = df_match[c1].isin(left).to_frame(c1)
    tmp[c2] = df_match[c2].isin(right)
    keep = tmp.max(1) if union else tmp.min(1)
    return df_match[keep]


def get_degs_one2one_dfs(all_genes, degs, one2one):
    reference_degs = one2one[one2one.iloc[:, 0].isin(degs[0])]
    query_degs = one2one[one2one.iloc[:, 1].isin(degs[1])]
    degs = pd.concat([reference_degs, query_degs], axis=0, ignore_index=True).drop_duplicates()
    degs = degs[degs.iloc[:, 0].isin(all_genes[0])]
    degs = degs[degs.iloc[:, 1].isin(all_genes[1])]
    return degs


def get_feature_counts(normlized_counts, features_genes):
    reference_feature = normlized_counts[0][features_genes[0]].values
    query_feature = normlized_counts[1][features_genes[1]].values
    concat_feature = np.concatenate((reference_feature, query_feature), axis=0)
    feature = zscore(concat_feature, with_mean=True, scale=True)
    return torch.Tensor(feature)


def get_gene_feature(nums_gene, dim, embeddingpath=None):
    if embeddingpath is not None:
        gene_feature = pd.read(embeddingpath)
    else:
        gene_feature = np.zeros((nums_gene, dim))
    return gene_feature


# ===
# process data
# ===
def get_type_counts_info(adatas, key_class, dsnames):
    type_counts_list = []
    for i in range(len(adatas)):
        type_counts_list.append(pd.value_counts(adatas[i].obs[key_class]))
    counts_info = pd.concat(type_counts_list, axis=1, keys=dsnames)

    return counts_info


def aligned_type(adatas, key_calss):
    adata1 = adatas[0].copy()
    adata2 = adatas[1].copy()
    counts_info = get_type_counts_info(adatas, key_calss, dsnames=['reference', 'query'])
    print('[aligned] raw '.ljust(50,'-'))
    print(counts_info)
    counts_info = counts_info.dropna(how='any')
    print('[aligned] new '.ljust(50,'-'))
    print(counts_info)

    com_type = counts_info.index.tolist()
    adata1 = adata1[adata1.obs[key_calss].isin(com_type)]
    adata2 = adata2[adata2.obs[key_calss].isin(com_type)]
    return adata1, adata2


def normalize(adata, is_norm=True, target_sum=None, is_log=False, is_scale=False):
    _adata = adata.copy()
    if is_norm:
        sc.pp.normalize_total(_adata, target_sum=target_sum)
    if is_log:
        sc.pp.log1p(_adata)
    if is_scale:
        sc.pp.scale(_adata)
    return _adata


def preprocess_for_adata(adata,
                         copy=True,
                         do_norm=True, target_sum=None,
                         is_norm=True, is_log=True, is_scale=False,
                         do_hvgs=True, n_hvgs=2000,
                         do_scale=True,
                         do_pca=True,
                         n_pcs=30,
                         do_neigh=True,
                         n_neighbors=5,
                         metric='cosine', ):
    _adata = adata.copy() if copy else adata
    if do_norm:
        _adata = normalize(_adata, is_norm=is_norm, target_sum=target_sum, is_log=is_log, is_scale=is_scale)
    _adata.raw = _adata
    if do_hvgs:
        sc.pp.highly_variable_genes(_adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_hvgs)
        _adata = _adata[:, _adata.var['highly_variable']].copy()
    if do_scale:
        # sc.pp.scale(_adata, max_value=10)
        sc.pp.scale(_adata, zero_center=True, max_value=None)
    if do_pca:
        sc.tl.pca(_adata, n_comps=n_pcs)
        if do_neigh:
            sc.pp.neighbors(_adata, n_pcs=n_pcs, n_neighbors=n_neighbors, metric=metric)
    return _adata


# def preprocess_for_adata(adata,
#     copy=True,
#     do_norm=True, target_sum=None,
#     is_norm=True, is_log=True, is_scale=False,
#     do_hvgs=True, n_hvgs=2000,
#     do_scale=True,
#     do_pca=True,
#     n_pcs=40,
#     do_neigh=True,
#     n_neighbors=20,
#     metric='cosine', ):
#     _adata = adata.copy() if copy else adata

#     if do_norm:
#         time1 = time.time()
#         _adata = normalize(_adata, is_norm=is_norm, target_sum=target_sum, is_log=is_log, is_scale=is_scale)
#         time2 = time.time()
#         print(time2-time1)
#     _adata.raw = _adata
#     if do_hvgs:
#         time1 = time.time()
#         sc.pp.highly_variable_genes(_adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_hvgs)
#         time2 = time.time()
#         print(time2-time1)
#     # _adata = _adata[:, _adata.var['highly_variable']].copy()
#     if do_scale:
#         time1 = time.time()
#         sc.pp.scale(_adata, max_value=10)
#         time2 = time.time()
#         print(time2-time1)
#     if do_pca:
#         time1 = time.time()
#         sc.tl.pca(_adata, n_comps=n_pcs)
#         time2 = time.time()
#         print(time2-time1)
#     if do_neigh:
#         time1 = time.time()
#         sc.pp.neighbors(_adata, n_pcs=n_pcs, n_neighbors=n_neighbors, metric=metric)
#         time2 = time.time()
#         print(time2-time1)
#     return _adata


def get_leiden_labels(adata,do_neighbors=True,
                      n_neighbors=20, reso=0.4, n_pcs=30,
                      neighbors_key='clust',
                      key_added='leiden',
                      copy=False,
                      ):
    adata = adata.copy() if copy else adata
    if do_neighbors:
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=n_pcs, key_added=neighbors_key)

    sc.tl.leiden(adata, resolution=reso,
                 key_added=key_added,
                 neighbors_key=neighbors_key)
    lbs = adata.obs[key_added]

    # print(f"Leiden results:\n{lbs.value_counts()}")
    print("[Leiden results]".ljust(50,'-'))
    [print('\t{}\t{}'.format(i,row['counts']))
    for i,row in lbs.value_counts().to_frame(name='counts').iterrows()]
    return lbs

def select_gene_nodes(reference_all_gene, query_all_gene, higs_reference, higs_query, homo):
    # 物种1的higs与物种2的higs在物种1的同源 联合
    query_homo = homo[homo.iloc[:, 1].isin(higs_query)].iloc[:, 0]
    reference_gene_nodes = np.union1d(query_homo, higs_reference)

    reference_homo = homo[homo.iloc[:, 0].isin(higs_reference)].iloc[:, 1]
    query_gene_nodes = np.union1d(reference_homo, higs_query)

    # biomart和数据集基因名的一致性筛选
    reference_gene_nodes = np.intersect1d(reference_all_gene, reference_gene_nodes)
    query_gene_nodes = np.intersect1d(query_all_gene, query_gene_nodes)
    print('[gene nodes info]'.ljust(50,'-'))
    print("\t[gene nodes] reference_gene_node\t{0}".format(len(reference_gene_nodes)))
    print("\t[gene nodes] query_gene_node\t\t{0}".format(len(query_gene_nodes)))

    return reference_gene_nodes, query_gene_nodes


def get_nums_cell(adatas):
    return adatas[0].X.shape[0] + adatas[1].X.shape[0]


def make_anndata_from_df(df: DataFrame):
    index = df.index
    columns = df.columns
    x = np.matrix(df, dtype=np.float32)
    adata = sc.AnnData(x)
    adata.obs.index = index
    adata.var.index = columns
    return adata


def biomart_process(homo_data):
    """
    :param homo_data: homology relation csv
    """
    # 清除NaN
    homo_data = homo_data.iloc[:, 0:3]
    homo_data = homo_data.drop_duplicates(subset=[homo_data.columns[0],homo_data.columns[1]])
    homo_data = homo_data.dropna(axis=0, how='any')
    # 选出one2one
    one2one = homo_data[homo_data.iloc[:, 2] == 'ortholog_one2one']
    # 选出n2n
    n2n = homo_data
    print('[Homolog information follows]'.ljust(50,'-'))
    [print('\t[homo] {:20} {}'.format(i,row['counts']))
    for i,row in n2n.iloc[:, 2].value_counts().to_frame(name='counts').iterrows()]
    return one2one, n2n


def get_one2one_gene(adatas, one2one):
    """
    according to one2one homo select one2one genes from datas
    :param one2one: one2one homo rel
    :return: one2one genes of reference, query
    """
    reference_all_gene, query_all_gene = adatas[0].var.index.tolist(), adatas[1].var.index.tolist()
    one2one = one2one[(one2one.iloc[:, 0].isin(reference_all_gene))]
    one2one = one2one[(one2one.iloc[:, 1].isin(query_all_gene))]
    reference_one2one_gene = one2one.iloc[:, 0]
    query_one2one_gene = one2one.iloc[:, 1]
    return reference_one2one_gene, query_one2one_gene


def get_one2one_data(adatas, one2one, isCommon=False):
    """
    get one2one genes counts according to one2one genes
    :param isCommon: whether trans query genes name to reference names
    :return:reference one2one dataframe,query one2one data frame
    """
    reference_raw_counts, query_raw_counts = get_counts_from_adatas(adatas)
    reference_one2one_gene, query_one2one_gene = get_one2one_data(adatas, one2one)
    reference_one2one_data = reference_raw_counts[reference_one2one_gene]
    query_one2one_data = query_raw_counts[query_one2one_gene]
    if isCommon:
        query_one2one_data.columns = reference_one2one_data.columns
    return reference_one2one_data, query_one2one_data


def get_hvgs(adata,
             do_norm=False,
             do_hvgs=False,
             nums_hvgs=2000):
    """
    get high var genes from adata of single-cell matrix
    :param adata: a AnnData of single-cell matrix
    :param do_norm: True if do normalize else False
    :param do_hvgs: True if do high var genes else False
    :param nums_hvgs: numbers of high var genes
    :return: a list of high var genes
    """
    if do_norm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if do_hvgs:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=nums_hvgs)
    is_hvgs = adata.var['highly_variable']
    return adata.var[is_hvgs].index.tolist()


def get_cluster_labels(adata,
                       force_redo=False,
                       nneigh=20, reso=0.4, n_pcs=30,
                       neighbors_key=None,
                       key_added='leiden',
                       copy=False,
                       ):
    """ assume that the `X_pca` is already computed
    """
    adata = adata.copy() if copy else adata
    if 'X_pca' not in adata.obsm.keys():
        sc.tl.pca(adata, n_comps=n_pcs)
    _key_adj = 'connectivities'
    if neighbors_key is not None:
        _key_adj = f'{neighbors_key}_{_key_adj}'
    if force_redo or _key_adj not in adata.obsp.keys():
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=nneigh, n_pcs=n_pcs,
                        key_added=neighbors_key)

    sc.tl.leiden(adata, resolution=reso,
                 key_added=key_added,
                 neighbors_key=neighbors_key)
    lbs = adata.obs[key_added]
    print("[Leiden results]".ljust(50,'-'))
    [print('\t{}\t{}'.format(i,row['counts']))
    for i,row in lbs.value_counts().to_frame(name='counts').iterrows()]
    

    return lbs


def get_degs(adata,
             groupby,
             key_added="rank_genes_groups",
             method="wilcoxon",
             n_degs=50):
    """
    get different expression genes

    :param adata: a AnnData
    :param n_degs: numbers of different expression genes
    :return: a list of different expression genes
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, key_added=key_added, method=method)

    degs_all = adata.uns['rank_genes_groups']['names']
    degs_index = np.array(degs_all.tolist())[0:n_degs, :].flatten()
    return np.unique(degs_index)


def get_degs_hvgs_higs(adata, nums_hvgs=2000, nums_degs=50):
    hvgs = get_hvgs(adata, nums_hvgs=nums_hvgs)
    degs = get_degs(adata, n_degs=nums_degs)
    higs = np.union1d(hvgs,degs)
    return hvgs,degs, higs


def get_pair_from_adatas(counts, gene_features, N1=10, N2=10, N=5, n_jobs=10):
    """
    get inter-species net from a pair of adatas
    :param adatas:
    :param degs:
    :return:
    """
    a = counts[0][gene_features[0]].values
    b = counts[1][gene_features[1]].values
    pair, g1 = mnn_from_counts(a, b, N1=N1, N2=N2, N=N, n_jobs=n_jobs)
    mnn_idx1 = np.array(pair[0])
    mnn_idx2 = np.array(pair[1]) + counts[0].shape[0]
    return mnn_idx1, mnn_idx2


def get_one2one_node_index(reference_gene_nodes, query_gene_nodes, homo):
    adj = match_bigraph(reference_gene_nodes, query_gene_nodes, homo)
    gn = np.append(reference_gene_nodes, query_gene_nodes)
    gnind_1 = pd.DataFrame(data=np.arange(reference_gene_nodes.size)[None, :], columns=reference_gene_nodes)
    gnind_2 = pd.DataFrame(data=np.arange(gn.size)[None, :], columns=gn)
    gnind_2 = gnind_2.iloc[:, len(reference_gene_nodes):]
    a = np.array(adj.iloc[:, 0]).tolist()
    b = np.array(adj.iloc[:, 1]).tolist()
    reference_indic = pd.Index(gnind_1[a].values.flatten()).tolist()
    query_indic = pd.Index(gnind_2[b].values.flatten()).tolist()
    return reference_indic, query_indic


def get_counts_from_adatas(adatas):
    """
    Assum adatas has been normalized
    Get a pair of dataframes from a pair adatas

    :param adatas: a tuple of reference adata and query adata
    :return: a tuple of reference raw counts dataframe and query raw counts dataframe
    """
    # _adatas = [adatas[0].raw, adatas[1].raw]
    # 需要对稀疏矩阵进行判断
    if sp.issparse(adatas[0].raw.X) or sp.issparse(adatas[1].raw.X):
        a = np.array(adatas[0].raw.X.todense())
        b = np.array(adatas[1].raw.X.todense())
    else:
        a = adatas[0].raw.X
        b = adatas[1].raw.X
    reference_counts = pd.DataFrame(a, columns=adatas[0].raw.var.index)
    query_counts = pd.DataFrame(b, columns=adatas[1].raw.var.index)
    return reference_counts, query_counts


def get_labels_from_adatas(adatas, key_class='cell_type'):
    """
     get a pair of labels lists  from a pair adatas

    :param adatas: adatas: a tuple of reference adata and query adata
    :param key_class: key_class
    :return: a tuple of reference labels dataframe and query labels
    """
    reference_labels = adatas[0].obs[key_class].tolist()
    query_labels = adatas[1].obs[key_class].tolist()
    return reference_labels, query_labels


def get_intra_net_from_adata(adata: sc.AnnData):
    """ Extract the pre-computed single-cell KNN network

    :param adata: :class:`~anndata.AnnData`

    """
    key0 = 'neighbors'
    key = 'connectivities'
    return adata.obsp[key] if key in adata.obsp.keys() else adata.uns[key0][key]


# ===
# label encoder and split idx
# ===
# def get_labelEncoder(reference_labels, query_labels):
#     """
#     encoder type to nums

#     :param reference_labels: a list of reference labels
#     :param query_labels: a list of query labels
#     :return: a tuple of raw type array and encoder type array
#     """
#     lb = LabelEncoder()
#     label_1 = reference_labels
#     label_2 = query_labels
#     y = np.append(label_1, label_2)
#     lb.fit(y)
#     y = lb.transform(y)
#     return y

def get_labelEncoder(reference_labels, query_labels):
    """
    encoder type to nums

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: a tuple of raw type array and encoder type array
    """
    unique_label = np.unique([i for i in query_labels if i not in reference_labels])
    lb = LabelEncoder()
    lb.fit(reference_labels)
    lb.classes_ = np.array(list(lb.classes_)+list(unique_label))
    encode1 = lb.transform(reference_labels)
    encode2 = lb.transform(query_labels)
    y = np.append(encode1,encode2)
    return y


def get_typeEncoder_ref(reference_labels):
    """
    encoder type to nums

    :param reference_labels: a list of reference labels
    :return: a tuple of raw type array and encoder type array from ref
    """
    lb = LabelEncoder()
    lb.fit(reference_labels)
    y = lb.transform(reference_labels)
    return np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1))


def get_typeEncoder(reference_labels, query_labels):
    """
    encoder type to nums

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: a tuple of raw type array and encoder type array
    """
    lb = LabelEncoder()
    lb.fit(reference_labels)
    y = lb.transform(reference_labels)
    # cell_class = (np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1)))
    return np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1))


def get_typeEncoder_unaligned(reference_labels, query_labels):
    """
    encoder type to nums

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: a tuple of raw type array and encoder type array
    """
    unique_label = np.unique([i for i in query_labels if i not in reference_labels])
    lb = LabelEncoder()
    lb.fit(reference_labels)
    lb.classes_ = np.array(list(lb.classes_)+list(unique_label))
    encode1 = lb.transform(reference_labels)
    encode2 = lb.transform(query_labels)
    y = np.append(encode1,encode2)
    # cell_class = (np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1)))
    return np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1))


# def get_typeEncoder(reference_labels, query_labels):
#     """
#     encoder type to nums

#     :param reference_labels: a list of reference labels
#     :param query_labels: a list of query labels
#     :return: a tuple of raw type array and encoder type array
#     """
#     unique_label = np.unique([i for i in query_labels if i not in reference_labels])
#     lb = LabelEncoder()
#     lb.fit(reference_labels)
#     lb.classes_ = np.array(list(lb.classes_)+list(unique_label))
#     encode1 = lb.transform(reference_labels)
#     encode2 = lb.transform(query_labels)
#     y = np.append(encode1,encode2)
#     # cell_class = (np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1)))
#     return np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1))


# def get_typeEncoder(reference_labels, query_labels):
#     """
#     encoder type to nums

#     :param reference_labels: a list of reference labels
#     :param query_labels: a list of query labels
#     :return: a tuple of raw type array and encoder type array
#     """
#     lb = LabelEncoder()
#     label_1 = reference_labels
#     label_2 = query_labels
#     y = np.append(label_1, label_2)
#     lb.fit(y)
#     y = lb.transform(y)
#     # cell_class = (np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1)))
#     return np.arange(y.max() + 1), lb.inverse_transform(np.arange(y.max() + 1))


def idxtomask(train_idx, val_idx, pre_idx):
    """
    trans idx to mask

    :param train_idx: a list of train idx
    :param val_idx: a list of train idx
    :param pre_idx: a list of train idx
    :return:train_mask tensor, val_mask tnesor, pre_mask tensor
    """
    a = pre_idx.max() + 1
    idx_list = [train_idx, val_idx, pre_idx]
    mask_list = [None] * len(idx_list)
    for i, item in enumerate(idx_list):
        mask = torch.zeros(a, dtype=torch.bool)
        mask[item] = True
        mask_list[i] = mask
    train_mask, val_mask, pre_mask = mask_list
    return train_mask, val_mask, pre_mask


def get_idx(reference_labels, query_labels):
    """
    get train idx, val idx, pred idx from labels

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: train idx list, val idx list, pred idx list
    """
    a = len(reference_labels) + len(query_labels)
    b = len(reference_labels)
    index_all = np.arange(a).tolist()
    index_reference = index_all[:b]
    random.shuffle(index_reference)
    train_idx = index_reference[:round(8 * b / 10)]
    val_idx = index_reference[round(8 * b / 10):]
    pre_idx = index_all[b:]
    return train_idx, val_idx, pre_idx


def get_mask(reference_labels, query_labels):
    """
    get train mask, val mask, pred mask from labels

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: train idx tensor, val idx tensor, pred idx tensor
    """
    train_idx, val_idx, pre_idx = get_idx(reference_labels, query_labels)
    train_mask, val_mask, pre_mask = idxtomask(train_idx, val_idx, pre_idx)
    return train_mask, val_mask, pre_mask


def get_idx_cross_classes(reference_labels, query_labels, seed=123):
    """
    get train idx, val idx, pred idx from labels in every type

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: train idx list, val idx list, pred idx list
    """
    a = len(reference_labels) + len(query_labels)
    reference_labels = pd.Series(reference_labels)
    index_0 = np.arange(a).tolist()

    types = np.unique(reference_labels).tolist()
    index = []
    for i in types:
        type_index = reference_labels[reference_labels == i].index.tolist()
        index.append(type_index)
    train_idx = []
    val_idx = []

    for item in index:
        random.seed(seed)
        random.shuffle(item)
        b = len(item)
        train = item[:round(8 * b / 10)]
        val = item[round(8 * b / 10):]
        train_idx.append(train)
        val_idx.append(val)

    train_idx = np.concatenate(train_idx).tolist()
    val_idx = np.concatenate(val_idx).tolist()
    pre_idx = index_0[len(reference_labels):]
    return train_idx, val_idx, pre_idx


def get_mask_cross_classes(reference_labels, query_labels):
    """
    get train mask, val mask, pred mask from labels in every type

    :param reference_labels: a list of reference labels
    :param query_labels: a list of query labels
    :return: train idx tensor, val idx tensor, pred idx tensor
    """
    train_idx, val_idx, pre_idx = get_idx_cross_classes(reference_labels, query_labels)
    train_mask, val_mask, pre_mask = idxtomask(train_idx, val_idx, pre_idx)
    return train_mask, val_mask, pre_mask


if __name__ == '__main__':
    tessie = "pancreas"
    gse_id = ["GSE84113", "GSE84113"]
    species = ['human', 'mouse']
    print(f'Task: refernece:{species[0]} -> query:{species[1]} in {tessie}')

    homo_df = pd.read_csv('/public/workspace/ruru_97/projects/data/homo/biomart/human_to_mouse.txt')
    adata_species_1 = sc.read_h5ad('/public/workspace/ruru_97/projects/data/pancreas/GSE84113/human.h5ad')
    adata_species_2 = sc.read_h5ad('/public/workspace/ruru_97/projects/data/pancreas/GSE84113/mouse.h5ad')
