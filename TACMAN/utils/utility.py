import datetime
import random
import dgl
from typing import Mapping, List

import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .preprocess import get_typeEncoder, get_labels_from_adatas, get_type_counts_info



def seed_all(seed):
    if not seed:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)




def make_nowtime_tag(nowtime=None):
    if nowtime is None:
        nowtime = datetime.datetime.now()
    d = nowtime.strftime('%m-%d')
    t = str(nowtime.time()).split('.')[0].replace(':', '.')
    fmt = '{}-{}'
    return fmt.format(d, t)


"""
    Data structure process
"""


def detach2numpy(x):
    if isinstance(x, Tensor):
        x = x.cpu().clone().detach().numpy()
    elif isinstance(x, Mapping):
        x = {k: detach2numpy(v) for k, v in x.items()}
    elif isinstance(x, List):
        x = [detach2numpy(v) for v in x]
    return x


def q(x):
    return np.array(list(x))


"""
    model related
"""


def save_train_log(adatas,
                   curdir,
                   output,
                   train_loss_list,
                   val_loss_list,
                   train_acc_list,
                   val_acc_list,
                   pre_acc_list,
                   ami_list,
                   f1_list,
                   stage,
                   key_class='cell_type',):
    train_dict = {'train_loss': train_loss_list,
                  'val_loss': val_loss_list,
                  'train_acc': train_acc_list,
                  'val_acc': val_acc_list,
                  'pre_acc': pre_acc_list,
                  'ami':ami_list,
                  'weightedF1':f1_list}
    train_log = pd.DataFrame(train_dict)

    cell_name = adatas[0].obs.index.tolist() + adatas[1].obs.index.tolist()
    referen_label, query_label = get_labels_from_adatas(adatas, key_class)
    labels_class = get_typeEncoder(referen_label, query_label)
    print(labels_class[1])

    pre_out = pd.DataFrame(data=None, index=cell_name)
    pre_out['true_label'] = np.concatenate(get_labels_from_adatas(adatas, key_class))
    pre_out['pre_label'] = output.argmax(dim=-1).detach().numpy()
    pre_out['pre_label'].replace(labels_class[0], labels_class[1], inplace=True)
    pre_out['pre_label_NUM'] = output.argmax(dim=-1).detach().numpy()
    pre_prob = output.softmax(dim=1).detach().numpy()
    pre_out['max_prob'] = pre_prob.max(axis=1)
    pre_out[labels_class[1]] = pre_prob

    pred_out = pre_out.iloc[adatas[0].shape[0]:, :]
    pred_out['count'] = 1
    pred_out_view = pred_out.pivot_table(index='true_label', columns='pre_label', values='count', aggfunc='sum')


    counts_info = get_type_counts_info(adatas, key_class, dsnames=['a', 'b'])

    reference_out = pd.DataFrame(data=output[:adatas[0].shape[0], :].detach().numpy(), columns=labels_class[1])
    query_out = pd.DataFrame(data=output[adatas[0].shape[0]:, :].detach().numpy(), columns=labels_class[1])

    if stage == 0:
        counts_info.to_csv(f'{curdir}/counts_info_.csv')
    pred_out_view.to_csv(f'{curdir}/res_{stage}/pred_out_view_{stage}.csv')
    reference_out.to_csv(f'{curdir}/res_{stage}/reference_out_{stage}.csv')
    query_out.to_csv(f'{curdir}/res_{stage}/query_out_{stage}.csv')
    train_log.to_csv(f'{curdir}/res_{stage}/train_log_{stage}.csv')
    pre_out.to_csv(f'{curdir}/res_{stage}/pre_out_{stage}.csv')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


"""
    Graph propagate
"""
def hg_propagate(g, max_hops, echo=False):
    for hop in range(1, max_hops):
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)

            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

    if echo:
        for ntype in g.ntypes:
            for k in g.nodes[ntype].data.keys():
                print(f'{ntype} {k}')
    return g


def hg_propagate_weight(g, max_hops, echo=False):
    for hop in range(1, max_hops):
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)

            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.u_mul_e(k, 'w', 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

    if echo:
        for ntype in g.ntypes:
            for k in g.nodes[ntype].data.keys():
                print(f'{ntype} {k}')
    return g


def clear_hg(g, echo=False):
    if echo: print('Remove keys left after propagation')
    for ntype in g.ntypes:
        keys = list(g.nodes[ntype].data.keys())
        if len(keys):
            if echo: print(ntype, keys)
            for k in keys:
                if k not in ['C', 'G']:
                    g.nodes[ntype].data.pop(k)
    return g
