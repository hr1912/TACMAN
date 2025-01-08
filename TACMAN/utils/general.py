#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pathlib import Path
import json as json
import time

import collections
from collections.abc import Iterable
from collections import namedtuple

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


rng = np.random.default_rng()


# In[4]:


def module_exists(module_name):
    import importlib.util
    import sys
    return module_name in sys.modules or importlib.util.find_spec(module_name)

def subset_dict(data,keep_item=[],regex=None,keep_order=False):
    if isinstance(keep_item,str):
        keep_item = [keep_item]
    
    data = data.copy()
    keys = []
    if regex:
        keys = pd.Series(list(data.keys()))
        keys = keys[keys.str.match(regex)]
    keys = pd.Series(np.concatenate([keys,keep_item])).unique()

    if keep_order:
        return {k:v for k,v in  data.items() if k in keys}
    return { k:data[k] for k in keys}

def update_dict(data,update=None,**kvargs):
    """更新字典，覆盖顺序 kvargs > update > data
"""
    data = data.copy()
    if isinstance(update,dict):
        data.update(update)
    data.update(kvargs)
    return data

def handle_type_to_list(data,t=str):
    """若 data 为 t 则返回 [data]
若 data 为list 则返回 data
"""
    assert isinstance(data,(list,t)),'[Error] data is not a list or {}'.format(t)
    return [data] if isinstance(data,t) else data


# # show

# In[5]:


def show_str_arr(data,n_cols=4,order='F',return_str=False):
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
    if data.size % n_cols != 0:
        data = np.concatenate([data,np.repeat('',n_cols - data.size % n_cols)])
    data = pd.Series(data)
    data = pd.DataFrame(np.reshape(data,(data.size//n_cols,n_cols),order=order))
    for k in data.columns:
        data[k] = data[k].str.ljust(data[k].str.len().max())
    res = '\n'.join(data.apply(lambda row:'| {}|'.format('| '.join(row.values)),axis=1))
    print(res)

    if return_str:
        return res

def show_obj_attr(obj,regex_filter=['^_'],regex_select = [],
                  show_n_cols = 3,
                 group=False,
                extract_pat = '^([^_]+)_',return_series=False):
    from functools import reduce
    
    if isinstance(regex_filter,str):
        regex_filter = [regex_filter]
    if isinstance(regex_select,str):
        regex_select = [regex_select]
        
    attr = pd.Series([_ for _ in dir(obj)])
    # filter
    if len(regex_filter) ==1:
        attr = attr[~attr.str.match(regex_filter[0])]
    elif len(regex_filter) ==0:
        pass
    else:
        attr = attr[reduce(lambda x,y:x&y ,
            [~attr.str.match(_) for _ in regex_filter])]
    # select
    if len(regex_select) == 1:
        attr = attr[attr.str.match(regex_select[0])]
    elif len(regex_select) == 0:
        pass
    else:
        attr = attr[reduce(lambda x,y:x|y ,
            [attr.str.match(_) for _ in regex_select])]
    if show_n_cols:
        # print(*['\t'.join(_) for _ in np.array_split(
        # attr.to_numpy(),attr.size//4)],sep='\n')
        show_str_arr(attr,show_n_cols)
    if group:
        print('[group]'.ljust(15,'-'))
        print(attr.str.extract(extract_pat,expand=False).value_counts(dropna=False))
        
    if return_series:
        attr.index = np
        return attr



def show_dict_key(data, tag='', sort_key=True):
    print("> {}['']".format(tag).ljust(75, '-'))
    ks = list(data.keys())
    if sort_key:
        ks = np.sort(ks)
    print(*['  {}'.format(k) for k in ks], sep='\n')


# # Block
# 
# `块` 用于将代码分块
# 
# 因为在notebook中用注释将代码分块，无法实现代码的折叠，使得代码较为混乱
# 
# 故通过`with Block():`组合构造出可以折叠的with语句块，进而提高代码的可读性
# 
# + `with Block():`内部并未与外部进行隔离
# 
# + 实现了计时功能
# + 实现了上下文功能

# In[6]:


class Block:
    """用于在notebook中给代码划分区域(块),从而使代码能够折叠

Examples
--------


# 上下文功能
with Block('context',context={
    'a':'Block','b':':','c':'Hello World'
}) as context:
    print('inner')
    print('\t',' '.join(context.context.values()))
    print('\t',context.a,context.b,context.c)
# output
## inner
## 	 Block : Hello World
## 	 Block : Hello World

# 计时功能
import time
with Block('test',show_comment=True):
    print('inner')
    time.sleep(2)
# output
## [start][test] 0509-00:20:47------------------------------------------------
## inner
## [end][test] 0509-00:20:49--------------------------------------------------
        2.002 s used
    """

    def __init__(self, comment, show_comment=False,context=None):
        self.comment = comment
        self.show_comment = show_comment
        self.context = context
        self.time = 0

    def __enter__(self):
        if self.show_comment:
            self.time = time.time()
            print("[start][{}] {}".format(
                self.comment,
                time.strftime('%m%d-%H:%M:%S',
                              time.localtime())).ljust(75, '-'))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.show_comment:
            print(
                "[end][{}] {}".format(
                    self.comment,
                    time.strftime(
                        '%m%d-%H:%M:%S',
                        time.localtime())).ljust(
                    75,
                    '-'))
            print("\t{:.3f} s used".format(time.time()-self.time))
        # 释放content
        self.context = None
    def __str__(self):
        return """Block
\tcomment     : {}
\tcontext_key : {}
""".format(self.comment,','.join(self.context.keys()))
    
    # 对类及其实例未定义的属性有效
    # 若name 不存在于 self.__dir__中,则调用__getattr__
    def __getattr__(self,name):
        cls = type(self)
        res = self.context.setdefault(name,None)
        if res is None:
            raise AttributeError(
                '{.__name__!r} object has no attribute {!r}'\
                .format(cls, name))
        
        return res


# # other

# In[7]:


import platform
system = platform.system()
system == 'Linux'


# In[8]:


def rm_rf(p):
    if not p.exists():
        return

    if p.is_file():
        p.unlink()

    if p.is_dir():
        for i in p.iterdir():
            if i.is_file():
                i.unlink()
            if i.is_dir():
                rm_rf(i)  # 递归
        p.rmdir()        
def str_step_insert(s,step,insert='\n'):
    """于s的每step个字符后插入一个insert
在为图添加超长的文字时
使用str_step_insert('0123456789012345678901234567',5,insert='\n')
使字符串换行

"""
    return insert.join([ s[i*step:(i+1)*step] for i in range(len(s)//step + (0 if len(s)%step==0 else 1))])


# In[24]:


def archive_gzip(p_source,decompress=False,out_dir=None,remove_source = True,show=True):
    """gzip 的压缩和解压缩
    """
    def archive_gzip_default(p_source,decompress,out_dir,remove_source,show):
        import gzip
        from shutil import copyfileobj
        
        func_open_source,func_open_target = open,gzip.open
        if decompress:
            func_open_source,func_open_target = func_open_target,func_open_source
            
        with func_open_source(p_source, 'rb') as f_source:  
            with func_open_target(p_target, 'wb') as f_target:  
                copyfileobj(f_source, f_target)

    def archive_gzip_Linux(p_source,decompress,out_dir,remove_source,show):
        import os
        os.system('gzip -{}c {} > {}'.format(
            'd' if decompress else '',p_source,p_target)
        )
    import platform
    
    handel_func = {
        'Linux':archive_gzip_Linux
    }.setdefault(platform.system(),archive_gzip_default)
    
    p_source = Path(p_source)
    assert p_source.exists(),'[not exitst] {}'.format(p_source)
    
    out_dir = p_source.parent if out_dir is None else Path(out_dir)
    if decompress:
        assert p_source.match('*.gz'),'[Error] p_source must end with .gz when decompress=True'
        p_target = out_dir.joinpath(p_source.name[:-3])
    else:
        if p_source.name.endswith('.gz'):
            print("[archive_gzip] has commpress {}".format(p_source)) if show else None
            return
        p_target = out_dir.joinpath('{}.gz'.format(p_source.name))


    handel_func(p_source,decompress,out_dir,remove_source,show)
    print('[archive_gzip][{}compress] {} -> {}'.format(
            'de' if decompress else '',p_source.name,p_target.name)) if show else None
    p_source.unlink() if remove_source else None


# # TACMAN
# 

# In[ ]:


def get_time_tag(time_format = '%y%m%d-%H%M%S'):
    return time.strftime(time_format, time.localtime())


def get_1v1_matches(
        df_match,
        key_homology_type='homology_type',
        value_homology_type='ortholog_one2one'):
    l, r = df_match.columns[:2]
    l_unique = df_match[l].value_counts(
    ).to_frame().query("count == 1").index
    r_unique = df_match[r].value_counts(
    ).to_frame().query("count == 1").index
    keep = pd.DataFrame({
        'l_is_unique': df_match[l].isin(l_unique),
        'r_is_unique': df_match[r].isin(r_unique)
    }).min(axis=1)
    df_match = df_match[keep]
    df_match = df_match.query(
        "{} == '{}'".format(
            key_homology_type,
            value_homology_type))
    return df_match

def extract_model_gene_nodes_info_from_log(p_output):
    log_df = pd.DataFrame({'line':[l for l in 
        p_output.joinpath('log').read_text().split('\n')
    if l.startswith('[') or l.startswith('\t[')]})
    log_df = log_df[log_df['line'].str.startswith('\t[')]
    log_df = log_df['line'].str.extract('\t\[(?P<key1>[^\]]+)\]\\s+(?P<key2>[^\\s]+)\\s+(?P<value>\\d+)')
    log_df
    res = {k:{} for k in log_df['key1'].unique()
    }
    for i,row in log_df.iterrows():
        res[row['key1']][row['key2']] = int(row['value'])
    return res

def load_result_adata(p_output,load_X = False):
    p_output = Path(p_output)
    assert p_output.exists(),'not exits {}'.format(p_output)
    adata = None
    if load_X:
        adata = sc.read_h5ad(p_output.joinpath('adt.h5ad'))
        adata.obs = adata.obs.loc[:,[]].join(
            pd.read_csv(p_output.joinpath('obs.csv'),index_col=0)
        )
    else:
        adata = sc.AnnData(obs = pd.read_csv(p_output.joinpath('obs.csv'),index_col=0))
    adata.obs['model_label'] = adata.obs['true_label'].mask(
        adata.obs['type'] == 'que',adata.obs['pre_label']
    )
    adata.obsm['X_umap'] = adata.obs.loc[:,'UMAP1,UMAP2'.split(',')].to_numpy()
    adata.uns['p_item'] = p_output
    adata.uns['p_fig'] = p_output.joinpath('figs')
    adata.uns['parameters'] = json.loads(p_output.joinpath('parameters.json').read_text())
    return adata


def get_accuracy(adata):
    from sklearn.metrics import accuracy_score
    assert pd.Series('type,true_label,pre_label'.split(',')).isin(adata.obs.columns).all(),'type,true_label,pre_label must be in adata.obs'
    data_ref = adata.obs.query("type == '{}'".format('ref'))
    data_que = adata.obs.query("type == '{}'".format('que'))
    return {
        'ref':accuracy_score(data_ref['true_label'].values,
                   data_ref['pre_label'].values),
        'que':accuracy_score(data_que['true_label'].values,
                   data_que['pre_label'].values)
    }    
def get_weighted_f1_score(adata):
    from sklearn.metrics import f1_score
    assert pd.Series('type,true_label,pre_label'.split(',')).isin(adata.obs.columns).all(),'type,true_label,pre_label must be in adata.obs'
    data_ref = adata.obs.query("type == '{}'".format('ref'))
    data_que = adata.obs.query("type == '{}'".format('que'))
    return {
        'ref':f1_score(data_ref['true_label'].values,
                   data_ref['pre_label'].values,average='weighted'),
        'que':f1_score(data_que['true_label'].values,
                   data_que['pre_label'].values,average='weighted')
    }


# In[ ]:


def run(adata_ref,adata_que,key_cell_type,
    sp_ref,sp_que,p_homo,
    tissue,aligned,p_output,tag_output,**kvargs
):
    from .preprocess import aligned_type,process_for_graph,make_graph
    from .train import Trainer

    assert key_cell_type in adata_ref.obs.columns,'[Error] {} is not in adata_ref.obs'
    
    homo_method = 'biomart'
    n_hvgs = kvargs.setdefault('n_hvgs', 2000)
    n_degs = kvargs.setdefault('n_degs', 50)
    seed = 123
    stages = kvargs.setdefault('stages', [100, 200, 200])  # [200, 200, 200]
    nfeats = kvargs.setdefault('nfeats', 64)  # 64  # embedding size #128
    hidden = kvargs.setdefault('hidden', 64)  # 64  # 128
    is_1v1 = kvargs.setdefault('is_1v1', False)
    input_drop = 0.2
    att_drop = 0.2
    residual = True
    
    threshold = 0.9  # 0.8
    lr = 0.01  # lr = 0.01
    weight_decay = 0.001
    patience = 100
    enhance_gama = 10
    simi_gama = 0.1
    
    tag_output = "{};{}-corss-{};{}".format(tissue, sp_ref, sp_que, tag_output) if len(
            tag_output) > 0 else "{};{}-corss-{}".format(tissue, sp_ref, sp_que)
    tag_output
    
    p_output = Path(p_output).joinpath(tag_output)
    p_model = p_output.joinpath('model')
    p_fig = p_output.joinpath('figs')
    p_checkpt = p_model.joinpath("mutistages")
    
    [p.mkdir(exist_ok=True, parents=True) for p in [p_output, p_model, p_fig,p_checkpt]]
    [p_output.joinpath('res_{}'.format(i)).mkdir(exist_ok=True, parents=True)
        for i in range(len(stages))]

    # whether complete
    p_finish = p_output.joinpath("finish")
    if p_finish.exists():
        print("[{}][has finished] {}".format(get_time_tag(),p_output.name))
        return

    with RedirectStdStreamToLog(p_output.joinpath('log')) as log:
        print("[{}][start] {}".format(get_time_tag(),p_output.name),file=log.stdout)
        print("[{}][start] {}".format(get_time_tag(),p_output.name))
        print('[path_varmap] {}'.format(p_homo.name))
        finish_content = ["[{}][strat]".format(get_time_tag())]
        
        adata_ref.obs[key_cell_type] = adata_ref.obs[key_cell_type].astype(str)
        adata_que.obs[key_cell_type] = adata_que\
            .obs[key_cell_type].astype(str) if key_cell_type in adata_que.obs.columns else 'NA'
        
        if aligned:
            adata_ref, adata_que = aligned_type(
                [adata_ref, adata_que], key_cell_type
            )
        
        df_count = pd.concat([
            adata_ref.obs[key_cell_type].value_counts()\
                .to_frame(name='{}_{}'.format(tissue,sp_ref)),
            adata_que.obs[key_cell_type].value_counts()\
                .to_frame(name='{}_{}'.format(tissue,sp_que))
            ],axis=1)
        print(df_count)
        df_count.to_csv(p_output.joinpath("cell_type_counts.csv"), index=True)
        del df_count
        
        adata_ref.obs.to_csv(p_output.joinpath("obs_ref.csv"), index=True)
        adata_que.obs.to_csv(p_output.joinpath("obs_que.csv"), index=True)
        
        homo = pd.read_csv(p_homo,names='gn_ref,gn_que,homology_type'.split(','),
                           skiprows=1,).dropna()
        homo = get_1v1_matches(homo) if is_1v1 else homo
        
            
        kvargs.update(
            key_cell_type=key_cell_type,
            sp_ref=sp_ref,
            sp_que=sp_que,
            p_homo=str(p_homo),
            tissue=tissue,
            aligned=aligned,
            p_output=str(p_output),
            tag_output=tag_output,
            # modle parameter
            stages=stages,
            is_1v1=is_1v1,
            n_hvgs=n_hvgs,
            n_degs=n_degs,
            nfeats=nfeats,
            hidden=hidden,
        )
        p_output.joinpath("parameters.json").write_text(json.dumps(kvargs))
        
        finish_content.append("[{}][finish before run]".format(get_time_tag()))
        print("[{}][process_for_graph]".format(get_time_tag()).ljust(75, '-'),file=log.stdout)
        print("[{}][process_for_graph]".format(get_time_tag()).ljust(75, '-'))
        adatas, features_genes, nodes_genes, scnets, one2one, n2n = process_for_graph(
            [adata_ref, adata_que], homo, key_cell_type, 'leiden', n_hvgs=n_hvgs, n_degs=n_degs)
        g, inter_net, one2one_gene_nodes_net, cell_label, n_classes, list_idx = make_graph(
            adatas, aligned, key_cell_type, features_genes, nodes_genes, scnets, one2one, n2n, has_mnn=True, seed=seed)
        
        print("[{}][construct Trainer]".format(get_time_tag()).ljust(75, '-'),file=log.stdout)
        print("[{}][construct Trainer]".format(get_time_tag()).ljust(75, '-'))
        trainer = Trainer(adatas,
                                 g,
                                 inter_net,
                                 list_idx,
                                 cell_label,
                                 n_classes,
                                 threshold=threshold,
                                 key_class=key_cell_type)
        print("[{}][train]".format(get_time_tag()).ljust(75, '-'),file=log.stdout)
        print("[{}][train]".format(get_time_tag()).ljust(75, '-'))
        trainer.train(curdir=str(p_output),
                      checkpt_file=str(p_checkpt),
                      nfeats=nfeats,
                      hidden=hidden,
                      enhance_gama=enhance_gama,
                      simi_gama=simi_gama,stages=stages)
        
        finish_content.append("[{}][finish run]".format(get_time_tag()))
        
        adt = sc.AnnData(
            trainer.embedding_hidden.detach().numpy(),
            obs=pd.concat([
            adata_ref.obs.loc[:,[key_cell_type]].assign(species=sp_ref,type='ref'),
            adata_que.obs.loc[:,[key_cell_type]].assign(species=sp_que,type='que')
        ]).rename(columns = {key_cell_type:'cell_type'})
        )
        adt.write_h5ad(p_output.joinpath('adt.h5ad'))

        adt.obs = adt.obs.join(pd.read_csv(p_output.joinpath("res_2", "pre_out_2.csv"), index_col=0)
                               .loc[:,'true_label,pre_label,max_prob'.split(',')])
        adt.obs['is_right'] = adt.obs.eval("true_label == pre_label")
        sc.pp.neighbors(adt, n_neighbors=15, metric="cosine", use_rep="X")
        sc.tl.umap(adt)
        adt.obs = adt.obs.join(pd.DataFrame(adt.obsm['X_umap'],index=adt.obs.index,columns='UMAP1,UMAP2'.split(',')))
        adt.obs = adt.obs.loc[:,'UMAP1,UMAP2,cell_type,type,species,true_label,pre_label,max_prob,is_right'.split(',')]
        adt.obs.to_csv(p_output.joinpath('obs.csv'),index=True)
        print("[{}][finish] {}".format(get_time_tag(),p_output.name))

    p_output.joinpath('model_gene_nodes_info.json').write_text(
        json.dumps(extract_model_gene_nodes_info_from_log(p_output)))
    # 完成标记
    finish_content.append("[{}][end]".format(get_time_tag()))
    p_finish.write_text("\n".join(finish_content))
    print("[{}][finish] {}".format(get_time_tag(),p_output.name))


# In[ ]:


class RedirectStdStreamToLog:
    """将stdout和stderr重定向至日志文件
使用with RedirectStdStreamToLog('path of log'):
确保在正常结束/发生错误时
stdout, stdout被置换为初始状态
以及日志文件流释放
    """
    def __init__(self,log_path):
        log_path = Path(log_path)
        # assert log_path.is_file(), '[Error] log_path must be a path of file'
        self.path = log_path
        self.log = None

        self.stdout = sys.stdout
        self.stderr = sys.stdout
    def __enter__(self):
        self.log = self.path.open(mode='w')
        sys.stdout = self.log
        sys.stderr = self.log
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout 
        sys.stderr = self.stderr
        self.log.close()

