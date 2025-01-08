#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""df
用于操作pandas.DataFrame

> function
    show
    reindex_with_unique_col
    apply_merge_field

>> Path
    iter_dir
    walk_dir

>> gourp
    gourp_agg
    group_transform
    group_yield

>> sort
    sort_key_with_str_list

>> random
    random_choice
    random_choice_by_group

"""


# In[2]:


from .general import Path, sys, np, pd
from .general import handle_type_to_list


# In[1]:


def show(df, n=2):
    from IPython.display import display
    display(df.head(n), df.shape)


def reindex_with_unique_col(df, key, drop=False):
    assert df[key].is_unique, "[Error][not unique] {} column".format(key)
    df.index = df[key].to_numpy()
    if drop:
        df = df.drop(columns=[key])
    return df


def apply_merge_field(df, str_format, check_unique=False):
    res = df.apply(lambda row: str_format.format(**row), axis=1)
    if check_unique:
        assert res.is_unique, "[Error][not unique] {}".format(str_format)
    return res


def to_dict(df, key_key, key_value, check_key_unique=False):
    if check_key_unique:
        assert df[key_key].is_unique, "[Error] col '{}' is not unique".format(
            key_key)
    res = {k: v for k, v in zip(df[key_key], df[key_value])}
    return res


def get(df, indexs=None, columns=None):
    if indexs is None and columns is None:
        return df
    if indexs is None:
        indexs = df.index
    else:
        temp_indexs = [i for i in indexs if not i in df.index]
        if len(temp_indexs) > 0:
            print("[Waring] index not in df : {}".format(','.join(temp_indexs)))
            indexs = [i for i in indexs if not i in temp_indexs]
        del temp_indexs
    if columns is None:
        columns = df.columns
    else:
        temp_columns = [i for i in columns if not i in df.columns]
        if len(temp_columns) > 0:
            print("[Waring] index not in df : {}".format(','.join(temp_columns)))
            columns = [i for i in columns if not i in temp_columns]
        del temp_columns

    return df.loc[indexs, columns]


def ravel(df, name_value='value', order='C', sep=';'):
    assert df.index.is_unique and df.columns.is_unique, "[Error] index or columns is not unique"
    return pd.DataFrame({name_value: np.ravel(df.to_numpy(), order=order)},
                        index=pd.MultiIndex.from_product([df.index, df.columns],
                                                         names=['ind', 'col']).to_series().str.join(sep).values)
def product_seq(seq2d,names=None):
    """ 笛卡尔积
Parameters
----------
seq2d : list | np.arr | arr-like
    [[1,2,3,4],['a','b','c']]
"""
    names = ['col_{}'.format(i) for i in range(len(seq2d))] if names is None else names
    return pd.MultiIndex.from_product(seq2d,names=names).to_frame().reset_index(drop=True)

def ones(size,shape,index=None,columns=None,value=1):
    """构造并返回一个size个元素，形状为shape，由value填充的DataFrame
"""
    return pd.DataFrame(np.repeat(value,size).reshape(shape),
                 index=index,columns=columns)
def ones_from_df(df,value=1):
    return ones(df.size,df.shape,df.index,df.columns,value)


# # path

# In[ ]:


def iter_dir(p, path_match='', path_match_filter=[], select='f'):
    p = Path(p)
    assert p.exists(), '[not exists] {}'.format(p)
    assert p.is_dir(), '[Error] p is not a dir'

    res = pd.DataFrame({'path': p.iterdir()})

    # select
    if select == 'file' or select[0] == 'f':
        res = res[res['path'].apply(lambda x: x.is_file())]
    elif select == 'dir' or select[0] == 'd':
        res = res[res['path'].apply(lambda x: x.is_dir())]
    else:
        # file and dir
        pass
    if res.shape[0] == 0:
        print('[Error] no item')
        return pd.DataFrame({'path':[],'name':[]})

    if path_match:
        res = res[res['path'].apply(lambda x: x.match(path_match))]

    if path_match_filter and isinstance(path_match_filter, str):
        path_match_filter = [path_match_filter]
    for _ in path_match_filter:
        res = res[res['path'].apply(lambda x: not x.match(_))]

    res['name'] = res['path'].apply(lambda x: x.name)
    res.index = np.arange(res.shape[0])
    return res.copy()


def walk_dir(p, select='f', drop_type=False):
    """深度遍历dir
"""
    def walk_with_os(p):
        import os
        files = np.array([])
        dirs = np.array([])
        for root, ds, fs in os.walk(p):
            if len(ds) > 0:
                dirs = np.concatenate(
                    [dirs, [os.path.join(root, d) for d in ds]])
            if len(fs) > 0:
                files = np.concatenate(
                    [files, [os.path.join(root, f) for f in fs]])
        return files, dirs

    def walk_with_Path(p):
        """
    Python 3.12以上才有 Path().walk
    sys.version_info.major >= 3  and  sys.version_info.minor >= 12
        """
        files = np.array([])
        dirs = np.array([])
        for root, ds, fs in p.walk():
            if len(ds) > 0:
                dirs = np.concatenate(
                    [dirs, [root.joinpath(d) for d in ds]])
            if len(fs) > 0:
                files = np.concatenate(
                    [files, [root.joinpath(f) for f in fs]])
        return files, dirs

    p = Path(p)
    assert p.exists(), "[not exists] {}".format(p)
    files, dirs = walk_with_Path(p) if sys.version_info.major >= 3 and\
        sys.version_info.minor >= 12 else walk_with_os(p)

    res = pd.concat([pd.DataFrame({'path': files, 'type': 'f'}),
                     pd.DataFrame({'path': dirs, 'type': 'd'})
                     ])
    res['path'] = res['path'].apply(lambda x: Path(x))
    res['name'] = res['path'].apply(lambda x: x.name)
    res = res.loc[:, 'path,name,type'.split(',')]
    if select in 'f,d'.split(','):
        res = res.query("type == '{}' ".format(select))
        if drop_type:
            res = res.drop(columns='type')

    return res.reset_index(drop=True).copy()


# # group

# In[ ]:


def group_agg(df, groupby_list, agg_dict=None, dropna=True,
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


def group_transform(df, groupby_list, key, func):
    """DataFrame.groupby.transform
相比于agg, transform 会将各个元素对应的分组计算结果返回，其索引保持不变
适合用于将分组统计的结果作为新的一列添加至df中
Parameters
----------
func : str | func
    count,sum,max,min,mean,median,std,var
    'sum' -> lambda x:x.sum()
"""
    return df.groupby(groupby_list)[key].transform(func)


def group_yield(df, key_group, keys_value, order=None, return_group=False, check_order=True):
    """按组yield
Parameters
----------
order : str | list | None

check_order : bool | default: True
    检查order 中的项 均存在于 df[key_group]
"""
    keys_value = handle_type_to_list(keys_value)
    order = df[key_group].unique() if order is None else order
    if check_order:
        assert pd.Series(order).isin(df[key_group]).all(), """[Error] not all ele of group in order
order special {}
group special {}
""".format(','.join(np.setdiff1d(order, df[key_group])),
           ','.join(np.setdiff1d(df[key_group], order)))

    if return_group:
        for g in order:
            yield g, df.loc[df[key_group] == g, keys_value]
    else:
        for g in order:
            yield df.loc[df[key_group] == g, keys_value]


# # sort

# In[ ]:


def sort_key_with_str_list(str_list,ascending=True,return_order_dict=False):
    def str_list_to_order_dict(str_list,ascending=True):
        if isinstance(str_list, str):
            str_list = str_list.split(',')
        str_list = str_list if ascending else str_list[::-1]
        return {k: i for i, k in enumerate(str_list)}

    order_dict = str_list_to_order_dict(str_list,ascending=ascending)
    if return_order_dict:
        return order_dict

    return lambda s: s.map(lambda k: order_dict.setdefault(k, k))


# # Matrix

# In[3]:


def matrix_classify(df, key_index, key_column, func_agg='count', fill_na=0):
    """获取两个分类变量的混淆矩阵"""
    df = group_agg(df, [key_index, key_column], {key_column: [func_agg]})
    key_value = '__{}'.format(func_agg) if isinstance(func_agg, str) else '__values'

    df.columns = [key_index, key_column, key_value]
    df = df.pivot(index=key_index, columns=key_column, values=key_value)
    if not fill_na is None:
        df = df.fillna(fill_na)
    return df


def matrix_numeric(df, groupby_list, key_value, func_agg, fill_na=0):
    """以groupby_list对df进行分组,使用func_agg进行聚合
Parameters
----------
groupby_list : list
    str list
    分组字段

key_value : str
    数值字段

func_agg : func | str
    min,max,mean,median,std,var
"""
    assert len(groupby_list) == 2
    df = df.groupby(groupby_list).agg({key_value: func_agg})
    key_value = 'value_{}'.format(key_value)
    df.columns = [key_value]
    df = df.reset_index().pivot(index=groupby_list[0], columns=groupby_list[1], values=key_value)
    if not fill_na is None:
        df = df.fillna(fill_na)
    return df


# # random

# In[ ]:


def random_choice(arr, size, seed=None, replace=False):
    arr = np.array(arr)
    _rng = np.random.default_rng(seed)  # 若为None,则相当于没有设置种子
    return _rng.choice(arr, size, replace=replace)


def random_choice_by_group(df, key, percentage, seed=None, replace=False):
    """按组抽样 , 返回抽样后的index
要求df.index是唯一的

"""
    assert df.index.is_unique
    assert key in df.columns

    df = df.loc[:, [key]].copy()
    df[key] = df[key].astype(str)
    df_stat = df[key].value_counts().to_frame('__count').reset_index(key)
    df_stat['__count_choice'] = np.ceil(df_stat['__count']*percentage)
    df_stat['__count_choice'] = df_stat['__count_choice'].astype(int)
    if df_stat.query("__count <= __count_choice").shape[0] > 0:
        print('[Waring][random_choice_by_group] __count <= __count_choice')
        display(df_stat.query("__count <= __count_choice"))
    return np.concatenate(df_stat.apply(lambda row:
                                        random_choice(
                                            df[df[key] == row[key]].index, row['__count_choice'],
                                            seed=seed, replace=replace), axis=1))

