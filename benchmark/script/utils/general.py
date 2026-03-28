#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""general
utils的基础, 公共的包在genral中导入

> variable
    rng

> function
    module_exists
    subset_dict
    rm_rf
    str_step_insert
    【show】
    show_str_arr
    show_obj_attr
    show_dict_key

> class
    Block

"""


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

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


rng = np.random.default_rng()


# In[4]:


def module_exists(module_name):
    import importlib.util
    import sys

    return module_name in sys.modules or importlib.util.find_spec(module_name)


def subset_dict(data, keep_item=[], regex=None, keep_order=False):
    if isinstance(keep_item, str):
        keep_item = [keep_item]

    data = data.copy()
    keys = []
    if regex:
        keys = pd.Series(list(data.keys()))
        keys = keys[keys.str.match(regex)]
    keys = pd.Series(np.concatenate([keys, keep_item])).unique()

    if keep_order:
        return {k: v for k, v in data.items() if k in keys}
    return {k: data[k] for k in keys}


def update_dict(data, update=None, **kvargs):
    """更新字典，覆盖顺序 kvargs > update > data"""
    data = data.copy()
    if isinstance(update, dict):
        data.update(update)
    data.update(kvargs)
    return data


def handle_type_to_list(data, t=str):
    """若 data 为 t 则返回 [data]
    若 data 为list 则返回 data
    """
    assert isinstance(data, (list, t)), "[Error] data is not a list or {}".format(t)
    return [data] if isinstance(data, t) else data


# # show

# In[5]:


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
    if data.size % n_cols != 0:
        data = np.concatenate([data, np.repeat("", n_cols - data.size % n_cols)])
    data = pd.Series(data)
    data = pd.DataFrame(np.reshape(data, (data.size // n_cols, n_cols), order=order))
    for k in data.columns:
        data[k] = data[k].str.ljust(data[k].str.len().max())
    res = "\n".join(
        data.apply(lambda row: "| {}|".format("| ".join(row.values)), axis=1)
    )
    print(res)

    if return_str:
        return res


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


def show_dict_key(data, tag="", sort_key=True):
    print("> {}['']".format(tag).ljust(75, "-"))
    ks = list(data.keys())
    if sort_key:
        ks = np.sort(ks)
    print(*["  {}".format(k) for k in ks], sep="\n")


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

    def __init__(self, comment, show_comment=False, context=None):
        self.comment = comment
        self.show_comment = show_comment
        self.context = context
        self.time = 0

    def __enter__(self):
        if self.show_comment:
            self.time = time.time()
            print(
                "[start][{}] {}".format(
                    self.comment, time.strftime("%m%d-%H:%M:%S", time.localtime())
                ).ljust(75, "-")
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.show_comment:
            print(
                "[end][{}] {}".format(
                    self.comment, time.strftime("%m%d-%H:%M:%S", time.localtime())
                ).ljust(75, "-")
            )
            print("\t{:.3f} s used".format(time.time() - self.time))
        # 释放content
        self.context = None

    def __str__(self):
        return """Block
\tcomment     : {}
\tcontext_key : {}
""".format(
            self.comment, ",".join(self.context.keys())
        )

    # 对类及其实例未定义的属性有效
    # 若name 不存在于 self.__dir__中,则调用__getattr__
    def __getattr__(self, name):
        cls = type(self)
        res = self.context.setdefault(name, None)
        if res is None:
            raise AttributeError(
                "{.__name__!r} object has no attribute {!r}".format(cls, name)
            )

        return res


# # other

# In[7]:


import platform

system = platform.system()
system == "Linux"


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


def str_step_insert(s, step, insert="\n"):
    """于s的每step个字符后插入一个insert
    在为图添加超长的文字时
    使用str_step_insert('0123456789012345678901234567',5,insert='\n')
    使字符串换行

    """
    return insert.join(
        [
            s[i * step : (i + 1) * step]
            for i in range(len(s) // step + (0 if len(s) % step == 0 else 1))
        ]
    )


# In[24]:


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


# # jupyter UI
#
# 引入jquery

# In[8]:


from IPython.display import display_javascript

display_javascript(
    """
    if (document.getElementsByClassName("user-container-lcc").length == 0) {
        var ele_body = document.getElementsByTagName("body")[0]
        var ele_div = document.createElement("div")
        var ele_scr = document.createElement("script")
        ele_scr.src = "https://code.jquery.com/jquery-3.6.2.min.js"
        ele_div.className = "user-container-lcc"
        ele_div.insertBefore(ele_scr, ele_div.firstChild)
        ele_body.insertBefore(ele_div, ele_body.firstChild)
    }
""",
    raw=True,
)
del display_javascript
