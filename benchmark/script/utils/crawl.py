#!/usr/bin/env python
# coding: utf-8

# In[17]:


"""crawl
爬虫相关函数

核心包:requests,lxml
"""


# In[ ]:


from .general import Path, json
import requests
from lxml import etree

session = requests.session()

headers_default = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
}


# # requests发送网络请求

# In[ ]:


def get_update_headers(dict_update={}, headers=headers_default):
    """使用dict_update对headers进行更新
    首先对headers进行copy, 再update, 不会对外部的headers进行修改

    Parameters
    ----------
    dict_update : dict (default: {})

    headers : dict (default: ut.crawl.headers_default)
    """
    headers = headers.copy()
    headers.update(dict_update)
    return headers


def get(
    url, headers=None, data={}, session=session, status_allow=[200], *args, **kvargs
):
    """由session调用get

    Parameters
    ----------
    headers : dict (default: None)
        若为None,则get_update_headers()为其赋值
        该函数默认返回ut.crawl.headers_default
    """
    if headers is None:
        headers = get_update_headers()
    r = session.get(url=url, headers=headers, data=data, *args, **kvargs)
    assert r.status_code in status_allow, "[status is not allowed] {}".format(
        r.status_code
    )
    return r


# # 使用lxml.etree进行xpath解析

# In[16]:


def load_etree(str_or_path):
    txt = None
    if isinstance(str_or_path, str):
        try:
            str_or_path = (
                Path(str_or_path) if Path(str_or_path).exists() else str_or_path
            )
        except Exception as e:
            txt = str_or_path
    if isinstance(str_or_path, Path):
        txt = Path(str_or_path).read_text()
    assert not txt is None
    return etree.HTML(txt)
