#!/usr/bin/env python
# coding: utf-8

# # 清除utils 下的py文件
#
# ipynb 和 py 混在一起实在时太难受了
#
#
# ```bash
# conda activate
#
# cd ~/link/csMAHN_Spatial
# jupyter nbconvert utils/*.ipynb --to python && rm utils/del_py.py
#
# :
#
# ```

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd


# In[2]:


# 来自df
def iterdir(p, path_match="", path_match_filter=[], select="f"):
    p = Path(p)
    assert p.is_dir(), "[Error] p is not a dir"

    res = pd.DataFrame({"path": p.iterdir()})

    # select
    if select == "file" or select[0] == "f":
        res = res[res["path"].apply(lambda x: x.is_file())]
    elif select == "dir" or select[0] == "d":
        res = res[res["path"].apply(lambda x: x.is_dir())]
    else:
        # file and dir
        pass

    if path_match:
        res = res[res["path"].apply(lambda x: x.match(path_match))]

    if path_match_filter and isinstance(path_match_filter, str):
        path_match_filter = [path_match_filter]
    for _ in path_match_filter:
        res = res[res["path"].apply(lambda x: not x.match(_))]

    res["name"] = res["path"].apply(lambda x: x.name)
    res.index = np.arange(res.shape[0])
    return res.copy()


# In[3]:


p_item = Path(".")

df = pd.concat(
    [
        iterdir(p, select="f", path_match="*.py")
        for p in iterdir(
            p_item,
            select="d",
            path_match_filter=["*__pycache__", "*.ipynb_checkpoints"],
        )["path"]
    ]
)
df = pd.concat([iterdir(".", select="f", path_match="*.py"), df])
df


# In[4]:


display(df)
for i, row in df.iterrows():
    row["path"].unlink()
pass
