#!/usr/bin/env python
# coding: utf-8

# <script src="https://unpkg.com/mermaid/dist/mermaid.min.js"></script>
#
# ```mermaid
# graph LR;
#     __init__{{__init__}};
#     general[general];
#     general --> df[df];
#     general --> arr[arr];
#     df -.-> __init__
#     arr -.-> __init__
#     general --> crawl[crawl]
#     crawl -.-> __init__
#     general -.-> |*| __init__
#
#     subgraph scanpy
#         scanpy.__init__{{__init__}};
#         scanpy.sc[sc];
#         scanpy.pl[pl];
#         scanpy.sc -->scanpy.pl
#         scanpy.pl -.-> scanpy.__init__
#         scanpy.sc -.-> |*| scanpy.__init__
#     end
#     general --> scanpy.sc
#     scanpy.__init__ -.-> __init__
#
#     subgraph plot
#         plot.__init__{{__init__}};
#         plot.figure[figure];
#         plot.pl[pl];
#
#         plot.figure -.-> plot.pl
#         plot.pl -.-> plot.__init__
#
#         plot.cmap[cmap];
#         plot.cmap -.-> plot.pl
#         plot.figure --> plot.cmap
#
#         plot.path[path];
#         plot.figure --> plot.path -.-> plot.pl
#     end
#     general --> plot.figure
#     plot.__init__ -.-> __init__
#
# ```

# In[ ]:


"""
LCC 的python 函数库
为实验室留下些什么吧
0.0.2 2024年10月29日

import utils as ut
# help(ut)
print(ut.__doc__)
print(ut.__version__)
"""


# In[ ]:


__version__ = "0.0.1"


# In[ ]:


from . import general
from . import df, arr
from .general import *


# In[ ]:


with Block(
    "[import utils.crawl]", context={"module": "requests,lxml".split(",")}
) as context:
    if all([module_exists(_) for _ in context.module]):
        from utils import crawl
    else:
        crawl = "[module has not installed] {}".format(",".join(context.module))


# In[ ]:


with Block(
    "[import utils.plot]", context={"module": "matplotlib,scipy".split(",")}
) as context:
    if all([module_exists(_) for _ in context.module]):
        from utils import plot as pl
    else:
        pl = "[module has not installed] {}".format(",".join(context.module))


# In[ ]:


with Block("[import utils.scanpy]", context={"module": "scanpy".split(",")}) as context:
    if all([module_exists(_) for _ in context.module]):
        from utils import scanpy as sc
    else:
        sc = "[module has not installed] {}".format(",".join(context.module))


# In[ ]:


del context
