#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""path
路径


> function
    rectangle

> codes

+ CLOSEPOLY = 79
+ CURVE3 = 3
+ CURVE4 = 4
+ LINETO = 2
+ MOVETO = 1
+ STOP = 0

"""


# In[ ]:


from .figure import *
import matplotlib.path as mpath


# In[ ]:


def rectangle(length_width_ratio = 1):
    """返回一个
(0,1) <- (length_width_ratio,1)
|          |
(0,0) -> (length_width_ratio,0)
的矩形路径
"""
    return mpath.Path(np.array([[0,0],[length_width_ratio,0],
                                [length_width_ratio,1],[0,1],[0,0]]),
        np.array([ 1,  2,  2,  2, 79]))

