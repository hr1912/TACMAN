#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""df
用于操作numpy.array
> function
    scale
"""


# In[ ]:


from .general import Path, np, pd


# In[ ]:


def scale(arr, arr_min=None, arr_max=None,
          edge_min=None, edge_max=None, func=None):
    """
将arr 从[arr_min,arr_max]映射到[edge_min,edge_max]
[可选]在线性映射完成后使用func对结果进行第二次映射

# Example
df_plot = pd.concat(
    [pd.DataFrame(
        {'data': rng.normal(3, 1, size=10)}
    ).assign(**{'type': 'a'}),
        pd.DataFrame(
        {'data': rng.normal(5, 1, size=90)}
    ).assign(**{'type': 'b'})])

df_plot = df_plot.sort_values('data')
df_plot.index = np.arange(df_plot.shape[0])
df_plot['data_2'] = np.sort(rng.random(size=df_plot.shape[0]) -5)
df_plot['data_2_scale'] = ut.arr.scale(df_plot['data_2'],
                                edge_min=df_plot['data'].min(),
                                edge_max=df_plot['data'].max())

df_plot.head(2)

with mpl.rc_context(ut.pl.figure.rc_frame):
    fig, ax = ut.pl.figure.subplots_get_fig_axs()

    ax.axhline(np.min(df_plot['data']), **{'linewidth': .5, 'color': 'black','linestyle':':'})
    ax.axhline(np.max(df_plot['data']), **{'linewidth': .5, 'color': 'black','linestyle':':'})
    
    for i, (k,c) in enumerate(
        zip('data,data_2,data_2_scale'.split(','),
            ut.pl.colormap.get_color(3))):
        ax.plot(df_plot.index, df_plot[k],label=k,
               linewidth= .5, color= c)
fig.legend()

fig
"""
    if arr_min is None or arr_max is None:
        arr_min, arr_max = np.min(arr), np.max(arr)
    if edge_min is None or edge_max is None:
        # print('edge_min,edge_max = arr_min,arr_max')
        edge_min, edge_max = arr_min, arr_max

    res = (edge_max-edge_min)/(arr_max-arr_min) * (arr- arr_min) + edge_min
    if func:
        res = np.vectorize(func)(res)
    return res

def yield_ele_by_count(arr,counts):
    """
Parameters
----------
counts : list
    每次要取出的元素数量
    [2,3,2] 第一/二/三次取2/3/2个元素
"""
    index = 0
    for c in counts:
        yield arr[index:index+c]
        index += c

