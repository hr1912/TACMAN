#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""pl
绘图函数
需传入ax

> function
>> tool
    tl_rc
    tl_fontdict
    tl_savefig
    tl_jitter
    tl_get_significance_marker
    tl_help_stats
    tl_GaussianKDE
    tl_GaussianKDE_evaluate
    tl_GaussianKDE_evaluate_get_coords

> variable
>> rc [import from plot.figure]
    rc_mpl
    rc_default
    rc_blank
    rc_frame
    rc_box
    rc_tl_notick
    rc_tl_off_x
    rc_tl_off_y
    rc_tl_xt
    rc_tl_yr
"""


# In[ ]:


from . import figure
from . import cmap
from . import path

from .figure import np,pd,mpl,plt,Path
from ..general import rng,Block,Iterable
from ..general import handle_type_to_list,update_dict
from .. import df as ut_df

from scipy import stats


# # rc

# In[ ]:


from .figure import rc_mpl,rc_default,rc_blank
from .figure import rc_frame,rc_frame_lt,rc_box
from .figure import rc_tl_notick,rc_tl_off_x,rc_tl_off_y,rc_tl_xt,rc_tl_yr


# # tool

# In[ ]:


from .figure import tl_rc,tl_fontdict,tl_savefig

def tl_help_stats():
    from IPython.display import display_markdown

    display_markdown("""
> pl.tl_help_stats
>> 非参数检验不要求样本分布服从正态分布

|函数||参数检验|简要说明|
|:-|:-|:-|:-|
|[scipy.stats.ttest_1samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)|`X=popmean`||t检验(n>=30)|
|[scipy.stats.ttest_1samp]((https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html))(X-Y,0)|`X-Y=0`||配对t检验|
|[scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)|`X=Y`||独立样本t检验|
|[scipy.stats.wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)|`X-Y=0`|非参数检验|**秩和检验**,配对t检验非参数形式|
|[scipy.stats.mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)||非参数检验|**U检验**,用于检验分布之间位置的差异|
|[scipy.stats.spearmanr](https://scipy.github.io/devdocs/tutorial/stats/hypothesis_spearmanr.html)|X,Y||相关性检验|
|[scipy.stats.pearsonr](https://scipy.github.io/devdocs/reference/generated/scipy.stats.pearsonr.html)|X,Y||相关性检验|

[p值校正 scipy.stats.false_discovery_control](https://scipy.github.io/devdocs/reference/generated/scipy.stats.false_discovery_control.html#scipy.stats.false_discovery_control)
""",
    raw=True)

def tl_get_significance_marker(pvalues, markers=None, not_significance_marker='ns'):
    """

Parameters
----------
pvalues : float | Iterable
markers : dict
        not_significance_marker:1
        '*': 0.05
        '**': 0.01,
        '**': 0.001,

"""
    marker = update_dict({not_significance_marker:1,'*': 0.05,'**': 0.01,'***': 0.001},
                         markers)
    
    marker = pd.DataFrame({
        'marker':marker.keys(),
        'value':marker.values(),
    }).sort_values('value',ascending=False)
    res = pd.DataFrame(
        {'pvalue':pvalues if isinstance(pvalues,Iterable) else [pvalues]})
    res['marker'] = not_significance_marker
    for i,row in marker.iterrows():
        res['marker'] = res['marker'].mask(res['pvalue']<row['value'],row['marker'])

    res = res['marker'].to_numpy()
    if len(res) == 1:
        res = res[0]
    return res

def tl_format_stats_res(res,tag='statistic',
                        format="{tag} = {statistic:.2e}\npvalue = {pvalue:.4f}\n{significance_marker}"):
    return format.format(
        tag=tag,statistic=res.statistic,
           pvalue=res.pvalue,
           significance_marker = tl_get_significance_marker(res.pvalue))

def tl_jitter(n,range=1):
    """抖动，生成n个微小的抖动

-1/2*range <-- 0 --> 1/2*range
"""
    return (rng.random(n)-.5)*range

def tl_GaussianKDE(X,bw_method='scott'):
    import matplotlib.mlab as mlab
    return mlab.GaussianKDE(X,bw_method)
def tl_GaussianKDE_evaluate(X,X_evaluate=None,bw_method='scott',return_X_evaluate = False):
    X_evaluate =  np.arange(X.min(),X.max(),(X.max()-X.min())/50) if X_evaluate is None else X_evaluate
    if return_X_evaluate:
        return X_evaluate,tl_GaussianKDE(X,bw_method).evaluate(X_evaluate)
    else:
        return tl_GaussianKDE(X,bw_method).evaluate(X_evaluate)
def tl_GaussianKDE_evaluate_get_coords(X,X_evaluate=None,bw_method='scott'):
    return tl_GaussianKDE_evaluate(X,X_evaluate,bw_method,True)

def tl_stats_bar(ax,X,bottom=.92,top=.95,vertical=False,kvarg_line=None,
                 text=None,text_offect=.01,fontdict=None):
    """ 绘制统计bar
   text
-----------
|         |


Parameters
----------
vertical : bool
    defalut False
    if True
        --
          |
          | text
          |
        --

kvarg_line : dict
    dict(
        lw=.5,
        color='black'
        )

fontdict : dict
    pl.tl_fontdict('cb')

    vertical is True
        pl.tl_fontdict('lc')
    """
    
    def _func(pct,_min,_max):
        return _min+(_max-_min)*pct

    kvarg_line = update_dict(dict(lw=.5,color='black'),kvarg_line)
    fontdict = update_dict(tl_fontdict('lc' if vertical  else 'cb'),fontdict)
    _min,_max = ax.get_xlim() if vertical else ax.get_ylim()
    coor = [[X[0],X[0],X[1],X[1]],
            [_func(bottom,_min,_max),_func(top,_min,_max),_func(top,_min,_max),_func(bottom,_min,_max)]]
    coor_text =  [sum(X[:2])/2,_func(top+text_offect,_min,_max)]
    if vertical:
        coor = coor[::-1]
        coor_text = coor_text[::-1]
    ax.step(*coor,**kvarg_line)
    if isinstance(text,str):
        fontdict = {} if fontdict is None else fontdict
        ax.text(*coor_text,text,fontdict=fontdict)

def tl_expand_ax(ax,pct=.1,axis='x',orientation = '+'):
    """延伸按比例 ax
Parameters
----------
axis : str
    延伸的轴
    x, y, both
orientation : str
    延伸方向, 默认 + 为正向延伸
    +, -, both
"""
    assert orientation in '+,-,both'.split(',')
    assert axis in 'x,y,both'.split(',')
    
    if axis == 'x':
        _min,_max = ax.get_xlim()
    elif axis == 'y':
        _min,_max = ax.get_ylim()
    elif axis == 'both':
        tl_expand_ax(ax,pct,orientation,axis='x')
        tl_expand_ax(ax,pct,orientation,axis='y')
        return

    offset = (_max-_min)*pct
    
    if orientation == '+':
        _min = _min
        _max = _max + offset
    elif orientation == '-':
        _min = _min - offset
        _max = _max
    elif orientation == 'both':
        _min = _min - offset
        _max = _max + offset
        
    if axis == 'x':
        ax.set_xlim(_min,_max)
    elif axis == 'y':
        ax.set_ylim(_min,_max)

def tl_str_next_line(s,index=30,s_find = ' '):
    """将字符串s每index个字符分为一组,
除了第0组外，其余组将第一个s_find替换为\n

Parameters
----------
s : str | Iterable
    """
    def handel(s,index,s_find=' '):
        _left = np.arange(0, len(s), index)
        _right = np.concatenate([_left[1:], [len(s)]])
        return ''.join([s[_l:_r] if _i == 0 else s[_l:_r].replace(
            s_find, '\n', 1) for _i, (_l, _r) in enumerate(zip(_left, _right))])
    if isinstance(s,str):
        return handel(s,index,s_find)
    elif isinstance(s,Iterable):
        return [handel(i,index,s_find) for i in s]

def tl_brokenaxes(axs,lims,func_plot,kw_func_plot=None,axis='y',**kvarg):
    assert len(axs) == len(lims),"[Error] length of axs and lims is not equal"
    kw_func_plot = update_dict({},kw_func_plot)
    for ax,lim in zip(axs,lims):
        func_plot(ax,**kw_func_plot)
        
        if axis == 'x':
            ax.set_xlim(*lim[:2])
        elif axis == 'y':
            ax.set_ylim(*lim[:2])
    return axs


# # scatter

# In[ ]:


def scatter_confidence_bands(X,Y,ax,kvarg_ax_plot=None,kvarg_ax_fill_between=None):
    """
[Example: Confidence bands](https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#example-confidence-bands)

Parameters
----------
kvarg_ax_plot : dict
    [详见 ax.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html)
kvarg_ax_fill_between :dict
    [详见 ax.fill_between](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html)
"""
    kvarg_ax_plot = {} if kvarg_ax_plot is None else kvarg_ax_plot
    kvarg_ax_fill_between = dict(alpha=.5) if kvarg_ax_fill_between is None else kvarg_ax_fill_between
    
    a, b = np.polyfit(X, Y, deg=1)
    y_est = a * X + b
    y_err = X.std() * np.sqrt(1/len(X) +
                              (X - X.mean())**2 / np.sum((X - X.mean())**2))
    ax.plot(X, y_est,**kvarg_ax_plot)
    ax.fill_between(X, y_est - y_err, y_est + y_err,**kvarg_ax_fill_between)


# # bar

# In[ ]:


def bar_count_to_ratio(df_plot):
    return df_plot.transpose()/df_plot.sum(axis=1)

def bar_cumsum_df_plot(df_plot,prefix='bottom_',zero_first = True):
    df_cumsum = df_plot.cumsum()
    if zero_first:
        df_cumsum = np.vstack((np.zeros(df_cumsum.shape[1]),
           df_cumsum.iloc[:-1,:].to_numpy()))
        df_cumsum = pd.DataFrame(df_cumsum,index = df_plot.index)
        
    df_cumsum.columns=['bottom_{}'.format(i) for i in df_plot.columns]
    return df_plot.join(df_cumsum)

def bar_transpose_df_cumsum(df_plot,cumsum_prefix='bottom_',reindex=False):
    
    if reindex or (not df_plot.index.to_series().apply(lambda x:isinstance(x,str)).all()):
        df_plot.index = ['I{}'.format(i) for i in np.arange(df_plot.shape[0])]
        
    df_cumsum = df_plot.filter(regex='^{}'.format(cumsum_prefix)).transpose()
    df_plot = df_plot.filter(regex='^(?!{})'.format(cumsum_prefix)).transpose()
    assert df_cumsum.shape == df_plot.shape
    df_cumsum.index = df_cumsum.index.to_series().str.replace('^{}'.format(cumsum_prefix),'',regex = True).to_numpy()
    df_cumsum.columns=['bottom_{}'.format(i) for i in df_cumsum.columns]
    return df_plot.join(df_cumsum)

def bar_add_ticks(ax,df_plot,
        group_counts = 1,
        gap_between_group = 1,
        width_one_bar=1,
        width_ratio=.8,
        offset=0,
        to_horizontal=False,
        fontdict=None):
    
    fontdict = update_dict({},fontdict)
    
    x = np.arange(df_plot.shape[0])
    x = np.array(
        [(x*group_counts+0)*width_one_bar + x*gap_between_group + offset,
        (x*group_counts+group_counts-1)*width_one_bar + x*gap_between_group + offset])\
        .mean(axis=0)
    
    if to_horizontal:
        ax.set_yticks(x,df_plot.index,**fontdict)
    else:
        ax.set_xticks(x,df_plot.index,**fontdict)
        
    return x

def bar(ax,df_plot,
        key_height,
        cmap = None,
        bottom = 0,
        group_counts = 1,
        ngroup = 0,
        gap_between_group = 1,
        width_one_bar=1,
        width_ratio=.8,
        to_horizontal =False,**kwargs):
    """封装了ax.bar 和 ax.barh (to_horizontal = True)

> 详见
+ [ax.bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html)
+ [ax.barh](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html)

Parameters
----------
cmap : None|list|dict
    颜色控制
bottom : str|float|np.array
    底
group_counts : int
    共有多少组
ngroup : int
    当前绘制的是第几组
gap_between_group : float
    组与组之间的间隔长度
width_one_bar : float
    一个柱子的区域的长度
width_ratio : float
    一个柱子占区域的比例 (可>1, 但不建议)
kwargs : 
    见上文链接

Examples
----------
import utils as ut
pl = ut.pl

df_plot = pd.DataFrame(
    np.reshape(np.repeat(np.arange(5)+1, 4), (4, 5), order='F') *
    np.reshape(rng.random(20), (4, 5)),
    columns=['V{}'.format(i) for i in range(5)],
    index=['I{}'.format(i) for i in range(4)])
cmap = pl.cmap.Customer.get_cmap("5_1",df_plot.columns)
para = dict(gap_between_group=2,width_one_bar=1,
           width_ratio=.8, to_horizontal=False)

display(df_plot)
fig, axs = pl.figure.subplots_get_fig_axs(2, 2,ratio_ncols=1.5,ratio_nrows=3,rc=pl.rc_frame)
# [plot] count ------------------------------------------------------------
ax = axs[0]
for i, (k) in enumerate(df_plot.columns):
    pl.bar(ax, df_plot, key_height=k,cmap=cmap,bottom=0,
           group_counts=df_plot.shape[0],ngroup=i,**para)
pl.bar_add_ticks(ax, df_plot,group_counts=df_plot.shape[0],
                fontdict=dict(fontsize=6),**para)

# [plot] ratio ------------------------------------------------------------
df_plot = df_plot.pipe(pl.bar_count_to_ratio)\
    .pipe(pl.bar_cumsum_df_plot)\
    .pipe(pl.bar_transpose_df_cumsum)
para['to_horizontal'] = True
ax = axs[1]

display(df_plot)
for i, (k) in enumerate(df_plot.filter(regex='^(?!bottom_)').columns):
    pl.bar(ax, df_plot, key_height=k,cmap=cmap,bottom='bottom_{}'.format(k),
           group_counts=0,ngroup=0,**para)
pl.bar_add_ticks(ax, df_plot,group_counts=0, offset=.5,
                fontdict=dict(fontsize=6),**para)

pl.cmap.show(cmap,marker='s',ax=axs[2],text_x=.025)
pl.cmap.show(cmap,marker='s',ax=axs[3],text_x=.025)
"""
    
    x = np.arange(df_plot.shape[0])
    x = (x*group_counts + ngroup) *width_one_bar + x*gap_between_group

    if isinstance(cmap,list):
        kwargs.update(color = cmap[ngroup])
    elif isinstance(cmap,dict):
        kwargs.update(color = cmap[key_height])
    
    if isinstance(bottom,str):
        bottom = df_plot[bottom]
    if to_horizontal:
        ax.barh(x,df_plot[key_height],
                height = width_ratio*width_one_bar,
                left = bottom,
                **kwargs)
    else:
        ax.bar(x,df_plot[key_height],
               width = width_ratio*width_one_bar,
               bottom = bottom,
               **kwargs)


# # boxplot

# In[ ]:


def boxplot_help():
    """
[详见boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html)

box,whisker,cap 分别对应 框--线--|帽子

"""
    data = np.array([1,5,6,7,8,5,4,6,5,4,9])
    
    with plt.rc_context(rc=tl_rc(
        {
            'boxplot.boxprops.linewidth':.5,
            'boxplot.whiskerprops.linewidth':.5,
            'boxplot.capprops.linewidth':.5,
        },
        figure.rc_box
    )):
        fig,ax = figure.subplots_get_fig_axs()
        
        boxplot(data,ax=ax)
        ax.annotate('box',xy=(1.08,6),xytext=(1.25,6),arrowprops=dict(
                headwidth=3,headlength=3,width=.1,color='red'),
                ha='left',va='center')
        ax.annotate('whisker',xy=(1,8),xytext=(1.25,8),arrowprops=dict(
                headwidth=3,headlength=3,width=.1,color='red'),
                ha='left',va='center')
        ax.annotate('cap',xy=(1.04,9),xytext=(1.25,9),arrowprops=dict(
                headwidth=3,headlength=3,width=.1,color='red'),
                ha='left',va='center')
        ax.annotate('fliers',xy=(1.04,1),xytext=(1.25,1),arrowprops=dict(
                headwidth=3,headlength=3,width=.1,color='red'),
                ha='left',va='center')
    return fig


def boxplot(df_plot,ax,positions=None,widths=.5,
            rc_boxplot=None,**kwargs):

    """封装了ax.boxplot

> 详见
+ [ax.boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html)
+ [matplotlibrc BOXPLOT](https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file)

由rc_boxplot实现细节的控制，也可通过**kwargs将参数传给ax.boxplot

rc_boxplot = tl_rc(rc_boxplot,{
    'boxplot.boxprops.linewidth':.5,
    'boxplot.whiskerprops.linewidth':.5,
    'boxplot.capprops.linewidth':.5,
    'boxplot.flierprops.marker':'.',
    'boxplot.flierprops.markersize':4,
    'boxplot.flierprops.markeredgewidth':.25,
    'boxplot.medianprops.color':'red',
    'boxplot.meanprops.color':'deeppink',
})

box,whisker,cap 分别对应 框--线--|帽子

Parameters
----------
positions : array-like | int | None
    每个box的坐标

"""
    if positions is None:
        if len(df_plot.shape) == 2:
            positions = np.arange(df_plot.shape[1]) + 1
        else:
            positions = [1]
    else:
        positions = handle_type_to_list(positions,int)
    rc_boxplot = update_dict({
        'boxplot.boxprops.linewidth':.5,
        'boxplot.whiskerprops.linewidth':.5,
        'boxplot.capprops.linewidth':.5,
        'boxplot.flierprops.marker':'.',
        'boxplot.flierprops.markersize':4,
        'boxplot.flierprops.markeredgewidth':.25,
        'boxplot.medianprops.color':'red',
        'boxplot.meanprops.color':'deeppink',
    },rc_boxplot)
    
    paras = dict(widths=widths,positions=positions)
    paras.update(kwargs)
    
    with plt.rc_context(rc = rc_boxplot):
        ax.boxplot(df_plot,**paras)


# # violinplot

# In[ ]:


def violinplot(df_plot,ax,positions = None,widths = None,vertical = True,
    dict_violin= None,
    dict_box = None,
    plot_scatter_jitter=False,dict_scatter_jitter=None
):
    """

封装了ax.violinplot

ax.violinplot 绘制小提琴

ax.scatter 和 ax.hlines/ax.vlines 绘制中间的 类箱型图

> 详见 violinplot
[ax.violinplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html)

Parameters
----------
positions : None | array-like default: [ 0, 1, 2, ...]

widths : None | array-like default: [ .5, .5, .5 ... ]

dict_violin : dict
    color='#CFCFCF',
    alpha=1,
    linecolor='#c2f9ff',
    linewidth = 0.1,
    
dict_box : dict
    scatter_key_center = 'mean',
    scatter_size = 10,
    scatter_marker = '.',
    scatter_color = '#696969',
    lines_color = '#A9A9A9',
    lines_lw1 = 2,
    lines_lw2 = .5

dict_scatter_jitter : dict
    size = .5,
    alpha=.1,
    color = '#000000'
"""
    
    dict_violin = update_dict(dict(
        color='#CFCFCF',
        alpha=1,
        linewidth = 0.1,
        linecolor='#c2f9ff'),dict_violin)
    dict_box = update_dict(dict(
        scatter_key_center = 'mean',
        scatter_size = 10,
        scatter_marker = '.',
        scatter_color = '#696969',
        lines_color = '#A9A9A9',
        lines_lw1 = 2,
        lines_lw2 = .5),dict_box)
    dict_scatter_jitter = update_dict(dict(
        size = .5,
        alpha=.1,
        color = '#000000'),dict_scatter_jitter)
    
    
    keys = df_plot.columns.to_numpy()
    df_para = df_plot.quantile([0,.25,.5,.75,1])
    df_para.index = 'min,q4_1,median,q4_3,max'.split(',')
    df_para = df_para.transpose().join(df_plot.mean().to_frame('mean'))
    
    df_para['positions']  = np.arange(df_para.shape[0]) + 1 if positions is None else positions
    df_para['widths'] = np.repeat(.5,df_para.shape[0]) if widths is None else widths

    with Block("violin",context=dict_violin) as context:
        parts = ax.violinplot(df_plot,
                              vert=vertical,positions=df_para['positions'].to_numpy(),
                              widths=df_para['widths'],
                              showextrema=False)
    
        for k in 'cmeans,cmins,cmaxes,cbars,cmedians,cquantiles'.split(','):
            if k in parts.keys():
                parts[k].set_lw(0)
        for pc in parts['bodies']:
            pc.set_linewidth(context.linewidth)
            pc.set_color(context.linecolor)
            pc.set_facecolor(context.color)
            pc.set_alpha(context.alpha)

    dict_box.update(zorder=4)
    with Block("box",context=dict_box) as context:
        context.context.update(lines_func = ax.vlines if vertical else ax.hlines)
        for i,row in df_para.iterrows():
            context.lines_func(row['positions'],row['q4_1'],row['q4_3'],
                               color = context.lines_color,
                               lw=context.lines_lw1,
                              zorder=context.zorder)
            context.lines_func(row['positions'],row['max'],row['q4_3'],
                               color = context.lines_color,
                               lw=context.lines_lw2,
                              zorder=context.zorder)
            context.lines_func(row['positions'],row['min'],row['q4_1'],
                               color = context.lines_color,
                               lw=context.lines_lw2,
                              zorder=context.zorder)

        ax.scatter(
            x = 'positions' if vertical else 'mean',
            y = 'mean' if vertical else 'positions',
            marker=context.scatter_marker,
            s=context.scatter_size,c=context.scatter_color,
            zorder=context.zorder,data=df_para)
    if plot_scatter_jitter:
        dict_scatter_jitter.update(zorder = 2)
        with Block("scatter_jitter",context=dict_scatter_jitter) as context:
            df_plot = df_plot.assign(**{
        'x_{}'.format(k) : tl_jitter(df_plot.shape[0],df_para.at[k,'widths']*.65) + df_para.at[k,'positions']
        for k in df_plot.columns})
            for k in keys:
                ax.scatter(x='x_{}'.format(k) if vertical else k,
                           y=k if vertical else 'x_{}'.format(k),
                           s=context.size,c=context.color,alpha=context.alpha,
                           data=df_plot,zorder=context.zorder)


# # bubble and heatmap

# In[ ]:


def bubble(df_size, ax, df_color='pink',fontdict_xtick=None,fontdict_ytick=None,
           nacolor='grey',
           **kwargs):
    """气泡图
在rc_default下，当size = 470 相邻的气泡开始重叠

Examples
----------
from utils.general import *
import utils as ut

# Example 0 --------------------------------------------------
df_size= pd.DataFrame(
    np.reshape(np.repeat(470,16),(4,4)),
   index=['I{}'.format(i) for i in range(4)],
   columns=['C{}'.format(i) for i in range(4)]
)
fig, ax = pl.figure.subplots_get_fig_axs(rc = pl.figure.rc_frame)
pl.bubble(df_size, ax)

# Example 1 --------------------------------------------------
df_size = pd.DataFrame(rng.random((4, 4)) * 470,
                       index=['I{}'.format(i) for i in range(4)],
                       columns=['C{}'.format(i) for i in range(4)])
df_color = pd.DataFrame(rng.random((4, 4)),
                        index=['I{}'.format(i) for i in range(4)],
                        columns=['C{}'.format(i) for i in range(4)])
                        
fig, axs = pl.figure.subplots_get_fig_axs(1,3,rc = pl.figure.rc_frame)
pl.bubble(df_size, axs[0])
pl.bubble(df_size, axs[1], df_color)
pl.bubble(df_size, axs[2], df_color,cmap='bwr')

"""
    fontdict_xtick = update_dict({},fontdict_xtick)
    fontdict_ytick = update_dict({},fontdict_ytick)
    
    df_plot = ut_df.ravel(df_size,'s')
    df_plot['x'] = np.arange(df_size.size) // df_size.shape[1] +1
    df_plot['y'] = df_size.shape[1] - np.arange(df_size.size) % df_size.shape[1]
    if isinstance(df_color, pd.DataFrame):
        assert df_size.shape == df_color.shape, "[Error] shape of df_size and df_color is not the same"
        df_plot = df_plot.join(ut_df.ravel(df_color,'c'))
    else:
        df_plot['c'] = df_color
    df_plot_na = df_plot[df_plot['c'].isna()].copy()
    df_plot_na['c'] = nacolor
    df_plot = df_plot[~df_plot['c'].isna()]
    scatter_return = ax.scatter('x', 'y', s='s', c='c', data=df_plot, **kwargs)
    if df_plot_na.shape[0] > 0:
        kwargs.pop('cmap') if 'cmap' in kwargs.keys() else None
        kwargs.pop('vmax') if 'vmax' in kwargs.keys() else None
        kwargs.pop('vmin') if 'vmin' in kwargs.keys() else None
        ax.scatter('x', 'y', s='s', c='c', data=df_plot_na, **kwargs)
    ax.set_xlim(.5, df_size.shape[0]+.5)
    ax.set_ylim(.5, df_size.shape[1]+.5)
    ax.set_yticks(np.arange(df_size.shape[1])+1, df_size.columns[::-1],**fontdict_ytick)
    ax.set_xticks(np.arange(df_size.shape[0])+1, df_size.index,**fontdict_xtick)

    return scatter_return


def heatmap(ax,df_plot,size_scale = 350,**kvargs):
    """借助bubble实现，marker='s' (正方形 square)
Parameters
----------
size_scale
    df_szie的缩放倍数
    75  占a4p .5 个单位
    350 占a4p  1 个单位
"""
    df_size = df_plot.copy()
    df_size[:] = 1*size_scale
    return bubble(df_size,
          ax,df_color=df_plot,fontdict_xtick=dict(rotation=90),
                     marker='s',**kvargs)


# # scanpy

# In[ ]:


def __assert_all_value_in_cmap(adata,key,cmap):
    return adata.obs[key].value_counts().index.isin(cmap.keys()).all()

def umap(adata,key,ax,ax_cmap=None,cmap=None,marker='.',
         sort=False,kw_scatter=None,key_obsm='X_umap'):
    assert key in adata.obs.columns,'{} is not in adata.obs'.format(key)
    if cmap is None:
            cmap = pl.cmap.get(adata.obs[key].sort_values().unique())
    assert __assert_all_value_in_cmap(adata,key,cmap),"[Error] not all ele of adata.obs[key] in cmap.keys()"
    kw_scatter = update_dict({},kw_scatter)
    
    df_plot = pd.DataFrame(adata.obsm[key_obsm][:,:2],
                 index=adata.obs.index,
                 columns='_obsm1,_obsm2'.split(','))\
            .join(adata.obs.loc[:, [key]]).copy()
    df_plot[key] = df_plot[key].astype(str)
    # scatter
    if sort:
        df_plot = df_plot.sort_values(key,key=lambda s:s.map({k:i for i,k in enumerate(cmap.keys())}))
    else:
        df_plot = df_plot.sample(df_plot.shape[0],replace=False)

    df_plot['_color'] = df_plot[key].map(cmap)
    ax.scatter('_obsm1', '_obsm2',
                marker=marker, c='_color',
                data=df_plot,**kw_scatter)
    
    ax.set_axis_off()
    if ax_cmap:
        cmap.show(cmap,ax=ax_cmap)


# # TACMAN

# In[ ]:


def max_prob(adata,dataset_type='que',factor_size = 200,
    factor_shape = 1,
    fontdict_xtick=dict(rotation=30),
    fontdict_ytick=dict(rotation=0),
    cmap='GnBu',
    order_row=None,order_col=None,
            vmax=1,vmin=0):
    
    kw_cbar = dict(label='median')

    df_size = ut_df.matrix_classify(
        adata.obs.query("type == '{}'".format(dataset_type)),
        'true_label','pre_label'
    )
    order_row = np.sort(df_size.index) if order_row is None else order_row
    order_col = np.sort(df_size.columns) if order_col is None else order_col
    df_size = df_size.loc[order_row,order_col]
    df_size = df_size.div(df_size.sum(axis=1),axis=0).transpose()

    
    df_color = ut_df.matrix_numeric(
        adata.obs.query("type == '{}'".format(dataset_type)),
        'true_label,pre_label'.split(','),'max_prob',func_agg='median'
    ).transpose()
    df_color = df_color.loc[df_size.index,df_size.columns]
    
    shape = np.array(df_size.shape)*factor_shape
    
    plt.close('all')
    pfdb = figure.DrawingBoard(
        shape[0]*2,shape[1]*3
    )
    
    x = 3
    y = 1
    ax = pfdb.add_ax(x,y,*shape,rc=rc_frame)
    cbar = bubble(df_size*factor_size,ax=ax,df_color=df_color,
                     fontdict_xtick=fontdict_xtick,
                     fontdict_ytick=fontdict_ytick,
                      vmax=vmax,vmin=vmin,
                     cmap=cmap,edgecolors='grey',linewidths=.8)
    ax.set_xlabel('predict label')
    ax.set_ylabel('original label')
    x += shape[0]+.5
    
    # legend
    df_legend = pd.DataFrame({'percentage': [.25, .5, .75]})
    df_legend.index = df_legend['percentage'].apply(lambda x: '{:.0f}%'.format(x*100)).to_numpy()
    df_legend = df_legend.transpose()
    shape_legend = np.array(df_legend.shape)*factor_shape
    # shape_legend[1] *= 
    
    ax_legend = pfdb.add_ax(x, y, *shape_legend,
                     rc=update_dict(rc_blank, {
        'ytick.labelright': True, 'xtick.labelbottom': True,
                     }))
    
    bubble(df_legend*factor_size,ax=ax_legend,df_color='black')
    
    y+= shape_legend[1]+.75
    
    
    cbar = pfdb.fig.colorbar(cbar, cax=pfdb.add_ax(x+.5,y,.2, 2,rc_blank),
                             **kw_cbar)
    cbar.ax.tick_params('both', width=.5, length=1.5, color='black')
    return ax


# In[ ]:


def confusion_matrix(adata,dataset_type='que',factor_size = 250,
    factor_shape = 1,
    fontdict_xtick=dict(rotation=30),
    fontdict_ytick=dict(rotation=0),
    cmap='coolwarm',
    order_row=None,order_col=None,
        vmax=1,vmin=0):

    kw_cbar = dict(label='percentage')
    
    df_color = ut_df.matrix_classify(
        adata.obs.query("type == '{}'".format(dataset_type)),
        'true_label','pre_label'
    )
    order_row = np.sort(df_color.index) if order_row is None else order_row
    order_col = np.sort(df_color.columns) if order_col is None else order_col
    df_color = df_color.loc[order_row,order_col]
    
    df_color = df_color.div(df_color.sum(axis=1),axis=0).transpose()
    df_size = pd.DataFrame(np.ones(df_color.shape),index=df_color.index,
                          columns=df_color.columns)
    
    shape = np.array(df_size.shape)*factor_shape
    
    plt.close('all')
    pfdb = figure.DrawingBoard(
        shape[0]*2,shape[1]*3
    )
    
    x = 2
    y = 1
    ax = pfdb.add_ax(x,y,*shape,rc=rc_frame)
    cbar = bubble(df_size*factor_size,ax=ax,df_color=df_color,
                     fontdict_xtick=fontdict_xtick,
                     fontdict_ytick=fontdict_ytick,
                      marker='s',
                     cmap=cmap,vmax=vmax,vmin=vmin)
    ax.set_xlabel('predict label')
    ax.set_ylabel('original label')
    x += shape[0]+.5

    cbar = pfdb.fig.colorbar(cbar, cax=pfdb.add_ax(x+.5,y,.2, 2,rc_blank),
                             **kw_cbar)
    cbar.ax.tick_params('both', width=.5, length=1.5, color='black')
    return ax

