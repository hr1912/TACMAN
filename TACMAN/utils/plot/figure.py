#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""figure

绘图初始化
分图

> variable
>> rc
    rc_mpl
    rc_default
    rc_default
    rc_blank
    rc_frame
    rc_box

> function
    subplots_get_fig_axs

> class
    A4Page
"""


# In[ ]:


__all__ = 'Path,np,pd,mpl,plt,json,Iterable'.split(',') +\
    'subplots_get_fig_axs,rc_blank'.split(',')


# In[ ]:


from ..general import Path, np, pd
from ..general import mpl, plt,json, Iterable
from ..general import update_dict,subset_dict


# In[ ]:


# SourceHanSansCN-Medium 思源黑体 体积极小
# from https://github.com/iizyd/SourceHanSansCN-TTF-Min

for i in 'arial,SourceHanSansCN-Medium'.split(','):
    p_item = Path(__file__).parent.joinpath('font/{}.ttf'.format(i))
    if p_item.exists():
        mpl.font_manager.fontManager.addfont(str(p_item)) 
del i,p_item


# # rc

# In[ ]:


rc_default = {}
# fig
rc_default.update({

    'figure.dpi': 200
    
})
# font
rc_default.update({
    'font.family': [f.name for f in mpl.font_manager.fontManager.ttflist if
 Path(f.fname).match('*/plot/font/*ttf')],
    'font.size': 6  # 磅 (points)
})
# axes
rc_default.update({
    # 'axes.facecolor': 'white',
    'axes.facecolor': '#00000000',  # 透明背景
    'axes.labelsize': rc_default['font.size'],
    'axes.titlesize': rc_default['font.size'] + 2,
    'axes.titleweight': 'bold',
    # axes.edge
    'axes.edgecolor': '#00000000',
    'axes.linewidth': .5,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    # polaraxes 极坐标
    'polaraxes.grid':False,
})
# legend
rc_default.update({
    'legend.title_fontsize': rc_default['font.size']
})
# tick
rc_default.update({
    'xtick.major.size': 0,
    'xtick.major.width': .2,
    'xtick.color': 'black',
    'xtick.major.pad': 1,
    'xtick.labelsize': rc_default['font.size'],

    'ytick.major.size': 0,
    'ytick.major.width': .2,
    'ytick.color': 'black',
    'ytick.major.pad': 1,
    'ytick.labelsize': rc_default['font.size']
})
# patch
rc_default.update({
    'patch.edgecolor': '#00000000',
    'patch.linewidth': .5
})
# legend
rc_default.update({
    'legend.fontsize': rc_default['font.size'],
    'legend.frameon': False
})


# In[ ]:


import seaborn as sns
sns.set_theme(style="ticks", rc=mpl.rcParams)
rc_mpl = mpl.rc_params()
rc_mpl.update(
    subset_dict(rc_default,'figure.dpi,axes.facecolor,font.family,axes.edgecolor'.split(','))
)
#,font.size,axes.labelsize,axes.titlesize,axes.titleweight

rc_blank = update_dict(rc_default,{
    'xtick.top': False,
    'xtick.bottom': False,
    'xtick.labeltop': False,
    'xtick.labelbottom': False,

    'ytick.left': False,
    'ytick.right': False,
    'ytick.labelleft': False,
    'ytick.labelright': False,
})

rc_frame = update_dict(rc_default,{
    'axes.edgecolor': 'black',
    'axes.linewidth': .5,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    'xtick.major.size': 2,
    'ytick.major.size': 2
})

rc_frame_lt = update_dict(rc_frame,{
    'xtick.bottom': False,'xtick.top': True,
    'xtick.labelbottom': False,'xtick.labeltop': True,
    'axes.spines.bottom': False,'axes.spines.top': True})

rc_box = update_dict(rc_frame,{
    "axes.spines.right": True,
    "axes.spines.top": True,
})

rc_tl_notick = {'xtick.top': False,
                'xtick.bottom': False,
                'xtick.labeltop': False,
                'xtick.labelbottom': False,
                'ytick.left': False,
                'ytick.right': False,
                'ytick.labelleft': False,
                'ytick.labelright': False}

rc_tl_off_x={
    'xtick.top': False,
    'xtick.bottom': False,
    'xtick.labeltop': False,
    'xtick.labelbottom': False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
}
rc_tl_off_y={
    'ytick.left': False,
    'ytick.right': False,
    'ytick.labelleft': False,
    'ytick.labelright': False,
    "axes.spines.left": False,
    "axes.spines.right": False,
}

rc_tl_xt={
    'xtick.top': True,
    'xtick.labeltop': True,
    "axes.spines.top": True
}

rc_tl_yr={
    'ytick.right': True,
    'ytick.labelright': True,
    "axes.spines.right": True,
}


# In[ ]:


# sns.set_theme(style="ticks", rc=mpl.rcParams)
mpl.rcParams.update(rc_mpl)


# # tl

# In[ ]:


def tl_rc(update=rc_default,*args,**kwargs):
    """ 更新rc，覆盖顺序 kvargs > args ( dict ) > update
"""
    update= update.copy()
    [update.update(i) for i in args if isinstance(i,dict) ]
    update.update(kwargs)
    return update

def tl_fontdict(alignment='lc',rotation=0,color='black',**kwargs):
    """生成fontdict
将最为常用的verticalalignment,horizontalalignment,rotaiton,color进行封装

详见[Text](https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text)

Parameters
----------
alignment : str
    assert len(alignment) == 2 or len(alignment) == 3
    "ha va" or "ha va ma"
    
    ma 为multialignment, 取值范围为 lrc
    
    l -> left
    r -> right
    c -> center
    t -> top
    b -> bottom
"""
    def alignment_convert(c):
        return {k:v for k,v in zip(list('lrctb'),'left,right,center,top,bottom'.split(','))}[c]
    assert len(alignment) == 2 or len(alignment) == 3
    if len(alignment) == 3:
        kwargs.update(ma = alignment_convert(alignment[2]))

    kwargs.update(ha=alignment_convert(alignment[0]),
                  va=alignment_convert(alignment[1]),
                  rotation=rotation,color=color)
    return kwargs

def tl_savefig(fig, fig_name, p_plot=None,transparent=True):
    if p_plot is None:
        fig_name = Path(fig_name)
        p_plot = fig_name.parent
        fig_name = fig_name.name
    else:    
        p_plot = Path(p_plot)
    
    fig.savefig(
        p_plot.joinpath(fig_name),
        transparent=transparent,
        dpi=200,
        bbox_inches='tight')
    print('[out][plot] {} \n\tin {}'.format(fig_name, p_plot))


# In[ ]:


def subplots_get_fig_axs(nrows=1, ncols=1,
                         ratio_nrows=2, ratio_ncols=2,
                         width_ratios=None, height_ratios=None,
                         rc=None,
                         kw_subplot=None, kw_gridspec=None,
                         kw_fig=None, ravel=True, kw_ravel=None):
    rc = {} if rc is None else rc
    kw_fig = {} if kw_fig is None else kw_fig
    kw_ravel = {} if kw_ravel is None else kw_ravel
    
    with plt.rc_context(rc=rc):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                gridspec_kw=kw_gridspec, subplot_kw=kw_subplot,
                                figsize=(ratio_nrows*ncols, ratio_ncols*nrows), **kw_fig)

    if ravel:
        axs = np.ravel(axs, **kw_ravel)
    if axs.size == 1:
        axs = axs[0]

    return fig, axs


# # DrawingBoard 和 A4Page
# 
# 
# > DrawingBoard
# 
# 以0.25 为单位长度的画板
# 
# 可在其上任意位置添加ax
# 
# > A4Page 继承 DrawingBoard
# 
# ```python
# # A4大小的figure,使用网格控制子图
# display([8.27*.97, 11.96*.96])
# display([8.27*.97/.25, 11.96*.96/.25])
# # 以0.25为单位1，将width分为32等分，higth分为46等分
# fig = plt.figure(figsize=(8.27, 11.69))
# spec = fig.add_gridspec(nrows=46, ncols=32,
#                     left=0.03, right=1,  # 设置边距
#                     bottom=0.02, top=0.98,  # 设置边距
#                     wspace=0, hspace=0)  # 设置子图间距
# ```

# In[ ]:


class DrawingBoard:
    """
以0.25 为单位长度的画板
可在其上任意位置添加ax

area

    """
    __area = {}
    
    def __init__(self,nrows =1,ncols=1,margin=dict(left=0,right=1,bottom=0,top=1)):
        """
Parameters
----------
nrows,ncols : int
    控制页面的宽度和高度
margin : dict
    控制页边距
"""        
        self.nrows = nrows
        self.ncols = ncols
        self.margin = margin
        
        self.unit_width = (
            self.margin['right']-self.margin['left'])/self.ncols
        self.unit_higth = (
            self.margin['top']-self.margin['bottom'])/self.nrows
    
        self.fig = plt.figure(figsize=(ncols*.25,nrows*.25))
        self.spec = self.fig.add_gridspec(nrows=self.nrows, ncols=self.ncols,
                                          wspace=0, hspace=0,  # 设置子图间距
                                          **margin)
        

    # 对类及其实例未定义的属性有效
    # 若name 不存在于 self.__dir__中,则调用__getattr__
    def __getattr__(self, name):
        return self.__area.setdefault(name, None)

    def __length_to_absolute_coordinate(self, width, height):
        return (width*self.unit_width, height*self.unit_higth)

    def __point_to_absolute_coordinate(self, x, y):
        w, h = self.__length_to_absolute_coordinate(x, y)
        return (self.margin['left']+w, self.margin['top']-h)

    def __subset_spec(self, x, y, width=1, height=1):
        return self.spec[y:y+height, x:x+width]

    def point_and_length_to_absolute_coordinate(self,x,y,width=1,height=1):
        """"""
        x, y = self.__point_to_absolute_coordinate(x, y)
        w, h = self.__length_to_absolute_coordinate(width, height)
        return [x,y,w,h]
        
    # [add_grid]--------------------------------------------------

    def add_grid(self, step=2, alpha=0.15):
        """添加网格
        """
        # |
        # |
        # |
        kw_line = dict(linestyle='--', alpha=alpha, c='#00BFFF')
        fontdict = dict(fontsize=10, color='blue',
                        alpha=alpha, ha='center', va='center')
        for i in np.arange(self.ncols+1):
            v, _ = self.__point_to_absolute_coordinate(i, 0)
            self.fig.add_artist(mpl.lines.Line2D([v, v],
                                                 [self.margin['bottom'],
                                                  self.margin['top']],
                                                 **kw_line))
            if i % step == 0 or i % (self.ncols // 2) == 0:
                for y in np.linspace(.15, .85, 3):
                    self.fig.text(
                        v, y, 'x={:.0f}'.format(i) if i %
                        (self.ncols // 2) == 0 else '{:.0f}'.format(i), fontdict=fontdict)
        # ------
        kw_line.update(dict(c='#D02090')),
        fontdict.update(dict(color='red'))
        for i in np.arange(self.nrows+1):
            _, v = self.__point_to_absolute_coordinate(0, i)
            self.fig.add_artist(mpl.lines.Line2D([self.margin['left'],
                                                  self.margin['right']],
                                                 [v, v], **kw_line))
            if i % step == 0 or i % (self.nrows // 2) == 0:
                for x in np.linspace(.15, .85, 3):
                    self.fig.text(
                        x, v, 'y={:.0f}'.format(i) if i %
                        (self.nrows // 2) == 0 else '{:.0f}'.format(i), fontdict=fontdict)

    def add_grid_absolute_coordinate(self, alpha=0.15):
        """标出self.fig的绝对坐标系
    """
        for i, (v) in enumerate(np.linspace(0, 1, 10+1)):
            # |
            self.fig.add_artist(mpl.lines.Line2D(
                [v, v], [0, 1], linestyle='--', alpha=alpha, c='#2F4F4F'))
            for y in [0.1, 0.5, 0.9]:
                self.fig.text(
                    v+0.01,
                    y,
                    'x=.{:.0f}'.format(
                        i) if v == 0 or v == .5 else '.{:.0f}'.format(i),
                    fontdict={
                        'fontsize': 10,
                        'color': 'black',
                        'alpha': alpha})
        fontdict = {'fontsize': 10, 'color': 'blue', 'alpha': alpha}

        for i, (v) in enumerate(np.linspace(0, 1, 10 + 1)):
            # ------
            self.fig.add_artist(mpl.lines.Line2D(
                [0, 1], [v, v], linestyle='--', alpha=alpha, c='#FFA500'))
            for x in [0.1, 0.5, 0.9]:
                self.fig.text(
                    x,
                    v+0.01,
                    'y=.{:.0f}'.format(
                        i) if v == 0 or v == .5 else '.{:.0f}'.format(i),
                    fontdict={
                        'fontsize': 10,
                        'color': 'orange',
                        'alpha': alpha})

    def add_ax(self, x, y, width=1, height=1, rc=None, *args, **kwargs):
        """添加ax, 其中x,y,width,height均可为小数
        """
        y = y + height
        x,y,w,h = self.point_and_length_to_absolute_coordinate(x,y,width,height)
        with plt.rc_context(rc=rc):
            ax = self.fig.add_axes((x, y, w, h), *args, **kwargs)
        return ax

    def add_text_with_ax(self, ax, text, x=0, y=1, fontdict=None):
        """
Parameters
----------
fontdict : dict | None
    dict(
        fontsize = 14,
        fontweight = 'bold'
    )
"""
        fontdict = update_dict(dict(fontsize = 14,fontweight = 'bold'),fontdict)
        ax.text(x, y, text, fontdict=fontdict)
        ax.set_axis_off()

    # [area]--------------------------------------------------
    def area_update(self, x, y, nrows, ncols, width, height,
                    gap_width=0, gap_height=0):
        """area为块状
a4p.area_get_ax可以通过area获取ax
因其使用了add_ax,
故x,y,width,height,gap_width,gap_height均可以为小数

"""
        self.__area.update(
            area_x=x, area_y=y,
            area_nrows=nrows,
            area_ncols=ncols,
            area_width=width,
            area_height=height,
            area_gap_width=gap_width,
            area_gap_height=gap_height)

    def area_get_ax(self, index, order='C', with_xy  = False, *args, **kwargs):
        """"""
        def get_index_index_x_index_y(index, nrows, ncols,order):
            sep = {'C':ncols,'F':nrows}.setdefault(order,ncols)
            return index//sep, index % sep

        assert index < self.area_nrows*self.area_ncols
        index_x, index_y = get_index_index_x_index_y(
            index, self.area_nrows, self.area_ncols,order=order)
        if order == 'C':
            index_x, index_y = index_y,index_x
        x = self.area_x + (self.area_width+self.area_gap_width)*index_x
        y = self.area_y + (self.area_height+self.area_gap_height)*index_y
        ax = self.add_ax(x, y, self.area_width, self.area_height,*args, **kwargs)
        if with_xy:
            return x,y,ax 
        return ax

    def area_yield_ax(self, order='C', with_xy  = False,*args, **kwargs):
        """
Parameters
----------
with_xy : bool | default = Flase
    if with_xy:
        yield x,y,ax
    else:
        yield ax
args,kwargs : list,dict
    transfer to add_ax
"""
        for i in np.arange(self.area_nrows*self.area_ncols):
            yield self.area_get_ax(i, order,with_xy = with_xy,*args, **kwargs)

    def area_show(self, order='C'):
        """"""
        for i in np.arange(self.area_nrows*self.area_ncols):
            ax = self.area_get_ax(i, order,rc=tl_rc(rc_box,rc_tl_notick))
            ax.text(
                np.mean(
                    ax.get_xlim()), np.mean(
                    ax.get_ylim()), "axs[{}]".format(i), fontdict=dict(
                    ha='center', va='center'))

    # [save]--------------------------------------------------
    def save_as_pdf(self, p, close=True, **kwargs):
        """"""
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(p) as pdf:
            pdf.savefig(self.fig, **kwargs)
            plt.close('all') if close else None

    def add_transparent_box(self,
                            x,
                            y,
                            width=1,
                            height=1,
                            title='',
                            fontdict={'color': 'blue',
                                      'loc': 'right',
                                      'fontsize': 6,
                                      'ha': 'left',
                                      'va': 'top',
                                      'rotation': -90},
                            rc={'axes.edgecolor': 'blue'}):
        """"""

        ax = self.add_ax(x, y, width, height, rc=tl_rc(rc_box,rc_tl_notick,rc))
        ax.set_title(title, y=0.9, **fontdict) if title else None
        return ax

class A4Page(DrawingBoard):
    """
A4大小的figure,使用网格控制ax
display([8.27*.97, 11.96*.96])
display([8.27*.97/.25, 11.96*.96/.25])
# 以0.25为单位1，将width分为32等分，higth分为46等分
# 余下的部分充当页边距, 计算其占比
#     margin_width : (8.27-0.25*32)/8.27    = 0.27/8.27  = 0.032
#     margin_height: (11.69-0.25*46)/11.69) = 0.19/11.69 = 0.016

margin = dict(
    left=0.032/2,
    right=1-0.032/2,
    bottom=0.016/2,
     top=1-0.016/2) #居中
# margin = dict(left=0.03, right=1,bottom=0.02, top=0.96) # 曾用

fig = plt.figure(figsize=(8.27, 11.69))
spec = fig.add_gridspec(nrows=46, ncols=32,
                    wspace=0, hspace=0,# 设置子图间距
                    **margin)

# 注意1：A4Page对象函数的absolute_coordinate，以右下角为(0,0),左上角为(1,1),
#     为matplotlib的绝对坐标系
#     而非页面长度的绝对坐标系
# 注意2：而所使用的坐标系则如add_grid所示，以右上角为(0,0),左上角为(nrows,ncols)
# 注意3：unit_width,unit_higth 为1个单位占figsize(width,hight)的比例, 并非实际长度

    """

    def __init__(
        self,
        nrows = 46,
        margin=dict(
            left=0.032/2,
            right=1-0.032/2,
            bottom=0.016/2,
            top=1-0.016/2)):
        """
Parameters
----------
nrows : int
    控制页面的长度，(0,46]
margin : dict
    控制页边距
"""

        assert nrows >0 and nrows <= 46,"[Error] nrows need to be (0,46] , get {}".format(nrows)
        
        if nrows == 46:
            figsize = (8.27, 11.69)
        else:
            figsize = (8.27, 11.69*(1-margin['top']) +\
                   11.69*(margin['top']-margin['bottom'])/46 * nrows)
            margin = margin.copy()
            margin['bottom'] = 0
            margin['top'] = 1 -  11.69*(1-margin['top'])/figsize[1]
            
        
        self.margin = margin
        self.nrows = nrows
        self.ncols = 32
        self.unit_width = (
            self.margin['right']-self.margin['left'])/self.ncols
        self.unit_higth = (
            self.margin['top']-self.margin['bottom'])/self.nrows
    
        self.fig = plt.figure(figsize=figsize)
        self.spec = self.fig.add_gridspec(nrows=self.nrows, ncols=self.ncols,
                                          wspace=0, hspace=0,  # 设置子图间距
                                          **margin)

