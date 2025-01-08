#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .figure import *
from .. import df as ut_df
from ..general import str_step_insert, show_str_arr,update_dict


# In[ ]:


def show(color_map,marker='.', size=40, text_x=0.1, kw_scatter=None,
         fontdict=None,axis_off=True,ax=None, return_ax=False):
    """
Parameters
----------
text_x : float
    控制text的横坐标
    marker 的横坐标为0
    ax的横坐标被锁定为(-0.05, 0.25)
    """
    if ax:
        fig = ax.figure
    else:
        fig, ax = subplots_get_fig_axs()
    if isinstance(size, (int, float)) or (not isinstance(size, Iterable)):
        size = np.repeat(size, len(color_map.keys()))
    if isinstance(marker, str) or (not isinstance(marker, Iterable)):
        marker = np.repeat(marker, len(color_map.keys()))

    fontdict = update_dict(dict(ha='left',va='center'),fontdict)
    kw_scatter = update_dict({},kw_scatter)
    
    for i, ((k, v), m, s) in enumerate(
            zip(color_map.items(), marker, size)):
        ax.scatter(0, len(color_map.keys())-i,
                   label=k, c=v, s=s, marker=m,**kw_scatter)
        ax.text(text_x, len(color_map.keys())-i, k, fontdict=fontdict)
    ax.set_xlim(-0.05, 0.25)
    ax.set_ymargin(.5)
    ax.set_axis_off() if axis_off else None

    return ax if return_ax else fig
def get_from_list(colors, is_categories=True, name='', **kvarg):
    """由颜色列表生成cmap
Examples
----------
colors = 'darkorange,#279e68,gold,#d62728,lawngreen,#aa40fc,lightseagreen,#8c564b'.split(',')
display(get_from_list(colors,True))
display(get_from_list(colors,False))
"""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    res = None
    if is_categories:
        res = ListedColormap(colors, name, **kvarg)
    else:
        res = LinearSegmentedColormap.from_list(name, colors, **kvarg)
    return res


# # matplotlib_qualitative_colormaps

# In[ ]:


class Qcmap:
    """matplotlib_qualitative_colormaps
>[详见](https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative)

> function
    get_colors
    get_cmap
    show
"""
    item_size_max = 20
    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/matplotlib_qualitative_colormaps.csv'),index_col=0)
    def get_colors(self,name):
        colors = np.array(self.df.at[name,'colors'].split(','))
        return colors[colors != 'white']
    
    def get_cmap(self,name,keys):
        colors = self.get_colors(name)
        if len(keys) > len(colors):
            print('[Warning][Qcmap][get_cmap] length of keys is greater than colors')
        return {k:v for k,v in zip(keys,colors)}

    # def show(self):
    #     data = self.df['colors'].str.extract(','.join(['(?P<c{}>[^,]+)'.format(i) for i in range(self.item_size_max)]))
    #     show_cmap_df_with_js(data)


# In[ ]:


class Customer(Qcmap):
    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/customer.csv'),index_col=0)
        self.item_size_max = self.df.index.str.extract("^(\\d+)_",expand=False).astype(int).max()
    def append_colors(self,colors):
        assert isinstance(colors,str) and ',' in colors,"[cmap][Customer][append_colors] colors must be a string colors separated by ,"
        
        if self.df['colors'].str.replace(',white','').isin([colors]).any():
            print("[cmap][Customer][append_colors] colors is exists")
            return
        info = self.df.index.str.extract("(?P<length>\\d+)_(?P<id>\\d+)").astype(int)['length'].value_counts()
        colors_length = len(colors.split(','))
        colors_id = info.at[colors_length] if colors_length in info.index else 0
        colors_id = '{}_{}'.format(colors_length,colors_id)
        self.df = pd.concat([self.df,pd.DataFrame({'colors':[colors]},index=[colors_id])])
        if colors_length > self.item_size_max:
            self.item_size_max = colors_length
            self.df['colors'] = self.df['colors'].str.replace(',white','')
            self.df = self.df.join(self.df.index.to_frame(name='index')['index']\
                .str.extract("^(?P<length>\\d+)_(?P<id>\\d+)").astype(int))
            for i,row in self.df.iterrows():
                self.df.at[i,'colors'] = self.df.at[i,'colors'] + (',white' * (self.df['length'].max() - row['length']))
            self.df = self.df.sort_values('length,id'.split(',')).loc[:,['colors']]
        else:
            self.df.at[colors_id,'colors'] = self.df.at[colors_id,'colors'] + (',white'*(self.item_size_max - colors_length ) )

    def save(self):
        self.df = self.df.join(self.df.index.to_frame(name='index')['index']\
                .str.extract("^(?P<length>\\d+)_(?P<id>\\d+)").astype(int))
        self.df = self.df.sort_values('length,id'.split(',')).loc[:,['colors']]
        self.df.to_csv(Path(__file__).parent
            .joinpath('color/customer.csv'),index=True)
        print("[cmap][Customer][append_colors][out] customer.csv\n in {}".format(
            Path(__file__).parent.joinpath('color')))

class ggsci(Qcmap):
    def __init__(self):
        self.df = pd.read_csv(Path(__file__).parent
                              .joinpath('color/ggsci.csv'),index_col=0)
        self.item_size_max = self.df['length'].max()


# In[ ]:


ggsci = ggsci()

COLOR_DF=ggsci.df

def get_colors(name):
    return ggsci.get_colors(name)
def get(name,keys):
    return ggsci.get_cmap(name,keys)

