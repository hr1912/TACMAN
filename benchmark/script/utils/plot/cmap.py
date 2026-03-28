#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""cmap
颜色
> variable
    ColorLisa
> function
    get
    show
    get_from_list
"""


# In[ ]:


from .figure import *
from .. import df as ut_df
from ..general import str_step_insert, show_str_arr, update_dict


# In[ ]:


palettes = json.loads(
    Path(__file__).parent.joinpath("color/scanpy.plotting.palettes.json").read_text()
)
palettes.update(
    {k: v.split(",") for k, v in palettes.items() if k.startswith("default")}
)


# In[ ]:


def get_color(count):
    palette = None
    if count <= 20:
        palette = palettes["default_20"]
    elif count <= 28:
        palette = palettes["default_28"]
    elif count <= len(palettes["default_102"]):  # 103 colors
        palette = palettes["default_102"]
    else:
        raise Exception("[categories too long] {}".format(count))
    return palette[:count]


def get(serise, color_missing_value="lightgray", offset=2, filter_offset=True):

    serise = pd.Series(pd.Series(serise).unique())
    has_missing_value = serise.isna().any()
    serise = pd.Series(
        np.concatenate(
            (["_{}".format(i) for i in range(offset)], serise.dropna().astype(str))
        )
    )

    palette = get_color(serise.size)

    colormap = {k: v for k, v in zip(serise, palette)}
    if has_missing_value:
        colormap.update({"nan": color_missing_value})

    if filter_offset:
        colormap = {
            k: v
            for _, (k, v) in zip(
                ~pd.Series(colormap.keys()).str.match("_\\d+"), colormap.items()
            )
            if _
        }
    return colormap


def show(
    color_map,
    marker=".",
    size=40,
    text_x=0.1,
    kw_scatter=None,
    fontdict=None,
    axis_off=True,
    ax=None,
    return_ax=False,
):
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

    fontdict = update_dict(dict(ha="left", va="center"), fontdict)
    kw_scatter = update_dict({}, kw_scatter)

    for i, ((k, v), m, s) in enumerate(zip(color_map.items(), marker, size)):
        ax.scatter(
            0, len(color_map.keys()) - i, label=k, c=v, s=s, marker=m, **kw_scatter
        )
        ax.text(text_x, len(color_map.keys()) - i, k, fontdict=fontdict)
    ax.set_xlim(-0.05, 0.25)
    ax.set_ymargin(0.5)
    ax.set_axis_off() if axis_off else None

    return ax if return_ax else fig


def show_cmap_df_with_js(df):
    from IPython.display import display, display_javascript

    display(
        df.style.set_table_styles(
            [
                {
                    "selector": ".ColorLisa-item",
                    "props": "color:white;background-color:grey;",
                }
            ]
        ).set_td_classes(
            pd.DataFrame(
                np.full(df.shape, "ColorLisa-item"), index=df.index, columns=df.columns
            )
        )
    )
    display_javascript(
        """
        $.each($(".ColorLisa-item"),function(i,ele){
        $(ele).css("background-color",$(ele).text())
    })
        """,
        raw=True,
    )


# In[ ]:


def get_from_list(colors, is_categories=True, name="", **kvarg):
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


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def func_rgba_to_hex(rgba):
#     if len(rgba) == 4:
#         r, g, b, a = rgba
#     elif len(rgba) == 3:
#         r, g, b = rgba
#     r_hex = hex(int(r * 255))[2:].zfill(2)
#     g_hex = hex(int(g * 255))[2:].zfill(2)
#     b_hex = hex(int(b * 255))[2:].zfill(2)
#     return "#{}{}{}".format(r_hex, g_hex, b_hex)

# df_color = pd.DataFrame({
#     'name':'Pastel1,Pastel2,Paired,Accent,Dark2,Set1,Set2,Set3,tab10,tab20,tab20b,tab20c'.split(',')
# })
# df_color['colors'] = df_color['name'].apply(
#     lambda x:','.join(pd.Series(plt.colormaps[x].colors).apply(func_rgba_to_hex))
# )
# df_color['length'] = df_color['colors'].apply(lambda x:len(x.split(',')))
# for i,row in df_color.iterrows():
#     df_color.at[i,'colors'] = df_color.at[i,'colors'] + ',white'*(df_color['length'].max() - row['length'])

# df_color.index = df_color['name'].to_numpy()
# df_color = df_color.drop(columns='name,length'.split(','))
# display(df_color)
# df_color.to_csv(Path('~/link/utils/plot/color/').joinpath('matplotlib_qualitative_colormaps.csv'),index=True)


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
        self.df = pd.read_csv(
            Path(__file__).parent.joinpath(
                "color/matplotlib_qualitative_colormaps.csv"
            ),
            index_col=0,
        )

    def get_colors(self, name):
        colors = np.array(self.df.at[name, "colors"].split(","))
        return colors[colors != "white"]

    def get_cmap(self, name, keys):
        colors = self.get_colors(name)
        if len(keys) > len(colors):
            print("[Warning][Qcmap][get_cmap] length of keys is greater than colors")
        return {k: v for k, v in zip(keys, colors)}

    def show(self):
        data = self.df["colors"].str.extract(
            ",".join(["(?P<c{}>[^,]+)".format(i) for i in range(self.item_size_max)])
        )
        show_cmap_df_with_js(data)


# # ColorLisa

# In[ ]:


class ColorLisa:
    """从ColorLisa获取的颜色,共110项, 每项中有5种颜色
    ColorLisa取材于艺术家的作品

    详见[ColorLisa](https://colorlisa.com/)

    > function
        get_colors
        get_cmap
        show
        show_author
        show_all
        show_all_as_df
    """

    common_color_n_cols = 5

    def __init__(self):
        self.df = (
            pd.read_csv(Path(__file__).parent.joinpath("color/ColorLisa.csv"))
            .sort_values("author,id".split(","))
            .reset_index(drop=True)
        )
        self.df.index = self.df.apply(
            lambda row: "{author}_{id}".format(**row), axis=1
        ).to_numpy()

    def __get_item(self, author, id=0):
        item = None
        if author in self.df.index:
            item = self.df.loc[author, :]
        if "{}_{}".format(author, id) in self.df.index:
            item = self.df.loc["{}_{}".format(author, id), :]
        assert not item is None, "[Error] can not get item with author={} id={}".format(
            author, id
        )
        return item

    def __show_df_cmap(self, df=None, ncols=4, flexible_ncols=True):
        if df is None:
            df = self.df
        if flexible_ncols:
            ncols = min(ncols, df.shape[0])
        nrows = df.shape[0] // ncols + (0 if df.shape[0] % ncols == 0 else 1)

        with plt.rc_context(rc=rc_blank):
            fig, axs = subplots_get_fig_axs(
                nrows, ncols, ratio_nrows=0.7, ratio_ncols=1.4
            )
            axs = [axs] if nrows * ncols == 1 else axs

        for ax, (i, row) in zip(axs, df.iterrows()):
            show({i: i for i in row["color"].split(",")}, ax=ax, text_x=0.05)
            ax.set_title(
                str_step_insert(i, 10), fontdict=dict(fontsize=6, ha="center", va="top")
            )
        return fig

    def help(self):
        print(self.__doc__)

    def get_colors(self, author, id=0):
        return np.array(self.__get_item(author, id)["color"].split(","))

    def get_cmap(self, author, id=0, keys=None):
        colors = self.get_colors(author, id)
        keys = [str(i) for i in range(len(colors))] if keys is None else keys

        if len(keys) > len(colors):
            print(
                "[Warning][ColorLisa][get_cmap] length of keys is greater than colors"
            )
        return {k: v for k, v in zip(keys, colors)}

    def show(self):
        ut_df.show(self.df)

    def show_author(self):
        show_str_arr(self.df["author"].unique())

    def show_all(self):
        display(self.__show_df_cmap(ncols=6))

    def show_all_as_df(self):
        show_cmap_df_with_js(
            self.df["color"].str.extract(
                ",".join(["(?P<c{}>[^,]+)".format(i) for i in range(5)])
            )
        )


# # customer

# In[ ]:


# df =  pd.DataFrame({'colors' : ['#40DAFF,#FF5c5c', '#6262FF,#FF6060',
# '#D59B3A,#3D4A78', '#1A908C,#D17133',
# '#387DB8,#E11A1D', '#179B73,#D48AAF',
# '#FFDD14,#AC592A', '#C381A8,#407BAE',
# '#FB8D62,#8DA0CD,#66C2A5',
# '#2DABB2,#DAAB36,#F0552B',
# '#CCD6BC,#EBC4B8,#CACDE8',
# '#F5AD65,#91CCAE,#795291,#F6C6D6',
# '#DB80AE,#8C96B8,#EC8360,#54B097',
# '#A5D3ED,#ED949A,#EEC48A,#B5AAD5,#5382BA',
# '#6194C9,#FE8D00,#0E5FDB,#970030,#681A98',
# '#E64B35,#4DBBD5,#00A087,#3C5488,#F39B7F']})
# df['length'] = df['colors'].apply(lambda x:len(x.split(',')))
# df['id'] = df.groupby('length').cumcount()
# df.index = df.apply(lambda row:"{length}_{id}".format(**row),axis=1).to_numpy()
# for i,row in df.iterrows():
#     df.at[i,'colors'] = df.at[i,'colors'] + (',white' * (df['length'].max() - row['length']))
# df = df.loc[:,['colors']]
# display(df)
# df.to_csv('/public/workspace/licanchengup/link/utils/plot/color/customer.csv',index=True)


# In[ ]:


class Customer(Qcmap):
    def __init__(self):
        self.df = pd.read_csv(
            Path(__file__).parent.joinpath("color/customer.csv"), index_col=0
        )
        self.item_size_max = (
            self.df.index.str.extract("^(\\d+)_", expand=False).astype(int).max()
        )

    def append_colors(self, colors):
        assert (
            isinstance(colors, str) and "," in colors
        ), "[cmap][Customer][append_colors] colors must be a string colors separated by ,"

        if self.df["colors"].str.replace(",white", "").isin([colors]).any():
            print("[cmap][Customer][append_colors] colors is exists")
            return
        info = (
            self.df.index.str.extract("(?P<length>\\d+)_(?P<id>\\d+)")
            .astype(int)["length"]
            .value_counts()
        )
        colors_length = len(colors.split(","))
        colors_id = info.at[colors_length] if colors_length in info.index else 0
        colors_id = "{}_{}".format(colors_length, colors_id)
        self.df = pd.concat(
            [self.df, pd.DataFrame({"colors": [colors]}, index=[colors_id])]
        )
        if colors_length > self.item_size_max:
            self.item_size_max = colors_length
            self.df["colors"] = self.df["colors"].str.replace(",white", "")
            self.df = self.df.join(
                self.df.index.to_frame(name="index")["index"]
                .str.extract("^(?P<length>\\d+)_(?P<id>\\d+)")
                .astype(int)
            )
            for i, row in self.df.iterrows():
                self.df.at[i, "colors"] = self.df.at[i, "colors"] + (
                    ",white" * (self.df["length"].max() - row["length"])
                )
            self.df = self.df.sort_values("length,id".split(",")).loc[:, ["colors"]]
        else:
            self.df.at[colors_id, "colors"] = self.df.at[colors_id, "colors"] + (
                ",white" * (self.item_size_max - colors_length)
            )

    def save(self):
        self.df = self.df.join(
            self.df.index.to_frame(name="index")["index"]
            .str.extract("^(?P<length>\\d+)_(?P<id>\\d+)")
            .astype(int)
        )
        self.df = self.df.sort_values("length,id".split(",")).loc[:, ["colors"]]
        self.df.to_csv(Path(__file__).parent.joinpath("color/customer.csv"), index=True)
        print(
            "[cmap][Customer][append_colors][out] customer.csv\n in {}".format(
                Path(__file__).parent.joinpath("color")
            )
        )


# # ggsci

# > r save all color as a table file
# ```r
# library(tidyverse)
# library(ggsci)
# 'name,palette_type,colors' %>% write('ggsci_color',append = F)
# sprintf("NPG,nrc,%s",str_c(str_replace_na(pal_npg('nrc')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("AAAS,default,%s",str_c(str_replace_na(pal_aaas('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("NEJM,default,%s",str_c(str_replace_na(pal_nejm('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Lancet,lanonc,%s",str_c(str_replace_na(pal_lancet('lanonc')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("JAMA,default,%s",str_c(str_replace_na(pal_jama('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("BMJ,default,%s",str_c(str_replace_na(pal_bmj('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("JCO,default,%s",str_c(str_replace_na(pal_jco('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("UCSCGB,default,%s",str_c(str_replace_na(pal_ucscgb('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("D3,category10,%s",str_c(str_replace_na(pal_d3('category10')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("D3,category20,%s",str_c(str_replace_na(pal_d3('category20')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("D3,category20b,%s",str_c(str_replace_na(pal_d3('category20b')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("D3,category20c,%s",str_c(str_replace_na(pal_d3('category20c')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Observable,observable10,%s",str_c(str_replace_na(pal_observable('observable10')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("LocusZoom,default,%s",str_c(str_replace_na(pal_locuszoom('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("IGV,default,%s",str_c(str_replace_na(pal_igv('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("IGV,alternating,%s",str_c(str_replace_na(pal_igv('alternating')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("COSMIC,hallmarks_light,%s",str_c(str_replace_na(pal_cosmic('hallmarks_light')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("COSMIC,hallmarks_dark,%s",str_c(str_replace_na(pal_cosmic('hallmarks_dark')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("COSMIC,signature_substitutions,%s",str_c(str_replace_na(pal_cosmic('signature_substitutions')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("UChicago,default,%s",str_c(str_replace_na(pal_uchicago('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("UChicago,light,%s",str_c(str_replace_na(pal_uchicago('light')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("UChicago,dark,%s",str_c(str_replace_na(pal_uchicago('dark')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Star Trek,uniform,%s",str_c(str_replace_na(pal_startrek('uniform')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tron Legacy,legacy,%s",str_c(str_replace_na(pal_tron('legacy')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Futurama,planetexpress,%s",str_c(str_replace_na(pal_futurama('planetexpress')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Rick and Morty,schwifty,%s",str_c(str_replace_na(pal_rickandmorty('schwifty')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("The Simpsons,springfield,%s",str_c(str_replace_na(pal_simpsons('springfield')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Flat UI,default,%s",str_c(str_replace_na(pal_flatui('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Flat UI,flattastic,%s",str_c(str_replace_na(pal_flatui('flattastic')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Flat UI,aussie,%s",str_c(str_replace_na(pal_flatui('aussie')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Frontiers,default,%s",str_c(str_replace_na(pal_frontiers('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("GSEA,default,%s",str_c(str_replace_na(pal_gsea('default')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,blue,%s",str_c(str_replace_na(pal_bs5('blue')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,indigo,%s",str_c(str_replace_na(pal_bs5('indigo')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,purple,%s",str_c(str_replace_na(pal_bs5('purple')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,pink,%s",str_c(str_replace_na(pal_bs5('pink')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,red,%s",str_c(str_replace_na(pal_bs5('red')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,orange,%s",str_c(str_replace_na(pal_bs5('orange')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,yellow,%s",str_c(str_replace_na(pal_bs5('yellow')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,green,%s",str_c(str_replace_na(pal_bs5('green')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,teal,%s",str_c(str_replace_na(pal_bs5('teal')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,cyan,%s",str_c(str_replace_na(pal_bs5('cyan')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Bootstrap 5,gray,%s",str_c(str_replace_na(pal_bs5('gray')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,red,%s",str_c(str_replace_na(pal_material('red')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,pink,%s",str_c(str_replace_na(pal_material('pink')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,purple,%s",str_c(str_replace_na(pal_material('purple')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,deep-purple,%s",str_c(str_replace_na(pal_material('deep-purple')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,indigo,%s",str_c(str_replace_na(pal_material('indigo')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,blue,%s",str_c(str_replace_na(pal_material('blue')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,light-blue,%s",str_c(str_replace_na(pal_material('light-blue')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,cyan,%s",str_c(str_replace_na(pal_material('cyan')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,teal,%s",str_c(str_replace_na(pal_material('teal')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,green,%s",str_c(str_replace_na(pal_material('green')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,light-green,%s",str_c(str_replace_na(pal_material('light-green')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,lime,%s",str_c(str_replace_na(pal_material('lime')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,yellow,%s",str_c(str_replace_na(pal_material('yellow')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,amber,%s",str_c(str_replace_na(pal_material('amber')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,orange,%s",str_c(str_replace_na(pal_material('orange')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,deep-orange,%s",str_c(str_replace_na(pal_material('deep-orange')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,brown,%s",str_c(str_replace_na(pal_material('brown')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,grey,%s",str_c(str_replace_na(pal_material('grey')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Material Design,blue-grey,%s",str_c(str_replace_na(pal_material('blue-grey')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,slate,%s",str_c(str_replace_na(pal_tw3('slate')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,gray,%s",str_c(str_replace_na(pal_tw3('gray')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,zinc,%s",str_c(str_replace_na(pal_tw3('zinc')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,neutral,%s",str_c(str_replace_na(pal_tw3('neutral')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,stone,%s",str_c(str_replace_na(pal_tw3('stone')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,red,%s",str_c(str_replace_na(pal_tw3('red')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,orange,%s",str_c(str_replace_na(pal_tw3('orange')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,amber,%s",str_c(str_replace_na(pal_tw3('amber')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,yellow,%s",str_c(str_replace_na(pal_tw3('yellow')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,lime,%s",str_c(str_replace_na(pal_tw3('lime')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,green,%s",str_c(str_replace_na(pal_tw3('green')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,emerald,%s",str_c(str_replace_na(pal_tw3('emerald')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,teal,%s",str_c(str_replace_na(pal_tw3('teal')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,cyan,%s",str_c(str_replace_na(pal_tw3('cyan')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,sky,%s",str_c(str_replace_na(pal_tw3('sky')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,blue,%s",str_c(str_replace_na(pal_tw3('blue')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,indigo,%s",str_c(str_replace_na(pal_tw3('indigo')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,violet,%s",str_c(str_replace_na(pal_tw3('violet')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,purple,%s",str_c(str_replace_na(pal_tw3('purple')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,fuchsia,%s",str_c(str_replace_na(pal_tw3('fuchsia')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,pink,%s",str_c(str_replace_na(pal_tw3('pink')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
# sprintf("Tailwind CSS,rose,%s",str_c(str_replace_na(pal_tw3('rose')(100),';'),collapse = ';')) %>%
#     write('ggsci_color',append = T)
#
# ```
# > process with python
# ```python
# def handel_str_fill_for_split(s,seq,count,fill,front = False):
#     """使用fill填充s, 使得使用seq对s进行split时，结果的长度为count
# Parameters
# ----------
# front : bool
#     填充在前面
#     默认填充在后
#
# """
#     s_fill = ''
#     if seq in s:
#         count_add = count - len(s.split(seq))
#     else:
#         count_add = count-1
#
#     if front:
#         return seq.join([seq.join([fill for i in  range(count_add)]),s])
#     return seq.join([s,seq.join([fill for i in  range(count_add)])])
#
#
# df_color = pd.read_csv('ggsci_color')
# df_color['colors'] = df_color['colors'].str.replace(';+$','',regex=True)
# df_color['length'] = df_color['colors'].apply(lambda x:len(x.split(';')))
# df_color['colors'] = df_color['colors'].apply(lambda x: handel_str_fill_for_split(x,';',df_color['length'].max(),'white'))
# df_color['colors'] = df_color['colors'].str.replace('FF;',';',regex=False)\
#     .str.replace('FF$','',regex=True)\
#     .str.replace(';$','',regex=True)\
#     .str.replace(';',',',regex=False)
# df_color['__index'] = ut.df.apply_merge_field(df_color,'{name}-{palette_type}')
# df_color = ut.df.reindex_with_unique_col(df_color,'__index',drop=True)
# ut.df.show(df_color)
#
# df_color.to_csv('..../color/ggsci.csv',index=True)
# ```
#

# In[ ]:


class ggsci(Qcmap):
    def __init__(self):
        self.df = pd.read_csv(
            Path(__file__).parent.joinpath("color/ggsci.csv"), index_col=0
        )
        self.item_size_max = self.df["length"].max()


# In[ ]:


# 单例模式
Qcmap = Qcmap()
ColorLisa = ColorLisa()
Customer = Customer()
ggsci = ggsci()
