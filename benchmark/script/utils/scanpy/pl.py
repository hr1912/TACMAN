#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""utils.scanpy.pl
用于为adata绘图

+ qc
+ umap
+ spatial
+ monocle2

"""


# In[ ]:


from .sc import Path, np, pd, plt, mpl
from .sc import sc
from .sc import get_scalefactor, get_spot_size
from .. import plot as pl

from ..general import update_dict, subset_dict, Block


# # standard_process

# In[ ]:


def qc(adata):
    plt.close("all")
    keys = "total_counts,n_genes_by_counts,pct_counts_mt,pct_counts_ribo,pct_counts_hb".split(
        ","
    )
    assert (
        pd.Series(keys).isin(adata.obs.columns).all()
    ), "[Error] please defind {}".format(",".join(keys))

    a4p = pl.figure.DrawingBoard(nrows=20, ncols=20)

    y = 0
    with Block("qc violin", context=dict(x=2, y=y)) as context:

        a4p.area_update(context.x, context.y, 1, 5, 2, 5, gap_width=1.5)
        for k, ylabel, ax in zip(
            keys,
            "count,count,pct,pct,pct".split(","),
            a4p.area_yield_ax(rc=pl.rc_frame),
        ):
            sc.pl.violin(adata, k, ax=ax, ylabel=ylabel, jitter=0.25, show=False)

    y += 6
    with Block("qc scatter", context=dict(x=1.5, y=y)) as context:
        a4p.area_update(context.x, context.y, 2, 3, 5, 5, gap_width=1.5, gap_height=1.5)

        sc.pl.scatter(
            adata,
            "total_counts",
            "n_genes_by_counts",
            show=False,
            ax=a4p.area_get_ax(0, rc=pl.rc_frame),
        )
        sc.pl.scatter(
            adata,
            "total_counts",
            "pct_counts_mt",
            show=False,
            ax=a4p.area_get_ax(3, rc=pl.rc_frame),
        )
        sc.pl.scatter(
            adata,
            "total_counts",
            "pct_counts_ribo",
            show=False,
            ax=a4p.area_get_ax(4, rc=pl.rc_frame),
        )
        sc.pl.scatter(
            adata,
            "total_counts",
            "pct_counts_hb",
            show=False,
            ax=a4p.area_get_ax(5, rc=pl.rc_frame),
        )

    return a4p.fig


def standard_process(
    adata, key_batch="sample", key_cluster="leiden", kw_pl_umap=dict(size=0.25)
):
    plt.close("all")
    with plt.rc_context(rc=pl.rc_frame):
        a4p = pl.figure.DrawingBoard(nrows=25, ncols=25)

    y = 1
    with Block("batch effect", context=dict(x=0, y=y)) as context:
        ax = a4p.add_ax(context.x, context.y, rc=pl.rc_blank)
        a4p.add_text_with_ax(ax, context.comment, fontdict=dict(fontsize=8))
        context.x += 1

        cmap = pl.cmap.ggsci.get_cmap("IGV", adata.obs[key_batch].unique())
        a4p.area_update(context.x, context.y, 1, 2, 6, 6)
        for ax, k in zip(
            a4p.area_yield_ax(rc=pl.rc_blank), "X_pca,X_pca_harmony".split(",")
        ):
            if not k in adata.obsm_keys():
                continue
            umap(adata, key=key_batch, cmap=cmap, ax=ax, key_obsm=k, **kw_pl_umap)
            ax.set_title(k)

        a4p.area_update(context.x + 13, context.y, 1, 2, 1, 6, gap_width=2)
        for ax, k in zip(
            a4p.area_yield_ax(rc=pl.rc_blank), np.array_split(list(cmap.keys()), 2)
        ):
            pl.cmap.show(subset_dict(cmap, k), ax=ax)

        del cmap, ax, k

    y += 7
    with Block("cluster", context=dict(x=0, y=y)) as context:
        ax = a4p.add_ax(context.x, context.y, rc=pl.rc_blank)
        a4p.add_text_with_ax(ax, context.comment, fontdict=dict(fontsize=8))
        context.x += 1

        a4p.area_update(context.x, context.y, 2, 1, 6, 6, gap_height=2)
        _x, _y, ax = a4p.area_get_ax(0, rc=pl.rc_blank, with_xy=True)

        for (_x, _y, ax), k in zip(
            list(a4p.area_yield_ax(rc=pl.rc_blank, with_xy=True)),
            [key_cluster, "cell_type"],
        ):
            if not k in adata.obs_keys():
                continue
            cmap = pl.cmap.ggsci.get_cmap("IGV", adata.obs[k].unique())
            umap(adata, key=k, cmap=cmap, ax=ax, key_obsm="X_umap", **kw_pl_umap)
            ax.set_title(k)

            a4p.area_update(_x + 7, _y, 1, 3, 1, 6, gap_width=2)
            for ax, k in zip(
                a4p.area_yield_ax(rc=pl.rc_blank), np.array_split(list(cmap.keys()), 3)
            ):
                pl.cmap.show(subset_dict(cmap, k), ax=ax)

    return a4p.fig


# # umap

# In[ ]:


def __assert_all_value_in_cmap(adata, key, cmap):
    return adata.obs[key].value_counts().index.isin(cmap.keys()).all()


def __get_default_cmap(adata, key):
    assert key in adata.obs.columns
    return pl.cmap.ggsci.get_cmap("IGV", adata.obs[key].unique())


def umap(
    adata,
    key,
    ax,
    ax_cmap=None,
    cmap=None,
    marker=".",
    size=1,
    sort=False,
    kw_scatter=None,
    key_obsm="X_umap",
    text_on_data=False,
    text_fontdict=None,
):
    assert key in adata.obs.columns, "{} is not in adata.obs".format(key)
    if cmap is None:
        cmap = __get_default_cmap(adata, key)
    assert __assert_all_value_in_cmap(
        adata, key, cmap
    ), "[Error] not all ele of adata.obs[key] in cmap.keys()"
    kw_scatter = update_dict({}, kw_scatter)

    df_plot = pd.DataFrame(
        adata.obsm[key_obsm][:, :2],
        index=adata.obs.index,
        columns="_obsm1,_obsm2".split(","),
    )
    df_plot = df_plot.join(adata.obs.loc[:, [key]].rename(columns={key: "_key"}))
    df_plot["_key"] = df_plot["_key"].astype(str)
    # scatter
    if sort:
        df_plot = df_plot.sort_values(
            "_key", key=lambda s: s.map({k: i for i, k in enumerate(cmap.keys())})
        )
    else:
        df_plot = df_plot.sample(df_plot.shape[0], replace=False)

    df_plot["_color"] = df_plot["_key"].map(cmap)
    ax.scatter(
        "_obsm1",
        "_obsm2",
        s=size,
        marker=marker,
        c="_color",
        data=df_plot,
        **kw_scatter
    )

    ax.set_axis_off()
    if ax_cmap:
        pl.cmap.show(cmap, ax=ax_cmap)
    if text_on_data:
        text_fontdict = update_dict(
            dict(va="center", ha="center", fontsize=8), text_fontdict
        )
        df_text = df_plot.groupby("_key").agg({"_obsm1": "mean", "_obsm2": "mean"})
        for i, row in df_text.iterrows():
            ax.text(row["_obsm1"], row["_obsm2"], i, **text_fontdict)


# In[1]:


def umap_gene(
    adata,
    key,
    ax,
    marker=".",
    layer=None,
    vmin=None,
    vmax=None,
    cmap="Reds",
    size=10,
    sort=True,
    draw_cbar=True,
    kw_scatter=None,
    kw_cbar=None,
    key_obsm="X_umap",
):
    """将连续变量(如基因的表达量)映射到umap坐标
    Parameters
    ----------
    kw_cbar : dict | None
        dict(location='right',
            fraction=0.025, # 在对应方向上的宽度与父轴宽度的比值
            aspect= 40,     # cbar的长宽比
            pad= 0.02,      # 与父轴的间距
            format='{x:.1f}')
    """
    if not (key in adata.obs.columns or key in adata.var_names):
        print("[Waring][umap_gene] '{}' is not in adata ".format(key))
        return
    kw_scatter = {} if kw_scatter is None else kw_scatter

    kw_cbar = update_dict(
        dict(location="right", fraction=0.025, aspect=40, pad=0.02, format="{x:.1f}"),
        {} if kw_cbar is None else kw_cbar,
    )

    df_plot = (
        pd.DataFrame(
            adata.obsm[key_obsm][:, :2],
            index=adata.obs.index,
            columns="obsm1,obsm2".split(","),
        )
        .join(sc.get.obs_df(adata, [key], layer=layer))
        .copy()
    )
    df_plot = (
        df_plot.sort_values(key)
        if sort
        else df_plot.sample(df_plot.shape[0], replace=False)
    )
    # scatter
    cbar = ax.scatter(
        "obsm1",
        "obsm2",
        s=size,
        c=df_plot[key],
        marker=marker,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        data=df_plot,
        **kw_scatter
    )
    ax.figure.colorbar(cbar, ax=ax, **kw_cbar) if draw_cbar else None
    ax.set_axis_off()

    return cbar


# # spatial

# In[ ]:


def spatial(
    adata,
    key,
    ax,
    ax_cmap=None,
    key_uns_spatial="spatial",
    key_img="img",
    cmap=None,
    size=1,
    spot_size=None,
    scale_factor=None,
    marker=".",
    draw_img=True,
    draw_scatter=True,
    kw_scatter={},
    kw_imshow={},
    kw_get_scalefactor={},
):
    """由空间信息和图像信息绘制图形, key对应的数据为离散变量

    Parameters
    ----------
    scale_factor : int,float (default : None)
        若为None, 尝试由util.scanpy.sc.default get_scalefactor()获取
        该函数默认值为1

    size,spot_size : int,float (default : 1,None)
        控制scatter点的大小

        spot_size为None, 尝试由util.scanpy.sc.default get_spot_size()获取
        该函数默认值为1

        最后将size * scale_factor * spot_size * 0.5
        传入 ax.scatter
    """

    # 前处理， 判断与赋值
    assert key in adata.obs.columns
    assert key_uns_spatial in adata.uns["spatial"]
    assert key_img in adata.uns["spatial"][key_uns_spatial]["images"]
    assert (
        draw_scatter or draw_img
    ), "[Error] at least one of draw_img and draw_scatter must be True"
    if cmap is None:
        cmap = __get_default_cmap(adata, key)
    if scale_factor is None:
        scale_factor = get_scalefactor(
            adata,
            key_uns_spatial=key_uns_spatial,
            key_img=key_img,
            **kw_get_scalefactor
        )
    if spot_size is None:
        spot_size = get_spot_size(adata, key_uns_spatial)
    circle_radius = size * scale_factor * spot_size * 0.5

    df_spatial = (
        pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs.index,
            columns="spatial1,spatial2".split(","),
        )
        .mul(scale_factor)
        .join(adata.obs.loc[:, [key]])
        .copy()
    )
    df_spatial[key] = df_spatial[key].astype(str)
    dict_spatial = adata.uns["spatial"][key_uns_spatial]
    # scatter
    [
        ax.scatter(
            "spatial1",
            "spatial2",
            label=label,
            s=circle_radius,
            marker=marker,
            c=cmap[label],
            data=df_spatial.query("{} == '{}'".format(key, label)),
            **kw_scatter
        )
        for label in cmap.keys()
    ]
    img_edge = np.concatenate([ax.get_xlim(), ax.get_ylim()])
    ax.clear() if not draw_scatter else None
    # im show
    if draw_img:
        ax.imshow(dict_spatial["images"][key_img], **kw_imshow)
        ax.grid(False)
        ax.set_xlim(img_edge[0], img_edge[1])
        ax.set_ylim(img_edge[2], img_edge[3])

    if not ax.yaxis_inverted():  # 确保y轴是转置的，符合img的坐标习惯
        ax.invert_yaxis()
    ax.set_axis_off()

    if ax_cmap:
        pl.cmap.show(cmap, ax=ax_cmap)


def spatial_gene(
    adata,
    key,
    key_uns_spatial,
    key_img,
    scale_factor,
    ax,
    vmin=None,
    vmax=None,
    cmap="Reds",
    size=1,
    spot_size=None,
    marker=".",
    draw_img=True,
    draw_scatter=True,
    draw_cbar=True,
    kw_scatter={},
    kw_imshow={},
    kw_cbar={
        "location": "right",
        "fraction": 0.025,  # 在对应方向上的宽度与父轴宽度的比值
        "aspect": 40,  # cbar的长宽比,fraction*aspect 即为cbar的长的占比
        "pad": 0.02,  # 与父轴的间距
        # 'format': '{x:.1f}'
    },
):
    """由空间信息和图像信息绘制图形, key对应的数据为连续性变量

    Parameters
    ----------
    scale_factor : int,float (default : None)
        若为None, 尝试由util.scanpy.sc.default get_scalefactor()获取
        该函数默认值为1

    size,spot_size : int,float (default : 1,None)
        控制scatter点的大小

        spot_size为None, 尝试由util.scanpy.sc.default get_spot_size()获取
        该函数默认值为1

        最后将size * scale_factor * spot_size * 0.5
        传入 ax.scatter
    """
    # 前处理， 判断与赋值
    assert key in adata.obs.columns or key in adata.var_names
    assert key_uns_spatial in adata.uns["spatial"]
    assert key_img in adata.uns["spatial"][key_uns_spatial]["images"]
    assert (
        draw_scatter or draw_img
    ), "[Error] at least one of draw_img and draw_scatter must be True"
    if scale_factor is None:
        scale_factor = get_scalefactor(
            adata,
            key_uns_spatial=key_uns_spatial,
            key_img=key_img,
            **kw_get_scalefactor
        )
    if spot_size is None:
        spot_size = get_spot_size(adata, key_uns_spatial)
    circle_radius = size * scale_factor * spot_size * 0.5

    df_spatial = (
        pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs.index,
            columns="spatial1,spatial2".split(","),
        )
        .mul(scale_factor)
        .join(sc.get.obs_df(adata, [key]))
    )

    dict_spatial = adata.uns["spatial"][key_uns_spatial]

    # scatter
    cbar = ax.scatter(
        "spatial1",
        "spatial2",
        s=circle_radius,
        marker=marker,
        c=df_spatial[key],
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        data=df_spatial,
        **kw_scatter
    )
    ax.figure.colorbar(cbar, ax=ax, **kw_cbar) if draw_cbar else None

    img_edge = np.concatenate([ax.get_xlim(), ax.get_ylim()])
    ax.clear() if not draw_scatter else None
    # im show
    if draw_img:
        ax.imshow(dict_spatial["images"][key_img], **kw_imshow)
        ax.grid(False)
        ax.set_xlim(img_edge[0], img_edge[1])
        ax.set_ylim(img_edge[2], img_edge[3])

    if not ax.yaxis_inverted():  # 确保y轴是转置的，符合img的坐标习惯
        ax.invert_yaxis()
    ax.set_axis_off()

    return cbar


def spatial_3d(
    adata,
    key,
    ax,
    cmap=None,
    scale_factor=None,
    marker=".",
    size=10,
    kw_scatter={},
    height=5,
    query_3d_line="",
    kw_line={"linewidth": 0.5, "color": "grey"},
    kw_view_init={"elev": 35, "azim": -90, "roll": 0},
    key_obsm="X_umap",
):
    """
    Parameters
    ----------
    scale_factor : int,float (default : None)
        若为None, 尝试由util.scanpy.sc.default get_scalefactor()获取
        该函数默认值为1

    size,spot_size : int,float (default : 1,None)
        控制scatter点的大小

        spot_size为None, 尝试由util.scanpy.sc.default get_spot_size()获取
        该函数默认值为1

        最后将size * scale_factor * spot_size * 0.5
        传入 ax.scatter
    """
    assert key in adata.obs.columns
    if cmap is None:
        cmap = __get_default_cmap(adata, key)
    if scale_factor is None:
        scale_factor = get_scalefactor(
            adata,
            key_uns_spatial=key_uns_spatial,
            key_img=key_img,
            **kw_get_scalefactor
        )
    if spot_size is None:
        spot_size = get_spot_size(adata, key_uns_spatial)
    circle_radius = size * scale_factor * spot_size * 0.5

    df_plot = (
        pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs.index,
            columns="spatial1,spatial2".split(","),
        )
        .mul(scale_factor)
        .join(
            pd.DataFrame(
                adata.obsm[key_obsm],
                index=adata.obs.index,
                columns="obsm1,obsm2".split(","),
            )
        )
        .join(adata.obs.loc[:, [key]])
    )
    df_plot[key] = df_plot[key].astype(str)

    from utils.arr import scale

    # 将UMAP 的数值 映射到spatial2上
    df_plot["scale_obsm1"] = scale(
        df_plot["obsm1"],
        edge_min=df_plot["spatial1"].min(),
        edge_max=df_plot["spatial1"].max(),
    )
    df_plot["scale_obsm2"] = scale(
        df_plot["obsm2"],
        edge_min=df_plot["spatial2"].min(),
        edge_max=df_plot["spatial2"].max(),
    )
    ## 以0点颠倒
    df_plot["scale_spatial2"] = scale(
        df_plot["spatial2"].mul(-1),
        edge_max=df_plot["spatial2"].max(),
        edge_min=df_plot["spatial2"].min(),
    )
    # scatter
    ## scatter [spatial]
    [
        ax.scatter(
            "spatial1",
            "scale_spatial2",
            zs=height,
            label=label,
            s=circle_radius,
            marker=marker,
            c=cmap[label],
            data=df_plot.query("{} == '{}'".format(key, label)),
            **kw_scatter
        )
        for label in cmap.keys()
    ]

    ## scatter [UMAP]
    [
        ax.scatter(
            "scale_obsm1",
            "scale_obsm2",
            zs=0,
            label=label,
            s=circle_radius,
            marker=marker,
            c=cmap[label],
            data=df_plot.query("{} == '{}'".format(key, label)),
            **kw_scatter
        )
        for label in cmap.keys()
    ]

    # line
    if not query_3d_line:
        query_3d_line = "{0} == {0}".format(key)
    for i_plot, row_plot in df_plot.query(query_3d_line).iterrows():
        kw_line.update({"color": cmap[row_plot[key]]})
        ax.plot3D(
            xs=[row_plot["scale_obsm1"], row_plot["spatial1"]],
            ys=[row_plot["scale_obsm2"], row_plot["scale_spatial2"]],
            zs=[0, height],
            alpha=0.25,
            **kw_line
        )

    # 调节视图
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ax.set_axis_off()
    ax.view_init(**kw_view_init)

    return df_plot


# # monocle2

# In[ ]:


def monocle2(
    adata,
    key,
    ax,
    ax_cmap=None,
    cmap=None,
    marker=".",
    size=10,
    kw_scatter={},
    show_branch_points=True,
    line_style={"color": "black", "linewidth": 0.8},
    show_trajectory_tree=True,
    kw_scatter_branch_points={"color": "black", "s": 30},
    fontdict_branch_points={
        "ha": "center",
        "va": "center",
        "color": "white",
        "fontsize": 4,
    },
):

    # adata.uns['monocle2']['data_df'] = adata.obs.loc[:,[]].join(adata.uns['monocle2']['data_df'])
    dict_monocle2 = adata.uns["monocle2"]
    df_plot = dict_monocle2["data_df"].copy()

    if key not in df_plot:
        df_plot = adata.obs.loc[:, [key]].join(df_plot)
    df_plot[key] = df_plot[key].astype(str)
    if cmap is None:
        cmap = pl.cmap.get(df_plot[key].sort_values().unique())

    # scatter
    [
        ax.scatter(
            "data_dim_1",
            "data_dim_2",
            label=label,
            s=size,
            marker=marker,
            c=cmap[label],
            data=df_plot.query("{} == '{}'".format(key, label)),
            **kw_scatter
        )
        for label in cmap.keys()
    ]

    # trajectory tree
    if show_trajectory_tree:
        df_edge = dict_monocle2["edge_df"]
        arr_edge = df_edge.loc[
            :,
            "source_prin_graph_dim_1,source_prin_graph_dim_2,target_prin_graph_dim_1,target_prin_graph_dim_2".split(
                ","
            ),
        ].to_numpy()
        arr_edge = np.reshape(arr_edge, (arr_edge.shape[0], 2, 2), order="F")
        for _ in arr_edge:
            ax.plot(*_, **line_style)

    df_branch_points = df_edge.filter(regex="^source")[
        df_edge["source"].isin(adata.uns["monocle2"]["branch_points"])
    ].drop_duplicates("source")
    df_branch_points["text"] = np.arange(df_branch_points.shape[0]) + 1
    df_branch_points["text"] = df_branch_points["text"].astype(str)
    # branch_points
    if show_branch_points:
        ax.scatter(
            "source_prin_graph_dim_1",
            "source_prin_graph_dim_2",
            data=df_branch_points,
            **kw_scatter_branch_points
        )
        for i, row in df_branch_points.iterrows():
            ax.text(
                row["source_prin_graph_dim_1"],
                row["source_prin_graph_dim_2"],
                row["text"],
                fontdict=fontdict_branch_points,
            )

    ax.set_axis_off()

    # legend
    if ax_cmap:
        pl.cmap.show(cmap, ax=ax_cmap)


# In[ ]:


def monocle2_gene(
    adata,
    key,
    ax,
    marker=".",
    vmin=None,
    vmax=None,
    cmap="Reds",
    size=10,
    draw_cbar=True,
    kw_scatter={},
    kw_cbar={
        "location": "right",
        "fraction": 0.025,  # 在对应方向上的宽度与父轴宽度的比值
        "aspect": 40,  # cbar的长宽比
        "pad": 0.02,  # 与父轴的间距
        "format": "{x:.1f}",
    },
    show_branch_points=True,
    line_style={"color": "black", "linewidth": 0.8},
    show_trajectory_tree=True,
    kw_scatter_branch_points={"color": "black", "s": 30},
    fontdict_branch_points={
        "ha": "center",
        "va": "center",
        "color": "white",
        "fontsize": 4,
    },
):
    """将连续变量(如基因的表达量)映射到monocle2的拟时序空间"""
    adata.uns["monocle2"]["data_df"] = adata.obs.loc[:, []].join(
        adata.uns["monocle2"]["data_df"]
    )
    dict_monocle2 = adata.uns["monocle2"]
    df_plot = dict_monocle2["data_df"]
    assert key in df_plot.columns or key in adata.obs.columns or key in adata.var.index
    if key not in df_plot.columns:
        df_plot = df_plot.join(sc.get.obs_df(adata, [key]))
    df_plot = df_plot.sort_values(key)
    # scatter
    cbar = ax.scatter(
        "data_dim_1",
        "data_dim_2",
        s=size,
        c=df_plot[key],
        marker=marker,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        data=df_plot,
        **kw_scatter
    )
    ax.figure.colorbar(cbar, ax=ax, **kw_cbar) if draw_cbar else None

    # trajectory tree
    if show_trajectory_tree:
        df_edge = dict_monocle2["edge_df"]
        arr_edge = df_edge.loc[
            :,
            "source_prin_graph_dim_1,source_prin_graph_dim_2,target_prin_graph_dim_1,target_prin_graph_dim_2".split(
                ","
            ),
        ].to_numpy()
        arr_edge = np.reshape(arr_edge, (arr_edge.shape[0], 2, 2), order="F")
        for _ in arr_edge:
            ax.plot(*_, **line_style)

    df_branch_points = df_edge.filter(regex="^source")[
        df_edge["source"].isin(adata.uns["monocle2"]["branch_points"])
    ].drop_duplicates("source")
    df_branch_points["text"] = np.arange(df_branch_points.shape[0]) + 1
    df_branch_points["text"] = df_branch_points["text"].astype(str)
    # branch_points
    if show_branch_points:
        ax.scatter(
            "source_prin_graph_dim_1",
            "source_prin_graph_dim_2",
            data=df_branch_points,
            **kw_scatter_branch_points
        )
        for i, row in df_branch_points.iterrows():
            ax.text(
                row["source_prin_graph_dim_1"],
                row["source_prin_graph_dim_2"],
                row["text"],
                fontdict=fontdict_branch_points,
            )

    ax.set_axis_off()
    return cbar
