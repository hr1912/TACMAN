import sys
from pathlib import Path

p_benchmark = Path(".").absolute()
p_script = p_benchmark.joinpath("script")

assert p_script.exists()
if not str(p_script) in sys.path:
    sys.path.append(str(p_script))

p_results = p_benchmark.joinpath("results")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import utils as ut
from utils import pl
from fun_three_model import get_test_result_df, get_res_stat, Info

MAP_MODEL2COLOR = {
    "Seurat": "#466983",
    "TACTiCS": "#5050FF",
    "TOSICA": "#749B58",
    "SAMap": "#F0E685",
    "CAME": "#FF6F00",
    "TACMAN": "#E71012",
}
MAP_MEASURE2LABEL = {"F1-score": "weight F1-score", "Accuracy": "Accuracy"}


def ge_df_res(
    tissue,
    sp_ref,
    sp_que,
):
    df_res = get_test_result_df(p_results)
    df_res = df_res.loc[:, ["dir", "name"]]
    df_res = df_res.join(
        df_res["name"].str.extract(
            "(?P<tissue>[^_;]+)_(?P<sp_ref>[^_;]+)-corss-(?P<sp_que>[^_;]+);(?P<model>[^_;]+);(?P<resdir_tag>[^_;]+)"
        )
    )
    df_res = df_res[df_res["tissue"] == tissue]
    df_res = df_res[df_res["sp_ref"] == sp_ref]
    df_res = df_res[df_res["sp_que"] == sp_que]

    df_res["batch"] = df_res["name"].str.extract("(?P<batch>batch\d+)", expand=False)
    df_res["F1-score"] = df_res.apply(
        get_res_stat, key="F1-score", q="dataset_type == 'que'", axis=1
    )
    df_res["Accuracy"] = df_res.apply(
        get_res_stat, key="Accuracy", q="dataset_type == 'que'", axis=1
    )
    return df_res


def calc_statistics(value, position, confidence=0.95, jitter_range=0.05, **kvargs):
    df_plot = pd.DataFrame(
        dict(x=position + pl.tl_jitter(value.size, jitter_range), value=value)
    )
    df_plot = df_plot.dropna(axis=0)

    ##################################################
    # mean ± SD
    ##################################################
    mean = np.mean(df_plot["value"])
    sd = np.std(df_plot["value"], ddof=1)

    ##################################################
    # mean ± 95% t CI
    ##################################################

    n = df_plot["value"].size
    se = np.std(df_plot["value"], ddof=1) / np.sqrt(n)
    t_critical = scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    t_margin = t_critical * se

    ##################################################
    # 95% Bootstrap Percentile CI
    ##################################################

    bootstrap_res = scipy.stats.bootstrap(
        (df_plot["value"],),
        statistic=np.mean,
        n_resamples=5000,
        confidence_level=0.95,
        method="percentile",
        random_state=42,
    )
    bootstrap_res.confidence_interval.low
    bootstrap_res.confidence_interval.high

    return Info(
        df_plot=df_plot,
        position=position,
        mean=mean,
        sd=sd,
        t_margin=t_margin,
        bootstrap_low=bootstrap_res.confidence_interval.low,
        bootstrap_high=bootstrap_res.confidence_interval.high,
        **kvargs,
    )


pass


def plot_scatter(ax, info, color_point):
    ax.scatter(info.df_plot["x"], info.df_plot["value"], s=2, c=color_point, alpha=0.5)


def plot_item(
    ax, info, errorbar_height, errorbar_low, color_point="red", color_error_bar="grey"
):

    ax.errorbar(
        info.position,
        info.mean,
        yerr=[
            [errorbar_height],
            [errorbar_low],
        ],
        fmt="o",
        ms=2,
        capsize=3,
        color=color_error_bar,
    )
    plot_scatter(ax, info, color_point)


def plot_sd(infos, models, ax, yticks=np.arange(0.5, 1.01, 0.1)):
    for info, model in zip(infos, models):
        plot_item(
            ax,
            info,
            info.sd,
            info.sd,
            color_point=MAP_MODEL2COLOR[model],
            color_error_bar=MAP_MODEL2COLOR[model],
        )

    ax.set_xticks(np.arange(len(models)) + 1, models, rotation=90)
    ax.set_xlim(0, len(models) + 1)
    ax.set_ylim(yticks.min(), 1)
    ax.set_yticks(yticks, ["{:.1f}".format(y) for y in yticks])


def plot_t_CI(infos, models, ax, yticks=np.arange(0.5, 1.01, 0.1)):
    for info, model in zip(infos, models):
        plot_item(
            ax,
            info,
            info.t_margin,
            info.t_margin,
            color_point=MAP_MODEL2COLOR[model],
            color_error_bar=MAP_MODEL2COLOR[model],
        )

    ax.set_xticks(np.arange(len(models)) + 1, models, rotation=90)
    ax.set_xlim(0, len(models) + 1)
    ax.set_ylim(yticks.min(), 1)
    ax.set_yticks(yticks, ["{:.1f}".format(y) for y in yticks])


def plot_bootstrap_CI(infos, models, ax, yticks=np.arange(0.5, 1.01, 0.1)):
    for info, model in zip(infos, models):
        plot_item(
            ax,
            info,
            info.mean - info.bootstrap_low,
            info.bootstrap_high - info.mean,
            color_point=MAP_MODEL2COLOR[model],
            color_error_bar=MAP_MODEL2COLOR[model],
        )

    ax.set_xticks(np.arange(len(models)) + 1, models, rotation=90)
    ax.set_xlim(0, len(models) + 1)
    ax.set_ylim(yticks.min(), 1)
    ax.set_yticks(yticks, ["{:.1f}".format(y) for y in yticks])


def get_title(sp_ref, sp_que):
    return "{} R  > {} Q".format(sp_ref.title(), sp_que.title())


models = [
    "Seurat",
    "TOSICA",
    "TACTiCS",
    "SAMap",
    "CAME",
    "TACMAN",
]

res = dict()


plt.close("all")


a4p = pl.figure.A4Page()

##################################################
# Human > Mouse
##################################################
tissue = "Spleen"
sp_ref = "mouse"
sp_que = "human"
df_res = ge_df_res(tissue, sp_ref, sp_que)
res["df_res_{}_{}_{}".format(tissue, sp_ref, sp_que)] = df_res


fig_x = 5
fig_y = 1
yticks = np.arange(0, 1.01, 0.2)
a4p.area_update(fig_x, fig_y, 2, 1, len(models) * 1.5, 5, gap_height=3)
for ax, measure in zip(a4p.area_yield_ax(rc=pl.rc_frame), ["F1-score", "Accuracy"]):
    df_stat = df_res.pivot(index="batch", columns="model", values=measure)
    infos = [
        calc_statistics(
            df_stat[model].to_numpy(), position=position, jitter_range=0.4, model=model
        )
        for position, model in enumerate(models, start=1)
    ]
    res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)] = infos
    plot_sd(infos, models, ax, yticks)

    ax.set_title(get_title(sp_ref, sp_que), fontdict=dict(fontsize=8))
    ax.set_ylabel("{}\n(mean ± SD)".format(MAP_MEASURE2LABEL[measure]))

fig_x += 15
a4p.area_update(fig_x, fig_y, 2, 1, len(models) * 1.5, 5, gap_height=3)
for ax, measure in zip(a4p.area_yield_ax(rc=pl.rc_frame), ["F1-score", "Accuracy"]):
    df_stat = df_res.pivot(index="batch", columns="model", values=measure)
    infos = [
        calc_statistics(
            df_stat[model].to_numpy(), position=position, jitter_range=0.4, model=model
        )
        for position, model in enumerate(models, start=1)
    ]
    res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)] = infos
    plot_t_CI(infos, models, ax, yticks)

    ax.set_title(get_title(sp_ref, sp_que), fontdict=dict(fontsize=8))
    ax.set_ylabel("{}\n(mean ± SD)".format(MAP_MEASURE2LABEL[measure]))


##################################################
# Human > Mouse
##################################################
tissue = "Spleen"
sp_ref = "human"
sp_que = "mouse"
df_res = ge_df_res(tissue, sp_ref, sp_que)
res["df_res_{}_{}_{}".format(tissue, sp_ref, sp_que)] = df_res


fig_x = 5
fig_y = 18
yticks = np.arange(0, 1.01, 0.2)
a4p.area_update(fig_x, fig_y, 2, 1, len(models) * 1.5, 5, gap_height=3)
for ax, measure in zip(a4p.area_yield_ax(rc=pl.rc_frame), ["F1-score", "Accuracy"]):
    df_stat = df_res.pivot(index="batch", columns="model", values=measure)
    infos = [
        calc_statistics(
            df_stat[model].to_numpy(), position=position, jitter_range=0.4, model=model
        )
        for position, model in enumerate(models, start=1)
    ]
    res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)] = infos
    plot_sd(infos, models, ax, yticks)

    ax.set_title(get_title(sp_ref, sp_que), fontdict=dict(fontsize=8))
    ax.set_ylabel("{}\n(mean ± SD)".format(MAP_MEASURE2LABEL[measure]))

fig_x += 15
a4p.area_update(fig_x, fig_y, 2, 1, len(models) * 1.5, 5, gap_height=3)
for ax, measure in zip(a4p.area_yield_ax(rc=pl.rc_frame), ["F1-score", "Accuracy"]):
    df_stat = df_res.pivot(index="batch", columns="model", values=measure)
    infos = [
        calc_statistics(
            df_stat[model].to_numpy(), position=position, jitter_range=0.4, model=model
        )
        for position, model in enumerate(models, start=1)
    ]
    res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)] = infos
    plot_t_CI(infos, models, ax, yticks)

    ax.set_title(get_title(sp_ref, sp_que), fontdict=dict(fontsize=8))
    ax.set_ylabel("{}\n(mean ± SD)".format(MAP_MEASURE2LABEL[measure]))

fig_stem = "benchmark_{}".format(tissue)
pl.tl_savefig(a4p.fig, "{}.png".format(fig_stem), p_benchmark)
pl.tl_savefig(a4p.fig, "{}.jpg".format(fig_stem), p_benchmark)
pl.tl_savefig(a4p.fig, "{}.pdf".format(fig_stem), p_benchmark)


p_out = p_benchmark.joinpath("benchmark_{}.xlsx".format(tissue))
key_save_res = ["tissue", "sp_ref", "sp_que", "model", "batch", "F1-score", "Accuracy"]
key_save_stat_info = ["model", "mean", "sd", "t_margin"]
with pd.ExcelWriter(p_out, mode="w") as pdew:
    tissue = "Spleen"
    sp_ref = "human"
    sp_que = "mouse"
    df_res = res["df_res_{}_{}_{}".format(tissue, sp_ref, sp_que)]
    df_res.loc[:, key_save_res].to_excel(
        pdew, sheet_name="{}_{}".format(sp_ref, sp_que), index=False
    )
    for measure in ["F1-score", "Accuracy"]:
        infos = res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)]
        [info.drop("df_plot") for info in infos]
        df_stat_info = pd.DataFrame([info._data for info in infos])
        df_stat_info.loc[:, key_save_stat_info].to_excel(
            pdew, sheet_name="{}_{}_{}".format(sp_ref, sp_que, measure), index=False
        )

    tissue = "Spleen"
    sp_ref = "mouse"
    sp_que = "human"
    df_res = res["df_res_{}_{}_{}".format(tissue, sp_ref, sp_que)]
    df_res.loc[:, key_save_res].to_excel(
        pdew, sheet_name="{}_{}".format(sp_ref, sp_que), index=False
    )
    for measure in ["F1-score", "Accuracy"]:
        infos = res["infos_{}_{}_{}_{}".format(tissue, sp_ref, sp_que, measure)]
        [info.drop("df_plot") for info in infos]
        df_stat_info = pd.DataFrame([info._data for info in infos])
        df_stat_info.loc[:, key_save_stat_info].to_excel(
            pdew, sheet_name="{}_{}_{}".format(sp_ref, sp_que, measure), index=False
        )

print("[out] {}".format(p_out))
pass
