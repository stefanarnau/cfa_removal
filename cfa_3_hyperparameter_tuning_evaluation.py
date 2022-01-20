#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import sys
import numpy as np
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datapaths
model_eval_path = "add_path_here"
path_plots = "add_path_here"
path_file = "add_path_here"

# insert path to color module
sys.path.insert(1, path_file)
from cool_colormaps import cga_p1_light as ccm


# Subjects to use
subject_list = [
    "VP02",
    "VP03",
    "VP04",
    "VP05",
    "VP06",
    "VP07",
    "VP08",
    "VP09",
    "VP10",
    "VP11",
    "VP12",
    "VP13",
    "VP14",
    "VP15",
    "VP16",
    "VP17",
    "VP18",
    "VP19",
    "VP20",
    "VP21",
    "VP22",
    "VP23",
    "VP24",
    "VP25",
    "VP26",
    "VP27",
    "VP28",
    "VP29",
    "VP30",
    "VP31",
    "VP32",
    "VP33",
    "VP34",
    "VP35",
    "VP36",
    "VP37",
    "VP38",
    "VP39",
    "VP40",
    "VP41",
]

# Channels to use
channel_list = [40, 48, 14]

# Specify number of hyperparameter combinations
n_hyperparameter_combinations = 384


def get_metric_df(metric_name):

    # Create a data frame for metric
    index_array = [
        [128, 2048] * 16,
        [5, 5, 10, 10, 20, 20, 50, 50] * 4,
        [1] * 8 + [2] * 8 + [4] * 8 + [8] * 8,
    ]
    row_indices = pd.MultiIndex.from_arrays(
        index_array, names=("batch size", "n neurons", "n layer")
    )
    index_array = [
        [1, 0.25] * 6,
        ["dropdown", "dropdown", "proper", "proper", "both", "both",] * 2,
        [True] * 6 + [False] * 6,
    ]
    col_indices = pd.MultiIndex.from_arrays(
        index_array, names=("trainset size", "ecg source", "phase")
    )
    df = pd.DataFrame(np.zeros((32, 12)), index=row_indices, columns=col_indices)
    df_out = pd.DataFrame(
        np.zeros((n_hyperparameter_combinations, 8)),
        columns=[
            "model number",
            "batch size",
            "n neurons",
            "n layer",
            "train size",
            "ecg",
            "phase",
            "metric",
        ],
    )

    # Load metrics
    counter = -1
    for subject in subject_list:
        for channel in channel_list:
            for model_number in np.arange(0, n_hyperparameter_combinations):
                counter += 1
                dat = load(
                    os.path.join(
                        model_eval_path,
                        "model_eval_scores_"
                        + subject
                        + "_channel_"
                        + str(channel)
                        + "_model_nr_"
                        + str(model_number)
                        + ".joblib",
                    ),
                )
                df.loc[
                    (
                        dat["parameter_batch_size"],
                        dat["parameter_n_neurons"],
                        dat["parameter_n_layer"],
                    ),
                    (
                        float(dat["parameter_train_size"]),
                        dat["parameter_ecg_elec"],
                        dat["parameter_use_phase"],
                    ),
                ] += dat[metric_name]

                # A second df for sorted models
                df_out["model number"][model_number] = model_number
                df_out["batch size"][model_number] = dat["parameter_batch_size"]
                df_out["n neurons"][model_number] = dat["parameter_n_neurons"]
                df_out["n layer"][model_number] = dat["parameter_n_layer"]
                df_out["train size"][model_number] = float(dat["parameter_train_size"])
                df_out["ecg"][model_number] = dat["parameter_ecg_elec"]
                df_out["phase"][model_number] = dat["parameter_use_phase"]
                df_out["metric"][model_number] += dat[metric_name]

    # Scale
    df = df.div(len(subject_list) * len(channel_list))
    df_out["metric"] = df_out["metric"].div(len(subject_list) * len(channel_list))

    # Sort
    df_out.sort_values(by=["metric"], inplace=True)

    # Plot metric as heatmap
    fig = plt.figure(num=None, figsize=(20, 20), dpi=300, facecolor="w", edgecolor="k")
    sns.set(font_scale=1.4)
    res = sns.heatmap(
        df,
        annot=False,
        cmap=ccm,
        cbar_kws={"shrink": 0.82},
        linewidths=0.1,
        linecolor="gray",
        robust=True,
    )
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=16)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=16)
    plt.title(metric_name)
    plt.savefig(f"{path_plots}{metric_name}_heatmap.png")
    plt.show()

    # Write dataframe as csv
    df.to_csv(
        path_or_buf=f"{path_plots}{metric_name}_heatmap.csv",
        sep=",",
        header=False,
        index=False,
    )
    return df_out.reset_index()


# Plot erps
def get_erp(channel_number, model_number_best, model_number_worst, titlestring):
    erp_best = {}
    for s, subject in enumerate(subject_list):
        dat = load(
            os.path.join(
                model_eval_path,
                "model_eval_scores_"
                + subject
                + "_channel_"
                + str(channel_number)
                + "_model_nr_"
                + str(model_number_best)
                + ".joblib",
            ),
        )
        if s == 0:
            erp_best["times_rlock"] = dat["erp_times_rlock"]
            erp_best["times_stimlock"] = dat["erp_times_stimlock"]
            erp_best["rlock_before"] = dat["erp_rlock_before"] / len(subject_list)
            erp_best["rlock_after"] = dat["erp_rlock_after"] / len(subject_list)
            erp_best["rlock_predicted"] = dat["erp_rlock_predicted"] / len(subject_list)
            erp_best["stimlock_before"] = dat["erp_stimlock_before"] / len(subject_list)
            erp_best["stimlock_after"] = dat["erp_stimlock_after"] / len(subject_list)
            erp_best["stimlock_predicted"] = dat["erp_stimlock_predicted"] / len(
                subject_list
            )
        else:
            erp_best["rlock_before"] = erp_best["rlock_before"] + dat[
                "erp_rlock_before"
            ] / len(subject_list)
            erp_best["rlock_after"] = erp_best["rlock_after"] + dat[
                "erp_rlock_after"
            ] / len(subject_list)
            erp_best["rlock_predicted"] = erp_best["rlock_predicted"] + dat[
                "erp_rlock_predicted"
            ] / len(subject_list)
            erp_best["stimlock_before"] = erp_best["stimlock_before"] + dat[
                "erp_stimlock_before"
            ] / len(subject_list)
            erp_best["stimlock_after"] = erp_best["stimlock_after"] + dat[
                "erp_stimlock_after"
            ] / len(subject_list)
            erp_best["stimlock_predicted"] = erp_best["stimlock_predicted"] + dat[
                "erp_stimlock_predicted"
            ] / len(subject_list)

    erp_worst = {}
    for s, subject in enumerate(subject_list):
        dat = load(
            os.path.join(
                model_eval_path,
                "model_eval_scores_"
                + subject
                + "_channel_"
                + str(channel_number)
                + "_model_nr_"
                + str(model_number_worst)
                + ".joblib",
            ),
        )
        if s == 0:
            erp_worst["times_rlock"] = dat["erp_times_rlock"]
            erp_worst["times_stimlock"] = dat["erp_times_stimlock"]
            erp_worst["rlock_before"] = dat["erp_rlock_before"] / len(subject_list)
            erp_worst["rlock_after"] = dat["erp_rlock_after"] / len(subject_list)
            erp_worst["rlock_predicted"] = dat["erp_rlock_predicted"] / len(
                subject_list
            )
            erp_worst["stimlock_before"] = dat["erp_stimlock_before"] / len(
                subject_list
            )
            erp_worst["stimlock_after"] = dat["erp_stimlock_after"] / len(subject_list)
            erp_worst["stimlock_predicted"] = dat["erp_stimlock_predicted"] / len(
                subject_list
            )
        else:
            erp_worst["rlock_before"] = erp_worst["rlock_before"] + dat[
                "erp_rlock_before"
            ] / len(subject_list)
            erp_worst["rlock_after"] = erp_worst["rlock_after"] + dat[
                "erp_rlock_after"
            ] / len(subject_list)
            erp_worst["rlock_predicted"] = erp_worst["rlock_predicted"] + dat[
                "erp_rlock_predicted"
            ] / len(subject_list)
            erp_worst["stimlock_before"] = erp_worst["stimlock_before"] + dat[
                "erp_stimlock_before"
            ] / len(subject_list)
            erp_worst["stimlock_after"] = erp_worst["stimlock_after"] + dat[
                "erp_stimlock_after"
            ] / len(subject_list)
            erp_worst["stimlock_predicted"] = erp_worst["stimlock_predicted"] + dat[
                "erp_stimlock_predicted"
            ] / len(subject_list)

    # A plot
    fig = plt.figure(num=None, figsize=(8, 8), dpi=300, facecolor="w", edgecolor="k")
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(titlestring, fontsize=10)

    axs[0, 0].plot(
        erp_best["times_rlock"],
        erp_best["rlock_before"],
        label="before",
        color="#00AAAA",
    )
    axs[0, 0].plot(
        erp_best["times_rlock"],
        erp_best["rlock_predicted"],
        label="predicted",
        color="#AA00AA",
    )
    axs[0, 0].plot(
        erp_best["times_rlock"],
        erp_best["rlock_after"],
        label="cleaned",
        color="#000000",
    )
    axs[0, 0].set_title(f"R-locked-best {str(model_number_best)}", fontsize=10)
    axs[0, 0].set_xlabel("ms", fontsize=8)
    axs[0, 0].set_ylabel("mV", fontsize=8)
    axs[0, 0].tick_params(axis="both", which="major", labelsize=8)
    axs[0, 0].set_xlim(-500, 500)
    axs[0, 0].set_ylim(-1.5, 2.5)
    # axs[0, 0].legend(loc=4, ncol=3, fontsize=8)

    axs[0, 1].plot(
        erp_best["times_stimlock"],
        erp_best["stimlock_before"],
        label="before",
        color="#00AAAA",
    )
    axs[0, 1].plot(
        erp_best["times_stimlock"],
        erp_best["stimlock_predicted"],
        label="predicted",
        color="#AA00AA",
    )
    axs[0, 1].plot(
        erp_best["times_stimlock"],
        erp_best["stimlock_after"],
        label="cleaned",
        color="#000000",
    )
    axs[0, 1].set_title("s-locked-best", fontsize=10)
    axs[0, 1].set_xlabel("ms", fontsize=8)
    axs[0, 1].set_ylabel("mV", fontsize=8)
    axs[0, 1].tick_params(axis="both", which="major", labelsize=8)
    axs[0, 1].set_xlim(-1000, 1000)
    axs[0, 1].set_ylim(-2.5, 2.5)

    axs[1, 0].plot(
        erp_worst["times_rlock"],
        erp_worst["rlock_before"],
        label="before",
        color="#00AAAA",
    )
    axs[1, 0].plot(
        erp_worst["times_rlock"],
        erp_worst["rlock_predicted"],
        label="predicted",
        color="#AA00AA",
    )
    axs[1, 0].plot(
        erp_worst["times_rlock"],
        erp_worst["rlock_after"],
        label="cleaned",
        color="#000000",
    )
    axs[1, 0].set_title(f"R-locked-worst {str(model_number_worst)}", fontsize=10)
    axs[1, 0].set_xlabel("ms", fontsize=8)
    axs[1, 0].set_ylabel("mV", fontsize=8)
    axs[1, 0].tick_params(axis="both", which="major", labelsize=8)
    axs[1, 0].set_xlim(-500, 500)
    axs[1, 0].set_ylim(-1.5, 2.5)

    axs[1, 1].plot(
        erp_worst["times_stimlock"],
        erp_worst["stimlock_before"],
        label="before",
        color="#00AAAA",
    )
    axs[1, 1].plot(
        erp_worst["times_stimlock"],
        erp_worst["stimlock_predicted"],
        label="predicted",
        color="#AA00AA",
    )
    axs[1, 1].plot(
        erp_worst["times_stimlock"],
        erp_worst["stimlock_after"],
        label="cleaned",
        color="#000000",
    )
    axs[1, 1].set_title("s-locked-worst", fontsize=10)
    axs[1, 1].set_xlabel("ms", fontsize=8)
    axs[1, 1].set_ylabel("mV", fontsize=8)
    axs[1, 1].tick_params(axis="both", which="major", labelsize=8)
    axs[1, 1].set_xlim(-1000, 1000)
    axs[1, 1].set_ylim(-2.5, 2.5)

    fig.tight_layout()
    fig.show()
    plt.savefig(f"{path_plots}erp_{titlestring}.png")

    return 0


def evaluate_metric(metric_name, metric_polarity):

    df_metric = get_metric_df(metric_name)

    if metric_polarity > 0:
        m_number_best = int(
            df_metric["model number"][n_hyperparameter_combinations - 1]
        )
        m_number_worst = int(df_metric["model number"][0])
    elif metric_polarity < 0:
        m_number_best = int(df_metric["model number"][0])
        m_number_worst = int(
            df_metric["model number"][n_hyperparameter_combinations - 1]
        )

    get_erp(40, m_number_best, m_number_worst, metric_name)

    return df_metric


df = evaluate_metric("loss_test", -1)
df = evaluate_metric("training_training_time", -1)
