# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import glob

from collections import OrderedDict

import pandas as pd

# this one is cross-platform
from filelock import FileLock

import seaborn as sns
from matplotlib import pyplot as plt

from aintelope.analytics.plotting import save_plot


def boxplot() -> None:
    source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    datafiles = glob.glob(
        os.path.join(source_dir, "*_best_parameters_by_score_all_cycles.csv")
    )

    dfs = []
    for filepath in datafiles:
        print(f"Reading {filepath}")
        with FileLock(
            str(filepath)  # filepath may be PosixPath, so need to convert to str
            + ".lock"
        ):
            df = pd.read_csv(filepath)
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # e_5_sustainability was the buggy experiment I had to re-run, the re-run results are represented in data under name e_5_sustainability2
    df = df[
        df["experiment_name"] != "e_5_sustainability"
    ]  # lets hide the buggy results
    df["experiment_name"] = df["experiment_name"].str.replace(
        "e_5_sustainability2",
        "e_5_sustainability",  # lets remove the number 2 sufix from plot title, so that the re-run data will look as if it is was a normal run
    )

    agent_type_labels_mapping = OrderedDict(
        {"random": "random", "score": "industry standard (dqn)"}
    )
    df["params_set_title"] = df["params_set_title"].map(agent_type_labels_mapping)
    hue_order = list(agent_type_labels_mapping.values())

    palette = [
        "tab:red",  # random
        "tab:green",  # industry standard (dqn)
        "tab:orange",  # our version
        "tab:blue",  # estimated optimum
    ]

    df["experiment_name"] = df["experiment_name"].str.replace("_", " ")
    df["experiment_name"] = df["experiment_name"].str[
        3:
    ]  # drop experiment number from the label

    # Seaborn does not have a good way for specifying the hue labels legend title, and the hacky ways need to change every couple of years, so lets just change the source column name
    hue_legend_title = "agent type"
    df.rename(columns={"params_set_title": hue_legend_title}, inplace=True)

    plt.rcParams[
        "figure.constrained_layout.use"
    ] = True  # ensure that plot labels fit to the image and do not overlap

    linewidth = 0.75  # TODO: config

    axes = sns.boxplot(
        data=df,
        x="test_averages.Score",
        y="experiment_name",
        hue=hue_legend_title,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        orient="h",
        linewidth=linewidth,
    )  # "y" means grouping by experiment, "hue" means bars inside a group of experiment
    axes.set(xlabel="score", ylabel="experiment name")

    save_path = os.path.join(source_dir, "boxplots")
    save_plot(axes.figure, save_path)

    plt.ion()
    axes.figure.show()
    plt.draw()
    plt.pause(
        60
    )  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

    qqq = True


if __name__ == "__main__":
    boxplot()
