# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import sys
import copy
import distutils
import shutil
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

# from command_runner.elevate import elevate

import hydra
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten, unflatten
from flatten_dict.reducers import make_reducer

# this one is cross-platform
from filelock import FileLock

import seaborn as sns
from matplotlib import pyplot as plt

from aintelope.analytics.plotting import save_plot, maximise_plot
from aintelope.config.config_utils import register_resolvers
from aintelope.utils import wait_for_enter, try_df_to_csv_write, RobustProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "aintelope", "config"),
    config_name="config_experiment",
)
def gridsearch_analytics(cfg: DictConfig) -> None:
    # TODO: command line arguments
    gridsearch_cycle_count = 10  # max cycle count   # TODO: read from config
    eval_cycle_count = 100  # gridsearch_cycle_count if gridsearch_cycle_count is not None else 25    # min cycle count       # TODO: read from config

    create_specialised_pipeline_config_with_best_parameters = True
    copy_log_files_of_best_parameter_combinations_to_separate_folder = False

    compute_average_evals_scores_per_parameter_combination = False  # if set to false then all cycles of evals with best gridsearch parameter combination are outputted in separate rows of the output CSV file
    create_box_plots = False  # if you set this to True then you need to set compute_average_evals_scores_per_parameter_combination = False
    return_n_best_combinations = (
        1  # used when compute_average_evals_scores_per_parameter_combination == True
    )

    # TODO: warn when there is less data than gridsearch_cycle_count or eval_cycle_count require

    # TODO: refactor into a shared method
    # TODO: automatically select correct gridsearch config file based on main cfg
    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    # if gridsearch_config is None:
    #    gridsearch_config = "initial_config_gridsearch.yaml"
    initial_config_gridsearch = OmegaConf.load(
        os.path.join("aintelope", "config", gridsearch_config_file)
    )

    OmegaConf.update(cfg, "hparams", initial_config_gridsearch.hparams, force_add=True)

    use_separate_parameters_for_each_experiment = (
        cfg.hparams.use_separate_models_for_each_experiment
    )
    use_separate_parameters_for_each_experiment = (
        True  # TODO: override function argument
    )

    if not compute_average_evals_scores_per_parameter_combination:
        return_n_best_combinations = 1

    # extract list parameters and compute cross product over their values
    # dict_config = OmegaConf.to_container(initial_config_gridsearch, resolve=True) # convert DictConfig to dict # NB! DO resolve references here since we DO want to handle references to lists as lists
    dict_config = OmegaConf.to_container(
        initial_config_gridsearch, resolve=False
    )  # convert DictConfig to dict # NB! do NOT resolve references here since we do NOT want to handle references to lists as lists. Gridsearch should loop over each list only once.
    flattened_config = flatten(
        dict_config, reducer=make_reducer(delimiter=".")
    )  # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    list_entries = OrderedDict(
        sorted(
            {
                key: value
                for key, value in flattened_config.items()
                if isinstance(value, list)
                or value
                is None  # value is None means that a column was previously part of grid search, but has been fixed to a best value per experiment
            }.items()
        )
    )  # select only entries of list type
    list_entries[
        "hparams.gridsearch_trial_no"
    ] = (
        initial_config_gridsearch.hparams.gridsearch_trial_no
    )  # this is an OmegaConf resolver that generates a list

    # preserve only the config parameters that were NOT searched over
    # TODO: is this needed?
    flattened_config_without_gridsearch_keys = dict(
        flattened_config
    )  # Omegaconf does not have entry removal method, so we need to work with dict
    gridsearch_cols = set()
    non_gridsearch_cols = set()
    for key, value in list_entries.items():
        if (
            value is None or len(value) > 1
        ):  # value is None means that a column was previously part of grid search, but has been fixed to a best value per experiment
            del flattened_config_without_gridsearch_keys[key]
            gridsearch_cols.add(key)
        else:
            flattened_config_without_gridsearch_keys[key] = value[0]

    unflattened_config_without_gridsearch_keys = unflatten(
        flattened_config_without_gridsearch_keys, splitter="dot"
    )
    config_gridsearch_without_lists = OmegaConf.create(
        unflattened_config_without_gridsearch_keys
    )

    test_summaries = []
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)

        # parts = os.path.splitext(aggregated_results_file)
        # aggregated_results_file2 = parts[0] + "_recalculated" + parts[1]
        # if False and os.path.exists(aggregated_results_file2):
        #    aggregated_results_file = aggregated_results_file2
        #    print(f"Using recalculated results file: {aggregated_results_file2}")
        # else:
        print(f"Using results file: {aggregated_results_file}")

        if os.path.exists(aggregated_results_file):
            aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
            with aggregated_results_file_lock:
                with open(aggregated_results_file, mode="r", encoding="utf-8") as fh:
                    data = fh.read()

            lines = data.split("\n")
            with RobustProgressBar(
                max_value=len(lines), granularity=10
            ) as bar:  # this takes a few moments of time
                for line_index, line in enumerate(lines):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    test_summary = json.loads(line)
                    # make nested dictionaries into one level
                    flattened_test_summary = flatten(
                        test_summary, reducer=make_reducer(delimiter=".")
                    )
                    for key, value in flattened_test_summary.items():
                        if isinstance(value, list):
                            flattened_test_summary[key] = str(
                                value
                            )  # pandas cannot handle list types when they are used as groupby keys
                        elif not isinstance(value, str) and np.isnan(
                            value
                        ):  # in current setup the values can overflow only in -inf direction    # TODO: remove nan-s in the main program
                            flattened_test_summary[key] = -np.inf

                    test_summaries.append(flattened_test_summary)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file not found")
    else:
        raise Exception("Aggregated results file not configured")

    if len(test_summaries) == 0:
        raise Exception("Aggregated results file is empty")

    df = pd.DataFrame.from_records(test_summaries)

    score_cols = [
        col
        for col in df.columns
        if col.startswith("test_totals.")
        or col.startswith("test_averages.")
        or col.startswith("test_variances.")
        # per score dimension results
        or col.startswith("test_sfella_totals.")
        or col.startswith("test_sfella_averages.")
        or col.startswith("test_sfella_variances.")
        # TODO: rename to test_*
        or col
        in ["sfella_score_total", "sfella_score_average", "sfella_score_variance"]
    ]

    df[score_cols] = df[score_cols].fillna(
        0
    )  # fill missing score dimensions with zeros

    # columns in file:

    # "timestamp", "experiment_name", "title", "params_set_title", "gridsearch_params", "score_dimensions", "test_totals", "test_averages", "test_variances", "test_sfella_totals", "test_sfella_averages", "test_sfella_variances"

    evals_parameter_grouping_cols = [
        col
        for col in df.columns
        if
        # consider experiment name in the output groupings, but not necessarily during best parameter selection
        (
            col == "experiment_name"
        )  # TODO: add also params_set_title to the grouping cols
        or (
            col.startswith("gridsearch_params.")
            and not col.endswith(
                ".gridsearch_trial_no"
            )  # calculate mean over gridsearch_trial_no
        )
    ]

    gridsearch_parameter_grouping_cols = [
        col
        for col in df.columns
        if
        # consider experiment name in the output groupings, but not necessarily during best parameter selection
        (
            use_separate_parameters_for_each_experiment and col == "experiment_name"
        )  # TODO: add also params_set_title to the grouping cols
        or (
            col.startswith("gridsearch_params.")
            and not col.endswith(
                ".gridsearch_trial_no"
            )  # calculate mean over gridsearch_trial_no
        )
    ]

    df_pre_filtering_cycle_count_transforms = [
        df,
        df.groupby(evals_parameter_grouping_cols, sort=False)
        .transform(  # NB! here we transform(), not agg() so that the input rows are preserved and just a new column is added
            "size"
        )  # count rows per group
        .rename("evals_count_per_experiment", inplace=False),
    ]
    df = pd.concat(
        df_pre_filtering_cycle_count_transforms,  # NB! Here we do not reindex, else it would not be possible to apply gridsearch_mask to gridsearch_df_params_and_results. Also we do not need to reindex here since this list contains only transforms and not aggregations.
        axis=1,
    )

    df = df[
        df["gridsearch_params.hparams.gridsearch_trial_no"] < eval_cycle_count
    ]  # ignore results past eval_cycle_count
    # df = df[
    #    df["gridsearch_params.hparams.env_params.map_max"] == 7
    # ]  # TODO: function argument

    if (
        gridsearch_cycle_count is not None
    ):  # keep only rows up to given trial no in gridsearch df
        gridsearch_df = df[
            (
                df["gridsearch_params.hparams.gridsearch_trial_no"]
                < gridsearch_cycle_count
            )
            & (
                df["evals_count_per_experiment"] >= eval_cycle_count
            )  # select only gridsearch results that have also sufficient evals cycle count available
        ]
    else:
        gridsearch_df = df

    gridsearch_df_params_and_results = gridsearch_df[
        gridsearch_parameter_grouping_cols
        + ["test_averages.Score", "sfella_score_average"]
    ]
    gridsearch_df_params_and_results_with_experiment_name = gridsearch_df[
        gridsearch_parameter_grouping_cols + ["experiment_name"]
    ]
    gridsearch_df_params_and_results_with_timestamp_pid_uuid = gridsearch_df[
        gridsearch_parameter_grouping_cols + ["timestamp_pid_uuid"]
    ]

    evals_params_and_results_cols = (
        evals_parameter_grouping_cols
        + (
            ["gridsearch_params.hparams.gridsearch_trial_no", "timestamp_pid_uuid"]
            if not compute_average_evals_scores_per_parameter_combination
            else []
        )
        + score_cols
    )
    evals_df_params_and_results = df[evals_params_and_results_cols]
    evals_df_params_and_results_with_timestamp_pid_uuid = df[
        evals_params_and_results_cols
        + (
            # add "timestamp_pid_uuid" column only if not already as part of evals_params_and_results_cols, else duplicate column would appear
            ["timestamp_pid_uuid"]
            if "timestamp_pid_uuid" not in evals_params_and_results_cols
            else []
        )
    ]

    score_cols_mean_renames = {col: "mean." + col for col in score_cols}
    score_cols_std_renames = {col: "std." + col for col in score_cols}

    if not use_separate_parameters_for_each_experiment:
        # NB! Ensure that all experiments have sufficient cycle count.
        # Evan if we will find common parameters over all experiments, we still need to ensure that all
        # experiments with a given parameter set have sufficient count of cycles.
        # In case of "not use_separate_parameters_for_each_experiment" this filtering can be done only
        # before running aggregating evals_aggregations on gridsearch data.

        gridsearch_cycle_count_transforms = [
            gridsearch_df_params_and_results_with_experiment_name[
                gridsearch_parameter_grouping_cols + ["experiment_name"]
            ],
            gridsearch_df_params_and_results_with_experiment_name.groupby(
                gridsearch_parameter_grouping_cols + ["experiment_name"], sort=False
            )
            .transform(  # NB! here we transform(), not agg() so that the input rows are preserved and just a new column is added
                "size"
            )  # count rows per group
            .rename("count_per_experiment", inplace=False),
        ]
        gridsearch_df_params_and_results_with_experiment_name = pd.concat(
            gridsearch_cycle_count_transforms,  # NB! Here we do not reindex, else it would not be possible to apply gridsearch_mask to gridsearch_df_params_and_results. Also we do not need to reindex here since this list contains only transforms and not aggregations.
            axis=1,
        )

        gridsearch_mask = (
            gridsearch_df_params_and_results_with_experiment_name[
                "count_per_experiment"
            ]
            >= gridsearch_cycle_count
        )
        gridsearch_df_params_and_results = gridsearch_df_params_and_results[
            gridsearch_mask
        ]
        gridsearch_df_params_and_results_with_experiment_name = (
            gridsearch_df_params_and_results_with_experiment_name[gridsearch_mask]
        )
        gridsearch_df_params_and_results_with_timestamp_pid_uuid = (
            gridsearch_df_params_and_results_with_timestamp_pid_uuid[gridsearch_mask]
        )

    # / if not use_separate_parameters_for_each_experiment:

    gridsearch_aggregations = [
        # preserve the groupby cols after aggregation by mean(). See https://stackoverflow.com/questions/40139184/keeping-key-column-when-using-groupby-with-transform-in-pandas
        gridsearch_df_params_and_results[
            gridsearch_parameter_grouping_cols
        ].drop_duplicates(),
        gridsearch_df_params_and_results.groupby(
            gridsearch_parameter_grouping_cols, sort=False
        )
        .agg("size")  # count rows per group  # count rows per group
        .rename("count", inplace=False),
        gridsearch_df_params_and_results.groupby(
            gridsearch_parameter_grouping_cols, sort=False
        )
        .agg("mean")  # returns averaged score_cols
        .rename(columns=score_cols_mean_renames, inplace=False),
        gridsearch_df_params_and_results.groupby(
            gridsearch_parameter_grouping_cols, sort=False
        )
        .agg("std", ddof=0)  # TODO: ddof
        .rename(columns=score_cols_std_renames, inplace=False),
        gridsearch_df_params_and_results_with_timestamp_pid_uuid.groupby(
            gridsearch_parameter_grouping_cols, sort=False
        )["timestamp_pid_uuid"]
        .apply(list)
        .rename("timestamp_pid_uuids", inplace=False),
    ]

    if (
        not use_separate_parameters_for_each_experiment
    ):  # count the number of different experiments in each parameter combination
        gridsearch_aggregations += [
            gridsearch_df_params_and_results_with_experiment_name.groupby(
                gridsearch_parameter_grouping_cols, sort=False
            )["experiment_name"]
            .agg("nunique")
            .rename("nunique_experiment_name", inplace=False),
            gridsearch_df_params_and_results_with_experiment_name.groupby(
                gridsearch_parameter_grouping_cols, sort=False
            )["experiment_name"]
            .apply(set)
            .rename("experiment_names", inplace=False),  # this is used for debugging
        ]

    gridsearch_averages_per_parameter_combination = pd.concat(
        [x.reset_index(drop=True) for x in gridsearch_aggregations],
        axis=1,
    )
    gridsearch_averages_per_parameter_combination["score_mean_minus_std"] = (
        gridsearch_averages_per_parameter_combination["mean.test_averages.Score"]
        - gridsearch_averages_per_parameter_combination["std.test_averages.Score"]
    )

    if use_separate_parameters_for_each_experiment:
        gridsearch_averages_per_parameter_combination = (
            gridsearch_averages_per_parameter_combination[
                gridsearch_averages_per_parameter_combination["count"]
                >= gridsearch_cycle_count
            ]
        )

    else:
        # check how many unique experiment names remained per gridsearch parameter combination after removing rows with too few eval_cycle_count

        # if "not use_separate_parameters_for_each_experiment" then the count-based filtering is already done above before aggregation

        max_nunique_experiment_name_count = evals_df_params_and_results[
            "experiment_name"
        ].nunique()

        # AFTER removing low-cycle count gridsearch groups, keep only gridsearch parameter combinations that have same maximum unique experiment count as maximum unique experiment count of eval dataset
        gridsearch_averages_per_parameter_combination = gridsearch_averages_per_parameter_combination[
            gridsearch_averages_per_parameter_combination["nunique_experiment_name"]
            == max_nunique_experiment_name_count
        ]  # Even though above code line already multiplies by max_nunique_experiment_name_count, we need to recheck here. It may happen that some gridsearch has many cycles on one experiment (more than gridsearch_cycle_count parameter), but is missing cycles on other environment

    # / if not use_separate_parameters_for_each_experiment:

    if compute_average_evals_scores_per_parameter_combination:
        evals_aggregations = [
            # preserve the groupby cols after aggregation by mean(). See https://stackoverflow.com/questions/40139184/keeping-key-column-when-using-groupby-with-transform-in-pandas
            evals_df_params_and_results[
                evals_parameter_grouping_cols
            ].drop_duplicates(),
            evals_df_params_and_results.groupby(
                evals_parameter_grouping_cols, sort=False
            )
            .agg("size")  # count rows per group
            .rename("count", inplace=False),
            evals_df_params_and_results.groupby(
                evals_parameter_grouping_cols, sort=False
            )
            .agg("mean")  # returns averaged score_cols
            .rename(columns=score_cols_mean_renames, inplace=False),
            evals_df_params_and_results.groupby(
                evals_parameter_grouping_cols, sort=False
            )
            .agg("std", ddof=0)  # TODO: ddof
            .rename(columns=score_cols_std_renames, inplace=False),
            evals_df_params_and_results_with_timestamp_pid_uuid.groupby(
                evals_parameter_grouping_cols, sort=False
            )["timestamp_pid_uuid"]
            .apply(list)
            .rename("timestamp_pid_uuids", inplace=False),
        ]

        # TODO: refactor this into a separate helper function since this code does not seem to be quite straightforward, but is a general use case
        evals_averages_per_parameter_combination = pd.concat(
            [x.reset_index(drop=True) for x in evals_aggregations],
            axis=1,
        )

        # keep only parameter combinations with sufficient cycle count
        evals_averages_per_parameter_combination = (
            evals_averages_per_parameter_combination[
                evals_averages_per_parameter_combination["count"] >= eval_cycle_count
            ]
        )

        evals_averages_per_parameter_combination["score_mean_minus_std"] = (
            evals_averages_per_parameter_combination["mean.test_averages.Score"]
            - evals_averages_per_parameter_combination["std.test_averages.Score"]
        )

    else:  # / if compute_average_evals_scores_per_parameter_combination:
        evals_cycle_count_transforms = [
            evals_df_params_and_results,  # [evals_parameter_grouping_cols],
            evals_df_params_and_results.groupby(
                evals_parameter_grouping_cols, sort=False
            )
            .transform(  # NB! here we transform(), not agg() so that the input rows are preserved and just a new column is added
                "size"
            )  # count rows per group
            .rename("count_per_experiment", inplace=False),
            evals_df_params_and_results_with_timestamp_pid_uuid["timestamp_pid_uuid"]
            .map(
                lambda x: [x]
            )  # convert uuid to single-entry list in order to maintain same format as in case of compute_average_evals_scores_per_parameter_combination == True
            .rename("timestamp_pid_uuids", inplace=False),
        ]
        evals_averages_per_parameter_combination = pd.concat(
            evals_cycle_count_transforms,  # NB! Here we do not reindex, else it would not be possible to apply gridsearch_mask to gridsearch_df_params_and_results. Also we do not need to reindex here since this list contains only transforms and not aggregations.
            axis=1,
        )

        # keep only parameter combinations with sufficient cycle count
        evals_averages_per_parameter_combination = (
            evals_averages_per_parameter_combination[
                evals_averages_per_parameter_combination["count_per_experiment"]
                >= eval_cycle_count
            ]
        )

    # / if compute_average_evals_scores_per_parameter_combination:

    # group by columns: hparams, map_max, experiment_name

    # TODO: in case multiple rows have same best value, take all of them

    evals_environment_grouping_dims = ["gridsearch_params.hparams.env_params.map_max"]
    evals_environment_grouping_dims.append("experiment_name")
    if not compute_average_evals_scores_per_parameter_combination:
        evals_environment_grouping_dims += [
            "gridsearch_params.hparams.gridsearch_trial_no",
            "timestamp_pid_uuid",
        ]

    gridsearch_environment_grouping_dims = [
        "gridsearch_params.hparams.env_params.map_max"
    ]
    if use_separate_parameters_for_each_experiment:
        gridsearch_environment_grouping_dims.append("experiment_name")

    # select evals rows that have same parameter sets as best gridsearch parameter sets

    test = (
        gridsearch_averages_per_parameter_combination.groupby(
            gridsearch_environment_grouping_dims, sort=False
        )
        .agg("max")
        .reset_index(drop=True)
    )  # for debugging

    gridsearch_best_parameters_by_score_sorted = (
        gridsearch_averages_per_parameter_combination.sort_values(
            by="mean.test_averages.Score", ascending=False
        )
    )
    gridsearch_best_parameters_by_score = (
        gridsearch_best_parameters_by_score_sorted.groupby(
            gridsearch_environment_grouping_dims, sort=False
        )
        .head(return_n_best_combinations)
        .sort_values(
            gridsearch_environment_grouping_dims + ["mean.test_averages.Score"],
            ascending=([True] * len(gridsearch_environment_grouping_dims) + [False]),
        )
    )  # sorting is needed in case of writing specialised config file(s) at the end. TODO: If two rows have same scores then it might happen that the order of rows in evals-based CSV file and in gridsearch-based config files is different (more precisely, first row may end up in config file with suffix _2 and second row in config file with suffix _1).

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(evals_averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            evals_averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_score[gridsearch_parameter_grouping_cols]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.test_averages.Score"] == gridsearch_best_parameters_by_score[mask]["mean.test_averages.Score"].item()
                rows.append(row)
            else:
                qqq = True  # for debugging
            bar.update(index + 1)
    if (
        use_separate_parameters_for_each_experiment
        and compute_average_evals_scores_per_parameter_combination
        and return_n_best_combinations == 1
        # and gridsearch_cycle_count == eval_cycle_count
    ):
        assert len(rows) == len(gridsearch_best_parameters_by_score)
    evals_best_parameters_by_score = pd.concat(rows, axis=1).transpose()
    # evals_best_parameters_by_score = evals_averages_per_parameter_combination.loc[best_parameters_by_score_row_indexes]
    # assert list(test["mean.test_averages.Score"]) == list(evals_best_parameters_by_score["mean.test_averages.Score"])

    gridsearch_best_parameters_by_sfella_score_sorted = (
        gridsearch_averages_per_parameter_combination.sort_values(
            by="mean.sfella_score_average", ascending=False
        )
    )
    gridsearch_best_parameters_by_sfella_score = (
        gridsearch_best_parameters_by_sfella_score_sorted.groupby(
            gridsearch_environment_grouping_dims, sort=False
        )
        .head(return_n_best_combinations)
        .sort_values(
            gridsearch_environment_grouping_dims + ["mean.sfella_score_average"],
            ascending=([True] * len(gridsearch_environment_grouping_dims) + [False]),
        )
    )  # sorting is needed in case of writing specialised config file(s) at the end. TODO: If two rows have same scores then it might happen that the order of rows in evals-based CSV file and in gridsearch-based config files is different (more precisely, first row may end up in config file with suffix _2 and second row in config file with suffix _1).

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(evals_averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            evals_averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_sfella_score[
                    gridsearch_parameter_grouping_cols
                ]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.sfella_score_average"] == gridsearch_best_parameters_by_sfella_score[mask]["mean.sfella_score_average"].item()
                rows.append(row)
            else:
                qqq = True  # for debugging
            bar.update(index + 1)
    if (
        use_separate_parameters_for_each_experiment
        and compute_average_evals_scores_per_parameter_combination
        and return_n_best_combinations == 1
        and gridsearch_cycle_count == eval_cycle_count
    ):
        assert len(rows) == len(gridsearch_best_parameters_by_sfella_score)
    evals_best_parameters_by_sfella_score = pd.concat(rows, axis=1).transpose()
    # evals_best_parameters_by_sfella_score = evals_averages_per_parameter_combination.loc[best_parameters_by_sfella_score_row_indexes]

    if compute_average_evals_scores_per_parameter_combination:
        # if evals_best_parameters_by_sfella_score contain -inf values then ignore these rows
        infinities = (
            evals_best_parameters_by_sfella_score["mean.sfella_score_average"]
            == -np.inf
        )
        evals_best_parameters_by_sfella_score = (
            evals_best_parameters_by_sfella_score.drop(
                evals_best_parameters_by_sfella_score[infinities].index
            )
        )
        # assert list(test["mean.sfella_score_average"]) == list(evals_best_parameters_by_sfella_score["mean.sfella_score_average"])

    gridsearch_best_parameters_by_score_minus_std_sorted = (
        gridsearch_averages_per_parameter_combination.sort_values(
            by="score_mean_minus_std", ascending=False
        )
    )
    gridsearch_best_parameters_by_score_minus_std = (
        gridsearch_best_parameters_by_score_minus_std_sorted.groupby(
            gridsearch_environment_grouping_dims, sort=False
        )
        .head(return_n_best_combinations)
        .sort_values(
            gridsearch_environment_grouping_dims + ["score_mean_minus_std"],
            ascending=([True] * len(gridsearch_environment_grouping_dims) + [False]),
        )
    )  # sorting is needed in case of writing specialised config file(s) at the end. TODO: If two rows have same scores then it might happen that the order of rows in evals-based CSV file and in gridsearch-based config files is different (more precisely, first row may end up in config file with suffix _2 and second row in config file with suffix _1).

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(evals_averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            evals_averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_score_minus_std[
                    gridsearch_parameter_grouping_cols
                ]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.sfella_score_average"] == gridsearch_best_parameters_by_sfella_score[mask]["mean.sfella_score_average"].item()
                rows.append(row)
            else:
                qqq = True  # for debugging
            bar.update(index + 1)
    if (
        use_separate_parameters_for_each_experiment
        and compute_average_evals_scores_per_parameter_combination
        and return_n_best_combinations == 1
        and gridsearch_cycle_count == eval_cycle_count
    ):
        assert len(rows) == len(gridsearch_best_parameters_by_score_minus_std)
    evals_best_parameters_by_score_minus_std = pd.concat(rows, axis=1).transpose()
    # evals_best_parameters_by_score_minus_std = evals_averages_per_parameter_combination.loc[best_parameters_by_score_minus_std_row_indexes]
    # assert list(test["score_mean_minus_std"]) == list(best_parameters_by_score_minus_std_row_indexes["score_mean_minus_std"])

    # / # select evals rows that have same parameter sets as best gridsearch parameter sets

    # copy log files of selected grid search and evals results to a separate output folder
    if copy_log_files_of_best_parameter_combinations_to_separate_folder:
        log_dir_root = os.path.normpath(
            cfg["log_dir_root"]
        )  # os.path.normpath removes any trailing slashes
        log_folder_copy_target = (
            log_dir_root + "_" + cfg.hparams.params_set_title + "_selected"
        )  # TODO: function argument for the suffix

        os.makedirs(log_folder_copy_target, exist_ok=True)

        # TODO: config option for including all gridsearch combinations (before selecting the best ones) into the selected logs folder

        for pd_index, eval_row in evals_best_parameters_by_score.iterrows():
            experiment_name = eval_row["experiment_name"]
            for timestamp_pid_uuid in eval_row["timestamp_pid_uuids"]:
                copy_log_folder(
                    experiment_name,
                    timestamp_pid_uuid,
                    log_dir_root,
                    log_folder_copy_target,
                    cfg,
                )

    # keep only selected columns in the CSV output

    if compute_average_evals_scores_per_parameter_combination:
        total_score_dims = [
            "mean.test_averages.Score",
            "std.test_averages.Score",
            "mean.sfella_score_average",
            "std.sfella_score_average",
            "score_mean_minus_std",
        ]
        reward_dims = [
            "mean.test_averages.Reward",
            "std.test_averages.Reward",
            "mean.test_sfella_averages.Reward",
            "std.test_sfella_averages.Reward",
        ]
        average_score_subdims = [
            col
            for col in evals_averages_per_parameter_combination.columns
            if (
                col.startswith("mean.test_averages.") or col == "score_mean_minus_std"
            )  # or col.startswith("std.test_averages."))
            and col not in total_score_dims
            and col not in reward_dims
        ]
        sfella_average_score_subdims = [
            col
            for col in evals_averages_per_parameter_combination.columns
            if (
                col.startswith("mean.test_sfella_averages.")
            )  # or col.startswith("std.test_sfella_averages."))
            and col not in total_score_dims
            and col not in reward_dims
        ]

    else:  # / if compute_average_evals_scores_per_parameter_combination:
        total_score_dims = ["test_averages.Score", "sfella_score_average"]
        reward_dims = ["test_averages.Reward", "test_sfella_averages.Reward"]
        average_score_subdims = [
            col
            for col in evals_averages_per_parameter_combination.columns
            if col.startswith("test_averages.")
            and col not in total_score_dims
            and col not in reward_dims
        ]
        sfella_average_score_subdims = [
            col
            for col in evals_averages_per_parameter_combination.columns
            if col.startswith("test_sfella_averages.")
            and col not in total_score_dims
            and col not in reward_dims
        ]

    # / if compute_average_evals_scores_per_parameter_combination:

    gridsearch_cols = list(gridsearch_cols)
    gridsearch_cols.sort()
    gridsearch_cols = [
        "gridsearch_params." + col
        for col in gridsearch_cols
        if not col.endswith(".gridsearch_trial_no")  # this one is averaged out
        and ("gridsearch_params." + col)
        not in evals_environment_grouping_dims  # map_max, experiment_name
    ]

    non_gridsearch_cols = list(flattened_config_without_gridsearch_keys.keys())
    non_gridsearch_cols.sort()
    non_gridsearch_cols = [
        "gridsearch_params." + col
        for col in non_gridsearch_cols
        if ("gridsearch_params." + col)
        not in evals_environment_grouping_dims  # exclude map_max, experiment_name here as well, even if they happen to be not part of gridsearch, else we get duplicate columns below
    ]

    evals_best_parameters_by_score = evals_best_parameters_by_score[
        evals_environment_grouping_dims
        + gridsearch_cols
        + [
            "count"
            if compute_average_evals_scores_per_parameter_combination
            else "count_per_experiment"
        ]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    evals_best_parameters_by_sfella_score = evals_best_parameters_by_sfella_score[
        evals_environment_grouping_dims
        + gridsearch_cols
        + [
            "count"
            if compute_average_evals_scores_per_parameter_combination
            else "count_per_experiment"
        ]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    evals_best_parameters_by_score_minus_std = evals_best_parameters_by_score_minus_std[
        evals_environment_grouping_dims
        + gridsearch_cols
        + [
            "count"
            if compute_average_evals_scores_per_parameter_combination
            else "count_per_experiment"
        ]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    evals_best_parameters_by_score = evals_best_parameters_by_score.sort_values(
        by=evals_environment_grouping_dims
        + (
            ["mean.test_averages.Score"]
            if compute_average_evals_scores_per_parameter_combination
            else ["test_averages.Score"]
        ),
        ascending=([True] * len(evals_environment_grouping_dims) + [False]),
    )

    evals_best_parameters_by_sfella_score = (
        evals_best_parameters_by_sfella_score.sort_values(
            by=evals_environment_grouping_dims
            + (
                ["mean.sfella_score_average"]
                if compute_average_evals_scores_per_parameter_combination
                else ["sfella_score_average"]
            ),
            ascending=([True] * len(evals_environment_grouping_dims) + [False]),
        )
    )

    evals_best_parameters_by_score_minus_std = (
        evals_best_parameters_by_score_minus_std.sort_values(
            by=evals_environment_grouping_dims
            + (
                ["score_mean_minus_std"]
                if compute_average_evals_scores_per_parameter_combination
                else []
            ),
            ascending=(
                [True] * len(evals_environment_grouping_dims)
                + (
                    [False]
                    if compute_average_evals_scores_per_parameter_combination
                    else []
                )
            ),
        )
    )

    if (
        True
    ):  # keep only config parameter names, remove their path to make column names shorter in the CSV file   # TODO: config option
        renames = {
            col: col.split(".")[-1]
            for col in evals_best_parameters_by_score.columns
            if col.startswith("gridsearch_params.")
        }
        best_parameters_by_score_shortened = evals_best_parameters_by_score.rename(
            columns=renames, inplace=False
        )

        renames = {
            col: col.split(".")[-1]
            for col in evals_best_parameters_by_sfella_score.columns
            if col.startswith("gridsearch_params.")
        }
        best_parameters_by_sfella_score_shortened = (
            evals_best_parameters_by_sfella_score.rename(columns=renames, inplace=False)
        )
        best_parameters_by_score_minus_std_shortened = (
            evals_best_parameters_by_score_minus_std.rename(
                columns=renames, inplace=False
            )
        )
    else:
        best_parameters_by_score_shortened = evals_best_parameters_by_score
        best_parameters_by_sfella_score_shortened = (
            evals_best_parameters_by_sfella_score
        )
        best_parameters_by_score_minus_std_shortened = (
            evals_best_parameters_by_score_minus_std
        )

    filepath = (
        cfg.hparams.params_set_title
        + "_best_parameters_by_score"
        + (
            "_all_cycles"
            if not compute_average_evals_scores_per_parameter_combination
            else ""
        )
        + ".csv"
    )
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_score_shortened, filepath, index=False, mode="w", header=True
    )

    filepath = (
        cfg.hparams.params_set_title
        + "_best_parameters_by_sfella_score"
        + (
            "_all_cycles"
            if not compute_average_evals_scores_per_parameter_combination
            else ""
        )
        + ".csv"
    )
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_sfella_score_shortened,
        filepath,
        index=False,
        mode="w",
        header=True,
    )

    filepath = (
        cfg.hparams.params_set_title
        + "_best_parameters_by_score_minus_std"
        + (
            "_all_cycles"
            if not compute_average_evals_scores_per_parameter_combination
            else ""
        )
        + ".csv"
    )
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_score_minus_std_shortened,
        filepath,
        index=False,
        mode="w",
        header=True,
    )

    if create_box_plots:
        axes = sns.boxplot(
            data=best_parameters_by_score_shortened,
            x="test_averages.Score",
            y="experiment_name",
            hue="params_set_title",
            showfliers=False,
            orient="h",
        )  # "y" means grouping by experiment, "hue" means bars inside a group of experiment

        source_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", ".."
        )
        save_path = os.path.join(source_dir, "boxplots")
        save_plot(axes.figure, save_path)

        plt.ion()
        maximise_plot()
        axes.figure.show()
        plt.draw()
        plt.pause(
            60
        )  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

    # / if create_box_plots:

    if create_specialised_pipeline_config_with_best_parameters:
        # create a new specialised pipeline config files based on the best parameters per experiment
        pipeline_config_file = os.environ.get("PIPELINE_CONFIG")
        using_default_pipeline_config_file = False
        if pipeline_config_file is None:
            pipeline_config_file = "config_pipeline.yaml"
            using_default_pipeline_config_file = True

        pipeline_config = OmegaConf.load(
            os.path.join("aintelope", "config", pipeline_config_file)
        )

        for combination_i in range(0, return_n_best_combinations):
            # TODO: ensure to do not use special pipeline config when doing initial gridsearch
            if using_default_pipeline_config_file:
                parts = os.path.splitext(pipeline_config_file)
                specialised_pipeline_config_file = (
                    parts[0]
                    + "_"
                    + cfg.hparams.params_set_title
                    + (
                        "_common"
                        if not use_separate_parameters_for_each_experiment
                        else ""
                    )
                    + (f"_{combination_i + 1}" if combination_i > 0 else "")
                    + parts[1]
                )
            else:
                parts = os.path.splitext(pipeline_config_file)
                specialised_pipeline_config_file = (
                    parts[0]
                    + f"_{combination_i + 1}"  # NB! If PIPELINE_CONFIG is specified then do not overwrite it. Therefore we need to append combination_i for combination_i == 0 as well.
                    + parts[1]
                )

            specialised_config_params_source = gridsearch_best_parameters_by_score  # TODO: config for choosing between gridsearch_best_parameters_by_score, gridsearch_best_parameters_by_sfella_score, gridsearch_best_parameters_by_score_minus_std
            specialised_config_params_source = (
                specialised_config_params_source.groupby(
                    gridsearch_environment_grouping_dims, sort=False
                )
                .nth(combination_i)
                .dropna()
            )  # dropna() will drop rows which match indexes missing in the source

            specialised_pipeline_config = copy.deepcopy(pipeline_config)
            for map_size in [
                7
            ]:  # TODO: config, or select all available map sizes from gridsearch config
                for env_conf_name in pipeline_config:
                    result_row_selector = (
                        specialised_config_params_source[
                            "gridsearch_params.hparams.env_params.map_max"
                        ]
                        == map_size
                    )
                    if "experiment_name" in evals_parameter_grouping_cols:
                        result_row_selector &= (
                            specialised_config_params_source["experiment_name"]
                            == env_conf_name
                        )

                    result_row = specialised_config_params_source[result_row_selector]

                    if (
                        len(result_row) > 0
                    ):  # check whether the grid search produced data for this map_max and experiment_name pair
                        key = env_conf_name + ".env_params.map_max"
                        value = map_size
                        OmegaConf.update(
                            specialised_pipeline_config, key, value, force_add=True
                        )

                        for gridsearch_col in gridsearch_cols:
                            key = (
                                env_conf_name
                                + "."
                                + gridsearch_col[len("gridsearch_params.hparams.") :]
                            )
                            value = result_row[gridsearch_col].item()
                            OmegaConf.update(
                                specialised_pipeline_config, key, value, force_add=True
                            )

            # TODO: confirm overwriting existing file
            print(f"\nWriting to {specialised_pipeline_config_file}")
            OmegaConf.save(
                specialised_pipeline_config,
                os.path.join("aintelope", "config", specialised_pipeline_config_file),
                resolve=False,
            )

        # / for combination_i in range(0, return_n_best_combinations):

    # / if create_specialised_pipeline_config_with_best_parameters:

    wait_for_enter("\nAnalytics done. Press [enter] to continue.")
    qqq = True


# / def gridsearch_analytics():


def copy_log_folder(experiment_name, source_uuid, source_base, dest_base, cfg):
    source_pipeline_log = os.path.abspath(os.path.join(source_base, source_uuid))
    dest_pipeline_log = os.path.join(dest_base, source_uuid)

    os.makedirs(dest_pipeline_log, exist_ok=True)

    # copy experiment logs

    source_experiment_log = os.path.join(source_pipeline_log, experiment_name)
    dest_experiment_log = os.path.join(dest_pipeline_log, experiment_name)
    if not os.path.exists(dest_experiment_log):
        os.symlink(source_experiment_log, dest_experiment_log, target_is_directory=True)
    else:
        qqq = True  # for debugging

    # copy hydra logs

    source_hydra_log = os.path.join(source_pipeline_log, cfg.hydra_logs_root)
    dest_hydra_log = os.path.join(dest_pipeline_log, cfg.hydra_logs_root)
    if not os.path.exists(dest_hydra_log):
        os.symlink(source_hydra_log, dest_hydra_log, target_is_directory=True)

    # copy plots

    source_plot = os.path.join(source_pipeline_log, "plot_" + experiment_name)
    dest_plot = os.path.join(dest_pipeline_log, "plot_" + experiment_name)

    if (
        os.name == "nt"
    ):  # symbolic links to files do not work well in Windows when zipping
        if not os.path.exists(dest_plot + ".png"):
            shutil.copyfile(source_plot + ".png", dest_plot + ".png")
        if not os.path.exists(dest_plot + ".svg"):
            shutil.copyfile(source_plot + ".svg", dest_plot + ".svg")
    else:
        if not os.path.exists(dest_plot + ".png"):
            os.symlink(
                source_plot + ".png", dest_plot + ".png", target_is_directory=False
            )
        if not os.path.exists(dest_plot + ".svg"):
            os.symlink(
                source_plot + ".svg", dest_plot + ".svg", target_is_directory=False
            )

    # copy code archives
    if False:  # TODO: function parameter
        source_code_archive = os.path.join(
            source_pipeline_log, "aintelope_code_archive.zip"
        )
        dest_code_archive = os.path.join(
            source_pipeline_log, "aintelope_code_archive.zip"
        )

        if not os.path.exists(dest_code_archive):
            if (
                os.name == "nt"
            ):  # symbolic links to files do not work well in Windows when zipping
                shutil.copyfile(source_code_archive, dest_code_archive)
            else:
                os.symlink(
                    source_code_archive, dest_code_archive, target_is_directory=False
                )

        source_code_archive = os.path.join(
            source_pipeline_log, "gridworlds_code_archive.zip"
        )
        dest_code_archive = os.path.join(
            source_pipeline_log, "gridworlds_code_archive.zip"
        )

        if not os.path.exists(dest_code_archive):
            if (
                os.name == "nt"
            ):  # symbolic links to files do not work well in Windows when zipping
                shutil.copyfile(source_code_archive, dest_code_archive)
            else:
                os.symlink(
                    source_code_archive, dest_code_archive, target_is_directory=False
                )

    # TODO: copy checkpoints


# / def copy_log_folder(experiment_name, source_uuid, source_base, dest_base, cfg):


def main():
    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    gridsearch_analytics()


if __name__ == "__main__":
    # TODO: do the elevation only when linking to log folders is activated
    debugging = sys.gettrace() is not None
    if os.name == "nt":
        if debugging:
            from ctypes import windll

            if not windll.shell32.IsUserAnAdmin():
                print(
                    "You need to start your debugger in elevated mode. Creating symbolic links to selected log folders will not be possible. If you cannot get Administrator permissions, the alternative is to let the Administrator to create user permission to create symbolic links: gpedit.msc -> Computer Configuration -> Windows Settings -> Security Settings -> Local Policies -> User Rights Assignment -> Create symbolic links."
                )
                wait_for_enter("\nPress [enter] to continue.")
            main()
        else:
            from ctypes import windll

            if not windll.shell32.IsUserAnAdmin():
                print(
                    "Not running as Administrator. Creating symbolic links to selected log folders will not be possible. If you cannot get Administrator permissions, the alternative is to let the Administrator to create user permission to create symbolic links: gpedit.msc -> Computer Configuration -> Windows Settings -> Security Settings -> Local Policies -> User Rights Assignment -> Create symbolic links."
                )
                wait_for_enter("\nPress [enter] to continue.")

            main()

            # comment-out: elevate() does not preserve environment variables. TODO: fix that

            # elevate(main)  # Needed to create symbolic links to selected log folders. See https://docs.python.org/3/library/os.html#os.symlink

            # TODO: reimplement elevate() in such a way that the parent process stays alive after elevate() exists so we can check whether the elevation succeeded.

            # try:
            #    elevate()  # Needed to create symbolic links to selected log folders. See https://docs.python.org/3/library/os.html#os.symlink
            # except Exception as ex:
            #    print(
            #        "Elevation failed, creating symbolic links to selected log folders will not be possible. If you cannot get Administrator permissions, the alternative is to let the Administrator to create user permission to create symbolic links: gpedit.msc -> Computer Configuration -> Windows Settings -> Security Settings -> Local Policies -> User Rights Assignment -> Create symbolic links."
            #    )
            #    wait_for_enter("\nPress [enter] to continue.")
            #    main()
    else:  # Linux, Mac
        main()
