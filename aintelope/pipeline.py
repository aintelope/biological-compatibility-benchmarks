# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import copy
import logging
import sys
import torch
import gc
import time
import json
import itertools
import subprocess
import asyncio

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer

from diskcache import Cache

# this one is cross-platform
from filelock import FileLock

from aintelope.utils import RobustProgressBar, Semaphore, wait_for_enter

from matplotlib import pyplot as plt

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
    archive_code,
    get_pipeline_score_dimensions,
    get_score_dimensions,
    set_console_title,
)
from aintelope.experiments import run_experiment


logger = logging.getLogger("aintelope.__main__")

gpu_count = torch.cuda.device_count()
worker_count_multiplier = 1  # when running pipeline search, then having more workers than GPU-s will cause all sorts of Python and CUDA errors under Windows for some reason, even though there is plenty of free RAM and GPU memory. Yet, when the pipeline processes are run manually, there is no concurrency limit except the real hardware capacity limits. # TODO: why?
num_workers = max(1, gpu_count) * worker_count_multiplier


def aintelope_main() -> None:
    run_pipeline()


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def run_pipeline(cfg: DictConfig) -> None:
    do_not_show_plot = False  # TODO: config parameter

    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    archive_code(cfg)

    pipeline_config_file = os.environ.get("PIPELINE_CONFIG")
    if pipeline_config_file is None:
        pipeline_config_file = "config_pipeline.yaml"
    pipeline_config = OmegaConf.load(
        os.path.join("aintelope", "config", pipeline_config_file)
    )

    config_name = HydraConfig.get().job.config_name
    set_console_title(
        config_name + " : " + pipeline_config_file + " : " + timestamp_pid_uuid
    )

    test_summaries_to_return = []
    test_summaries_to_jsonl = []

    # use additional semaphore here since the user may launch multiple processes manually
    semaphore_name = (
        "AIntelope_pipeline_semaphore"
        + (
            "_" + cfg.hparams.params_set_title
            if cfg.hparams.params_set_title in ["handwritten_rules", "random"]
            else ""
        )
        + ("_debug" if sys.gettrace() is not None else "")
    )
    print("Waiting for semaphore...")
    with Semaphore(
        semaphore_name,
        max_count=num_workers,
        disable=(
            os.name != "nt"
            or gpu_count == 0
            or True  # TODO: config flag for disabling the semaphore
        ),  # Linux does not unlock semaphore after a process gets killed, therefore disabling Semaphore under Linux until this gets resolved.
    ) as semaphore:
        print("Semaphore acquired...")
        max_pipeline_cycle = (
            cfg.hparams.num_pipeline_cycles + 1
            if cfg.hparams.num_pipeline_cycles >= 1
            else 1
        )  # Last +1 cycle is for testing. In case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
        with RobustProgressBar(
            max_value=max_pipeline_cycle
        ) as pipeline_cycle_bar:  # this is a slow task so lets use a progress bar
            for i_pipeline_cycle in range(0, max_pipeline_cycle):
                test_mode = i_pipeline_cycle == cfg.hparams.num_pipeline_cycles

                with RobustProgressBar(
                    max_value=len(pipeline_config)
                ) as pipeline_bar:  # this is a slow task so lets use a progress bar
                    for env_conf_i, env_conf_name in enumerate(pipeline_config):
                        experiment_cfg = copy.deepcopy(
                            cfg
                        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
                        OmegaConf.update(
                            experiment_cfg, "experiment_name", env_conf_name
                        )

                        OmegaConf.update(
                            experiment_cfg,
                            "hparams",
                            pipeline_config[env_conf_name],
                            force_add=True,
                        )

                        logger.info("Running training with the following configuration")
                        logger.info(
                            os.linesep
                            + str(OmegaConf.to_yaml(experiment_cfg, resolve=True))
                        )

                        # Training
                        params_set_title = experiment_cfg.hparams.params_set_title
                        logger.info(
                            f"params_set: {params_set_title}, experiment: {env_conf_name}"
                        )

                        score_dimensions = get_score_dimensions(experiment_cfg)

                        num_actual_train_episodes = -1
                        if (
                            cfg.hparams.num_pipeline_cycles == 0
                        ):  # in case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
                            num_actual_train_episodes = run_experiment(
                                experiment_cfg,
                                experiment_name=env_conf_name,
                                score_dimensions=score_dimensions,
                                test_mode=False,
                                i_pipeline_cycle=i_pipeline_cycle,
                            )
                        elif test_mode:
                            pass  # TODO: optional: obtain num_actual_train_episodes. But this is not too important: in case of training a model over one or more pipeline cycles, the final test cycle gets its own i_pipeline_cycle index, therefore it is clearly distinguishable anyway

                        run_experiment(
                            experiment_cfg,
                            experiment_name=env_conf_name,
                            score_dimensions=score_dimensions,
                            test_mode=test_mode,
                            i_pipeline_cycle=i_pipeline_cycle,
                            num_actual_train_episodes=num_actual_train_episodes,
                        )

                        # torch.cuda.empty_cache()
                        # gc.collect()

                        if test_mode:
                            # Not using timestamp_pid_uuid here since it would make the title too long. In case of manual execution with plots, the pid-uuid is probably not needed anyway.
                            title = (
                                timestamp
                                + " : "
                                + params_set_title
                                + " : "
                                + env_conf_name
                            )
                            test_summary = analytics(
                                experiment_cfg,
                                score_dimensions,
                                title=title,
                                experiment_name=env_conf_name,
                                group_by_pipeline_cycle=cfg.hparams.num_pipeline_cycles
                                >= 1,
                                gridsearch_params=None,
                                do_not_show_plot=do_not_show_plot,
                            )
                            test_summaries_to_return.append(test_summary)
                            test_summaries_to_jsonl.append(test_summary)

                        pipeline_bar.update(env_conf_i + 1)

                    # / for env_conf_name in pipeline_config:
                # / with RobustProgressBar(max_value=len(pipeline_config)) as pipeline_bar:

                pipeline_cycle_bar.update(i_pipeline_cycle + 1)

            # / for i_pipeline_cycle in range(0, max_pipeline_cycle):
        # / with RobustProgressBar(max_value=max_pipeline_cycle) as pipeline_cycle_bar:
    # / with Semaphore('name', max_count=num_workers, disable=False) as semaphore:

    # Write the pipeline results to file only when entire pipeline has run. Else crashing the program during pipeline run will cause the aggregated results file to contain partial data which will be later duplicated by re-run.
    # TODO: alternatively, cache the results of each experiment separately
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)
        aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
        with aggregated_results_file_lock:
            with open(aggregated_results_file, mode="a", encoding="utf-8") as fh:
                for test_summary in test_summaries_to_jsonl:
                    # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                    json_text = json.dumps(test_summary)
                    fh.write(
                        json_text + "\n"
                    )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
                fh.flush()

    torch.cuda.empty_cache()
    gc.collect()

    # keep plots visible until the user decides to close the program
    if not do_not_show_plot:
        # uses less CPU on Windows than input() function. Note that the graph window will be frozen, but will still show graphs
        wait_for_enter("\nPipeline done. Press [enter] to continue.")

    return test_summaries_to_return


def analytics(
    cfg,
    score_dimensions,
    title,
    experiment_name,
    group_by_pipeline_cycle,
    gridsearch_params=DictConfig,
    do_not_show_plot=False,
):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name)
    events = recording.read_events(experiment_dir, events_fname)

    (
        test_totals,
        test_averages,
        test_variances,
        test_sfella_totals,
        test_sfella_averages,
        test_sfella_variances,
        sfella_score_total,
        sfella_score_average,
        sfella_score_variance,
        score_dimensions_out,
    ) = plotting.aggregate_scores(
        events,
        num_train_pipeline_cycles,
        score_dimensions,
        group_by_pipeline_cycle=group_by_pipeline_cycle,
    )

    test_summary = {
        "timestamp": cfg.timestamp,
        "timestamp_pid_uuid": cfg.timestamp_pid_uuid,
        "experiment_name": experiment_name,
        "title": title,  # timestamp + " : " + params_set_title + " : " + env_conf_name
        "params_set_title": cfg.hparams.params_set_title,
        "gridsearch_params": OmegaConf.to_container(gridsearch_params, resolve=True)
        if gridsearch_params is not None
        else None,  # Object of type DictConfig is not JSON serializable, neither can yaml.dump in plotting.prettyprint digest it, so need to convert it to ordinary dictionary
        "num_train_pipeline_cycles": num_train_pipeline_cycles,
        "score_dimensions": score_dimensions_out,
        "group_by_pipeline_cycle": group_by_pipeline_cycle,
        "test_totals": test_totals,
        "test_averages": test_averages,
        "test_variances": test_variances,
        # per score dimension results
        "test_sfella_totals": test_sfella_totals,
        "test_sfella_averages": test_sfella_averages,
        "test_sfella_variances": test_sfella_variances,
        # over score dimensions results
        # TODO: rename to test_*
        "sfella_score_total": sfella_score_total,
        "sfella_score_average": sfella_score_average,
        "sfella_score_variance": sfella_score_variance,
    }

    plotting.prettyprint(test_summary)

    plotting.plot_performance(
        events,
        num_train_episodes,
        num_train_pipeline_cycles,
        score_dimensions,
        save_path=savepath,
        title=title,
        group_by_pipeline_cycle=group_by_pipeline_cycle,
        do_not_show_plot=do_not_show_plot,
    )

    return test_summary


def aintelope_main() -> None:
    # return run_gridsearch_experiment(gridsearch_params=None)    # TODO: caching support
    run_pipeline()


if __name__ == "__main__":  # for multiprocessing support
    register_resolvers()

    if (
        sys.gettrace() is None
    ):  # do not set low priority while debugging. Note that unit tests also set sys.gettrace() to not-None
        set_priorities()

    set_memory_limits()

    # Need to choose GPU early before torch fully starts up. Else there may be CUDA errors later.
    select_gpu()

    aintelope_main()
