# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import pathlib
from typing import Dict, Tuple, Union

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf


def constants() -> DictConfig:
    constants_dict = {
        "PROJECT": "aintelope",
        "BASELINE": "run-training-baseline",
    }
    return OmegaConf.create(constants_dict)


# @pytest.fixture
# def root_dir() -> pathlib.Path:
#    return pathlib.Path(__file__).parents[1]


@pytest.fixture
# def tparams_hparams(root_dir: pathlib.Path) -> Union[DictConfig, ListConfig]:
def tparams_hparams() -> Union[DictConfig, ListConfig]:
    # full_params = OmegaConf.load(os.path.join(root_dir, "aintelope", "config", "config_experiment.yaml"))
    full_params = OmegaConf.load(
        os.path.join("aintelope", "config", "config_experiment.yaml")
    )

    # override some parameters during tests in order to speed up computations
    # TODO: move these overrides to a separate config?
    full_params.hparams.unit_test_mode = True
    full_params.hparams.num_episodes = min(5, full_params.hparams.num_episodes)
    full_params.hparams.env_params.num_iters = min(
        50, full_params.hparams.env_params.num_iters
    )
    full_params.hparams.warm_start_steps = min(10, full_params.hparams.warm_start_steps)

    return full_params
