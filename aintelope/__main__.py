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

from aintelope.config.config_utils import (
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)
from aintelope.gridsearch import (
    run_gridsearch_experiments,
    run_gridsearch_experiment_subprocess,
)
from aintelope.pipeline import run_pipeline


# logger = logging.getLogger("aintelope.__main__")


def aintelope_main() -> None:
    # return run_gridsearch_experiment(gridsearch_params=None)    # TODO: caching support
    run_pipeline()


if __name__ == "__main__":
    register_resolvers()

    if (
        sys.gettrace() is None
    ):  # do not set low priority while debugging. Note that unit tests also set sys.gettrace() to not-None
        set_priorities()

    set_memory_limits()

    # Need to choose GPU early before torch fully starts up. Else there may be CUDA errors later.
    select_gpu()

    gridsearch_params_json = os.environ.get(
        "GRIDSEARCH_PARAMS"
    )  # used by gridsearch subprocess
    gridsearch_config_file = os.environ.get(
        "GRIDSEARCH_CONFIG"
    )  # used by main/parent gridsearch process
    if gridsearch_params_json is not None:
        run_gridsearch_experiment_subprocess(gridsearch_params_json)
    elif gridsearch_config_file is not None:
        asyncio.run(
            run_gridsearch_experiments()
        )  # TODO: use separate python file for starting gridsearch
    else:
        aintelope_main()
