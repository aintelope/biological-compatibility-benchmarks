import os
from typing import Dict, Tuple

import numpy as np
import pytest
from omegaconf import OmegaConf

from aintelope.config.config_utils import register_resolvers
from aintelope.training.simple_eval import run_episode
from tests.conftest import tparams_hparams


@pytest.mark.parametrize("execution_number", range(1))
def test_randomwalkagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_randomwalkagent = {
        "agent": "random_walk_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_randomwalkagent)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(1))
def test_onestepperfectpredictionagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_perfectpredictionagent = {
        "agent": "one_step_perfect_prediction_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(
        full_params.hparams, params_perfectpredictionagent
    )
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(1))
def test_iterativeweightoptimizationagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_weightoptimizationagent = {
        "agent": "iterative_weight_optimization_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:" "SavannaGridworldSequentialEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(
        full_params.hparams, params_weightoptimizationagent
    )
    run_episode(full_params=full_params)


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file