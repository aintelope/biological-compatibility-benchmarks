# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import traceback

import numpy as np
import numpy.testing as npt
import pytest

from aintelope.environments import savanna_safetygrid as safetygrid
from aintelope.environments.savanna_safetygrid import SavannaGridworldParallelEnv

from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)
from zoo_to_gym_multiagent_adapter.multiagent_zoo_to_gym_adapter import (
    MultiAgentZooToGymAdapterGymSide,
    MultiAgentZooToGymAdapterZooSide,
)

from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.test import max_cycles_test, performance_benchmark, render_test
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test
from stable_baselines3.common.env_checker import check_env

# from pettingzoo.utils import parallel_to_aec


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_parallel_scalarized_rewards(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 1,  # for now only one agent
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "scalarize_rewards": True,  # Zoo parallel API tests work with multidimensional rewards as well, but Zoo sequential API tests do not. So just for consistency, test Zoo parallel API both with scalarized and multidimensional rewards. The tests for multidimensional rewars are below.
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    env.seed(execution_number)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_parallel_with_death_scalarized_rewards(execution_number):
    # TODO: refactor these values out to a test-params file
    # for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 2,  # needed for death test
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "test_death": True,
        "seed": execution_number,
        "scalarize_rewards": True,  # Zoo parallel API tests work with multidimensional rewards as well, but Zoo sequential API tests do not. So just for consistency, test Zoo parallel API both with scalarized and multidimensional rewards. The tests for multidimensional rewars are below.
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_parallel(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 1,  # for now only one agent
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    env.seed(execution_number)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_parallel_with_death(execution_number):
    # TODO: refactor these values out to a test-params file
    # for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 2,  # needed for death test
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "test_death": True,
        "seed": execution_number,
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_seed(execution_number):
    # override_infos: Zoo parallel_seed_test is unable to compare infos unless they have simple
    # structure.
    # seed: for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "override_infos": True,
        "seed": execution_number,
    }

    def get_env_instance() -> safetygrid.SavannaGridworldParallelEnv:
        """Method for seed_test"""
        return safetygrid.SavannaGridworldParallelEnv(env_params=env_params)

    try:
        parallel_seed_test(get_env_instance, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles
        # neither as named or positional argument
        parallel_seed_test(get_env_instance)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_step_result(execution_number):
    # default is 1 iter which means that the env is done after 1 step below and the
    # test will fail
    env = safetygrid.SavannaGridworldParallelEnv(
        env_params={
            "num_iters": 2,
            "seed": execution_number,
        }
    )
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.possible_agents[0]
    action = {agent: env.action_space(agent).sample()}

    observations, rewards, terminateds, truncateds, infos = env.step(action)
    dones = {
        key: terminated or truncateds[key] for (key, terminated) in terminateds.items()
    }

    assert not dones[agent]
    assert isinstance(observations, dict), "observations is not a dict"
    assert isinstance(
        observations[agent][0], np.ndarray
    ), "observations[0] of agent is not an array"
    assert isinstance(
        observations[agent][1], np.ndarray
    ), "observations[1] of agent is not an array"
    assert isinstance(rewards, dict), "rewards is not a dict"
    assert isinstance(rewards[agent], dict), "reward of agent is not a dict"


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_done_step(execution_number):
    env = safetygrid.SavannaGridworldParallelEnv(
        env_params={
            "amount_agents": 1,
            "seed": execution_number,
        }
    )
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]  # TODO: multi-agent iteration
    for _ in range(env.metadata["num_iters"]):
        action = {agent: env.action_space(agent).sample()}
        _, _, terminateds, truncateds, _ = env.step(action)
        dones = {
            key: terminated or truncateds[key]
            for (key, terminated) in terminateds.items()
        }

    assert dones[agent]
    with pytest.raises(ValueError):
        action = {agent: env.action_space(agent).sample()}
        env.step(action)


def test_gridworlds_agents():
    env = safetygrid.SavannaGridworldParallelEnv()

    # assert len(env.possible_agents) == env.metadata["amount_agents"]  # TODO: this is now determined by the environment, not by config
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_gridworlds_action_spaces():
    env = safetygrid.SavannaGridworldParallelEnv()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 5  # includes no-op


@pytest.mark.parametrize("execution_number", range(1))
def test_singleagent_zoo_to_gym_wrapper_scalarized_rewards(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 1,  # for now only one agent
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "scalarize_rewards": True,  # Gym test requires scalarised rewards
        "combine_interoception_and_vision": True,  # SB3 does not support complex observation spaces
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    env.seed(execution_number)

    env = SingleAgentZooToGymAdapter(env, "agent_0")
    # OpenAI Stable Baselines 3 Gym env checker
    # warn=False : disable additional warnings since they are handled in the agent side by:
    # * disabling normalization
    # * using custom feature extractor
    check_env(env, warn=False, skip_render_check=True)


def sb3_gym_test_thread_entry_point(
    pipe,
    gpu_index,
    num_total_steps,
    model_constructor,
    agent_id,
    checkpoint_filename,
    cfg,
    observation_space,
    action_space,
):
    env_wrapper = MultiAgentZooToGymAdapterGymSide(
        pipe, agent_id, checkpoint_filename, observation_space, action_space
    )
    try:
        # OpenAI Stable Baselines 3 Gym env checker
        # warn=False : disable additional warnings since they are handled in the agent side by:
        # * disabling normalization
        # * using custom feature extractor
        check_env(env_wrapper, warn=False, skip_render_check=True)

        env_wrapper.save_or_return_model(model=None, filename_timestamp_sufix_str=None)
    except (
        Exception
    ) as ex:  # NB! need to catch exception so that the env wrapper can signal the training ended
        info = str(ex) + os.linesep + traceback.format_exc()
        env_wrapper.terminate_with_exception(info)
        print(info)


@pytest.mark.parametrize("execution_number", range(1))
def test_multiagent_zoo_to_gym_wrapper_scalarized_rewards(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 2,  # NB!
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "scalarize_rewards": True,  # Gym test requires scalarised rewards
        "combine_interoception_and_vision": True,  # SB3 does not support complex observation spaces
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    env.seed(execution_number)

    env_wrapper = MultiAgentZooToGymAdapterZooSide(
        env, cfg=None
    )  # cfg is unused at sb3_gym_test_thread_entry_point() function
    _, exceptions = env_wrapper.train(
        num_total_steps=None,  # unused at sb3_gym_test_thread_entry_point() function
        agent_thread_entry_point=sb3_gym_test_thread_entry_point,
        model_constructor=None,  # unused at sb3_gym_test_thread_entry_point() function
        terminate_all_agents_when_one_excepts=True,
        checkpoint_filenames=None,
    )

    if exceptions:
        raise Exception(str(exceptions))


# TODO: single-agent env test with death
# TODO: multi-agent env wrapper test with death


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # test_gridworlds_api_parallel_with_death()
