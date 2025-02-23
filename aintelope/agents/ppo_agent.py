# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import logging
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import pandas as pd
from omegaconf import DictConfig

from aintelope.utils import RobustProgressBar

import numpy as np
import numpy.typing as npt
import os
import datetime

from aintelope.agents.sb3_base_agent import SB3BaseAgent, CustomCNN, vec_env_args
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer
from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

import torch
import stable_baselines3
from stable_baselines3 import PPO
import supersuit as ss

from typing import Any, Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.ppo_agent")


# need separate function outside of class in order to init multi-model training threads
def ppo_model_constructor(env, cfg):
    # policy_kwarg:
    # if you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False
    # see the following links:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    # https://github.com/DLR-RM/stable-baselines3/issues/1863
    # Also: make sure your image is in the channel-first format.
    return PPO(
        "CnnPolicy" if cfg.hparams.model_params.num_conv_layers > 0 else "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=(
            {
                "normalize_images": False,
                "features_extractor_class": CustomCNN,  # need custom CNN in order to handle observation shape 9x9
                "features_extractor_kwargs": {
                    "features_dim": 256,  # TODO: config parameter. Note this is not related to the number of features in the original observation (15), this parameter here is model's internal feature dimensionality
                    "num_conv_layers": cfg.hparams.model_params.num_conv_layers,
                },
            }
            if cfg.hparams.model_params.num_conv_layers > 0
            else {"normalize_images": False}
        ),
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Note, CUDA-based CPU performance is much better than Torch-CPU mode.
    )


class PPOAgent(SB3BaseAgent):
    """PPOAgent class from stable baselines
    https://pettingzoo.farama.org/tutorials/sb3/waterworld/
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(env=env, cfg=cfg, **kwargs)

        self.model_constructor = ppo_model_constructor

        if (
            self.test_mode
        ):  # during test, each agent has a separate in-process instance with its own model and not using threads/subprocesses
            env = SingleAgentZooToGymAdapter(env, self.id)
            self.model = self.model_constructor(env, cfg)
        elif self.env.num_agents == 1 or cfg.hparams.model_params.use_weight_sharing:
            # PPO supports weight sharing for multi-agent scenarios
            # TODO: Environment duplication support for parallel compute purposes. Abseil package needs to be replaced for that end.

            ss.vector.vector_constructors.vec_env_args = vec_env_args  # Since we need only one environment, there is no reason for cloning, so lets replace the cloning function with identity function.

            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(
                env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
            )  # NB! num_vec_envs=1 is important here so that we can use identity function instead of cloning in vec_env_args
            self.model = self.model_constructor(env, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.
