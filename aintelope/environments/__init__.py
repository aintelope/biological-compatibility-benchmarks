from typing import Mapping, Type, Union

import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
)

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


ENV_REGISTRY: Mapping[str, Type[Environment]] = {}


def register_env_class(env_id: str, env_class: Type[Environment]):
    if env_id in ENV_REGISTRY:
        raise ValueError(f"{env_id} is already registered")
    ENV_REGISTRY[env_id] = env_class


def get_env_class(env_id: str) -> Type[Environment]:
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"{env_id} is not found in env registry")
    return ENV_REGISTRY[env_id]


register_env_class("savanna-safetygrid-sequential-v1", SavannaGridworldSequentialEnv)
register_env_class("savanna-safetygrid-parallel-v1", SavannaGridworldParallelEnv)