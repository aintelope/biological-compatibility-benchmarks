from typing import Dict, Union

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv, ParallelEnv

ObservationFloat = np.float32
PositionFloat = np.float32
Action = int
AgentId = str
AgentStates = Dict[AgentId, np.ndarray]

Observation = np.ndarray
Reward = float
Info = dict

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]
