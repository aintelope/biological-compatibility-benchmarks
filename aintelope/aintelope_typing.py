# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

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
