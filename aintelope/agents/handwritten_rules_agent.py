# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import logging
from typing import List, Optional, Tuple
from collections import defaultdict
from gymnasium.spaces import Discrete

from omegaconf import DictConfig

import numpy as np
import numpy.typing as npt

from aintelope.environments.savanna_safetygrid import (
    GridworldZooBaseEnv,
    ACTION_RELATIVE_COORDINATE_MAP,
)

from aintelope.agents.handwritten_rules.savanna_safetygrid_handwritten_rules import (
    savanna_safetygrid_available_handwritten_rules_dict,
    format_float,
)

from aintelope.agents.abstract_agent import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]

logger = logging.getLogger("aintelope.agents.handwritten_rules_agent")


class HandwrittenRulesAgent(Agent):
    """Agent class with handwritten rules"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        env: Environment = None,
        cfg: DictConfig = None,
        target_handwritten_rules: List[str] = [],
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.hparams = trainer.hparams
        self.env = env
        self.cfg = cfg
        self.done = False
        self.last_action = None

        self.target_handwritten_rules = target_handwritten_rules
        self.handwritten_rules = {}

    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None
        self.state = state
        self.info = info
        self.env_class = env_class

        self.init_handwritten_rules()

    def get_action(
        self,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        test_mode: bool = False,
        *args,
        **kwargs,
    ) -> Optional[int]:
        """Given an observation, ask your model what to do.

        Returns:
            action (Optional[int]): index of action
        """

        if self.done:
            return None

        action_space = self.trainer.action_spaces[self.id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
            max_action = action_space.start + action_space.n - 1
        else:
            min_action = action_space.min_action
            max_action = action_space.max_action

        # calculate action reward predictions using handwritten rules
        predicted_action_rewards = defaultdict(float)

        for (
            handwritten_rule_name,
            handwritten_rule_object,
        ) in self.handwritten_rules.items():
            handwritten_rule_action_rewards = {}
            # predict reward for all available actions
            for action in range(
                min_action, max_action + 1
            ):  # NB! max_action is inclusive max
                agent_coordinate = info[ACTION_RELATIVE_COORDINATE_MAP][action]

                (
                    handwritten_rule_reward,
                    handwritten_rule_event,
                ) = handwritten_rule_object.calc_reward(
                    self,
                    observation,
                    info,
                    agent_coordinate=agent_coordinate,
                    predicting=True,
                )

                handwritten_rule_action_rewards[action] = handwritten_rule_reward
                predicted_action_rewards[
                    action
                ] += handwritten_rule_reward  # TODO: nonlinear aggregation

            # debug helper  # TODO: refactor into a separate method
            # if handwritten_rule_name == "gold":
            #    q_values = np.zeros([max_action - min_action + 1], np.float32)
            #    for action, bias in handwritten_rule_action_rewards.items():
            #        q_values[action - min_action] = bias
            #    print(f"gold q_values: {format_float(q_values)}")

        handwritten_rule_q_values = (
            predicted_action_rewards  # handwritten rules see only one step ahead
        )

        q_values = np.zeros([max_action - min_action + 1], np.float32)
        for action, bias in handwritten_rule_q_values.items():
            q_values[action - min_action] = bias
        action = (
            self.trainer.tiebreaking_argmax(q_values) + min_action
        )  # take best action predicted by handwritten rules

        # print(f"Action: {action}")
        self.last_action = action
        return action

    def update(
        self,
        env: PettingZooEnv = None,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        score: float = 0.0,
        done: bool = False,
        test_mode: bool = False,
    ) -> list:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to calculate handwritten rule rewards.

        Args:
            env: Environment
            observation: Tuple[ObservationArray, ObservationArray]
            score: Only baseline uses score as a reward
            done: boolean whether run is done
        Returns:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net
        """

        assert self.last_action is not None

        next_state = observation
        next_info = info
        # For future: add state (interoception) handling here when needed

        # calculate handwritten rules rewards
        if len(self.handwritten_rules) == 0:
            # use env reward if no handwritten_rules available
            handwritten_rule_events = []
            reward = score
        else:
            # interpret new_state and score to compute actual reward
            reward = 0
            handwritten_rule_events = []
            if next_state is not None:  # temporary, until we solve final states
                for (
                    handwritten_rule_name,
                    handwritten_rule_object,
                ) in self.handwritten_rules.items():
                    (
                        handwritten_rule_reward,
                        handwritten_rule_event,
                    ) = handwritten_rule_object.calc_reward(self, next_state, next_info)
                    reward += handwritten_rule_reward  # TODO: nonlinear aggregation
                    logger.debug(
                        f"Reward of {handwritten_rule_name}: {handwritten_rule_reward}; "
                        f"total reward: {reward}"
                    )
                    if handwritten_rule_event != 0:
                        handwritten_rule_events.append(
                            (handwritten_rule_name, handwritten_rule_event)
                        )

            # print(f"reward: {reward}")

        event = [self.id, self.state, self.last_action, reward, done, next_state]
        if not test_mode:  # TODO: do we need to update replay memories during test?
            self.trainer.update_memory(*event)
        self.state = next_state
        self.info = info
        return event

    def init_handwritten_rules(self) -> None:
        if issubclass(
            self.env_class, GridworldZooBaseEnv
        ):  # radically different types of environments may need different handwritten_rules
            available_handwritten_rules_dict_local = (
                savanna_safetygrid_available_handwritten_rules_dict
            )

        logger.debug(f"target_handwritten_rules: {self.target_handwritten_rules}")
        for handwritten_rule_name in self.target_handwritten_rules:
            if handwritten_rule_name not in available_handwritten_rules_dict_local:
                logger.warning(
                    f"Warning: could not find {handwritten_rule_name} "
                    "in available_handwritten_rules_dict"
                )
                continue

        self.handwritten_rules = {
            handwritten_rule: available_handwritten_rules_dict_local.get(
                handwritten_rule
            )()
            for handwritten_rule in self.target_handwritten_rules
            if handwritten_rule in available_handwritten_rules_dict_local
        }
        for handwritten_rule in self.handwritten_rules.values():
            handwritten_rule.reset()

    def init_model(
        self,
        observation_shape,
        action_space,
        unit_test_mode: bool,
        checkpoint: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.trainer.add_agent(
            self.id,
            observation_shape,
            action_space,
            unit_test_mode,
            checkpoint,
            *args,
            **kwargs,
        )

    def save_model(
        self,
        i_episode,
        path,
        experiment_name,
        use_separate_models_for_each_experiment,
        *args,
        **kwargs,
    ):
        pass
