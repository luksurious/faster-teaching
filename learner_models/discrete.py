from collections import deque

from concepts.concept_base import ConceptBase
from learner_models.model_base import ModelBase
from actions import Actions

import math
import numpy as np


class DiscreteMemoryModel(ModelBase):

    def __init__(self, states, prior, concept: ConceptBase, memory_size: int):
        super().__init__(states, prior, concept)

        self.transition_noise = 0.34  # pretty high
        self.production_noise = 0.046

        self.memory_size = memory_size

        self.memory = deque(maxlen=2)

    def see_action(self, action_type, action):
        self.memory.append((action_type, action))

    def get_state(self):
        return self.memory.copy()

    def set_state(self, state):
        self.memory = state

    def observation_model(self, observation, new_state, action_type, action):
        """
        Probability of seeing observation (i.e. response of the learner) in the (new) state given the taken action

        Should be the same as memoryless right?
        """
        # example action: there is no observation
        if observation is None:
            return 1

        # TODO how to handle question properly?
        # question action: there is an observation but it belongs to the old state; the feedback is supposed to change
        # the state which happens after the observation
        if action_type == Actions.QUESTION:
            return 1

        # quiz action: the observation is deterministic based on if it matches the state of the learner
        if self.is_observation_consistent(observation, new_state, action):
            return 1 - self.production_noise

        # TODO do I need to scale it so that Sum[p(z|s,a)] = 1 ? i.e. determine how many illegal answers exist
        #  and divide the production noise parameter by that number?
        #  although the number is already super small; so maybe not needed?
        return self.production_noise  # / self.state_matches[observation]

    def is_observation_consistent(self, observation, state, action):
        return self.concept.evaluate_concept(action, state) == observation

    def transition_model(self, new_state, new_idx, action_type, action):
        """
        Probability of going to new state given an action (and current state)
        """

        p_s = 0
        # for idx2, state in enumerate(self.states):
        #     b_s = self.belief_state[idx2]  # sums to 1 over all states
        #     # new state is consistent with action
        #     if self.concept.evaluate_concept(action, new_state) == action[1]:
        #         p_s += self.prior[new_idx] * b_s  # TODO assumption: uniform prior, negligible

        if self.concept.evaluate_concept(action, new_state) == action[1] and self.matches_memory(new_state):
            p_s += self.prior[new_idx]  # TODO assumption: uniform prior, negligible

        return p_s

    def matches_memory(self, new_state):
        matches = True
        for memory_item in self.memory:
            if memory_item[0] == Actions.QUIZ:
                continue

            if self.concept.evaluate_concept(memory_item[1], new_state) != memory_item[1][1]:
                matches = False
                break

        return matches

