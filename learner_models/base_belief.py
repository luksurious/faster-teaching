from typing import Dict, Any

import numpy as np
from abc import ABC, abstractmethod

from actions import Actions
from concepts.concept_base import ConceptBase, ActionResult

DEBUG = False


class BaseBelief(ABC):
    state_action_values: Dict[Any, np.ndarray]

    def __init__(self, belief_state, prior, concept: ConceptBase, trans_noise: float = 0, prod_noise: float = 0,
                 verbose: bool = True):
        self.belief_state = belief_state
        self.belief_state_orig = belief_state.copy()
        self.prior = prior
        self.concept = concept

        self.transition_noise = trans_noise
        self.production_noise = prod_noise
        self.obs_noise_prob = self.production_noise / len(concept.get_observation_space())

        self.prior_pos_len = np.count_nonzero(self.prior)

        self.hypotheses = concept.get_concept_space()

        self.verbose = verbose

        # pre-calculate state-action concept values
        # TODO duplicated as in letter addition?
        self.state_action_values = {}
        self.pre_calc_state_values()

    def pre_calc_state_values(self):
        for action in self.concept.get_rl_actions():
            self.state_action_values[action] = np.zeros(len(self.hypotheses))
            for idx, state in enumerate(self.hypotheses):
                self.state_action_values[action][idx] = self.concept.evaluate_concept(action, state, idx)

    def reset(self):
        self.belief_state = self.belief_state_orig.copy()

    def update_belief(self, action_type, result, response):
        if action_type == Actions.FEEDBACK:
            # first narrow down belief of previous state, ignoring the correct answer
            self.belief_state = self.calc_new_belief(action_type, response, (result[0], None))

        self.belief_state = self.calc_new_belief(action_type, response, result)

        self.see_action(action_type, result)

    def calc_new_belief(self, action_type, response, result):
        new_belief = self.belief_update_formula(action_type, result, response)
        new_belief = self.assert_belief_is_valid(action_type, new_belief, response, result)
        # scale to 1
        new_belief /= np.sum(new_belief)
        return new_belief

    def assert_belief_is_valid(self, action_type, new_belief, response, result):
        # TODO does it make sense to reset the belief like this?
        if np.max(new_belief) == 0:
            # quiz response inconsistent with previous state, calc only based on quiz now
            if DEBUG:
                print("Inconsistent quiz response")

            self.belief_state = self.prior.copy()
            new_belief = self.belief_update_formula(action_type, result, response)

            if np.max(new_belief) == 0:
                # still 0 means, invalid response; reset to prior probs
                new_belief = self.prior.copy()

        return new_belief

    def belief_update_formula(self, action_type, action: ActionResult, observation):
        new_belief = np.zeros_like(self.belief_state)

        for idx, new_state in enumerate(self.hypotheses):
            concept_val = self.concept.evaluate_concept(action[0], new_state, idx)

            p_z = self.observation_model(observation, new_state, action_type, action, concept_val)
            if p_z == 0:
                continue

            p_s = self.transition_model(new_state, idx, action_type, action, concept_val)

            if DEBUG:
                print("S=%s p_z=%.2f p_s=%.2f" % (new_state, p_z, p_s))

            new_belief[idx] = p_z * p_s

        return new_belief

    @abstractmethod
    def observation_model(self, observation, new_state, action_type, action, concept_val):
        pass

    @abstractmethod
    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        pass

    def see_action(self, action_type, action):
        pass

    def get_state(self):
        return self.belief_state.copy()

    def set_state(self, state):
        self.belief_state = state.copy()

    def get_concept_prob(self, index):
        return self.belief_state[index]

    @abstractmethod
    def get_observation_prob(self, action, observation):
        pass

    def copy(self):
        o = self.__copy__()
        return o

    def __copy__(self):
        return BaseBelief(self.belief_state.copy(), self.prior, self.concept, trans_noise=self.transition_noise,
                          prod_noise=self.production_noise, verbose=self.verbose)
