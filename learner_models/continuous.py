from learner_models.base_belief import BaseBelief
from concepts.concept_base import ConceptBase
from actions import Actions

import math
import numpy as np


class ContinuousModel(BaseBelief):

    def __init__(self, belief_state, prior, concept: ConceptBase, particles: int = 16, verbose: bool = True):
        super().__init__(belief_state, prior, concept, verbose)

        self.transition_noise = 0.14 / self.prior_pos_len
        self.production_noise = 0.12

        # pre-calculate state-action concept values
        # TODO duplicated as in letter addition?
        self.state_action_values = {}
        self.pre_calc_state_values()

    def pre_calc_state_values(self):
        for action in self.concept.get_rl_actions():
            self.state_action_values[action] = np.zeros(len(self.states))
            for idx, state in enumerate(self.states):
                self.state_action_values[action][idx] = self.concept.evaluate_concept((action,), state, idx)

    def observation_model(self, observation, new_state, action_type, action, concept_val):
        """
        Probability of seeing observation (i.e. response of the learner) in the (new) state given the taken action
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
        if concept_val == observation:
            return 1 - self.production_noise

        # TODO do I need to scale it so that Sum[p(z|s,a)] = 1 ? i.e. determine how many illegal answers exist
        #  and divide the production noise parameter by that number?
        #  although the number is already super small; so maybe not needed?
        return self.production_noise  # / self.state_matches[observation]

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        """
        Probability of going to new state given an action (and current state)
        """

        if action_type == Actions.QUIZ:
            # no state change expected - but we can rule out states that do not match her response
            # no need for a loop
            return self.belief_state[new_idx]  # transition prob only to same state is 1, only b(s) left in formula

        p_s = 0

        if concept_val == action[1]:
            p_s = self.calculate_ps(action, new_idx)

        return p_s

    def calculate_ps(self, action, new_idx):
        b_s = self.belief_state

        consistent_state_filter = self.state_action_values[action[0]] == action[1]

        # prob of going to new state
        noisy_prior = self.prior[new_idx] / np.sum(self.prior[consistent_state_filter]) - self.transition_noise

        p_s = np.ones(len(self.states)) * noisy_prior  # default transition with prior probability
        p_s[consistent_state_filter] = 0  # no transition from other consistent concepts

        p_s[new_idx] = 1  # instead it stays at the same concept
        p_s = np.sum(p_s * b_s)

        return p_s

    def see_action(self, action_type, action):
        pass

    def __copy__(self):
        return ContinuousModel(self.belief_state.copy(), self.prior, self.concept, self.verbose)
