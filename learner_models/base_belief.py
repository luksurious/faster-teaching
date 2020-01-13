import numpy as np
from abc import ABC, abstractmethod

from actions import Actions
from concepts.concept_base import ConceptBase


DEBUG = False


class BaseBelief(ABC):
    def __init__(self, belief_state, prior, concept: ConceptBase, verbose: bool = True):
        self.belief_state = belief_state
        self.prior = prior
        self.concept = concept

        self.transition_noise = 0
        self.production_noise = 0

        self.prior_pos_len = np.count_nonzero(self.prior)

        self.states = concept.get_concept_space()

        self.verbose = verbose

    def update_belief(self, action_type, result, response):
        # TODO should this be modeled inside the belief update?
        # transition noise probability: no state change;
        # if action_type == Actions.QUIZ or np.random.random() >= self.transition_noise:
        if True:

            if action_type == Actions.QUESTION:
                # Handle question as two step action
                new_belief = self.calc_new_belief(Actions.QUIZ, response, result)
                self.belief_state = new_belief
                new_belief = self.calc_new_belief(Actions.EXAMPLE, response, result)
            else:
                new_belief = self.calc_new_belief(action_type, response, result)

            self.belief_state = new_belief

        self.see_action(action_type, result)

    def calc_new_belief(self, action_type, response, result):
        new_belief = self.calc_belief_updates(action_type, result, response)
        new_belief = self.assert_belief_is_valid(action_type, new_belief, response, result)
        # scale to 1
        new_belief /= np.sum(new_belief)
        return new_belief

    def assert_belief_is_valid(self, action_type, new_belief, response, result):
        # TODO does it make sense to reset the belief like this?
        if np.sum(new_belief) == 0:
            # quiz response inconsistent with previous state, calc only based on quiz now
            # if self.verbose:
            #     print("Inconsistent quiz response")

            self.belief_state[:] = 1
            new_belief = self.calc_belief_updates(action_type, result, response)

            if np.sum(new_belief) == 0:
                # still 0 means, invalid response; reset to prior probs
                new_belief = self.prior.copy()

        return new_belief

    def calc_belief_updates(self, action_type, action, observation):
        new_belief = np.zeros_like(self.belief_state)

        for idx, new_state in enumerate(self.states):
            concept_val = self.concept.evaluate_concept(action, new_state, idx)

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

    def copy(self):
        o = self.__copy__()
        return o

    def __copy__(self):
        return BaseBelief(self.belief_state.copy(), self.prior, self.concept, self.verbose)
