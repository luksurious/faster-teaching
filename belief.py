import numpy as np
from abc import ABC, abstractmethod

from actions import Actions
from concepts.concept_base import ConceptBase
from learner_models.model_base import ModelBase


DEBUG = False


class Belief(ABC):
    def __init__(self, belief_state, prior, concept: ConceptBase, verbose: bool = True):
        self.belief_state = belief_state
        self.prior = prior
        self.start_prior = prior
        self.concept = concept

        self.transition_noise = 0
        self.production_noise = 0

        self.states = concept.get_concept_space()

        self.verbose = verbose

    def update_belief(self, action_type, result, response):
        # TODO should this be modeled inside the belief update?
        # transition noise probability: no state change;
        if action_type == Actions.QUIZ or np.random.random() >= self.transition_noise:
            # TODO is prior updated in every step?? It seems in the paper it refers to initial probability
            #  but it is somewhat counter intuitive
            # self.prior = self.belief_state

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
                new_belief = self.start_prior.copy()

        return new_belief

    def calc_belief_updates(self, action_type, action, observation):
        new_belief = np.zeros_like(self.belief_state)

        for idx, new_state in enumerate(self.states):
            concept_val = self.concept.evaluate_concept(action, new_state)

            p_z = self.observation_model(observation, new_state, action_type, action, concept_val)

            p_s = self.transition_model(new_state, idx, action_type, action, concept_val)

            if DEBUG:
                print("S=%s p_z=%.2f p_s=%.2f" % (new_state, p_z, p_s))

            new_belief[idx] = p_z * p_s

        # self.belief_state = new_belief / np.sum(new_belief)
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
        return self.belief_state

    def set_state(self, state):
        self.belief_state = state
