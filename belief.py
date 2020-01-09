import numpy as np

from actions import Actions
from concepts.concept_base import ConceptBase
from learner_models.model_base import ModelBase


class Belief:
    def __init__(self, belief_state, prior, concept: ConceptBase, model: ModelBase, verbose: bool = True):
        self.belief_state = belief_state
        self.prior = prior
        self.start_prior = prior
        self.model = model
        self.concept = concept

        self.states = concept.get_concept_space()

        self.verbose = verbose

    def update_belief(self, action_type, result, response):
        # TODO should this be modeled inside the belief update?
        if action_type != Actions.QUIZ and np.random.random() <= self.model.transition_noise:
            # transition noise probability: no state change;
            self.model.see_action(action_type, result)
            return

        # TODO is prior updated in every step?? It seems in the paper it refers to initial probability
        #  but it is somewhat counter intuitive
        # self.prior = self.belief_state

        new_belief = self.belief_update(action_type, result, response)

        # TODO does it make sense to reset the belief like this?
        if np.sum(new_belief) == 0:
            # quiz response inconsistent with previous state, calc only based on quiz now
            if self.verbose:
                print("Inconsistent quiz response")

            self.belief_state[:] = 1
            new_belief = self.belief_update(action_type, result, response)

            if np.sum(new_belief) == 0:
                # still 0 means, invalid response; reset to prior probs
                new_belief = self.start_prior.copy()

        # scale to 1
        new_belief /= np.sum(new_belief)

        self.belief_state = new_belief

        self.model.see_action(action_type, result)

    def belief_update(self, action_type, action, observation):
        new_belief = np.zeros_like(self.belief_state)

        for idx, new_state in enumerate(self.states):
            p_z = self.model.observation_model(observation, new_state, action_type, action)

            p_s = self.model.transition_model(new_state, idx, action_type, action)

            new_belief[idx] = p_z * p_s

        # self.belief_state = new_belief / np.sum(new_belief)
        return new_belief



