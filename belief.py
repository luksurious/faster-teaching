import numpy as np

from actions import Actions
from concepts.concept_base import ConceptBase


class Belief:
    def __init__(self, belief_state, prior, concept: ConceptBase):
        # for memoryless model
        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.belief_state = belief_state
        self.prior = prior
        self.concept = concept

    def update_belief(self, action_type, result, response):
        # TODO should this be modeled inside the belief update?
        if action_type != Actions.QUIZ and np.random.random() <= self.transition_noise:
            # transition noise probability: no state change;
            return

        new_belief = self.calc_unscaled_belief(action_type, result, response)

        # TODO does it make sense?
        if sum(new_belief) == 0:
            # quiz response inconsistent with previous state, calc only based on quiz now
            print("Inconsistent quiz response")
            self.belief_state[:] = 1
            new_belief = self.calc_unscaled_belief(action_type, result, response)

            if sum(new_belief) == 0:
                # still 0 means, invalid response; reset to prior probs
                new_belief = self.prior.copy()

        new_belief /= sum(new_belief)

        # is prior updated in every step??
        self.belief_state = new_belief

    def calc_unscaled_belief(self, action_type, result, response):
        new_belief = np.zeros_like(self.belief_state)
        for i, concept in enumerate(self.concept.get_concept_space()):
            concept_val = self.concept.evaluate_concept(result, concept)

            p_s = 0
            p_z = 0
            if action_type == Actions.EXAMPLE:
                # TODO do I need to calculate the probability of the learners concept
                #  being already consistent with the new action somehow?

                # TODO does prior from the paper here refer to the initial prior,
                #  or the prior previous to the current action?
                p_s = self.prior[i]
                if concept_val == result[1]:
                    p_z = 1 - self.production_noise
                else:
                    p_z = self.production_noise
            elif action_type == Actions.QUIZ:
                if response and concept_val == int(response) and self.belief_state[i] > 0:
                    # the true state of the learner doesn't change. but we can better infer which state he is in now
                    p_s = self.prior[i]
                    p_z = 1
            else:
                # TODO: not sure about this, but otherwise it doesnt make sense
                #  the observation is from the previous state, not from the next state
                #  type question with answer; or should somehow be taken into account that more likely now are
                #  concepts with overlap in old and new state?
                p_s = self.prior[i]
                if concept_val == result[1]:
                    p_z = 1 - self.production_noise
                else:
                    p_z = self.production_noise

            new_belief[i] = p_z * p_s

        return new_belief
