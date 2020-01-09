from concepts.concept_base import ConceptBase
from learner_models.model_base import ModelBase
from actions import Actions

import math
import numpy as np


class MemorylessModel(ModelBase):

    def __init__(self, states, prior, concept: ConceptBase):
        super().__init__(states, prior, concept)

        self.transition_noise = 0.15
        self.production_noise = 0.019

        self.state_matches = {}

        # not needed?
        if False:
            max_number = max(concept.numbers)
            numbers_to_assign = len(concept.numbers) - 2
            problem_len = len(concept.letters) - 2
            for n in range(1, max_number + max_number - 1):
                permutations = math.factorial(numbers_to_assign) / math.factorial(numbers_to_assign - problem_len)
                # number of non-matching states per possible observation
                self.state_matches[n] = (permutations * self.count_possible_pairs(n, max_number)) - len(self.states)

    def observation_model(self, observation, new_state, action_type, action):
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

        if self.concept.evaluate_concept(action, new_state) == action[1]:
            p_s += self.prior[new_idx]  # TODO assumption: uniform prior, negligible

        return p_s

    def calc_observation_prob(self, action_type, concept, i, result, response):
        concept_val = self.concept.evaluate_concept(result, concept)

        if action_type == Actions.EXAMPLE:
            # TODO do I need to calculate the probability of the learners concept
            #  being already consistent with the new action somehow?
            if concept_val == result[1]:
                p_z = 1 - self.production_noise
            else:
                p_z = self.production_noise
        elif action_type == Actions.QUIZ:
            if response and concept_val == int(response) and self.belief_state[i] > 0:
                # the true state of the learner doesn't change. but we can better infer which state he is in now
                p_z = 1  # production noise?
        else:
            # TODO: not sure about this, but otherwise it doesnt make sense
            #  the observation is from the previous state, not from the next state
            #  type question with answer; or should somehow be taken into account that more likely now are
            #  concepts with overlap in old and new state?
            if concept_val == result[1]:
                p_z = 1 - self.production_noise
            else:
                p_z = self.production_noise

        return p_z

    def calc_state_prob(self, action_type, concept, i, result, response):
        concept_val = self.concept.evaluate_concept(result, concept)

        if action_type == Actions.EXAMPLE:
            # TODO do I need to calculate the probability of the learners concept
            #  being already consistent with the new action somehow?
            p_s = self.prior[i]
        elif action_type == Actions.QUIZ:
            if response and concept_val == int(response) and self.belief_state[i] > 0:
                # the true state of the learner doesn't change. but we can better infer which state he is in now
                p_s = self.prior[i]
        else:
            # TODO: not sure about this, but otherwise it doesnt make sense
            #  the observation is from the previous state, not from the next state
            #  type question with answer; or should somehow be taken into account that more likely now are
            #  concepts with overlap in old and new state?
            p_s = self.prior[i]

        return p_s

    @staticmethod
    def count_possible_pairs(result, max_number_range):
        """
        Duplicated for convenience
        """
        possible_pairs = []

        for i in range(int(result) + 1):
            pair = (i, int(result) - i)
            if max(pair[0], pair[1]) > max_number_range:
                continue

            if pair[0] == pair[1]:
                continue

            possible_pairs.append(pair)

        return len(possible_pairs)

    def see_action(self, action_type, action):
        pass
