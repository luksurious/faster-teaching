import itertools
from collections import deque

from concepts.concept_base import ConceptBase
from learner_models.memoryless import MemorylessModel
from actions import Actions

import numpy as np


class DiscreteMemoryModel(MemorylessModel):

    def __init__(self, belief_state, prior, concept: ConceptBase, memory_size: int, verbose: bool = True):
        super().__init__(belief_state, prior, concept, verbose)

        # TODO incorporate memory into states?
        # expand regular states with memory combinations
        # expand prior accordingly
        # self.generate_memory_states(concept, memory_size)

        # TODO devolves into asking only quizzes at some point?

        self.transition_noise = 0.34 / self.prior_pos_len  # pretty high
        self.production_noise = 0.046

        self.memory_size = memory_size

        self.memory = deque(maxlen=memory_size)

    def generate_memory_states(self, concept, memory_size):
        concept_actions = concept.get_rl_actions()
        teaching_actions = Actions.all()
        action_combinations = list(itertools.product(concept_actions, teaching_actions))
        memory_combinations = list(itertools.product(action_combinations, repeat=memory_size))
        memory_combinations += list(itertools.product([None], action_combinations + [None]))  # add half-empty memory

        combined_prior = []
        combined_belief = []
        combined_states = []
        for state_i, state in enumerate(self.states):
            for memory_state in memory_combinations:
                combined_states.append((state, memory_state))
                combined_prior.append(self.prior[state_i])
                combined_belief.append(self.belief_state[state_i])

        self.concept_prior = self.prior
        self.concept_states = self.states

        # results in 670k combinations... seems unreasonable to work with
        self.prior = np.array(combined_prior)
        self.states = np.array(combined_states)
        self.belief_state = np.array(combined_belief)

    def see_action(self, action_type, action):
        self.memory.append((action_type, action))

    def get_state(self):
        return self.belief_state.copy(), self.memory.copy()

    def set_state(self, state):
        self.belief_state = state[0].copy()
        self.memory = state[1].copy()

    def calc_belief_updatesX(self, action_type, action, observation):
        new_belief = np.zeros_like(self.belief_state)

        for idx, new_state in enumerate(self.states):
            concept_val = self.concept.evaluate_concept(action, new_state[0], idx)

            p_z = self.observation_model(observation, new_state, action_type, action, concept_val)
            if p_z == 0:
                continue

            p_s = self.transition_model(new_state, idx, action_type, action, concept_val)

            # if DEBUG:
            #     print("S=%s p_z=%.2f p_s=%.2f" % (new_state, p_z, p_s))

            new_belief[idx] = p_z * p_s

        return new_belief

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        """
        Probability of going to new state given an action (and current state)
        """

        if action_type == Actions.QUIZ:
            # no state change expected - but we can rule out states that do not match her response
            # no need for a loop
            return self.belief_state[new_idx]  # transition prob only to same state is 1, only b(s) left in formula

            # If memory is added to the state
            # # check that the last memory item is matching the quiz
            # if new_state[1][-1] == (action, action_type):
            #     # quizzes cannot be ignored
            #     # TODO now I would need to loop to check previous states as well...
            #     return self.belief_state[new_idx]  # transition prob only to same state is 1, only b(s) left in formula
            # else:
            #     pass

        p_s = 0

        # only allow transition if memory matches new state
        if concept_val == action[1] and self.matches_memory(new_state):
            p_s = self.calculate_ps(action, new_idx)

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

