from collections import deque

from concepts.concept_base import ConceptBase
from learner_models.memoryless import MemorylessModel
from actions import Actions


class DiscreteMemoryModel(MemorylessModel):

    def __init__(self, states, prior, concept: ConceptBase, memory_size: int, verbose: bool = True):
        super().__init__(states, prior, concept, verbose)

        # TODO incorporate memory into states?

        self.transition_noise = 0.34 / self.prior_pos_len  # pretty high
        self.production_noise = 0.046

        self.memory_size = memory_size

        self.memory = deque(maxlen=2)

    def see_action(self, action_type, action):
        self.memory.append((action_type, action))

    def get_state(self):
        return self.belief_state.copy(), self.memory.copy()

    def set_state(self, state):
        self.belief_state = state[0].copy()
        self.memory = state[1].copy()

    def is_observation_consistent(self, observation, state, action):
        return self.concept.evaluate_concept(action, state) == observation

    def transition_model(self, new_state, new_idx, action_type, action, concept_val):
        """
        Probability of going to new state given an action (and current state)
        """

        if action_type == Actions.QUIZ:
            # no state change expected - but we can rule out states that do not match her response
            # no need for a loop
            return self.belief_state[new_idx]  # transition prob only to same state is 1, only b(s) left in formula

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

