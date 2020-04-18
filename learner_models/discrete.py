from collections import deque

from concepts.concept_base import ConceptBase, ActionResult
from learner_models.memoryless import MemorylessModel
from actions import Actions

import numpy as np

# TODO Paper note
IGNORE_QUIZ_MEMORY = True


class DiscreteMemoryModel(MemorylessModel):
    def __init__(self, belief_state, prior, concept: ConceptBase, memory_size: int, trans_noise=0.34, prod_noise=0.046,
                 verbose: bool = True):
        super().__init__(belief_state, prior, concept, trans_noise=trans_noise, prod_noise=prod_noise, verbose=verbose)

        # TODO check if still happens: devolves into asking only quizzes at some point?
        self.memory_size = memory_size

        # type: deque[ActionResult]
        self.memory = deque(maxlen=memory_size)

    def reset(self):
        super().reset()
        self.memory = deque(maxlen=self.memory_size)

    def see_action(self, action_type, action):
        if IGNORE_QUIZ_MEMORY and action_type == Actions.QUIZ:
            return

        self.memory.append((action_type, action))

    def get_state(self):
        return self.belief_state.copy(), self.memory.copy()

    def set_state(self, state):
        self.belief_state = state[0].copy()
        self.memory = state[1].copy()

    def __copy__(self):
        model = DiscreteMemoryModel(self.belief_state.copy(), self.prior, self.concept, memory_size=self.memory_size,
                                    trans_noise=self.transition_noise, prod_noise=self.production_noise,
                                    verbose=self.verbose)
        model.memory = self.memory.copy()

        return model

    def find_consistent_states_for_transition(self, action):
        # state might have changed
        consistent_states = self.state_action_values[action[0]] == action[1]

        if len(self.memory) > 0:
            const_indices = np.flatnonzero(consistent_states)
            for idx in const_indices:
                # check if consistent with memory
                consistent_states[idx] = self.matches_memory(self.hypotheses[idx], idx)

        return consistent_states

    # Model for explicit Bayesian formula
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
        # TODO not sure if correct
        if concept_val == action[1] and self.matches_memory(new_state):
            p_s = self.calculate_ps(action, new_idx)

        return p_s

    def matches_memory(self, new_state, idx: int = None):
        matches = True
        for memory_item in self.memory:
            if memory_item[0] == Actions.QUIZ:
                continue

            if self.concept.evaluate_concept(memory_item[1][0], new_state, idx) != memory_item[1][1]:
                matches = False
                break

        return matches
