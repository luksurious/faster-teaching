import itertools
import numpy as np

from collections import deque

from concepts.concept_base import ConceptBase
from learners.sim_memoryless_learner import SimMemorylessLearner


class SimDiscreteLearner(SimMemorylessLearner):
    def __init__(self, concept: ConceptBase, number_range: list, memory_size: int = 2):
        super().__init__(concept, number_range)

        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

        self.transition_noise = 0.34  # pretty high
        self.production_noise = 0.046

        self.mode = "stoch"

    def update_state(self, example):
        if np.random.random() < self.transition_noise:
            # TODO should also memory be ignored?
            # ignore change
            return

        if self.mode == "pair":
            self.find_match_by_pair(example)
        else:
            self.find_stochastically(example)

    def find_stochastically(self, example):
        concepts_results = np.array([self.concept.evaluate_concept(example, c) for c in self.concept_space])
        consistent_concepts_filter = concepts_results == example[1]
        consistent_concepts_prob = self.prior_distribution[consistent_concepts_filter]
        consistent_concepts = self.concept_space[consistent_concepts_filter]

        for memory_item in self.memory:
            if memory_item[1] is None:
                continue

            concepts_results = np.array([self.concept.evaluate_concept(memory_item, c) for c in consistent_concepts])
            consistent_concepts_filter = concepts_results == memory_item[1]
            consistent_concepts_prob = consistent_concepts_prob[consistent_concepts_filter]
            consistent_concepts = consistent_concepts[consistent_concepts_filter]

        consistent_concepts_prob /= np.sum(consistent_concepts_prob)
        self.concept_belief = consistent_concepts[np.random.choice(range(len(consistent_concepts)),
                                                                   p=consistent_concepts_prob)]

    def find_match_by_pair(self, example):
        # find pairs matching example
        possible_pairs = self.generate_possible_pairs(example[1])
        found_match = self.find_match_with_memory(example, possible_pairs)
        if not found_match:
            # starting concept was incorrect
            # reset values
            original_values = self.concept_belief.copy()
            # TODO or would it be better if the numbers were reset one by one?
            self.concept_belief = [-1] * self.problem_len
            found_match = self.find_match_with_memory(example, possible_pairs)

            if not found_match:
                # inconsistent memory?
                raise Exception("Unable to find logical match")

    def find_match_with_memory(self, example, possible_pairs):
        original_values = self.concept_belief.copy()

        # iterate (randomly) until a pair and replacements are found that match memory
        found_match = False
        for pair in possible_pairs:
            self.update_values_with_pair(example[0], pair)

            num_reassign, refill_idx = self.get_idx_val_to_fill()

            # get all refill combinations
            refill_combs = itertools.permutations(num_reassign, len(refill_idx))

            found_match = self.try_refill_combinations(refill_combs, refill_idx)

            if found_match:
                break
            else:
                self.concept_belief = original_values.copy()

        return found_match

    def try_refill_combinations(self, refill_combs, refill_idx):
        found_match = False

        for refill_comb in refill_combs:
            refill_comb = list(refill_comb)

            for i in refill_idx:
                self.concept_belief[i] = refill_comb.pop(0)

            fits_memory = True
            for memory_item in self.memory:
                if not self.evaluate_memory(memory_item):
                    fits_memory = False
                    break

            if fits_memory:
                found_match = True
                break

        return found_match

    def evaluate_memory(self, memory_item):
        if memory_item[1] is None:
            # ignore quizzes
            return True

        return self.self_evaluate(memory_item[0]) == memory_item[1]

    def finish_action(self, action_data):
        self.memory.append(action_data)
